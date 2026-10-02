defmodule ExPhil.Eval.Calibration do
  @moduledoc """
  Teacher-forced calibration of a windowed imitation policy on held-out
  batches (2026-10-02): when the model says "30 % chance of A", does the
  expert press A 30 % of the time?

  A calibrated model that is SAMPLED reproduces the expert's rates by
  construction, so miscalibration separates "the model's probabilities are
  wrong" from "the probabilities are right and the decode/feedback is the
  problem". Also reports the per-head share of the loss (where the
  remaining error lives) and, when the batch carries the prev-action slot,
  the two conditional button probabilities that govern flicker:
  P(down | up last frame) and P(down | down last frame), model vs expert.

  Everything here is under teacher forcing (the expert's history and the
  expert's same-frame prefix), i.e. the same conditions as validation loss.
  """
  alias ExPhil.Training.Imitation.Loss

  @buttons ~w(a b x y z l r d_up)
  @cats [:main_x, :main_y, :c_x, :c_y, :shoulder]
  @bins 10

  @doc "Run over `batches` (maps with :states, :actions) with a trainer's predict_fn/params/config."
  def run(trainer, batches) do
    config = trainer.config
    head = Loss.forward_head(config)
    offset = if config[:use_prev_action], do: prev_offset(config, trainer)

    acc =
      Enum.reduce(batches, %{n: 0, buttons: %{}, cats: %{}, cond: %{}}, fn batch, acc ->
        logits = trainer.predict_fn.(trainer.policy_params, Loss.policy_forward_inputs(head, true, batch.states, batch.actions))
        {b_l, mx, my, cx, cy, sh} = logits
        host = &(Nx.backend_copy(&1, Nx.BinaryBackend))

        probs = b_l |> Nx.as_type(:f32) |> Nx.sigmoid() |> host.() |> Nx.to_list()
        targets = batch.actions.buttons |> host.() |> Nx.to_list()
        prev = if offset, do: prev_buttons(batch.states, offset) |> host.() |> Nx.to_list()

        acc = %{acc | n: acc.n + length(probs), buttons: add_buttons(acc.buttons, probs, targets)}
        acc = if prev, do: %{acc | cond: add_cond(acc.cond, probs, targets, prev)}, else: acc

        Enum.zip(@cats, [mx, my, cx, cy, sh])
        |> Enum.reduce(acc, fn {name, l}, acc ->
          p = l |> Nx.as_type(:f32) |> then(&Axon.Activations.softmax(&1, axis: -1)) |> host.()
          conf = p |> Nx.reduce_max(axes: [-1]) |> Nx.to_list()
          pred = p |> Nx.argmax(axis: -1) |> Nx.to_list()
          tgt = batch.actions[name] |> host.()
          p_true = p |> Nx.take_along_axis(Nx.new_axis(tgt, -1), axis: -1) |> Nx.squeeze(axes: [-1]) |> Nx.to_list()
          %{acc | cats: Map.update(acc.cats, name, add_cat(new_cat(), conf, pred, Nx.to_list(tgt), p_true), &add_cat(&1, conf, pred, Nx.to_list(tgt), p_true))}
        end)
      end)

    summarize(acc)
  end

  # ---- accumulation --------------------------------------------------------------

  defp prev_offset(config, trainer) do
    config[:prev_action_offset] ||
      hd(ExPhil.Interp.Attribution.prev_action_dim_range(config: trainer.embed_config))
  end

  defp prev_buttons(states, offset) do
    states
    |> Nx.slice_along_axis(Nx.axis_size(states, 1) - 1, 1, axis: 1)
    |> Nx.squeeze(axes: [1])
    |> Nx.slice_along_axis(offset, 8, axis: 1)
  end

  defp bin(p), do: min(trunc(p * @bins), @bins - 1)

  defp add_buttons(acc, probs, targets) do
    Enum.zip(probs, targets)
    |> Enum.reduce(acc, fn {ps, ys}, acc ->
      Enum.zip([@buttons, ps, ys])
      |> Enum.reduce(acc, fn {b, p, y}, acc ->
        nll = -:math.log(max(if(y == 1, do: p, else: 1.0 - p), 1.0e-7))
        Map.update(acc, {b, bin(p)}, {1, p, y, nll}, fn {n, sp, sy, sl} -> {n + 1, sp + p, sy + y, sl + nll} end)
      end)
    end)
  end

  defp add_cond(acc, probs, targets, prev) do
    Enum.zip([probs, targets, prev])
    |> Enum.reduce(acc, fn {ps, ys, vs}, acc ->
      Enum.zip([@buttons, ps, ys, vs])
      |> Enum.reduce(acc, fn {b, p, y, v}, acc ->
        Map.update(acc, {b, v > 0.5}, {1, p, y}, fn {n, sp, sy} -> {n + 1, sp + p, sy + y} end)
      end)
    end)
  end

  defp new_cat, do: %{bins: %{}, n: 0, correct: 0, nll: 0.0}

  defp add_cat(c, conf, pred, tgt, p_true) do
    Enum.zip([conf, pred, tgt, p_true])
    |> Enum.reduce(c, fn {cf, pr, t, pt}, c ->
      hit = if pr == t, do: 1, else: 0

      %{c |
        n: c.n + 1,
        correct: c.correct + hit,
        nll: c.nll - :math.log(max(pt, 1.0e-7)),
        bins: Map.update(c.bins, bin(cf), {1, cf, hit}, fn {n, sc, sh} -> {n + 1, sc + cf, sh + hit} end)}
    end)
  end

  # ---- summary -------------------------------------------------------------------

  defp summarize(%{n: 0}), do: %{samples: 0}

  defp summarize(acc) do
    r = &Float.round(&1 * 1.0, 4)

    buttons =
      Map.new(@buttons, fn b ->
        rows = for i <- 0..(@bins - 1), v = acc.buttons[{b, i}], v != nil, do: {i, v}
        n = rows |> Enum.map(fn {_, {n, _, _, _}} -> n end) |> Enum.sum()
        ece = Enum.sum(Enum.map(rows, fn {_, {k, sp, sy, _}} -> k / n * abs(sp / k - sy / k) end))
        sum = fn f -> rows |> Enum.map(fn {_, v} -> f.(v) end) |> Enum.sum() end

        {b,
         %{
           expert_rate: r.(sum.(fn {_, _, sy, _} -> sy end) / n),
           model_mean_p: r.(sum.(fn {_, sp, _, _} -> sp end) / n),
           ece: r.(ece),
           nll: r.(sum.(fn {_, _, _, sl} -> sl end) / n),
           reliability: Enum.map(rows, fn {i, {k, sp, sy, _}} -> %{bin: i, n: k, model_p: r.(sp / k), expert_rate: r.(sy / k)} end)
         }}
      end)

    cats =
      Map.new(acc.cats, fn {name, c} ->
        ece = c.bins |> Map.values() |> Enum.map(fn {k, sc, sh} -> k / c.n * abs(sc / k - sh / k) end) |> Enum.sum()
        mean_conf = (c.bins |> Map.values() |> Enum.map(fn {_, sc, _} -> sc end) |> Enum.sum()) / c.n
        {name, %{accuracy: r.(c.correct / c.n), mean_confidence: r.(mean_conf), ece: r.(ece), nll: r.(c.nll / c.n)}}
      end)

    conditional =
      if acc.cond == %{} do
        nil
      else
        Map.new(@buttons, fn b ->
          row = fn held ->
            case acc.cond[{b, held}] do
              nil -> nil
              {n, sp, sy} -> %{n: n, model_p_down: r.(sp / n), expert_rate_down: r.(sy / n)}
            end
          end

          {b, %{prev_up: row.(false), prev_down: row.(true)}}
        end)
      end

    button_nll = buttons |> Map.values() |> Enum.map(& &1.nll) |> Enum.sum()
    cat_nll = Map.new(cats, fn {k, v} -> {k, v.nll} end)
    total = button_nll + (cat_nll |> Map.values() |> Enum.sum())

    %{
      samples: acc.n,
      loss_by_head: Map.merge(%{buttons: r.(button_nll), total: r.(total)}, cat_nll),
      buttons: buttons,
      categorical: cats,
      buttons_given_previous: conditional
    }
  end
end
