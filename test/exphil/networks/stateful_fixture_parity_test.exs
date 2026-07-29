defmodule ExPhil.Networks.StatefulFixtureParityTest do
  @moduledoc """
  Bit-parity gate for the Edifice.Stateful step path on FIXTURE-REPLAY
  windows at deployment embedding dimensions (slippi-ai harness parity
  step 2 — HANDOFF_2026-07-28).

  `stateful_step_equivalence_test.exs` pins windowed == stepwise on random
  frames at toy dims (embed 16). This gate re-pins it where it actually has
  to hold: real embedded game states (the multishine tech fixture, embedded
  through the live-agent `Embeddings.Game.embed/4` path) at the default
  embedding size, across multiple rolling window offsets — the exact inputs
  a deployed `--stateful-step` GRU agent would see. embed_path_parity_test
  is the prior art for this bug class: train/deploy path skew is invisible
  to val_loss and only surfaces as live-play pathology.

  Tolerance is ULP-scale, not exact-zero: the windowed forward (Axon
  sequence graph) and the per-frame step (Edifice.Recurrent.step) are
  different compiled graphs, so float reassociation yields ~5e-7 deltas
  (measured 2026-07-29 at toy dims). Structural bugs — wrong state
  threading, layer misrouting, cold-start pad drift — produce deltas of
  ~0.1-1.0, five orders of magnitude above @tol.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Embeddings
  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.Heads
  alias ExPhil.Test.ReplayFixtures
  alias ExPhil.Training.Utils

  @window 16
  @hidden_size 32
  @num_layers 2
  @axis_buckets 16
  @shoulder_buckets 4
  @fixture_frames 40
  @head_names [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

  # ULP-scale accumulation over 16 steps at real dims; see @moduledoc.
  @tol 1.0e-5

  defp fixture_stream(port) do
    config = Embeddings.config()

    states =
      ReplayFixtures.tech_fixture(:multishine, frames: @fixture_frames, period: 8)
      |> Enum.map(fn {gs, _cs} -> gs end)

    # Live-agent embedding path (per-frame embed/4), stacked to {1, T, E}
    states
    |> Enum.map(&Embeddings.Game.embed(&1, nil, port, config: config))
    |> Nx.stack()
    |> Nx.new_axis(0)
  end

  defp build_policy(embed_size, cell_type) do
    model =
      Policy.build_temporal(
        embed_size: embed_size,
        backbone: cell_type,
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        window_size: @window,
        dropout: 0.0,
        axis_buckets: @axis_buckets,
        shoulder_buckets: @shoulder_buckets
      )

    {init_fn, predict_fn} = Utils.build_compiled(model)

    params =
      init_fn.(
        Nx.template({1, @window, embed_size}, :f32),
        Axon.ModelState.empty()
      )

    {params, predict_fn}
  end

  defp build_heads_predict do
    input = Axon.input("features", shape: {nil, @hidden_size})
    model = Heads.build_controller_head(input, @axis_buckets, @shoulder_buckets)
    {_init_fn, predict_fn} = Utils.build_compiled(model)
    predict_fn
  end

  defp raw(%Axon.ModelState{data: data}), do: data

  defp init_state(params, cell_type) do
    Edifice.Recurrent.init_state(raw(params),
      batch_size: 1,
      hidden_size: @hidden_size,
      num_layers: @num_layers,
      cell_type: cell_type
    )
  end

  defp step_all(params, state, frames) do
    0..(Nx.axis_size(frames, 1) - 1)
    |> Enum.reduce({nil, state}, fn t, {_out, st} ->
      frame = frames |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      Edifice.Recurrent.step(raw(params), st, frame)
    end)
  end

  defp max_delta(a, b), do: a |> Nx.subtract(b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  defp head_deltas(tuple_a, tuple_b) do
    Enum.zip([@head_names, Tuple.to_list(tuple_a), Tuple.to_list(tuple_b)])
    |> Enum.map(fn {name, a, b} -> {name, max_delta(a, b)} end)
  end

  for cell_type <- [:gru, :lstm] do
    @cell_type cell_type

    test "#{cell_type}: step path matches windowed forward on every fixture window (bit-parity gate)" do
      port = 2
      stream = fixture_stream(port)
      embed_size = Nx.axis_size(stream, 2)

      {params, full_predict} = build_policy(embed_size, @cell_type)
      heads_predict = build_heads_predict()

      # Rolling offsets cover cycle-aligned AND cycle-straddling windows of
      # the period-8 multishine pattern.
      for offset <- [0, 5, 8, 16, @fixture_frames - @window] |> Enum.uniq() do
        window = Nx.slice_along_axis(stream, offset, @window, axis: 1)

        windowed_logits = full_predict.(params, window)

        state = init_state(params, @cell_type)
        {features, _state} = step_all(params, state, window)
        step_logits = heads_predict.(params, features)

        for {name, delta} <- head_deltas(windowed_logits, step_logits) do
          assert delta < @tol,
                 "#{name} logits diverge between windowed and step path at " <>
                   "window offset #{offset}: max |delta| = #{delta} (tol #{@tol}). " <>
                   "Structural skew — the deployed step path is NOT computing " <>
                   "the trained function."
        end
      end
    end
  end

  test "informational: carried-state drift vs sliding window at fixture end" do
    # NOT a parity bug — a semantic gap this suite documents on purpose. The
    # deployed windowed path truncates history to @window frames (matching
    # how BC training windows are cut); a carried state that never resets
    # has seen the whole game. The drift below measures how different those
    # two functions are on a periodic fixture. If it is ever ~0 for a
    # TRAINED policy, the truncation is behaviorally irrelevant and the
    # step path can carry state across the whole game with no train/deploy
    # divergence; if it is large, deployment should mirror training by
    # bounding effective history (slippi-ai solves this upstream by making
    # the training unroll and the deployment step the same function).
    port = 2
    stream = fixture_stream(port)
    embed_size = Nx.axis_size(stream, 2)

    {params, full_predict} = build_policy(embed_size, :gru)
    heads_predict = build_heads_predict()

    last_window =
      Nx.slice_along_axis(stream, @fixture_frames - @window, @window, axis: 1)

    windowed_logits = full_predict.(params, last_window)

    state = init_state(params, :gru)
    {features, _} = step_all(params, state, stream)
    carried_logits = heads_predict.(params, features)

    deltas = head_deltas(windowed_logits, carried_logits)
    worst = deltas |> Enum.map(&elem(&1, 1)) |> Enum.max()

    IO.puts(
      "[stateful-fixture] carried-state vs sliding-window drift after " <>
        "#{@fixture_frames} frames (random-init GRU): max |delta| = #{worst}"
    )
  end
end
