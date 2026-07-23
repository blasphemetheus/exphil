defmodule ExPhil.Training.ProbeRegularizer do
  @moduledoc """
  Probe-as-regularizer (r15, the steering-decided lever): penalize the
  trunk representation's alignment with the shield-lock direction DURING
  training, instead of projecting it out at inference.

  The steering A/B (commit 74e27bc) proved the direction causal: removing
  it at inference (alpha=1.0) produced the first 9/9 card, and alpha=0.5
  was WORSE than baseline — the axis is all-or-nothing, so the honest
  training-time analog is suppression, not attenuation.

  Because each DAgger round trains FRESH weights (the resume fingerprint
  refuses cross-recipe blends), a fixed r14 steer vector is in the wrong
  basis. The direction is therefore refit ONLINE every few epochs from
  the network's own current activations:

      v = normalize(mean(trunk | shielding) - mean(trunk | grounded-free))

  — the same class-mean contrast `scripts/extract_steer_vector.exs` uses,
  with the same shield family (178..182). Between refits the direction is
  constant and flows into the jitted loss as an ARGUMENT (never a closure
  capture — GOTCHAS closure-tensor rule); the penalty is
  `weight * mean((h . v)^2)`. A zero vector (pre-first-refit state) makes
  the penalty exactly 0, so training starts as plain BC.

  Like INLP, the contrast self-limits: as the penalty flattens the
  direction, the refit contrast shrinks toward zero and the regularizer
  fades out rather than fighting a moving target.
  """

  alias ExPhil.Networks
  alias ExPhil.Training.{Data, Utils}

  # Shield family per the steering extraction (GuardOn 178, Guard 179,
  # GuardOff 180, GuardSetOff 181, GuardReflect 182)
  @shield_states MapSet.new(178..182)

  @doc """
  Per-frame 0/1 shield labels for the learner side (players[1], the
  drill's normalized port), aligned with `dataset.frames`.
  """
  def frame_labels(frames) do
    Enum.map(frames, fn f ->
      pl = f.game_state.players[1]
      if pl && MapSet.member?(@shield_states, trunc(pl.action || 0)), do: 1, else: 0
    end)
  end

  @doc """
  Build the trunk-only predict fn for a trainer config. The trunk model
  shares parameter names with the full policy, so the trainer's own
  `policy_params` drive it directly (same mechanism as
  `ExPhil.Interp.Activations.load_trunk/2`).
  """
  def build_trunk_fn(config) do
    trunk =
      Networks.Policy.build_temporal_trunk(
        embed_size: config.embed_size,
        backbone: config.backbone,
        window_size: config.window_size,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        hidden_size: config.hidden_size,
        num_layers: config.num_layers,
        state_size: config.state_size,
        expand_factor: config.expand_factor,
        conv_size: config.conv_size,
        dropout: config.dropout
      )

    {_init_fn, predict_fn} = Utils.build_compiled(trunk, mode: :inference)
    predict_fn
  end

  @doc """
  Class-mean contrast direction from trunk activations `h` `{n, hidden}`
  and 0/1 `labels` (list or `{n}` tensor).

  Returns `{:ok, unit_v}`, or `{:error, {:insufficient, pos, neg}}` when
  either class has fewer than `:min_class` rows (default 64), or
  `{:error, :zero_contrast}` when the class means coincide.
  """
  def fit_direction(h, labels, opts \\ []) do
    min_class = Keyword.get(opts, :min_class, 64)

    h = Nx.as_type(h, :f32)
    labels = if is_list(labels), do: Nx.tensor(labels, type: :f32), else: Nx.as_type(labels, :f32)

    n = Nx.axis_size(h, 0)
    pos = labels |> Nx.sum() |> Nx.to_number() |> trunc()
    neg = n - pos

    cond do
      pos < min_class or neg < min_class ->
        {:error, {:insufficient, pos, neg}}

      true ->
        # Weighted row-sums give the class means without materializing masks
        mean_pos = Nx.dot(Nx.divide(labels, pos), h)
        mean_neg = Nx.dot(Nx.divide(Nx.subtract(1.0, labels), neg), h)
        diff = Nx.subtract(mean_pos, mean_neg)
        norm = diff |> Nx.LinAlg.norm() |> Nx.to_number()

        # Floor at 1e-4, not epsilon: f32 weighted-mean roundoff produces
        # ~1e-7 phantom contrasts between IDENTICAL classes (1/90 * 90 != 1),
        # and normalizing those yields a garbage direction. Real class
        # contrasts in trunk space are orders of magnitude above this.
        if norm < 1.0e-4 do
          {:error, :zero_contrast}
        else
          {:ok, Nx.divide(diff, norm)}
        end
    end
  end

  @doc """
  The regularization term: `mean((h . v)^2)` in f32. Pure tensor math —
  safe inside `Nx.Defn.value_and_grad`. A zero `v` yields exactly 0.
  """
  def alignment_penalty(h, v) do
    h = Nx.as_type(h, :f32)
    v = Nx.as_type(v, :f32)

    h
    |> Nx.dot(v)
    |> Nx.pow(2)
    |> Nx.mean()
  end

  @doc """
  Refit the trainer's probe direction from current activations.

  Streams ordered (shuffle: false) lazy sequence batches, keeps every Nth
  batch to cap the sample near `:target_rows` (default 4096), runs the
  trunk on the trainer's params, and fits the contrast against
  `frame_labels` (per-frame list aligned with `dataset.frames`; a
  window's label is its supervised last frame's).

  Returns `{trainer, stats}` — trainer unchanged when the fit fails
  (insufficient class rows / zero contrast), so early epochs without
  shield frames just stay on the previous (or zero) direction.
  """
  def refit(trainer, dataset, frame_labels, opts \\ []) do
    window = Keyword.get(opts, :window_size) || trainer.config.window_size
    stride = Keyword.get(opts, :stride, 1)
    batch_size = Keyword.get(opts, :batch_size, 256)
    target_rows = Keyword.get(opts, :target_rows, 4096)
    min_class = Keyword.get(opts, :min_class, 64)

    labels_tuple = :erlang.list_to_tuple(frame_labels)
    n_labels = tuple_size(labels_tuple)

    num_sequences = div(dataset.size - window, stride) + 1
    every = max(div(num_sequences, target_rows), 1)

    # Every-Nth-SEQUENCE systematic sample (2026-07-23). The previous
    # every-Nth-BATCH scheme kept a few CONTIGUOUS index clumps; at r16
    # scale (every=495 -> 16 clumps) they landed in shield-free stretches
    # and the regularizer silently no-opped all run (4 shield rows from a
    # 3.9%-shield pool). Same trunk cost, uniform coverage.
    {batches, indices} =
      Data.strided_sequence_batches(dataset,
        batch_size: batch_size,
        window_size: window,
        stride: stride,
        every: every,
        limit: target_rows
      )

    idx_tuple = :erlang.list_to_tuple(indices)
    n_idx = tuple_size(idx_tuple)

    {h_parts, label_parts} =
      batches
      |> Stream.with_index()
      |> Enum.reduce({[], []}, fn {batch, j}, {hs, ls} ->
        h =
          trainer.trunk_predict_fn.(
            Utils.ensure_model_state(trainer.policy_params),
            batch.states
          )

        rows = Nx.axis_size(batch.states, 0)
        base = j * batch_size

        labels =
          for k <- 0..(rows - 1) do
            row = base + k

            with true <- row < n_idx,
                 frame_idx = elem(idx_tuple, row) * stride + window - 1,
                 true <- frame_idx < n_labels do
              elem(labels_tuple, frame_idx)
            else
              _ -> 0
            end
          end

        # BinaryBackend copy: eager EXLA buffers accumulated across a whole
        # sweep otherwise linger until BEAM GC (the step-~3000 OOM pattern)
        {[Nx.backend_copy(Nx.as_type(h, :f32), Nx.BinaryBackend) | hs], [labels | ls]}
      end)

    if h_parts == [] do
      {trainer, %{rows: 0, shield_rows: 0, every: every, refit: false, reason: :no_batches}}
    else
      finish_refit(trainer, h_parts, label_parts, every, min_class)
    end
  end

  defp finish_refit(trainer, h_parts, label_parts, every, min_class) do
    h = h_parts |> Enum.reverse() |> Nx.concatenate()
    labels = label_parts |> Enum.reverse() |> List.flatten()
    pos = Enum.sum(labels)

    stats = %{rows: length(labels), shield_rows: pos, every: every}

    case fit_direction(h, labels, min_class: min_class) do
      {:ok, v} ->
        {%{trainer | probe_direction: v}, Map.put(stats, :refit, true)}

      {:error, reason} ->
        {trainer, Map.merge(stats, %{refit: false, reason: reason})}
    end
  end
end
