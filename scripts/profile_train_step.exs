# Full-training-step profile: where does the mamba-1 vs mamba-2 epoch
# gap live? (2.4 vs 4.2 min/epoch; block-level bench cleared the scan —
# bench_ssd_scan_20260721: SSD block FASTER than mamba-1's in isolation.)
#
#   mix run --no-compile scripts/profile_train_step.exs [--steps 30] \
#     [--backbones mamba,mamba_2_seq,mamba_2_c8] [--replays glob,glob]
#
# Uses the REAL drill graph — Imitation trainer, 6-head loss,
# lazy-sliced batches — not an isolated block, so full-graph fusion and
# scheduling effects are included. Per backbone:
#   compile_s     first-step wall time (XLA compile of the full graph)
#   batch_ms      median batch materialization (shared pipeline —
#                 should be backbone-independent; a sanity row)
#   step_ms       median Imitation.train_step (fwd+bwd+optimizer, synced)
#   tail_ms       the epoch-final PARTIAL batch's step (different batch
#                 shape = second compiled program; one-time cost)
#   params        parameter count
#
# Interpretation: if step_ms shows the 1.75x, the gap is in the compiled
# graph (suspect: backward through the unrolled sequential scan /
# O(chunks^2) inter-chunk propagation). If step_ms is equal across
# backbones, the gap is outside the step (recompiles, per-epoch
# overheads) — check compile_s and tail_ms.
#
# GPU-INTENSIVE + compiles graphs: run only with no training live (#67).

alias ExPhil.Training.{Data, Imitation, Output}
alias ExPhil.Data.Peppi

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [steps: :integer, backbones: :string, replays: :string]
  )

steps = opts[:steps] || 30

default_replays =
  "probes/newera8/mamba_full/r13/plain/p*/*.slp"

replay_paths =
  (opts[:replays] || default_replays)
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.filter(fn f -> File.stat!(f).size > 512_000 end)
  |> Enum.take(3)

if replay_paths == [], do: raise("no replays matched")

variants =
  case opts[:backbones] do
    nil -> ["mamba", "mamba_2_seq", "mamba_2_c8"]
    s -> String.split(s, ",", trim: true)
  end

variant_config = fn
  "mamba" -> {:mamba, []}
  "mamba_2_seq" -> {:mamba_2, []}
  "mamba_2_c8" -> {:mamba_2, [chunk_size: 8, training_mode: true]}
  other -> raise "unknown variant #{other} (mamba | mamba_2_seq | mamba_2_c8)"
end

Output.banner("Train-step profile (real drill graph)")
Output.config([
  {"Replays", Enum.map(replay_paths, &Path.basename/1)},
  {"Steps", steps},
  {"Variants", variants}
])

# Shared dataset — the drill pipeline minus relabeling (raw controllers
# as labels; identical tensor shapes and batching path)
frames =
  Enum.flat_map(replay_paths, fn path ->
    {:ok, replay} = Peppi.parse(path)

    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
  end)

Output.puts("#{length(frames)} frames")

dataset =
  frames
  |> Data.from_frames()
  |> Data.precompute_frame_embeddings(use_prev_action: true, prev_action_dropout: 0.0)

time_ms = fn fun ->
  t0 = System.monotonic_time(:microsecond)
  r = fun.()
  {(System.monotonic_time(:microsecond) - t0) / 1000, r}
end

median = fn xs ->
  s = Enum.sort(xs)
  Enum.at(s, div(length(s), 2))
end

results =
  Enum.map(variants, fn vname ->
    {backbone, extra} = variant_config.(vname)
    bb = ExPhil.Training.Config.backbone_defaults(backbone) || []
    window = bb[:window_size] || 60

    Output.puts("")
    Output.puts("#{vname}: building trainer...")

    trainer =
      Imitation.new(
        [
          embed_config: dataset.embed_config,
          use_prev_action: true,
          embed_size: ExPhil.Embeddings.embedding_size(dataset.embed_config),
          temporal: true,
          backbone: backbone,
          window_size: window,
          hidden_size: 256,
          num_layers: bb[:num_layers] || 2,
          state_size: bb[:state_size] || 16,
          expand_factor: bb[:expand_factor] || 2,
          conv_size: bb[:conv_size] || 4,
          learning_rate: 2.0e-4,
          max_grad_norm: 0.5,
          dropout: 0.0,
          label_smoothing: 0.0
        ] ++ extra
      )

    count_params = fn data ->
      walk = fn walk, v ->
        cond do
          is_struct(v, Nx.Tensor) -> Nx.size(v)
          is_struct(v) -> 0
          is_map(v) -> v |> Map.values() |> Enum.map(&walk.(walk, &1)) |> Enum.sum()
          true -> 0
        end
      end

      walk.(walk, data)
    end

    params =
      count_params.(ExPhil.Training.Utils.ensure_model_state(trainer.policy_params).data)

    {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

    batches =
      dataset
      |> Data.batched_sequences(
        batch_size: 64,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: true,
        drop_last: false,
        seed: 42
      )
      |> Enum.to_list()

    # Full batches for steady-state timing; the partial tail separately
    {full, tail} = Enum.split_with(batches, fn b -> Nx.axis_size(b.states, 0) == 64 end)
    Output.puts("#{vname}: #{length(full)} full batches + #{length(tail)} partial")

    # First step = XLA compile of the whole graph
    [b0 | _rest] = full
    {compile_ms, {trainer, _}} = time_ms.(fn ->
      {t, m} = Imitation.train_step(trainer, b0, loss_fn)
      _ = Nx.to_number(m.loss)
      {t, m}
    end)

    # Steady state
    {step_times, trainer} =
      full
      |> Stream.cycle()
      |> Enum.take(steps + 2)
      |> Enum.reduce({[], trainer}, fn batch, {ts, tr} ->
        {ms, {tr2, m}} = time_ms.(fn ->
          {t2, m} = Imitation.train_step(tr, batch, loss_fn)
          _ = Nx.to_number(m.loss)
          {t2, m}
        end)

        {[ms | ts], tr2}
      end)

    # Drop the 2 highest (stragglers/GC) then median
    step_ms = step_times |> Enum.sort() |> Enum.drop(-2) |> median.()

    # Partial-tail batch (second compiled program)
    tail_ms =
      case tail do
        [tb | _] ->
          {ms, _} = time_ms.(fn ->
            {t2, m} = Imitation.train_step(trainer, tb, loss_fn)
            _ = Nx.to_number(m.loss)
            {t2, m}
          end)

          Float.round(ms, 1)

        [] ->
          nil
      end

    # Batch materialization cost (shared pipeline sanity row)
    batch_times =
      dataset
      |> Data.batched_sequences(
        batch_size: 64,
        window_size: window,
        stride: 1,
        lazy: true,
        shuffle: true,
        drop_last: true,
        seed: 43
      )
      |> Stream.map(fn b -> elem(time_ms.(fn -> Nx.to_number(Nx.sum(Nx.slice_along_axis(b.states, 0, 1))) end), 0) end)
      |> Enum.take(10)

    row = %{
      variant: vname,
      params: params,
      compile_s: Float.round(compile_ms / 1000, 1),
      batch_ms: Float.round(median.(batch_times), 1),
      step_ms: Float.round(step_ms, 1),
      tail_ms: tail_ms
    }

    Output.puts(
      "#{vname}: params=#{params} compile=#{row.compile_s}s batch=#{row.batch_ms}ms " <>
        "step=#{row.step_ms}ms tail=#{inspect(row.tail_ms)}ms"
    )

    row
  end)

Output.puts("")
Output.puts("=== summary ===")
ref = Enum.find(results, &(&1.variant == "mamba"))

Enum.each(results, fn r ->
  ratio = if ref && ref.step_ms > 0, do: " (#{Float.round(r.step_ms / ref.step_ms, 2)}x mamba)", else: ""
  Output.puts("  #{String.pad_trailing(r.variant, 12)} step #{r.step_ms}ms#{ratio}  compile #{r.compile_s}s  params #{r.params}")
end)

out = "logs/profile_train_step_#{Date.utc_today() |> Date.to_iso8601(:basic)}.json"
File.write!(out, Jason.encode!(results, pretty: true))
Output.success("profile -> #{out}")
