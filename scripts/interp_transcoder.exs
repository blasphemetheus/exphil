# Flagship Stage 1 v2 (attribution-graph arc, 2026-08-06): TRUNK-UPDATE
# transcoder with ground-truth-scored features.
#
# v1 (input_t -> trunk_t) FAILED its R^2 gate at -0.01 — the trunk state
# integrates 60 frames of recurrent history, so a single-frame embedding
# cannot predict it (the P1 compression finding, re-encountered as a
# design error). v2 targets the trunk's own UPDATE MAP:
#
#     x = [trunk_t (256) ++ input_{t+1} (336)]  ->  y = trunk_{t+1} (256)
#
# — the GRU's actual computation, so reconstruction is a fair target, and
# the dictionary factors into state-carry vs input-driven features:
# exactly the vocabulary Stage 2's decision tracing needs. Pairs are
# built per replay (never across boundaries). Features are scored against
# the GroundTruth dictionary (labels aligned at t+1) exactly like the P6
# SAE run (strictly-binary labels, standardization), plus reconstruction
# R^2 — the graph-basis quality gate.
#
# The fit (params + standardization constants) is SAVED for Stage 2
# (frozen-gate attribution reads this dictionary).
#
#   mix run scripts/interp_transcoder.exs \
#     [--policies "checkpoints/ms_g10b_human.bin"] [--delay-id 3] \
#     [--replays "glob1,glob2"] [--dict-size 2048] [--top-k 16] \
#     [--steps 500] [--out eval_runs/interp/transcoder_scores.json]

require Logger
Logger.configure(level: :warning)

alias Edifice.Interpretability.SAETrainer
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policies: :string,
      replays: :string,
      delay_id: :integer,
      dict_size: :integer,
      top_k: :integer,
      steps: :integer,
      lr: :float,
      out: :string
    ]
  )

policies =
  (opts[:policies] || "checkpoints/ms_g10b_human.bin")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)

# Default corpus: clean FD cycling + the YS contrastive pair + the 0805
# human/netplay games — covers the phenomena Stage 2 wants to trace
# (platform silence, fight states, delay modes) with exact labels.
default_replays =
  "eval_runs/0804_cycle3b_stand/r*.slp," <>
    "test/fixtures/replays/ys_multishine_good_2026-08-04.slp," <>
    "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp," <>
    "eval_runs/0805_human_g10b/**/*.slp," <>
    "eval_runs/0805_direct_g10b/**/*.slp," <>
    "eval_runs/0805_direct_g10b_d4/**/*.slp"

replays =
  (opts[:replays] || default_replays)
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.sort()
  |> Enum.filter(&match?({:ok, _}, ExPhil.Data.Peppi.parse(&1)))

delay_id = opts[:delay_id] || 3
dict_size = opts[:dict_size] || 2048
top_k = opts[:top_k] || 16
steps = opts[:steps] || 500
out_path = opts[:out] || "eval_runs/interp/transcoder_scores.json"

Output.banner("Stage 1: input->trunk transcoder vs ground truth")
Output.puts("#{length(policies)} policies x #{length(replays)} replays, dict #{dict_size} k=#{top_k}")

standardize = fn t ->
  mean = Nx.mean(t, axes: [0], keep_axes: true)
  std = Nx.standard_deviation(t, axes: [0], keep_axes: true)
  {Nx.divide(Nx.subtract(t, mean), Nx.add(std, 1.0e-6)), mean, std}
end

results =
  for path <- policies do
    seed = Path.basename(path, ".bin")
    trunk = Activations.load_trunk(path)
    window = trunk.window

    # Per-replay pair construction (never across replay boundaries):
    # activation row i = trunk state at frame i+window-1; its update pair
    # is x = [act_i ++ emb_{i+window}] -> y = act_{i+1}, labels at i+1.
    {x_parts, y_parts, label_parts} =
      replays
      |> Enum.map(fn rp ->
        cap = Activations.capture_replay(trunk, rp, delay_id: delay_id)

        frames =
          rp
          |> Path.expand()
          |> then(fn p -> {:ok, r} = ExPhil.Data.Peppi.parse(p); r end)
          |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
          |> Enum.reject(&(&1.game_state.frame < 0))

        ds = Activations.embed_frames(frames, trunk.config, delay_id: delay_id)
        emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
        acts = Nx.backend_transfer(cap.activations, Nx.BinaryBackend)
        n_windows = Nx.axis_size(acts, 0)

        # pairs 0..n_windows-2: state_t, next input (frame index t+window), next state
        act_t = Nx.slice_along_axis(acts, 0, n_windows - 1, axis: 0)
        emb_next = Nx.slice_along_axis(emb, window, n_windows - 1, axis: 0)
        act_next = Nx.slice_along_axis(acts, 1, n_windows - 1, axis: 0)

        labels =
          Map.new(cap.labels, fn {k, t} ->
            {k, Nx.slice_along_axis(t, 1, n_windows - 1, axis: 0)}
          end)

        # Target the update RESIDUAL (act_{t+1} - act_t), not the raw next
        # state: the GRU update is dominated by near-identity state-carry,
        # and k active features cannot represent a rank-256 copy — the
        # same reason CLTs sit on residual streams (v2 R^2 was -0.03 on
        # the raw target; the copy has to be handled by the graph's
        # residual edge, the transcoder explains only the change).
        {Nx.concatenate([act_t, emb_next], axis: 1), Nx.subtract(act_next, act_t), labels}
      end)
      |> Enum.reduce({[], [], []}, fn {x, y, l}, {xs, ys, ls} ->
        {[x | xs], [y | ys], [l | ls]}
      end)

    x_raw = x_parts |> Enum.reverse() |> Nx.concatenate(axis: 0) |> Nx.backend_copy(EXLA.Backend)
    y_raw = y_parts |> Enum.reverse() |> Nx.concatenate(axis: 0) |> Nx.backend_copy(EXLA.Backend)

    labels =
      label_parts
      |> Enum.reverse()
      |> Enum.reduce(fn m, acc ->
        Map.new(acc, fn {k, t} -> {k, Nx.concatenate([t, m[k]], axis: 0)} end)
      end)

    {x, x_mean, x_std} = standardize.(x_raw)
    {y, y_mean, y_std} = standardize.(y_raw)
    {n, d_in} = Nx.shape(x)
    {_, d_out} = Nx.shape(y)
    Output.puts("#{seed}: {#{n}, #{d_in}} -> {#{n}, #{d_out}} pairs")

    fit =
      SAETrainer.fit(:transcoder, x,
        targets: y,
        input_size: d_in,
        output_size: d_out,
        dict_size: dict_size,
        top_k: top_k,
        steps: steps,
        lr: opts[:lr] || 1.0e-3,
        batch_size: 16_384,
        compiler: EXLA
      )

    # history is reverse-chronological? (built by prepend) — report both ends
    {h_first, h_last} = {List.last(fit.history), hd(fit.history)}

    Output.puts(
      "  fit done: dead=#{fit.dead_count} loss #{Float.round(h_first, 4)} -> #{Float.round(h_last, 4)} " <>
        "(standardized-target var ~1.0)"
    )

    model =
      Edifice.Interpretability.Transcoder.build(
        Keyword.put(fit.build_opts, :output, :container)
      )

    {_init, predict} = Axon.build(model, mode: :inference)
    input_key = model |> Axon.get_inputs() |> Map.keys() |> hd()
    out = predict.(fit.params, %{input_key => x})

    # Reconstruction R^2 — the graph-basis quality gate
    resid = Nx.subtract(y, out.reconstruction)
    r2 = 1.0 - Nx.to_number(Nx.sum(Nx.pow(resid, 2))) / Nx.to_number(Nx.sum(Nx.pow(y, 2)))
    Output.puts("  reconstruction R^2 = #{Float.round(r2, 4)}")

    hidden = out.hidden
    fired = Nx.greater(hidden, 0.0)
    fire_rate = fired |> Nx.mean(axes: [0]) |> Nx.to_flat_list()

    # STRICTLY binary labels (v2 lesson: multi-valued u8 -> F1 > 1 nonsense)
    labels =
      labels
      |> Enum.filter(fn {_k, t} ->
        Nx.type(t) == {:u, 8} and Nx.to_number(Nx.reduce_max(t)) <= 1
      end)
      |> Map.new(fn {k, t} -> {k, Nx.backend_copy(t, EXLA.Backend)} end)

    scored =
      for {label, yl} <- labels do
        base = Nx.to_number(Nx.mean(yl))
        yf = Nx.as_type(yl, :f32)
        fired_f = Nx.as_type(fired, :f32)
        n_fire = Nx.sum(fired_f, axes: [0])
        tp = Nx.dot(yf, fired_f)
        prec = Nx.divide(tp, Nx.max(n_fire, 1))
        rec = Nx.divide(tp, max(Nx.to_number(Nx.sum(yf)), 1))
        f1 = Nx.divide(Nx.multiply(2, Nx.multiply(prec, rec)), Nx.max(Nx.add(prec, rec), 1.0e-6))

        best = f1 |> Nx.argmax() |> Nx.to_number()

        %{
          label: label,
          base_rate: Float.round(base, 4),
          best_feature: best,
          f1: Float.round(Nx.to_number(f1[best]), 3),
          precision: Float.round(Nx.to_number(prec[best]), 3),
          recall: Float.round(Nx.to_number(rec[best]), 3),
          fire_rate: Float.round(Enum.at(fire_rate, best), 4)
        }
      end
      |> Enum.sort_by(&(-&1.f1))

    for s <- scored do
      Output.puts(
        "  #{String.pad_trailing(to_string(s.label), 22)} f#{String.pad_leading(to_string(s.best_feature), 4)} " <>
          "F1=#{s.f1} P=#{s.precision} R=#{s.recall} (base #{s.base_rate}, fires #{s.fire_rate})"
      )
    end

    # Persist the dictionary for Stage 2 (frozen-gate attribution)
    dict_path = "eval_runs/interp/transcoder_#{seed}.bin"

    File.write!(
      dict_path,
      :erlang.term_to_binary(%{
        build_opts: fit.build_opts,
        params: fit.params,
        x_mean: Nx.backend_transfer(x_mean, Nx.BinaryBackend),
        x_std: Nx.backend_transfer(x_std, Nx.BinaryBackend),
        y_mean: Nx.backend_transfer(y_mean, Nx.BinaryBackend),
        y_std: Nx.backend_transfer(y_std, Nx.BinaryBackend),
        r2: r2,
        replays: replays,
        delay_id: delay_id
      })
    )

    Output.success("  dictionary -> #{dict_path}")
    %{policy: seed, r2: r2, dead: fit.dead_count, scores: scored}
  end

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(%{results: results}, pretty: true))
Output.success("Scores -> #{out_path}")
