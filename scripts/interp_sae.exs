# P6 SAE half (task #14): ground-truth-scored dictionary features.
#
# Fit a BatchTopK SAE on trunk activations over a replay corpus, then
# score every learned feature against the GroundTruth binary label
# dictionary: precision/recall/F1 of "feature fires" as a predictor of
# each label. Melee's edge over language-domain SAE work: the scores are
# exact and falsifiable.
#
#   mix run scripts/interp_sae.exs \
#     [--policies "checkpoints/mewtwo_combo_newera_r10_policy.bin"] \
#     [--replays "~/Slippi/Game_20260714*.slp"] \
#     [--dict-size 1024] [--steps 500] [--out eval_runs/sae_scores.json]
#
# Cross-architecture comparison: pass a GRU and a mamba checkpoint in
# --policies (comma glob) — do the same ground-truth features emerge?

alias Edifice.Interpretability.SAETrainer
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, replays: :string, dict_size: :integer, steps: :integer, out: :string]
  )

policies =
  (opts[:policies] || "checkpoints/mewtwo_combo_newera_r10_policy.bin")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)

# Pre-filter SD-flake replays: Activations.capture/3 raises on parse
# failure (the 07-14 corpus has two truncated files)
replays =
  Path.wildcard(Path.expand(opts[:replays] || "~/Slippi/Game_20260714*.slp"))
  |> Enum.sort()
  |> Enum.filter(&match?({:ok, _}, ExPhil.Data.Peppi.parse(&1)))
dict_size = opts[:dict_size] || 1024
steps = opts[:steps] || 500

Output.banner("SAE features vs ground truth")
Output.puts("#{length(policies)} policies x #{length(replays)} replays, dict #{dict_size}")

results =
  for path <- policies do
    seed = Path.basename(path, ".bin")
    trunk = Activations.load_trunk(path)
    cap = Activations.capture(trunk, replays)
    acts = Nx.backend_copy(cap.activations, EXLA.Backend)
    {n, _d} = Nx.shape(acts)
    Output.puts("#{seed}: {#{n}, hidden} activations captured")

    fit =
      SAETrainer.fit(:batch_top_k_sae, acts,
        dict_size: dict_size,
        # batch_k = batch_size * per-sample k (8), per BatchTopKSAE docs
        batch_k: 16_384 * 8,
        steps: steps,
        batch_size: 16_384,
        compiler: EXLA
      )

    Output.puts("  fit done: dead=#{fit.dead_count} threshold=#{inspect(fit.threshold)}")

    # Rebuild with the FIXED threshold so firing is batch-independent
    model =
      Edifice.Interpretability.BatchTopKSAE.build(
        Keyword.merge(fit.build_opts,
          output: :container,
          inference_threshold: fit.threshold
        )
      )

    {_init, predict} = Axon.build(model, mode: :inference)
    hidden = predict.(fit.params, %{"batch_topk_sae_input" => acts}).hidden
    fired = Nx.greater(hidden, 0.0)
    fire_rate = fired |> Nx.mean(axes: [0]) |> Nx.to_flat_list()

    # Binary ground-truth labels only (u8 tensors)
    labels =
      cap.labels
      |> Enum.filter(fn {_k, t} -> Nx.type(t) == {:u, 8} end)
      |> Map.new(fn {k, t} -> {k, Nx.backend_copy(t, EXLA.Backend)} end)

    scored =
      for {label, y} <- labels do
        base = Nx.to_number(Nx.mean(y))
        yf = Nx.as_type(y, :f32)

        # Vectorized per-feature precision/recall against this label
        fired_f = Nx.as_type(fired, :f32)
        n_fire = Nx.sum(fired_f, axes: [0])
        tp = Nx.dot(yf, fired_f)
        prec = Nx.divide(tp, Nx.max(n_fire, 1))
        rec = Nx.divide(tp, max(Nx.to_number(Nx.sum(yf)), 1))
        f1 = Nx.divide(Nx.multiply(2, Nx.multiply(prec, rec)), Nx.max(Nx.add(prec, rec), 1.0e-6))

        best = Nx.argmax(f1) |> Nx.to_number()

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

    %{policy: seed, dead: fit.dead_count, scores: scored}
  end

out = opts[:out] || "eval_runs/sae_scores_#{System.os_time(:second)}.json"
File.write!(out, Jason.encode!(%{results: results}, pretty: true))
Output.success("Scores -> #{out}")
