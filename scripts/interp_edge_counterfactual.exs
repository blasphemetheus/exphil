# Flagship Stage 3, final rung (2026-08-06): REPLAY COUNTERFACTUAL edge
# validation — the standard the survey set ("validate every edge with
# replay counterfactuals"), and the one thing synthetic ablation cannot
# provide.
#
# Synthetic ablation asks "if I subtract this feature's decoded vector,
# does the logit move?" — informative, but the edited state may be off
# the manifold entirely (the LEACE lesson). A replay counterfactual asks
# the same question of REALITY: among real frames MATCHED on cycle phase
# (action family x action_frame x grounded), compare those where the
# upstream feature naturally fires high vs low, and test whether the
# downstream feature and the head logit differ as the edge predicts.
#
# Reported per edge: matched-pair counts, downstream delta, logit delta,
# and a PERMUTATION p-value (shuffle the high/low labels within matched
# strata — kills the "both just track cycle phase" confound).
#
#   mix run scripts/interp_edge_counterfactual.exs \
#     [--edges "1186:126,1665:126,958:126"] [--out ...json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, dict: :string, edges: :string, out: :string, seed: :integer]
  )

policy_path = opts[:policy] || "checkpoints/ms_g10b_human.bin"
dict_path = opts[:dict] || "eval_runs/interp/transcoder_ms_g10b_human.bin"
out_path = opts[:out] || "eval_runs/interp/edge_counterfactuals.json"
:rand.seed(:exsss, {opts[:seed] || 7, 7, 7})

edges =
  (opts[:edges] || "1186:126,1665:126,958:126,151:126")
  |> String.split(",", trim: true)
  |> Enum.map(fn s ->
    [a, b] = String.split(s, ":")
    {String.to_integer(a), String.to_integer(b)}
  end)

corpus =
  (Path.wildcard("eval_runs/0804_cycle3b_stand/r*.slp") ++
     Path.wildcard("eval_runs/0805_direct_g10b/**/*.slp") ++
     [
       "test/fixtures/replays/ys_multishine_absorbed_2026-08-04.slp",
       "test/fixtures/replays/ys_multishine_good_2026-08-04.slp"
     ])
  |> Enum.filter(&match?({:ok, _}, Peppi.parse(&1)))

Output.banner("Stage 3: replay counterfactuals")
Output.puts("#{length(corpus)} replays, #{length(edges)} edges")

trunk = Activations.load_trunk(policy_path)
heads = Activations.load_heads_only(policy_path)
window = trunk.window
dict = dict_path |> File.read!() |> :erlang.binary_to_term()
tc_data = if match?(%Axon.ModelState{}, dict.params), do: dict.params.data, else: dict.params
tc_params = ExPhil.Training.Utils.ensure_model_state(tc_data)
tc_model = Edifice.Interpretability.Transcoder.build(Keyword.put(dict.build_opts, :output, :container))
{_i, tc_predict} = Axon.build(tc_model, mode: :inference)
tc_key = tc_model |> Axon.get_inputs() |> Map.keys() |> hd()
x_mean = Nx.backend_copy(dict.x_mean, EXLA.Backend) |> Nx.squeeze()
x_std = Nx.backend_copy(dict.x_std, EXLA.Backend) |> Nx.squeeze()

# Collect (stratum, h_row, x_logit) over the corpus
rows =
  Enum.flat_map(corpus, fn path ->
    delay_id = if String.contains?(path, "_d4"), do: 4, else: 3
    cap = Activations.capture_replay(trunk, path, delay_id: delay_id, labels: false)
    acts = Nx.backend_copy(cap.activations, EXLA.Backend)

    frames =
      path
      |> then(fn p -> {:ok, r} = Peppi.parse(p); r end)
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    ds = Activations.embed_frames(frames, trunk.config, delay_id: delay_id)
    emb = Nx.backend_copy(Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend), EXLA.Backend)
    n = Nx.axis_size(acts, 0)

    pairs =
      Nx.concatenate(
        [
          Nx.slice_along_axis(acts, 0, n - 1, axis: 0),
          Nx.slice_along_axis(emb, window, n - 1, axis: 0)
        ],
        axis: 1
      )

    h =
      tc_predict.(tc_params, %{
        tc_key => Nx.divide(Nx.subtract(pairs, x_mean), Nx.add(x_std, 1.0e-6))
      }).hidden

    x_logits =
      heads.predict_fn.(heads.params, Nx.slice_along_axis(acts, 1, n - 1, axis: 0))
      |> elem(0)
      |> then(& &1[[.., 2]])
      |> Nx.to_flat_list()

    h_list = Nx.to_batched(h, 1) |> Enum.map(&Nx.squeeze/1)

    frames
    |> Enum.drop(window)
    |> Enum.take(n - 1)
    |> Enum.zip(Enum.zip(h_list, x_logits))
    |> Enum.map(fn {f, {h_row, xl}} ->
      p = f.game_state.players[1]

      %{
        stratum: {ShineChain.family(p.action), trunc(p.action_frame), p.on_ground},
        h: h_row,
        x: xl
      }
    end)
  end)

Output.puts("collected #{length(rows)} states")

by_stratum = Enum.group_by(rows, & &1.stratum)

analyze_edge = fn {from, to} ->
  # Within each stratum, split on the upstream feature's activation and
  # compare downstream activation + X logit. Strata with too few rows or
  # no variation in `from` are skipped.
  contrasts =
    by_stratum
    |> Enum.flat_map(fn {_stratum, rs} ->
      if length(rs) < 40 do
        []
      else
        vals = Enum.map(rs, fn r -> Nx.to_number(r.h[from]) end)
        sorted = Enum.sort(vals)
        q_hi = Enum.at(sorted, trunc(0.75 * length(sorted)))
        q_lo = Enum.at(sorted, trunc(0.25 * length(sorted)))

        if q_hi <= q_lo + 1.0e-6 do
          []
        else
          hi = Enum.filter(Enum.zip(rs, vals), fn {_r, v} -> v >= q_hi end) |> Enum.map(&elem(&1, 0))
          lo = Enum.filter(Enum.zip(rs, vals), fn {_r, v} -> v <= q_lo end) |> Enum.map(&elem(&1, 0))

          if hi == [] or lo == [] do
            []
          else
            mean = fn xs, f -> Enum.sum(Enum.map(xs, f)) / length(xs) end
            d_down = mean.(hi, fn r -> Nx.to_number(r.h[to]) end) - mean.(lo, fn r -> Nx.to_number(r.h[to]) end)
            d_x = mean.(hi, & &1.x) - mean.(lo, & &1.x)
            [%{n_hi: length(hi), n_lo: length(lo), d_down: d_down, d_x: d_x, rows: rs, from: from, to: to}]
          end
        end
      end
    end)

  if contrasts == [] do
    nil
  else
    n_strata = length(contrasts)
    w = Enum.map(contrasts, &(&1.n_hi + &1.n_lo))
    tot = Enum.sum(w)
    wavg = fn f -> Enum.sum(Enum.zip_with(contrasts, w, fn c, wi -> f.(c) * wi end)) / tot end
    obs_down = wavg.(& &1.d_down)
    obs_x = wavg.(& &1.d_x)

    # Permutation null: shuffle hi/lo labels WITHIN each stratum
    perms =
      for _ <- 1..200 do
        Enum.sum(
          Enum.zip_with(contrasts, w, fn c, wi ->
            shuffled = Enum.shuffle(c.rows)
            {a, b} = Enum.split(shuffled, div(length(shuffled), 2))
            mean = fn xs -> Enum.sum(Enum.map(xs, & &1.x)) / max(length(xs), 1) end
            (mean.(a) - mean.(b)) * wi
          end)
        ) / tot
      end

    p_val = Enum.count(perms, &(abs(&1) >= abs(obs_x))) / length(perms)

    Output.puts(
      "  f#{from} -> f#{to}: #{n_strata} strata, n=#{tot} | downstream delta #{Float.round(obs_down, 3)}, " <>
        "X-logit delta #{Float.round(obs_x, 3)} (permutation p=#{Float.round(p_val, 3)})"
    )

    %{from: from, to: to, strata: n_strata, n: tot, downstream_delta: obs_down, x_delta: obs_x, p_value: p_val}
  end
end

Output.puts("")
Output.puts("Matched-stratum contrasts (upstream high vs low, same cycle phase):")

results = edges |> Enum.map(analyze_edge) |> Enum.reject(&is_nil/1)

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Jason.encode!(%{edges: results}))
Output.success("Wrote #{out_path}")
