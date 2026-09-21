# EVALS_PROGRAM.md item 1 — regret maps.
#
# For every start of a policy-guided oracle run, the policy's OWN conversion
# probability is n_converted / n (its samples) and the oracle's is
# converted_any (best of n). Regret = oracle − policy. Binned by the start's
# situation labels (ExPhil.Situations, folded over the entry's history) and
# by geometry (distance band, defender height, defender percent band,
# defender grounded/airborne, defender action family), the map says WHERE the
# policy leaves value on the table. With --results-b (a second oracle run on
# the SAME pool, e.g. another checkpoint) it prints the per-bin diff: what a
# training step changed, by situation.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/regret_map.exs \
#     --pool-term eval_runs/0921_step8/pool_mix2/pool.term \
#     --results eval_runs/0921_step8/oracle_mix2_t10/results.jsonl [--results-b other/results.jsonl] \
#     --n 64 --out eval_runs/0921_evals/regret_mix2.json

alias ExPhil.Eval.Opening
alias ExPhil.Sim.Drill
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [pool_term: :string, results: :string, results_b: :string, n: :integer, out: :string, min_n: :integer])
n = opts[:n] || 64
min_n = opts[:min_n] || 15

pool = Drill.pool_from_file(opts[:pool_term] || raise("--pool-term required"))
load = fn path -> path |> File.stream!() |> Stream.map(&Jason.decode!/1) |> Map.new(fn r -> {r["id"], r} end) end
res_a = load.(opts[:results] || raise("--results required"))
res_b = if opts[:results_b], do: load.(opts[:results_b]), else: nil

Output.banner("Regret map")
Output.config([{"Pool", opts[:pool_term]}, {"Starts", length(pool)}, {"Results A", opts[:results]}, {"Results B", opts[:results_b] || "-"}, {"n per start", n}])

band = fn v, cuts -> Enum.find_index(cuts, &(v < &1)) || length(cuts) end
dist_names = ["d<20", "d20-35", "d35-50", "d50+"]
pct_names = ["p2%<30", "p2%30-70", "p2%70+"]

bins_for = fn entry ->
  states = Enum.map(entry.history, &elem(&1, 0))
  labels = if states == [], do: MapSet.new(), else: states |> ExPhil.Situations.label_states(1, as: :set) |> List.last()
  s = entry.summary
  p1 = s.p1
  p2 = s.p2
  dist = :math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))
  {fam, _} = Opening.classify(p2.action)
  def_state =
    cond do
      p2.action in 75..91 or p2.action in 223..232 -> :def_hitstun
      fam in [:aerial, :smash, :tilt, :jab, :grab, :dash_attack, :special] -> :"def_#{fam}"
      p2.action in 178..182 -> :def_shield
      true -> :def_neutral
    end
  height = cond do p2.y - p1.y > 8 -> :def_above; p1.y - p2.y > 8 -> :def_below; true -> :level end

  MapSet.new(labels)
  |> MapSet.put(String.to_atom(Enum.at(dist_names, band.(dist, [20, 35, 50]))))
  |> MapSet.put(String.to_atom(Enum.at(pct_names, band.(p2.percent, [30, 70]))))
  |> MapSet.put(if(p2.on_ground, do: :def_grounded, else: :def_airborne))
  |> MapSet.put(def_state)
  |> MapSet.put(height)
end

rows =
  pool
  |> Enum.filter(&Map.has_key?(res_a, &1.id))
  |> Enum.map(fn e ->
    a = res_a[e.id]
    b = res_b && res_b[e.id]
    %{
      id: e.id,
      bins: bins_for.(e),
      policy_a: a["n_converted"] / n,
      oracle_a: (if a["best"]["converted"] and a["best"]["alive"], do: 1.0, else: 0.0),
      policy_b: b && b["n_converted"] / n,
      oracle_b: b && (if b["best"]["converted"] and b["best"]["alive"], do: 1.0, else: 0.0)
    }
  end)

Output.puts("starts with results: #{length(rows)}")

mean = fn xs -> if xs == [], do: 0.0, else: Enum.sum(xs) / length(xs) end
all_bins = rows |> Enum.flat_map(&MapSet.to_list(&1.bins)) |> Enum.uniq()

table =
  all_bins
  |> Enum.map(fn bin ->
    rs = Enum.filter(rows, &MapSet.member?(&1.bins, bin))
    pa = mean.(Enum.map(rs, & &1.policy_a))
    oa = mean.(Enum.map(rs, & &1.oracle_a))
    base = %{bin: bin, n: length(rs), policy: Float.round(pa, 3), oracle: Float.round(oa, 3), regret: Float.round(oa - pa, 3), lost: Float.round((oa - pa) * length(rs), 1)}

    if res_b do
      pb = mean.(Enum.map(rs, & &1.policy_b))
      ob = mean.(Enum.map(rs, & &1.oracle_b))
      Map.merge(base, %{policy_b: Float.round(pb, 3), oracle_b: Float.round(ob, 3), delta_policy: Float.round(pb - pa, 3), delta_oracle: Float.round(ob - oa, 3)})
    else
      base
    end
  end)
  |> Enum.filter(&(&1.n >= min_n))

overall = %{policy: Float.round(mean.(Enum.map(rows, & &1.policy_a)), 3), oracle: Float.round(mean.(Enum.map(rows, & &1.oracle_a)), 3)}
Output.puts("overall: policy #{overall.policy}  oracle #{overall.oracle}  regret #{Float.round(overall.oracle - overall.policy, 3)}")

Output.puts("\nWhere value is lost (bins by total regret = regret × n; min n #{min_n}):")
Output.puts(String.pad_trailing("bin", 26) <> "    n  policy  oracle  regret   lost" <> if(res_b, do: "  | policy_b  Δpolicy  Δoracle", else: ""))
table
|> Enum.sort_by(&(-&1.lost))
|> Enum.take(30)
|> Enum.each(fn r ->
  line = String.pad_trailing(to_string(r.bin), 26) <> String.pad_leading(to_string(r.n), 5) <> String.pad_leading(to_string(r.policy), 8) <> String.pad_leading(to_string(r.oracle), 8) <> String.pad_leading(to_string(r.regret), 8) <> String.pad_leading(to_string(r.lost), 7)
  line = if res_b, do: line <> String.pad_leading(to_string(r.policy_b), 11) <> String.pad_leading(to_string(r.delta_policy), 9) <> String.pad_leading(to_string(r.delta_oracle), 9), else: line
  Output.puts(line)
end)

Output.puts("\nWhere the policy already selects well (lowest regret, n >= #{min_n}):")
table |> Enum.sort_by(& &1.regret) |> Enum.take(8) |> Enum.each(fn r -> Output.puts("  #{String.pad_trailing(to_string(r.bin), 26)} n #{r.n}  policy #{r.policy}  oracle #{r.oracle}  regret #{r.regret}") end)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{overall: overall, n_per_start: n, starts: length(rows), bins: Enum.map(table, &Map.update!(&1, :bin, fn b -> to_string(b) end))}, pretty: true))
  Output.success("wrote #{out}")
end
