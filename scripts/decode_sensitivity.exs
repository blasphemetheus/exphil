# D3 — decode-vs-model sensitivity (EVAL_DIRECTIONS).
#
# From BANKED sweeps (loop_report report.json files, one per arm), which
# metrics move monotonically with the decode knob (decode-driven) and
# which stay flat (model-driven)? A metric that a temperature sweep can
# steer is not evidence about the model.
#
#   mix run scripts/decode_sensitivity.exs \
#     --arm 0.5=eval_runs/0828_loop_rescore/buttons_sweep/report.json:0 \
#     ... (see --help note below)
#
# Simpler form — point at per-arm report.json files:
#   mix run scripts/decode_sensitivity.exs \
#     --json eval_runs/0828_loop_rescore/buttons_sweep/report.json \
#     --knobs 0.5,0.6,0.7,1.0 --out eval_runs/0830_decode_sensitivity/RESULTS.md
#
# The report.json is a LIST of group entries in sweep order; --knobs
# supplies the knob value per entry (same order). Spearman rank
# correlation |rho| >= 0.8 across >= 4 arms = decode-driven.
require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [json: :string, knobs: :string, out: :string]
  )

json = opts[:json] || raise "--json report.json required"
knobs =
  (opts[:knobs] || raise("--knobs v1,v2,... required (one per group entry, sweep order)"))
  |> String.split(",")
  |> Enum.map(&String.to_float(String.trim(&1)))

groups = json |> File.read!() |> Jason.decode!()

if length(groups) != length(knobs) do
  raise "#{length(groups)} group entries but #{length(knobs)} knob values"
end

metrics =
  groups
  |> Enum.flat_map(fn g -> Map.keys(g["stats"] || %{}) end)
  |> Enum.uniq()

rank = fn vals ->
  vals
  |> Enum.with_index()
  |> Enum.sort_by(&elem(&1, 0))
  |> Enum.with_index()
  |> Enum.sort_by(fn {{_v, i}, _r} -> i end)
  |> Enum.map(fn {{_v, _i}, r} -> r end)
end

spearman = fn xs, ys ->
  n = length(xs)

  if n < 3 do
    nil
  else
    rx = rank.(xs)
    ry = rank.(ys)
    d2 = Enum.zip(rx, ry) |> Enum.map(fn {a, b} -> (a - b) * (a - b) end) |> Enum.sum()
    1.0 - 6.0 * d2 / (n * (n * n - 1))
  end
end

f = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

rows =
  metrics
  |> Enum.map(fn m ->
    means = Enum.map(groups, fn g -> get_in(g, ["stats", m, "mean"]) end)

    if Enum.any?(means, &is_nil/1) do
      nil
    else
      rho = spearman.(knobs, means)
      lo = Enum.min(means)
      hi = Enum.max(means)
      range_ratio = if abs(lo) > 1.0e-9, do: hi / lo, else: nil

      verdict =
        cond do
          rho == nil -> "?"
          abs(rho) >= 0.8 and range_ratio != nil and range_ratio > 1.5 -> "**decode-driven**"
          abs(rho) >= 0.8 -> "decode-leaning (small range)"
          range_ratio != nil and range_ratio > 2.0 -> "noisy (big range, no order)"
          true -> "model-side / flat"
        end

      {m, rho, means, verdict}
    end
  end)
  |> Enum.reject(&is_nil/1)
  |> Enum.sort_by(fn {_m, rho, _means, _v} -> -abs(rho || 0.0) end)

table =
  Enum.map_join(rows, "\n", fn {m, rho, means, verdict} ->
    "| #{m} | #{if rho, do: f.(rho), else: "–"} | " <>
      Enum.map_join(means, " · ", &f.(&1)) <> " | #{verdict} |"
  end)

report = """
# D3 — decode-vs-model sensitivity

Sweep: #{json} · knob values #{inspect(knobs)} (per group entry, in order).
Spearman rho between knob and per-arm mean; |rho| ≥ 0.8 with range ratio
> 1.5 = the decode steers this metric — do not read it as model evidence.

| metric | rho(knob) | per-arm means | verdict |
|---|---:|---|---|
#{table}
"""

Output.banner("D3 — decode sensitivity")
IO.puts(report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
