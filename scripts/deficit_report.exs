# P5 deficit report (+ cross-checkpoint / cross-architecture comparison).
#
#   mix run scripts/deficit_report.exs \
#     --policies checkpoints/a_policy.bin,checkpoints/b_policy.bin \
#     --replays "probes/newera8/mamba_full/r13/plain/p1/*.slp,..." \
#     [--directions shield:checkpoints/r14_shield_steer.bin] \
#     [--no-controls] [--out logs/deficit_report.json]
#
# Per policy: probe suite (v1+v2) on trunk activations vs the input
# floor, random-init + shuffled controls, direction-projection
# monitoring, and a deficit ranking (trunk-minus-floor, saturated
# features excluded). See ExPhil.Interp.DeficitReport.
#
# NO-MIX: inference-only beam; do not run beside training (#67).

alias ExPhil.Interp.DeficitReport
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, replays: :string, directions: :string, out: :string, no_controls: :boolean]
  )

policies =
  (opts[:policies] || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)

replays =
  (opts[:replays] || "")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  # Crashed/restarted arms leave truncated strays that fail to parse —
  # size filter per the #69 rule (size, not mtime)
  |> Enum.filter(fn f -> File.stat!(f).size > 512_000 end)

directions =
  (opts[:directions] || "")
  |> String.split(",", trim: true)
  |> Enum.map(fn spec ->
    [name, path] = String.split(spec, ":", parts: 2)
    {name, Path.expand(path)}
  end)

if policies == [], do: raise("--policies required")
if length(replays) < 3, do: raise("need >= 3 replays (train/eval split by replay), got #{length(replays)}")

Output.banner("Deficit report")
Output.config([
  {"Policies", Enum.map(policies, &Path.basename/1)},
  {"Replays", length(replays)},
  {"Directions", Enum.map(directions, &elem(&1, 0))},
  {"Controls", not (opts[:no_controls] || false)}
])

reports =
  Enum.map(policies, fn policy ->
    Output.puts("")
    report = DeficitReport.run(policy, replays, directions: directions, controls: not (opts[:no_controls] || false))
    Output.puts(DeficitReport.format(report))
    report
  end)

# Cross-checkpoint deficit comparison: per feature, trunk BA across policies
if length(reports) > 1 do
  Output.puts("")
  Output.puts("=== cross-checkpoint trunk balanced-accuracy (non-saturated features) ===")

  features = reports |> hd() |> Map.get(:deficits) |> Enum.map(& &1.feature)

  Enum.each(features, fn f ->
    cells =
      Enum.map(reports, fn r ->
        case Enum.find(r.deficits, &(&1.feature == f)) do
          nil -> "     -"
          d -> d.trunk |> Float.round(3) |> :erlang.float_to_binary(decimals: 3) |> String.pad_leading(6)
        end
      end)

    Output.puts("  #{String.pad_trailing(to_string(f), 27)}#{Enum.join(cells, "  ")}")
  end)

  Output.puts("  (columns: #{reports |> Enum.map(& &1.policy) |> Enum.join(" | ")})")
end

out = opts[:out] || "logs/deficit_report_#{Date.utc_today() |> Date.to_iso8601(:basic)}.json"
File.write!(out, Jason.encode!(reports, pretty: true))
Output.success("report -> #{out}")
