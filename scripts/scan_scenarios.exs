#!/usr/bin/env elixir
# Scan .slp replays for scenario handoff candidates (task #18).
#
# Runs the ExPhil.Eval.ScenarioScan pathology detectors over each replay and
# prints candidates per type; optionally writes a manifest JSON of ALL
# candidates (curate by hand into scenarios/manifest.json — read the
# numbers, don't take the first).
#
# Usage:
#   mix run scripts/scan_scenarios.exs [options] FILE.slp [FILE.slp ...]
#   mix run scripts/scan_scenarios.exs --out /tmp/candidates.json ~/Slippi/Game_202607{14,15,16}*.slp
#
# Options:
#   --out PATH        Write all candidates as JSON manifest entries
#   --min-frame N     Earliest acceptable in-game frame (default 300)
#   --gap N           Minimum frames between same-type candidates (default 240)
#   --types a,b,c     Only run these detectors
#   --quiet           Errors only

alias ExPhil.Eval.ScenarioScan
alias ExPhil.Training.Output

{opts, files, _} =
  OptionParser.parse(System.argv(),
    strict: [out: :string, min_frame: :integer, gap: :integer, types: :string, quiet: :boolean]
  )

if files == [] do
  Output.error("No replay files given. Usage: mix run scripts/scan_scenarios.exs FILE.slp ...")
  System.halt(1)
end

types =
  case opts[:types] do
    nil -> ScenarioScan.types()
    s -> s |> String.split(",") |> Enum.map(&String.to_existing_atom/1)
  end

scan_opts = [
  types: types,
  min_frame: opts[:min_frame] || 300,
  gap: opts[:gap] || 240
]

unless opts[:quiet], do: Output.banner("Scenario candidate scan")

results =
  files
  |> Enum.sort()
  |> Enum.flat_map(fn path ->
    case ScenarioScan.load(path) do
      {:error, reason} ->
        Output.warning("#{Path.basename(path)}: unparseable (#{inspect(reason)}), skipping")
        []

      {:ok, %{frames: []}} ->
        Output.warning("#{Path.basename(path)}: no two-player frames, skipping")
        []

      {:ok, %{frames: frames}} ->
        last = List.last(frames).frame
        cands = ScenarioScan.scan(frames, scan_opts)

        unless opts[:quiet] do
          Output.puts("#{Path.basename(path)} (#{last} frames): #{length(cands)} candidates")

          for c <- cands do
            Output.puts("    #{String.pad_trailing(to_string(c.type), 16)} f=#{String.pad_leading(to_string(c.frame), 6)}  #{c.note}")
          end
        end

        Enum.map(cands, fn c ->
          %{slp: path, frame: c.frame, type: c.type, note: c.note}
        end)
    end
  end)

by_type = Enum.frequencies_by(results, & &1.type)

Output.puts("")
Output.puts("Totals across #{length(files)} file(s):")

for t <- ScenarioScan.types() do
  Output.puts("  #{String.pad_trailing(to_string(t), 16)} #{Map.get(by_type, t, 0)}")
end

if out = opts[:out] do
  File.write!(out, Jason.encode!(results, pretty: true))
  Output.success("Wrote #{length(results)} candidates to #{out}")
end
