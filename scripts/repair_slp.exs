# Repair truncated .slp replays (the SD-flake casualties) so peppi can
# parse them — see ExPhil.Data.SlpRepair.
#
# Usage:
#   mix run scripts/repair_slp.exs <replay.slp> [more.slp ...]
#   mix run scripts/repair_slp.exs "eval_runs/**/r*.slp" --check
#
# Default: writes <path>.repaired.slp next to each unparseable input.
# --check: report which files are truncated without writing anything.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.{Peppi, SlpRepair}
alias ExPhil.Training.Output

{opts, paths, _} = OptionParser.parse(System.argv(), strict: [check: :boolean])

paths = Enum.flat_map(paths, &Path.wildcard/1)
if paths == [], do: raise("no replays given")

check? = opts[:check] == true

for path <- paths do
  case Peppi.parse(path) do
    {:ok, _} ->
      Output.puts("#{path}: intact")

    {:error, _} when check? ->
      Output.warning("#{path}: TRUNCATED (repairable)")

    {:error, _} ->
      case SlpRepair.repair(path) do
        {:ok, out, stats} ->
          case Peppi.parse(out) do
            {:ok, replay} ->
              n = replay |> Peppi.to_training_frames(player_port: 1, opponent_port: 2) |> length()

              Output.success(
                "#{path}: repaired -> #{out} (#{stats.events} events, " <>
                  "#{stats.dropped_bytes}B dropped, #{n} frames recovered)"
              )

            {:error, reason} ->
              Output.error("#{path}: repaired file still unparseable: #{inspect(reason)}")
          end

        {:error, reason} ->
          Output.error("#{path}: repair failed: #{inspect(reason)}")
      end
  end
end
