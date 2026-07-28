# Break-phase signatures per seed (INIT_FORENSICS sustain follow-up).
# ShineChain.chains_detailed/2 already names the species of each chain
# ending; this tabulates the distribution per seed across its live replays:
#
#   :empty_hop   - jump-cancelled, then NEVER shined airborne (missed B on
#                  the first airborne frame — the canonical drop)
#   :air_shine   - aerial shine happened but the air stretch exceeded the
#                  gap (LATE B — shine on the way down, full-jump loop)
#   :aerial_jump - airborne with no shine, not straight from jumpsquat
#   :other_action- left the loop on the ground (landed and waited/crouched)
#
# Same species across seeds -> one shared weak joint (margin story);
# different species -> different weak joints per seed.
#
# Usage: XLA_TARGET=cpu mix run scripts/analyze_break_phases.exs \
#          [--replays "eval_runs/*crouch*_idle*/r*.slp"]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [replays: :string])
glob = opts[:replays] || "eval_runs/*crouch*_idle*/r*.slp"

replays = Path.wildcard(glob) |> Enum.sort()
Output.banner("Break-phase signatures (#{length(replays)} replays)")

species = [:empty_hop, :air_shine, :aerial_jump, :other_action, :end_of_input]

by_seed =
  replays
  |> Enum.group_by(fn path ->
    path |> Path.dirname() |> Path.basename() |> String.replace(~r/^\d+_/, "") |> String.replace(~r/_idle.*$/, "")
  end)

IO.puts(
  String.pad_trailing("seed", 14) <>
    String.pad_trailing("chains", 8) <>
    String.pad_trailing("maxlen", 8) <>
    Enum.map_join(species, "", &String.pad_trailing("#{&1}", 14))
)

for {seed, paths} <- Enum.sort(by_seed) do
  all =
    Enum.flat_map(paths, fn path ->
      case Peppi.parse(path) do
        {:ok, replay} ->
          replay
          |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(& &1.game_state.players[1].action)
          |> ShineChain.chains_detailed()

        _ ->
          []
      end
    end)

  # Only real multishine chains (length >= 2) tell us about SUSTAIN breaks;
  # length-1 "chains" are entry attempts.
  sustained = Enum.filter(all, &(&1.length >= 2))
  counts = Enum.frequencies_by(sustained, & &1.ended_by)
  maxlen = all |> Enum.map(& &1.length) |> Enum.max(fn -> 0 end)

  IO.puts(
    String.pad_trailing(seed, 14) <>
      String.pad_trailing("#{length(sustained)}", 8) <>
      String.pad_trailing("#{maxlen}", 8) <>
      Enum.map_join(species, "", &String.pad_trailing("#{Map.get(counts, &1, 0)}", 14))
  )
end
