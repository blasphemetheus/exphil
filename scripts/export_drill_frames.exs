# Export expert-relabeled drill frames for curriculum mixing into corpus
# training (train.exs --mix-frames).
#
#   mix run scripts/export_drill_frames.exs --expert fox_recovery \
#     --rollouts "~/Slippi/Game_*.slp" --out drills/fox_recovery.frames
#
# Output format: :erlang.term_to_binary of %{
#   expert: "fox_recovery",
#   exported_at: iso8601,
#   frame_lists: [[frame, ...], ...]   # one list per source replay — the
# }                                    # delay shift must not cross replays
#
# Frames carry the expert's corrected :controller and the policy's actual
# press in :prev_controller (same relabel protocol as dagger_drill.exs).

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.Output
alias ExPhil.Data.Peppi

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [expert: :string, rollouts: :string, fixture: :string, out: :string, port: :integer, action_delay: :integer]
  )

expert_mod =
  case opts[:expert] do
    "multishine" -> ExPhil.Agents.MultishineExpert
    "mewtwo_fair" -> ExPhil.Agents.MewtwoFairExpert
    "fox_recovery" -> ExPhil.Agents.FoxRecoveryExpert
    other ->
      Output.error("Unknown expert #{inspect(other)}")
      System.halt(1)
  end

out_path = opts[:out] || "drills/#{opts[:expert]}.frames"
port = opts[:port] || 1
action_delay = opts[:action_delay] || 2

rollout_paths =
  (opts[:rollouts] || "")
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)
  |> Enum.flat_map(&Path.wildcard/1)

if rollout_paths == [] do
  Output.error("No rollouts — pass --rollouts \"a.slp,b*.slp\"")
  System.halt(1)
end

expert =
  case opts[:fixture] do
    nil ->
      expert_mod.from_fixture()

    fixture ->
      expert_mod.from_fixture(fixture, player_port: port)
  end

frame_lists =
  Enum.map(rollout_paths, fn path ->
    {:ok, replay} = Peppi.parse(path)

    raw =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    recorded = Map.new(raw, fn f -> {f.game_state.frame, f.controller} end)

    opp_port = if port == 1, do: 2, else: 1

    relabeled =
      Enum.flat_map(raw, fn frame ->
        prev = recorded[frame.game_state.frame + action_delay - 1]

        case expert_mod.label(expert, frame.game_state.players[port], prev, frame.game_state.players[opp_port]) do
          {:ok, correction} ->
            [frame |> Map.put(:controller, correction) |> Map.put(:prev_controller, prev)]

          :skip ->
            []
        end
      end)

    Output.puts("  #{Path.basename(path)}: #{length(relabeled)} labeled frames")
    relabeled
  end)
  |> Enum.reject(&(&1 == []))

total = frame_lists |> Enum.map(&length/1) |> Enum.sum()

File.mkdir_p!(Path.dirname(out_path))

payload = %{
  expert: opts[:expert],
  exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  action_delay: action_delay,
  frame_lists: frame_lists
}

File.write!(out_path, :erlang.term_to_binary(payload, [:compressed]))
Output.success("#{total} frames (#{length(frame_lists)} replay segments) -> #{out_path}")
