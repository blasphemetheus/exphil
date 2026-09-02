# Drill-bank window slicer (DRILL_HITCONFIRM training half, 09-02).
#
# Extracts ONLY the continuation windows [handoff, handoff+window] from a
# drill bank's replays into a MixFrames .frames envelope. Raw drill .slp
# files must NEVER be fed to training whole: the scripted setup/reset
# phases would clone idle-watching, and --awbc-reward standard would
# credit every scripted dummy SD as a kill earned by standing still
# (drill.json training block: slice handoff_window_only, port explicit —
# both ports are Fox, character resolution is ambiguous).
#
# Ordering guarantee: windows are emitted globally DESCENDING by handoff
# frame, so every adjacent pair of lists breaks frame monotonicity —
# AdvantageWeighting.split_by_replay (cur <= prev => new group) and the
# sequence builder (frame discontinuity => boundary) both see each window
# as its own segment. Never "simplify" to ascending order: consecutive
# ascending windows MERGE into one AWBC group and one GRU stitch region.
#
#   mix run scripts/drill_slice.exs \
#     --bank eval_runs/0902_drill1_bank,eval_runs/0902_drill1_bank2 \
#     --out drills/drill1_hitconfirm.frames
#
# Options: --window (default: each episode's recorded span, else the
#   bank drill.json window, else 240) · --expert NAME (envelope tag)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [bank: :string, out: :string, window: :integer, expert: :string]
  )

banks = (opts[:bank] || raise("--bank required")) |> String.split(",", trim: true)
out_path = opts[:out] || "drills/drill1_hitconfirm.frames"
expert_tag = opts[:expert] || "drill1_hitconfirm"

Output.banner("Drill-bank window slicer")

episodes =
  Enum.flat_map(banks, fn bank ->
    drill =
      case File.read(Path.join(bank, "drill.json")) do
        {:ok, bin} -> Jason.decode!(bin)
        _ -> %{}
      end

    port = get_in(drill, ["training", "port"]) || 1
    window = opts[:window] || drill["window"] || 240

    bank
    |> Path.join("episodes.jsonl")
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&Jason.decode!/1)
    |> Enum.filter(& &1["slp"])
    |> Enum.map(fn row ->
      %{slp: Path.join(bank, row["slp"]), handoff: row["handoff"], window: window, port: port}
    end)
  end)

Output.puts("  #{length(episodes)} episodes across #{length(banks)} banks")

by_slp = Enum.group_by(episodes, & &1.slp)

{lists, dropped} =
  by_slp
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {{slp, eps}, i} ->
    if rem(i, 10) == 0, do: Output.progress_bar(i, map_size(by_slp), label: "slicing")

    case Peppi.parse(slp) do
      {:ok, replay} ->
        frames =
          replay
          |> Peppi.to_training_frames(
            player_port: hd(eps).port,
            opponent_port: if(hd(eps).port == 1, do: 2, else: 1),
            remap_ports: true
          )
          |> Enum.reject(&(&1.game_state.frame < 0))

        index = Map.new(Enum.with_index(frames), fn {f, j} -> {f.game_state.frame, j} end)
        frames_t = List.to_tuple(frames)
        n = tuple_size(frames_t)

        Enum.map(eps, fn ep ->
          case index[ep.handoff] do
            nil ->
              {:dropped, "handoff f#{ep.handoff} not found in #{Path.basename(slp)}"}

            j ->
              count = min(ep.window + 1, n - j)
              {ep.handoff, Enum.map(j..(j + count - 1), &elem(frames_t, &1))}
          end
        end)

      err ->
        [{:dropped, "parse failed #{Path.basename(slp)}: #{inspect(err)}"}]
    end
  end)
  |> Enum.split_with(fn
    {:dropped, _} -> false
    _ -> true
  end)

Output.progress_done()

for {:dropped, why} <- dropped, do: Output.warning("  dropped: #{why}")

# Global DESC by handoff — see the ordering guarantee in the header.
frame_lists =
  lists
  |> Enum.sort_by(fn {handoff, _} -> -handoff end)
  |> Enum.map(fn {_, l} -> l end)

total = frame_lists |> Enum.map(&length/1) |> Enum.sum()

envelope = %{
  expert: expert_tag,
  exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  action_delay: 0,
  frame_lists: frame_lists
}

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, :erlang.term_to_binary(envelope))

Output.success(
  "#{length(frame_lists)} windows / #{total} frames (#{length(dropped)} dropped) -> #{out_path}"
)
