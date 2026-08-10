# Edge-SD snippet miner: cut + relabel windows around DASH-OFF deaths.
#
# The queue-#2 arm (HANDOFF_2026-08-10): the generalist's SD loop is a
# coverage gap (probe_edge_attribution, 0809 — position IS read at
# near-edge dash sites), so harvest the bot's own dash-off deaths and
# relabel the approach with a scripted turnaround expert.
#
# Anchor (trajectory-classified, NOT the percent<20 heuristic — that
# misses high-percent SDs and miscounts spikes, 08-10 review): a
# grounded->airborne transition where the last grounded frame was
# DASHING/RUNNING toward the near edge at the teeter, followed by a
# stock loss within --lookahead frames with NO hitstun in between (an
# untouched fall — edgeguards and spikes are excluded at any percent).
#
# Relabel is a two-expert composite over the cut [anchor-pre, anchor+post]:
#   grounded frames -> EdgeTurnaroundExpert (dash-back toward center)
#   airborne frames -> FoxRecoveryExpert (jump / Firefox back to ledge)
# so one window teaches both "don't run off" and "if you did, recover".
# Frames both experts skip are dropped (same convention as snippet_mine)
# UNLESS --keep-unlabeled: then they keep the RECORDED controller, which
# preserves frame contiguity — required when the output feeds temporal
# window training (build_snippet_corpus.exs), where dropped frames would
# put time-skips inside GRU windows. The kept labels are mostly benign
# (away-dashes = the desired behavior; pre-window neutral movement); the
# known exception is a handful of airborne over-stage frames.
#
# Output: <out>/snippets.frames (MixFrames envelope) + manifest.json.
# Consume via train.exs --mix-frames or dagger_drill --snippet-frames.
#
#   mix run scripts/edge_snippet_mine.exs \
#     --replays "eval_runs/0810_v2_pool_ab/r*/*.slp" \
#     --out eval_runs/0810_edge_snippets [--pre 45] [--post 90] \
#     [--port 1] [--bot-code CODE#123] [--lookahead 300] [--margin 20]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.{EdgeTurnaroundExpert, FoxRecoveryExpert}
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :string,
      out: :string,
      pre: :integer,
      post: :integer,
      port: :integer,
      bot_code: :string,
      action_delay: :integer,
      lookahead: :integer,
      margin: :float,
      keep_unlabeled: :boolean
    ]
  )

replays =
  (opts[:replays] || raise("--replays is required"))
  |> String.split(",", trim: true)
  |> Enum.map(&String.trim/1)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.uniq()
  |> Enum.sort()

if replays == [] do
  Output.error("--replays matched no files — check the glob(s)")
  System.halt(2)
end

out_dir = opts[:out] || raise("--out is required")
pre = opts[:pre] || 45
post = opts[:post] || 90
default_port = opts[:port] || 1
bot_code = opts[:bot_code]
# Emitted UNSHIFTED; recorded in the envelope so MixFrames can catch a
# consumer-side delay mismatch (GOTCHA #86).
action_delay = opts[:action_delay] || 0
lookahead = opts[:lookahead] || 300
margin = opts[:margin] || 20.0
keep_unlabeled = opts[:keep_unlabeled] || false

# libmelee Action enum
dashing_running = MapSet.new([0x14, 0x15])
# GroundTruth's hitstun action set (same as snippet_mine)
hitstun = MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))

normalize_code = fn c ->
  c |> to_string() |> String.replace("＃", "#") |> String.upcase() |> String.trim()
end

bot_port_for = fn replay ->
  players = replay.metadata.players || []

  found =
    if bot_code do
      Enum.find(players, fn p ->
        normalize_code.(p.netplay_code || "") == normalize_code.(bot_code)
      end)
    end

  case found do
    nil -> default_port
    p -> p.port
  end
end

File.mkdir_p!(out_dir)
Output.banner("Edge-SD snippet miner (dash-off deaths)")
Output.config([
  {"Replays", length(replays)},
  {"Window", "-#{pre}/+#{post}"},
  {"Lookahead", lookahead},
  {"Danger margin", margin}
])

recovery = FoxRecoveryExpert.new()

{snippets, stats} =
  Enum.reduce(replays, {[], %{replays: 0, anchors: 0, frames: 0, skipped: 0}}, fn path,
                                                                                  {acc, st} ->
    try do
      case Peppi.parse(path) do
        {:ok, replay} ->
          port = bot_port_for.(replay)
          opp_port = if port == 1, do: 2, else: 1

          frames =
            replay
            |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
            |> Enum.reject(&(&1.game_state.frame < 0))

          arr = List.to_tuple(frames)
          n = tuple_size(arr)
          if n < 2, do: throw({:too_short, path})

          me = fn i -> elem(arr, i).game_state.players[port] end

          stage = elem(arr, 0).game_state.stage

          # external (Slippi) id -> atom; raw ids hit the wrong stage
          # (external YS 8 = internal FoD 8)
          edge =
            (trunc(stage || 0) |> Melee.Enums.Stage.from_external() |> Melee.Stages.edge_ground_position()) ||
              85.5656967163
          turnaround = EdgeTurnaroundExpert.new(edge_x: edge, danger_margin: margin)

          in_hitstun = fn i -> MapSet.member?(hitstun, trunc(me.(i).action || 0)) end

          # Untouched fall from anchor to a stock loss within lookahead
          dies_clean = fn i ->
            last = min(i + lookahead, n - 1)

            Enum.reduce_while((i + 1)..last, true, fn j, _ ->
              cond do
                in_hitstun.(j) -> {:halt, false}
                (me.(j).stock || 0) < (me.(i).stock || 0) -> {:halt, true}
                j == last -> {:halt, false}
                true -> {:cont, true}
              end
            end)
          end

          anchors =
            for i <- 1..(n - 1),
                p = me.(i - 1),
                p.on_ground == true and me.(i).on_ground == false,
                MapSet.member?(dashing_running, trunc(p.action || 0)),
                abs(p.x || 0.0) > edge - margin,
                dies_clean.(i),
                do: i

          ranges =
            anchors
            |> Enum.map(&{max(&1 - pre, 0), min(&1 + post, n - 1)})
            |> Enum.reduce([], fn {a, b}, racc ->
              case racc do
                [{pa, pb} | rest] when a <= pb + 1 -> [{pa, max(b, pb)} | rest]
                _ -> [{a, b} | racc]
              end
            end)
            |> Enum.reverse()

          recorded = Map.new(frames, &{&1.game_state.frame, &1.controller})

          cut_snippets =
            Enum.map(ranges, fn {a, b} ->
              Enum.slice(frames, a, b - a + 1)
              |> Enum.flat_map(fn frame ->
                prev = recorded[frame.game_state.frame - 1]
                p = frame.game_state.players[port]

                label =
                  case EdgeTurnaroundExpert.label(turnaround, p, prev) do
                    {:ok, c} -> {:ok, c}
                    :skip -> FoxRecoveryExpert.label(recovery, p, prev)
                  end

                case label do
                  {:ok, correction} ->
                    [frame |> Map.put(:controller, correction) |> Map.put(:prev_controller, prev)]

                  :skip when keep_unlabeled ->
                    [Map.put(frame, :prev_controller, prev)]

                  :skip ->
                    []
                end
              end)
            end)
            |> Enum.reject(&(length(&1) < 20))

          mined = Enum.sum(Enum.map(cut_snippets, &length/1))

          if anchors != [] do
            Output.puts(
              "  #{Path.basename(path)}: #{length(anchors)} dash-off death(s), #{mined} frames"
            )
          end

          {acc ++ cut_snippets,
           %{st | replays: st.replays + 1, anchors: st.anchors + length(anchors), frames: st.frames + mined}}

        _ ->
          {acc, %{st | skipped: st.skipped + 1}}
      end
    catch
      {:too_short, _path} -> {acc, %{st | skipped: st.skipped + 1}}
    end
  end)

out_file = Path.join(out_dir, "snippets.frames")

File.write!(
  out_file,
  :erlang.term_to_binary(
    %{
      expert: "edge_turnaround+fox_recovery",
      exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      action_delay: action_delay,
      frame_lists: snippets
    },
    [:compressed]
  )
)

File.write!(
  Path.join(out_dir, "manifest.json"),
  Jason.encode!(
    %{
      replays_scanned: stats.replays,
      replays_skipped: stats.skipped,
      anchors: stats.anchors,
      snippets: length(snippets),
      frames: stats.frames,
      pre: pre,
      post: post,
      lookahead: lookahead,
      danger_margin: margin,
      action_delay: action_delay,
      keep_unlabeled: keep_unlabeled
    },
    pretty: true
  )
)

Output.success(
  "#{stats.anchors} dash-off death(s) across #{stats.replays} replays -> " <>
    "#{length(snippets)} snippets / #{stats.frames} frames -> #{out_file}"
)

if stats.anchors == 0 do
  Output.warning("zero anchors — check --port/--bot-code and that the replays contain SDs")
end
