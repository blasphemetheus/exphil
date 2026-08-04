# Snippet-miner v1 (task #19): cut + relabel event-anchored windows.
#
# Event (v1): OWN player enters hitstun within 60 frames of being in the
# shine-cycle family — "hit while multishining", the fight-state gap.
# Cuts [anchor-pre, anchor+post] (defaults 32/120: lead-in covers window
# + prev threading; aftermath covers the exit path), merges overlaps,
# relabels every frame with the MultishineExpert (rule 2: recorded
# controllers are only valid labels when the recorder was the teacher).
#
# Output: <out>/snippet_NNN.frames (term_to_binary'd training-frame
# lists, consecutive frame numbers preserved so prev-action threads —
# rule 1) + manifest.json. Consumed later via the drill's snippet flag
# (integration lands when no grind loop is mid-flight) — mixed into the
# base pool, never replacing it (rule 3).
#
#   mix run scripts/snippet_mine.exs --replays "eval_runs/0803_pressure_pool/r*.slp" \
#     [--out eval_runs/0803_snippets] [--pre 32] [--post 120] [--port 1]

alias ExPhil.Agents.MultishineExpert
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :string,
      out: :string,
      pre: :integer,
      post: :integer,
      port: :integer,
      fixture: :string,
      action_delay: :integer,
      bot_code: :string
    ]
  )

# --replays accepts a COMMA-SEPARATED list of globs (the convention every
# other script here uses: --rollouts, --opening-replays, --snippet-frames).
# v1 ran Path.expand over the whole string, so a comma list expanded to one
# nonexistent path and silently mined ZERO replays while reporting success —
# found 2026-08-04 pointing it at the human corpus.
replays =
  (opts[:replays] || "eval_runs/0803_pressure_pool/r*.slp")
  |> String.split(",", trim: true)
  |> Enum.map(&String.trim/1)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.uniq()
  |> Enum.sort()

if replays == [] do
  ExPhil.Training.Output.error("--replays matched no files — check the glob(s)")
  System.halt(2)
end
out_dir = opts[:out] || "eval_runs/0803_snippets"
# Frames are emitted UNSHIFTED; the consumer applies the delay. Recorded in
# the envelope so MixFrames can warn on a mismatch (see the write below).
action_delay = opts[:action_delay] || 0
pre = opts[:pre] || 32
post = opts[:post] || 120
# --bot-code CODE#123: identify the BOT's port PER REPLAY from the netplay
# code, instead of assuming port 1. Slippi assigns ports per session, so a
# netplay corpus has the bot on 1 in some games and 2 in others — mining a
# fixed port would harvest the HUMAN's frames and relabel them with our
# expert (a worse cousin of the farm-5 poisoning: not just wrong labels,
# wrong player). Confirmed on the 0803/0804 corpus: bot on port 1 in 6
# games, port 2 in 6, plus two Fox-vs-Fox mirrors where character alone
# cannot disambiguate. Falls back to --port when no code is given.
default_port = opts[:port] || 1
bot_code = opts[:bot_code]

# Slippi writes a FULLWIDTH '#' (U+FF03) in netplay codes.
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
fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"

# GroundTruth's hitstun action set
hitstun = MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
cycle_families = [:ground_reflect, :air_reflect, :jumpsquat, :aerial_jump]

File.mkdir_p!(out_dir)
Output.banner("Snippet miner: hit-while-multishining")

{:ok, fx} = Peppi.parse(fixture)

fixture_frames =
  fx
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)

expert = MultishineExpert.from_frames(fixture_frames, player_port: 1)

{snippets, stats} =
  Enum.reduce(replays, {[], %{replays: 0, anchors: 0, frames: 0, skipped: 0}}, fn path, {acc, st} ->
    try do
      case Peppi.parse(path) do
      {:ok, replay} ->
        # Per-replay bot port (netplay scrambles it — see --bot-code above)
        port = bot_port_for.(replay)
        opp_port = if port == 1, do: 2, else: 1

        frames =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
          |> Enum.reject(&(&1.game_state.frame < 0))

        arr = List.to_tuple(frames)
        n = tuple_size(arr)
        in_hitstun = fn i -> MapSet.member?(hitstun, elem(arr, i).game_state.players[port].action) end
        in_cycle = fn i -> ShineChain.family(elem(arr, i).game_state.players[port].action) in cycle_families end

        # Truncated replays are common in this corpus (the mainline writer
        # dies mid-game; one 0804 game is a single frame). n < 2 makes the
        # anchor range 1..-1 and elem/2 blows up on an empty tuple.
        if n < 2 do
          throw({:too_short, path})
        end

        anchors =
          for i <- 1..(n - 1),
              in_hitstun.(i),
              not in_hitstun.(i - 1),
              Enum.any?(max(i - 60, 0)..(i - 1), in_cycle),
              do: i

        # Merge overlapping cuts
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

        # Recorded controller stream (for prev threading during relabel)
        recorded = Map.new(frames, &{&1.game_state.frame, &1.controller})

        cut_snippets =
          Enum.map(ranges, fn {a, b} ->
            Enum.slice(frames, a, b - a + 1)
            |> Enum.flat_map(fn frame ->
              prev = recorded[frame.game_state.frame - 1]

              case MultishineExpert.label(expert, frame.game_state.players[port], prev, frame.game_state.players[opp_port]) do
                {:ok, correction} ->
                  [frame |> Map.put(:controller, correction) |> Map.put(:prev_controller, prev)]

                :skip ->
                  []
              end
            end)
          end)
          |> Enum.reject(&(length(&1) < 20))

        mined = Enum.sum(Enum.map(cut_snippets, &length/1))

        {acc ++ cut_snippets,
         %{st | replays: st.replays + 1, anchors: st.anchors + length(anchors), frames: st.frames + mined}}

        _ ->
          {acc, %{st | skipped: st.skipped + 1}}
      end
    catch
      {:too_short, _path} -> {acc, %{st | skipped: st.skipped + 1}}
    end
  end)

# ONE file in the MixFrames envelope (unified 2026-08-03). v1 wrote a bare
# term_to_binary per snippet, a parallel format that silently dropped the
# :action_delay contract — ExPhil.Training.MixFrames warns when a file's
# export delay differs from the training delay, because a mismatch
# misaligns the :prev_controller channel by that many frames and corrupts
# training invisibly. Snippets are per-source lists (shift_actions must not
# cross a snippet boundary), which is exactly what :frame_lists means.
out_file = Path.join(out_dir, "snippets.frames")

File.write!(
  out_file,
  :erlang.term_to_binary(%{
    expert: "multishine",
    exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
    action_delay: action_delay,
    frame_lists: snippets,
    # snippet-miner provenance (ignored by MixFrames, useful in forensics)
    event: "hitstun_in_cycle",
    pre: pre,
    post: post
  })
)

File.write!(
  Path.join(out_dir, "manifest.json"),
  Jason.encode!(%{
    event: "hitstun_in_cycle",
    pre: pre,
    post: post,
    snippets: length(snippets),
    stats: stats
  }, pretty: true)
)

Output.puts("replays #{stats.replays} (skipped #{stats.skipped}) anchors #{stats.anchors}")
Output.success("#{length(snippets)} snippets, #{stats.frames} relabeled frames -> #{out_dir}")
