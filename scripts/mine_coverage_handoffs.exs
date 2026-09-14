# Mine held-out multishine handoffs from replays for the coverage round
# (2026-09-14): standing NEUTRAL starts stratified by the subject's x
# position and facing and the opponent's side, plus HIT starts (the first
# hitstun frame of each hitstun episode). Writes a scenario_suite manifest
# (type multishine_reentry) with a per-entry `class` note, and a sidecar
# JSON describing the strata so a train/held-out split can be drawn later.
#
#   mix run --no-start scripts/mine_coverage_handoffs.exs \
#     --out MANIFEST.json --per-replay-neutral 4 --per-replay-hit 4 \
#     --min-frame 240 --max-frame 4800 --gap 300 REPLAY.slp ...
#
# Neutral start = subject (port 1) grounded in an off-loop actionable state
# (wait/walk/turn/dash/run/crouch/landing) for >= 4 frames,
# grounded, zero hitstun, not invulnerable, stock unchanged over the next
# 120 frames. Strata: x bucket (left/center/right thirds of FD, |x| < 20
# is center), facing (+1/-1), opponent side relative to the subject. One
# candidate per stratum per replay, preferring the earliest frame (short
# prefixes drift least). Hit start = the frame BEFORE hitstun_frames_left
# turns positive (the interruption manifest's convention: hit502 -> episode
# start 503), spaced by --gap.
alias ExPhil.Data.Peppi

{opts, replays, []} =
  OptionParser.parse(System.argv(),
    strict: [
      out: :string,
      per_replay_neutral: :integer,
      per_replay_hit: :integer,
      min_frame: :integer,
      max_frame: :integer,
      gap: :integer
    ]
  )

out = Keyword.fetch!(opts, :out)
if File.exists?(out), do: raise("output exists: #{out}")
per_neutral = opts[:per_replay_neutral] || 4
per_hit = opts[:per_replay_hit] || 4
min_frame = opts[:min_frame] || 240
max_frame = opts[:max_frame] || 4800
gap = opts[:gap] || 300
# grounded, actionable, off the multishine loop: WAIT 14, WALK 15-17, TURN 18,
# DASH 20, RUN 21, CROUCH 40, LANDING 42 (the bot rarely idles for long)
neutral_actions = [14, 15, 16, 17, 18, 20, 21, 40, 42]
stable_frames = 4

x_bucket = fn x ->
  cond do
    x < -20 -> :left
    x > 20 -> :right
    true -> :center
  end
end

entries =
  Enum.flat_map(replays, fn path ->
    {:ok, replay} = Peppi.parse(path)
    frames = Peppi.to_training_frames(replay) |> Enum.reject(&(&1.game_state.frame < 0))
    by_frame = Map.new(frames, &{&1.game_state.frame, &1.game_state})
    p2_char = trunc(hd(frames).game_state.players[2].character)

    stable_wait? = fn f ->
      Enum.all?(0..(stable_frames - 1), fn d ->
        case by_frame[f + d] do
          nil -> false
          gs -> p = gs.players[1]; trunc(p.action) in neutral_actions and p.on_ground and p.hitstun_frames_left == 0 and not p.invulnerable
        end
      end)
    end

    stock_stable? = fn f ->
      case {by_frame[f], by_frame[f + 120]} do
        {%{players: %{1 => a}}, %{players: %{1 => b}}} -> a.stock == b.stock
        _ -> false
      end
    end

    neutral =
      frames
      |> Enum.map(& &1.game_state)
      |> Enum.filter(fn gs -> gs.frame >= min_frame and gs.frame <= max_frame end)
      |> Enum.filter(fn gs -> stable_wait?.(gs.frame) and stock_stable?.(gs.frame) end)
      |> Enum.map(fn gs ->
        p = gs.players[1]
        o = gs.players[2]
        stratum = {x_bucket.(p.x), if(p.facing > 0, do: :right, else: :left), if(o.x > p.x, do: :opp_right, else: :opp_left)}
        %{frame: gs.frame, class: "neutral", stratum: stratum, x: Float.round(p.x * 1.0, 1), facing: p.facing, opp_dx: Float.round((o.x - p.x) * 1.0, 1)}
      end)
      |> Enum.group_by(& &1.stratum)
      |> Enum.map(fn {_, cands} -> Enum.min_by(cands, & &1.frame) end)
      |> Enum.sort_by(& &1.frame)
      |> Enum.take(per_neutral)

    hits =
      frames
      |> Enum.map(& &1.game_state)
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [a, b] ->
        a.players[1].hitstun_frames_left == 0 and b.players[1].hitstun_frames_left > 0 and
          a.frame >= min_frame and a.frame <= max_frame and a.players[1].stock == b.players[1].stock
      end)
      |> Enum.map(fn [a, _] ->
        p = a.players[1]
        o = a.players[2]
        %{frame: a.frame, class: "hit", stratum: {x_bucket.(p.x), if(p.on_ground, do: :grounded, else: :airborne)}, x: Float.round(p.x * 1.0, 1), facing: p.facing, opp_dx: Float.round((o.x - p.x) * 1.0, 1), percent: p.percent}
      end)
      |> Enum.reduce([], fn c, acc ->
        if Enum.any?(acc, &(abs(&1.frame - c.frame) < gap)), do: acc, else: acc ++ [c]
      end)
      |> Enum.take(per_hit)

    IO.puts("#{path}: p2 char #{p2_char}, #{length(neutral)} neutral, #{length(hits)} hit")

    Enum.map(neutral ++ hits, fn c ->
      %{
        slp: path,
        frame: c.frame,
        type: "multishine_reentry",
        note: "#{c.class} #{inspect(c.stratum)} x=#{c.x} facing=#{c.facing} opp_dx=#{c.opp_dx} opp_char=#{p2_char}",
        class: c.class,
        stratum: Tuple.to_list(c.stratum) |> Enum.map(&to_string/1),
        opp_char: p2_char,
        x: c.x,
        facing: c.facing,
        opp_dx: c.opp_dx
      }
    end)
  end)

File.write!(out, Jason.encode!(%{entries: entries}, pretty: true), [:exclusive])
IO.puts("#{length(entries)} entries -> #{out}")
