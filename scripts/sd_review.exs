# SD review over .slp replays (2026-10-04): every bot death, read back to the
# frame the recovery decision was made from, judged by surviving-route count
# (ExPhil.Melee.Checkmate.routes/1: 0 checkmate / 1 forced / 2+ mixup —
# same bucketing as sim_closed_loop.exs), and what the bot DID from there.
#
#   mix run scripts/sd_review.exs --label L [--bot-port 1] [--out FILE.json] GAME.slp ...
#
# Per death: verdict, routes (means + fastest), position/velocity/jumps at
# the decision frame, then the inputs and action states on the way down:
# which Fox specials it entered, jumps spent, share of frames holding the
# stick toward the stage, frames offstage. The summary groups deaths by
# verdict and by what was attempted, which is what a death review needs:
# "threw a 3-route stock by side-B-ing low" vs "never pressed B".
alias ExPhil.{Data.Peppi, Training.Output}
alias ExPhil.Melee.Checkmate
alias ExPhil.Sim.GA
alias ExPhil.Interp.ActionNames

{opts, files, bad} =
  OptionParser.parse(System.argv(), strict: [label: :string, bot_port: :integer, out: :string, min_bytes: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
label = opts[:label] || "sd_review"
bot_port = opts[:bot_port] || 1
min_bytes = opts[:min_bytes] || 150_000

Output.banner("SD review: #{label}")

# Fox specials by action id (ActionNames): side-B 350..352 (FOX_ILLUSION*),
# up-B 353..356 (FIREFOX_WAIT_*, FIREFOX_GROUND/AIR), shine 360.., laser 341..348
special = fn a ->
  cond do
    a in 350..352 -> :side_b
    a in 353..356 -> :up_b
    a in 341..348 -> :laser
    a in 360..368 -> :shine
    true -> nil
  end
end
stun? = fn p -> (p.hitstun_frames_left || 0) > 0 end

deaths =
  files
  |> Enum.filter(&(File.stat!(&1).size >= min_bytes))
  |> Enum.flat_map(fn path ->
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(&1.port == bot_port))
    opp = Enum.find(meta.players, &(&1.port != bot_port))
    if own == nil or opp == nil, do: raise("#{path}: ports #{inspect(Enum.map(meta.players, & &1.port))}")
    stage = meta.stage
    edge = GA.stage_edge(stage)
    {:ok, replay} = Peppi.parse(path, player_port: own.port)
    frames =
      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&%{own: &1.game_state.players[own.port], opp: &1.game_state.players[opp.port], c: &1.controller, f: &1.game_state.frame})
      |> List.to_tuple()
    n = tuple_size(frames)
    offstage? = fn p -> not p.on_ground and (abs(p.x) > edge or p.y < -5.0) end

    # same state machine as sim_closed_loop.exs, over consecutive frames
    {_, rows} =
      Enum.reduce(1..(n - 1), {%{last_hit: -10_000, off: false, actionable: nil, since: nil}, []}, fn i, {e, out} ->
        p0 = elem(frames, i - 1).own
        p1 = elem(frames, i).own
        hit? = p1.percent > p0.percent or stun?.(p1)
        died? = (p1.stock || 0) < (p0.stock || 0)
        was_off = e.off
        now_off = offstage?.(p1) and not died?
        {actionable, since} =
          cond do
            hit? -> {nil, nil}
            now_off and not stun?.(p1) and (stun?.(p0) or not was_off) -> {p1, i}
            true -> {e.actionable, e.since}
          end
        last_hit = if hit?, do: i, else: e.last_hit

        out =
          if died? do
            {verdict, routes, means} =
              case e.actionable do
                nil -> {:kill, nil, []}
                p ->
                  r = Checkmate.routes(GA.checkmate_state(p, stage))
                  {r.verdict, r.count, Enum.map(r.routes, &Enum.join(&1.means, "+"))}
              end
            from = e.since || max(i - 120, 1)
            path_frames = for j <- from..(i - 1), do: elem(frames, j)
            specials = path_frames |> Enum.map(&special.(&1.own.action)) |> Enum.reject(&is_nil/1) |> Enum.dedup()
            toward = fn fr -> sign = if fr.own.x > 0, do: -1, else: 1; (fr.c.main_stick.x - 0.5) * sign > 0.2 end
            jumps0 = if e.actionable, do: e.actionable.jumps_left, else: nil
            jumps_used = if jumps0, do: jumps0 - Enum.min(Enum.map(path_frames, & &1.own.jumps_left || 0)), else: nil
            b_presses = path_frames |> Enum.map(& &1.c.button_b) |> Enum.chunk_every(2, 1, :discard) |> Enum.count(fn [a, b] -> b and not a end)
            actions = path_frames |> Enum.map(& &1.own.action) |> Enum.dedup() |> Enum.map(&ActionNames.name/1)
            row = %{
              game: Path.basename(path), frame: elem(frames, i).f, verdict: verdict, routes: routes, means: means,
              sd: i - last_hit > 90,
              at: if(e.actionable, do: %{x: Float.round(e.actionable.x * 1.0, 1), y: Float.round(e.actionable.y * 1.0, 1),
                vx: Float.round((e.actionable.speed_air_x_self || 0.0) * 1.0, 2), vy: Float.round((e.actionable.speed_y_self || 0.0) * 1.0, 2),
                jumps: e.actionable.jumps_left, facing: e.actionable.facing}),
              frames_offstage: if(e.since, do: i - e.since),
              specials: specials, jumps_used: jumps_used, b_presses: b_presses,
              toward_stage_share: Float.round(Enum.count(path_frames, toward) / max(length(path_frames), 1), 2),
              actions: Enum.take(actions, -8)
            }
            [row | out]
          else
            out
          end
        {%{e | last_hit: last_hit, off: now_off, actionable: actionable, since: since}, out}
      end)
    Output.puts("  #{Path.basename(path)}: #{n} frames, #{length(rows)} deaths")
    Enum.reverse(rows)
  end)

attempt = fn r ->
  cond do
    r.verdict == :kill -> "died in stun"
    r.specials == [] and (r.jumps_used || 0) == 0 -> "nothing (no jump, no special)"
    r.specials == [] -> "jump only"
    :up_b in r.specials -> "up-B"
    :side_b in r.specials -> "side-B, no up-B"
    true -> Enum.map_join(r.specials, "+", &to_string/1)
  end
end
rows = Enum.map(deaths, &Map.put(&1, :attempt, attempt.(&1)))
by = fn key -> rows |> Enum.group_by(key) |> Enum.map(fn {k, v} -> {k, length(v)} end) |> Enum.sort_by(&(-elem(&1, 1))) end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{label: label, deaths: rows}, pretty: true))
end

Output.puts("RESULT #{label}: #{length(rows)} deaths, #{Enum.count(rows, & &1.sd)} SDs (no hit within 90 f)")
Output.puts("RESULT #{label} by verdict: " <> Enum.map_join(by.(& &1.verdict), "  ", fn {k, n} -> "#{k} #{n}" end))
Output.puts("RESULT #{label} by attempt: " <> Enum.map_join(by.(& &1.attempt), "  ", fn {k, n} -> "#{k} #{n}" end))
Output.puts("RESULT #{label} mixup deaths by attempt: " <>
  Enum.map_join(by.(fn r -> if r.verdict == :mixup, do: r.attempt, else: :other end), "  ", fn {k, n} -> "#{k} #{n}" end))
Output.puts("")
Output.puts(String.pad_trailing("game@frame", 32) <> "verdict  routes  at(x,y,vx,vy,j)              off-f  attempt                       toward  actions (last)")
for r <- rows do
  at = if r.at, do: "(#{r.at.x},#{r.at.y},#{r.at.vx},#{r.at.vy},#{r.at.jumps})", else: "-"
  Output.puts(String.pad_trailing("#{String.slice(r.game, 5..19)}@#{r.frame}", 32) <>
    String.pad_trailing("#{r.verdict}", 9) <> String.pad_trailing("#{r.routes || "-"}", 8) <> String.pad_trailing(at, 30) <>
    String.pad_trailing("#{r.frames_offstage || "-"}", 7) <> String.pad_trailing(r.attempt, 30) <>
    String.pad_trailing("#{r.toward_stage_share}", 8) <> Enum.join(r.actions, ">"))
end
