# Set review export (2026-10-24): ONE of Bradley's own ranked sets, read the
# way a player reviews it — the big moments (stock losses, big combos), each
# with a clip, plus everything the review page needs to show the ghosts of
# the last 15 frames and both controllers frame by frame (what rwing shows).
#
#   mix run scripts/set_review_export.exs --me "tv／spikenarding" \
#     --slp ~/Slippi/2026-10-Mainline/Game_20261022T170737.slp --slp ... --out eval_runs/1024_set_review
#
# Per game: <id>.msltrace.json (the whole game for the viewer pane, index =
# frame), <id>.game.json (per-frame arrays for both players: position,
# facing, action, percent, stocks, hitstun, shield; both raw controllers;
# the moments; the stage geometry). manifest.json lists the set.
# No model in here: the moments are ground truth from the replay. The
# model's view (scripts/agreement_eval.exs rows) attaches later as the
# developer layer.

alias ExPhil.Data.Peppi
alias ExPhil.Sim.Trace
alias ExPhil.Training.Output
alias ExPhil.Interp.ActionNames

{opts, _, bad} = OptionParser.parse(System.argv(), strict: [me: :string, slp: [:string, :keep], out: :string, label: :string])
if bad != [], do: raise("bad options: #{inspect(bad)}")
me_name = opts[:me] || raise("--me NETPLAY_NAME required")
paths = Keyword.get_values(opts, :slp)
if paths == [], do: raise("--slp PATH (repeatable) required")
out = opts[:out] || "eval_runs/#{Date.utc_today() |> Calendar.strftime("%m%d")}_set_review"
File.mkdir_p!(out)

Output.banner("Set review export: #{length(paths)} games of #{me_name}")

# tuples -> lists so the geometry encodes
deep = fn
  deep, t when is_tuple(t) -> t |> Tuple.to_list() |> Enum.map(&deep.(deep, &1))
  deep, l when is_list(l) -> Enum.map(l, &deep.(deep, &1))
  deep, %{__struct__: _} = s -> s |> Map.from_struct() |> Map.new(fn {k, v} -> {k, deep.(deep, v)} end)
  deep, m when is_map(m) -> Map.new(m, fn {k, v} -> {k, deep.(deep, v)} end)
  _deep, v -> v
end

buttons_mask = fn c ->
  [c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r, c.button_start, c.button_d_up, c.button_d_down, c.button_d_left, c.button_d_right]
  |> Enum.with_index()
  |> Enum.reduce(0, fn {b, i}, acc -> if b, do: Bitwise.bor(acc, Bitwise.bsl(1, i)), else: acc end)
end

r1 = fn v -> if is_number(v), do: Float.round(v * 1.0, 2), else: nil end

games =
  paths
  |> Enum.with_index()
  |> Enum.map(fn {path, gi} ->
    {:ok, meta} = Peppi.metadata(path)
    me = Enum.find(meta.players, &(&1.netplay_name == me_name)) || raise("#{me_name} not in #{path}: #{inspect(Enum.map(meta.players, & &1.netplay_name))}")
    opp = Enum.find(meta.players, &(&1.port != me.port))
    {:ok, replay} = Peppi.parse(path)
    raw = replay.frames |> Enum.filter(&(&1.frame_number >= 0)) |> Enum.sort_by(& &1.frame_number)
    first = hd(raw).frame_number
    n = length(raw)

    # the viewer trace: me in slot 1 (remap), index = frame
    states =
      replay
      |> Peppi.to_training_frames(player_port: me.port, opponent_port: opp.port, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.game_state)

    id = Path.basename(path, ".slp")
    chars = [Trace.char_id(me.character_name), Trace.char_id(opp.character_name)]
    stage = states |> List.first() |> Map.get(:stage)
    label = "#{id} — #{me_name} (#{me.character_name}, red) vs #{opp.netplay_name} (#{opp.character_name})"
    Trace.from_game_states(states, label: label, chars: chars, stage: stage) |> Trace.write!(Path.join(out, "#{id}.msltrace.json"))
    # ghosts: players 3/4 = 1/2 fifteen frames ago, drawn by the viewer in the port-3/4 colours
    Trace.from_game_states(states, label: label <> " + ghosts (15 f ago)", chars: chars, stage: stage, ghosts: 15) |> Trace.write!(Path.join(out, "#{id}.ghost.msltrace.json"))

    # per-frame arrays, both players (raw ports)
    pf = fn f, port ->
      p = Map.get(f.players, port)
      if p == nil, do: nil, else: [r1.(p.x), r1.(p.y), (if p.facing && p.facing < 0, do: -1, else: 1), p.action, p.action_frame, r1.(p.percent), p.stock, (if p.on_ground, do: 1, else: 0), p.hitstun_frames_left || 0, r1.(p.shield_strength), (if p.invulnerable, do: 1, else: 0)]
    end
    cf = fn f, port ->
      p = Map.get(f.players, port)
      c = p && p.controller
      if c == nil, do: nil, else: [buttons_mask.(c), r1.(c.main_stick_x), r1.(c.main_stick_y), r1.(c.c_stick_x), r1.(c.c_stick_y), r1.(c.l_trigger), r1.(c.r_trigger)]
    end
    me_f = Enum.map(raw, &pf.(&1, me.port))
    op_f = Enum.map(raw, &pf.(&1, opp.port))
    me_c = Enum.map(raw, &cf.(&1, me.port))
    op_c = Enum.map(raw, &cf.(&1, opp.port))

    # ---- moments: stock losses + big combos, ground truth ----------------------------
    arr = List.to_tuple(raw)
    get = fn i, port -> Map.get(elem(arr, i).players, port) end
    ports = %{me: me.port, opp: opp.port}
    who = fn port -> if port == me.port, do: "me", else: "opp" end

    # hits taken: frames where a player's percent rose; the dealer's action at that frame names the move
    hits =
      for i <- 1..(n - 1), {_, port} <- ports, p0 = get.(i - 1, port), p1 = get.(i, port), p0 && p1, (p1.percent || 0) > (p0.percent || 0) + 0.01 do
        other = if port == me.port, do: opp.port, else: me.port
        # the move: the dealer's action on the hit frame, or its last ATTACK action within 12 frames
        # (a hit can land while the dealer is already dashing/jumping again; 44..74 = ground/aerial attacks + landings, >= 341 = specials)
        attack? = fn a -> a != nil and (a in 44..74 or a >= 341) end
        d = get.(i, other)
        recent_attack = fn -> Enum.find_value(1..12, fn k -> dk = if(i - k >= 0, do: get.(i - k, other)); if(dk && attack?.(dk.action), do: dk.action) end) end
        mv = if attack?.(d && d.action), do: d.action, else: recent_attack.() || (d && d.action)
        %{frame: elem(arr, i).frame_number, victim: who.(port), damage: Float.round((p1.percent - p0.percent) * 1.0, 1), to: r1.(p1.percent), move: mv, move_name: ActionNames.name(mv || 0)}
      end

    deaths =
      for i <- 1..(n - 1), {_, port} <- ports, p0 = get.(i - 1, port), p1 = get.(i, port), p0 && p1, (p1.stock || 0) < (p0.stock || 0) do
        f = elem(arr, i).frame_number
        # the killing blow: the last hit of ≥ 5 % in the 4 s before the death (a 1 % tap after the real hit is not it), else the last hit
        recent = hits |> Enum.filter(&(&1.victim == who.(port) and &1.frame <= f and &1.frame >= f - 240))
        last_hit = Enum.filter(recent, &(&1.damage >= 5)) |> List.last() || List.last(recent)
        %{type: "stock_loss", frame: f, who: who.(port), percent: r1.(p0.percent), stocks_left: p1.stock,
          killed_by: last_hit && last_hit.move_name, last_hit_frame: last_hit && last_hit.frame,
          clip: [max(first, f - 600), min(first + n - 1, f + 45)]}
      end

    # combos: hits on one victim with gaps ≤ 45 frames; keep ≥ 35 %
    combos =
      hits
      |> Enum.group_by(& &1.victim)
      |> Enum.flat_map(fn {victim, hs} ->
        hs
        |> Enum.sort_by(& &1.frame)
        |> Enum.chunk_while([], fn h, acc ->
          case acc do
            [] -> {:cont, [h]}
            [prev | _] when h.frame - prev.frame <= 45 -> {:cont, [h | acc]}
            _ -> {:cont, Enum.reverse(acc), [h]}
          end
        end, fn acc -> if acc == [], do: {:cont, []}, else: {:cont, Enum.reverse(acc), []} end)
        |> Enum.map(fn run ->
          dmg = run |> Enum.map(& &1.damage) |> Enum.sum() |> Float.round(1)
          %{type: "combo", victim: victim, who: (if victim == "me", do: "opp", else: "me"), frame: hd(run).frame, end_frame: List.last(run).frame, hits: length(run), damage: dmg,
            moves: run |> Enum.map(& &1.move_name) |> Enum.dedup(), from: hd(run).to - hd(run).damage, to: List.last(run).to,
            clip: [max(first, hd(run).frame - 120), min(first + n - 1, List.last(run).frame + 60)]}
        end)
        |> Enum.filter(&(&1.damage >= 35))
      end)
      # a combo that ends in a death belongs to the death moment
      |> Enum.reject(fn c -> Enum.any?(deaths, &(&1.who == c.victim and &1.frame >= c.end_frame and &1.frame - c.end_frame <= 90)) end)

    deaths = Enum.map(deaths, fn d ->
      c = hits |> Enum.filter(&(&1.victim == d.who and &1.frame <= d.frame)) |> Enum.reverse()
          |> Enum.chunk_while([], fn h, acc -> case acc do [] -> {:cont, [h]}; [prev | _] when prev.frame - h.frame <= 45 -> {:cont, [h | acc]}; _ -> {:halt, acc} end end, fn acc -> {:cont, acc, []} end)
          |> List.first()
      Map.merge(d, %{combo_hits: (c && length(c)) || 0, combo_damage: (c && Float.round(Enum.sum(Enum.map(c, & &1.damage)) * 1.0, 1)) || 0, combo_moves: (c && c |> Enum.map(& &1.move_name) |> Enum.dedup()) || []})
    end)

    # ---- situation episodes (ExPhil.Situations, ego = me) --------------------------------
    {sits, _} = Enum.map_reduce(states, ExPhil.Situations.new_context(), fn gs, ctx -> {c2, set} = ExPhil.Situations.fold(ctx, gs, 1); {set, c2} end)
    sit_arr = List.to_tuple(sits)
    # the training frames drop the trailing frame(s) without a successor input: fewer than n
    ns = tuple_size(sit_arr)
    frame_of = fn i -> first + i end
    episodes = fn label, max_gap ->
      Enum.reduce(0..(ns - 1), {[], nil}, fn i, {acc, open} ->
        has = MapSet.member?(elem(sit_arr, i), label)
        case {has, open} do
          {true, nil} -> {acc, {i, i}}
          {true, {a, _}} -> {acc, {a, i}}
          {false, {a, b}} when i - b > max_gap -> {[{a, b} | acc], nil}
          _ -> {acc, open}
        end
      end)
      |> then(fn {acc, open} -> if open, do: [open | acc], else: acc end)
      |> Enum.reverse()
    end
    death_frames = fn who -> deaths |> Enum.filter(&(&1.who == who)) |> Enum.map(& &1.frame) end
    died_within = fn who, f_end, k -> Enum.any?(death_frames.(who), &(&1 >= f_end - 5 and &1 <= f_end + k)) end
    pct_at = fn i, port -> (get.(i, port) && get.(i, port).percent) || 0 end

    # edgeguards: :edgeguard = we edgeguard them, :being_edgeguarded = they edgeguard us
    edgeguards =
      for {label, victim} <- [{:edgeguard, "opp"}, {:being_edgeguarded, "me"}], {a, b} <- episodes.(label, 45), b - a >= 20 do
        vport = if victim == "me", do: me.port, else: opp.port
        died = died_within.(victim, frame_of.(b), 90)
        %{type: "edgeguard", who: (if victim == "me", do: "opp", else: "me"), victim: victim, frame: frame_of.(a), end_frame: frame_of.(b), frames: b - a,
          outcome: (if died, do: "stock", else: "returned"), damage: Float.round((pct_at.(b, vport) - pct_at.(a, vport)) * 1.0, 1),
          clip: [max(first, frame_of.(a) - 60), min(first + n - 1, frame_of.(b) + 60)]}
      end

    # my recoveries (offstage low): made it back or not
    recoveries =
      for {a, b} <- episodes.(:recovery_low, 30), b - a >= 15 do
        died = died_within.("me", frame_of.(b), 60)
        %{type: "recovery", who: "me", victim: "me", frame: frame_of.(a), end_frame: frame_of.(b), frames: b - a, outcome: (if died, do: "stock", else: "returned"),
          clip: [max(first, frame_of.(a) - 90), min(first + n - 1, frame_of.(b) + 45)]}
      end
      # drop the recoveries already covered by an edgeguard episode on us
      |> Enum.reject(fn r -> Enum.any?(edgeguards, &(&1.victim == "me" and &1.frame <= r.end_frame and &1.end_frame >= r.frame)) end)

    # missed techs (knockdown bounce) by either player
    missed_techs =
      for {_, port} <- ports, {a, b} <- (Enum.reduce(0..(n - 1), {[], nil}, fn i, {acc, open} ->
            p = get.(i, port); bounce = p && trunc(p.action || 0) in [183, 191, 185, 193]
            case {bounce, open} do {true, nil} -> {acc, {i, i}}; {true, {x, _}} -> {acc, {x, i}}; {false, {x, y}} -> {[{x, y} | acc], nil}; _ -> {acc, open} end
          end) |> then(fn {acc, open} -> if open, do: [open | acc], else: acc end) |> Enum.reverse()) do
        %{type: "missed_tech", who: who.(port), victim: who.(port), frame: frame_of.(a), end_frame: frame_of.(b), percent: r1.(pct_at.(a, port)),
          clip: [max(first, frame_of.(a) - 90), min(first + n - 1, frame_of.(b) + 90)]}
      end

    # openings: the first hit after ≥ 90 frames with no hit either way (a neutral win)
    openings =
      hits
      |> Enum.sort_by(& &1.frame)
      |> Enum.chunk_every(2, 1, [nil])
      |> Enum.reduce({[], nil}, fn [h, _], {acc, last_f} ->
        if last_f == nil or h.frame - last_f >= 90, do: {[h | acc], h.frame}, else: {acc, h.frame}
      end)
      |> elem(0) |> Enum.reverse()
      |> Enum.map(fn h ->
        dealer = if h.victim == "me", do: "opp", else: "me"
        follow = hits |> Enum.filter(&(&1.victim == h.victim and &1.frame >= h.frame and &1.frame <= h.frame + 240)) |> Enum.map(& &1.damage) |> Enum.sum() |> Float.round(1)
        %{type: "opening", who: dealer, victim: h.victim, frame: h.frame, move: h.move_name, damage: h.damage, follow_up_damage: follow,
          clip: [max(first, h.frame - 180), min(first + n - 1, h.frame + 120)]}
      end)

    sds = deaths |> Enum.filter(&(&1.killed_by == nil)) |> Enum.map(&%{&1 | type: "sd"})
    deaths = Enum.reject(deaths, &(&1.killed_by == nil))

    moments = (deaths ++ sds ++ combos ++ edgeguards ++ recoveries ++ missed_techs ++ openings) |> Enum.sort_by(& &1.frame)
    last = elem(arr, n - 1)
    final = %{me: get.(n - 1, me.port) && get.(n - 1, me.port).stock, opp: get.(n - 1, opp.port) && get.(n - 1, opp.port).stock}
    winner = cond do (final.me || 0) > (final.opp || 0) -> "me"; (final.me || 0) < (final.opp || 0) -> "opp"; true -> "?" end
    geo = deep.(deep, ExPhil.Situations.geometry(stage))

    File.write!(Path.join(out, "#{id}.game.json"), Jason.encode!(%{
      id: id, path: path, started_at: meta.started_at && to_string(meta.started_at), stage: stage, frames: n, first_frame: first,
      me: %{port: me.port, name: me.netplay_name, code: me.netplay_code, character: me.character_name},
      opp: %{port: opp.port, name: opp.netplay_name, code: opp.netplay_code, character: opp.character_name},
      final: final, winner: winner, geometry: geo,
      moments: moments, hits: hits,
      p: %{me: me_f, opp: op_f}, c: %{me: me_c, opp: op_c},
      p_fields: ["x", "y", "facing", "action", "action_frame", "percent", "stock", "on_ground", "hitstun", "shield", "invulnerable"],
      c_fields: ["buttons(a,b,x,y,z,l,r,start,dup,ddown,dleft,dright bits)", "main_x", "main_y", "c_x", "c_y", "l", "r"]
    }))

    counts = moments |> Enum.frequencies_by(& &1.type) |> Enum.map_join(", ", fn {k, v} -> "#{v} #{k}" end)
    Output.puts("  #{gi + 1}/#{length(paths)} #{id}: #{n} f, #{me.character_name} vs #{opp.character_name} (#{opp.netplay_name}) on stage #{stage}, final #{final.me}-#{final.opp} (#{winner}); moments: #{counts}; last frame #{last.frame_number}")
    %{id: id, frames: n, stage: stage, opp: opp.netplay_name, opp_character: opp.character_name, me_character: me.character_name, final: final, winner: winner, moments: length(moments), trace: "#{id}.msltrace.json", ghost_trace: "#{id}.ghost.msltrace.json", game: "#{id}.game.json", started_at: meta.started_at && to_string(meta.started_at)}
  end)

wins = Enum.count(games, &(&1.winner == "me"))
manifest = %{created: DateTime.utc_now() |> DateTime.to_iso8601(), me: me_name, label: opts[:label] || "#{me_name} vs #{hd(games).opp}", games: games, score: "#{wins}-#{length(games) - wins}"}
File.write!(Path.join(out, "manifest.json"), Jason.encode!(manifest))
Output.puts("RESULT set review: #{length(games)} games, set #{manifest.score} -> #{out}/manifest.json")
