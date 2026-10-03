defmodule ExPhil.Melee.Checkmate do
  @moduledoc """
  Is this fighter checkmated for the stock? A position is checkmate when no
  combination of the recovery resources still available (drift, double jump,
  side special, up special, air dodge, one wall jump) can bring the fighter
  to a ledge, onto the stage, or onto a platform before the blast zone.

  The answer depends only on the state — how it was reached (hit away, air
  dodged offstage, side-B'd past the ledge) does not matter, and a partner's
  save in doubles undoes a checkmate without making it not one. Call it at
  the first actionable frame: hitstun, DI, SDI and teching come before that.

  The model is a frame-by-frame kinematic search over scripted plans. Ground
  truth is the sim (`scripts/recovery_probe.exs` TASes a replay position);
  this model is calibrated against it. Constants, Fox (PlFx.dat
  `ftCo_DatAttrs` / Fox special attributes, and sim measurements 2026-09-23
  on replay 170200 f12338 onward):

    * self velocity and knockback velocity are separate, as in the game;
      position moves by their sum. Knockback shrinks by 0.051/f in
      magnitude (replay speed_x/y_attack); jumps and specials reset only the
      self part.
    * gravity 0.23/f, terminal fall 2.8/f, drift toward the stage 0.08/f up
      to 0.83/f, aerial friction 0.02/f
    * double jump: self vy 4.416 (3.68 x 1.2)
    * Fire Fox: 42 f charge — self vy 0, falling from frame 15 at 0.015/f²,
      self vx kept and bled by friction; 30 f travel at 3.8/f slowing
      0.083/f² along the stick angle (~77 units); 20 f end carrying the exit
      velocity with gravity and friction; then helpless (drift x0.4)
    * Illusion (air): 19 f startup hover, ~48 units of dash over 12 f
      (coach trace 170200 f12384-12445), then helpless; shorten lengths
    * air dodge (recorded in the game): 49 f; for 29 f Fox slides along the
      stick at 2.79/f shrinking x0.9 a frame (~27 units), no gravity, no
      drift; then 20 f of falling with gravity and no horizontal control;
      no ledge grab until it ends; then helpless
    * wall jump: self (1.4 away from the wall, 3.3), once, on a real wall
    * blast zones: sides and bottom always kill; the top only while upward
      knockback exceeds 2.4/f (ftCo_800D3158)
    * stage: the static collision shell from ExPhil.StageCollision; walls
      and ceilings stop motion, crossing a ground line lands
    * ledge grab: the game's test (mpColl_80044164 / 800443C4) at the
      data's grabbable ledges — only while falling, only the ledge Fox
      faces, the corner inside a box swept from last frame's position to
      this one, reaching Fox's ledge-snap x (11) past his ECB side point and
      his snap height (9) centred snap y (13) above him, his ECB bottom
      outside and below the corner, and no stage line between him and it.
      The ECB comes frame by frame from ExPhil.Melee.FoxEcb (recorded from
      the game). Air dodge cannot grab (ftCo_EscapeAir_Coll).
    * facing: from the state (default: toward the stage); Illusion turns
      Fox to its dash, Fire Fox to its travel direction, nothing else does.
  """

  alias ExPhil.Melee.FoxEcb
  alias ExPhil.Situations

  @grav 0.23
  @terminal 2.8
  @drift_max 0.83
  @drift_accel 0.08
  @friction 0.02
  @helpless_mobility 0.4
  @kb_decay 0.051
  @jump_vy 3.68 * 1.2
  @illusion_startup 19
  @illusion_dash 12
  @illusion_dist 48.0
  @illusion_lengths [1.0, 0.66, 0.33]
  @firefox_charge 42
  @firefox_hover 15
  @firefox_charge_fall 0.015
  @firefox_frames 30
  @firefox_speed 3.8
  @firefox_decel 0.083
  @firefox_end 20
  @airdodge_frames 49
  @airdodge_slide 29
  @airdodge_v0 2.79
  @airdodge_decay 0.9
  @walljump_vx 1.4
  @walljump_vy 3.3
  @horizon 300

  @doc "True when no plan in the search brings the fighter back."
  def checkmate?(state, opts \\ []), do: analyze(state, opts).checkmate?

  @doc """
  `%{checkmate?: bool, plan: plan | nil, outcome: :ledge | :stage | :platform | nil, frames: n | nil}`.
  State keys: `:stage` (external id), `:x`, `:y`, `:vx`/`:vy` (self velocity),
  `:kb_vx`/`:kb_vy` (knockback velocity), `:jumps_left`, `:up_b`, `:side_b`,
  `:air_dodge`, `:wall_jump` (booleans, default true), `:facing` (+1 / -1,
  default toward the stage).
  """
  def analyze(state, opts \\ []) do
    s = normalize(state)
    geo = geometry(s.stage)

    case Enum.find_value(plans(s, opts), fn plan -> simulate(s, geo, plan) end) do
      nil -> %{checkmate?: true, plan: nil, outcome: nil, frames: nil}
      {plan, outcome, frames} -> %{checkmate?: false, plan: plan, outcome: outcome, frames: frames}
    end
  end

  @doc """
  Which recovery routes survive, and how many: `%{count: n, verdict: v, routes: [...]}`
  with `verdict` `:checkmate` (0 routes), `:forced` (1 — the opponent has one
  thing to cover) or `:mixup` (2+). Same state keys and grid options as
  `analyze/2`, but every plan in the search is simulated.

  A route is a means plus a destination: the resources a plan spends, in
  order (`[]` = drift only, `[:jump]`, `[:up_b]`, `[:jump, :side_b]`, ...)
  and where it arrives (`:ledge`, `:stage`, `:platform`). Timing variants of
  the same means to the same place are one route — a ledge reached by Fire
  Fox at frame 0 or frame 30 is covered by the same ledgehog — and the
  timing freedom is reported per route instead: `plans` (grid plans that
  succeed), `delays` (earliest and latest start of the final action) and
  `fastest` (frames to arrive). Routes come cheapest-means first.
  """
  def routes(state, opts \\ []) do
    s = normalize(state)
    geo = geometry(s.stage)

    routes =
      plans(s, opts)
      |> Enum.flat_map(fn plan ->
        case simulate(s, geo, plan) do
          nil -> []
          {plan, outcome, frames} -> [{plan, outcome, frames}]
        end
      end)
      |> Enum.group_by(fn {plan, outcome, _} -> {Enum.map(plan, &elem(&1, 0)), outcome} end)
      |> Enum.map(fn {{means, outcome}, hits} ->
        starts = for {plan, _, _} <- hits, plan != [], do: plan |> List.last() |> elem(1)

        %{
          means: means,
          outcome: outcome,
          plans: length(hits),
          delays: if(starts == [], do: nil, else: Enum.min_max(starts)),
          fastest: hits |> Enum.map(&elem(&1, 2)) |> Enum.min()
        }
      end)
      |> Enum.sort_by(&{length(&1.means), &1.fastest})

    verdict =
      case length(routes) do
        0 -> :checkmate
        1 -> :forced
        _ -> :mixup
      end

    %{count: length(routes), verdict: verdict, routes: routes}
  end

  defp normalize(st) do
    %{
      stage: trunc(st[:stage] || 32),
      x: st[:x] * 1.0,
      y: st[:y] * 1.0,
      vx: (st[:vx] || 0.0) * 1.0,
      vy: (st[:vy] || 0.0) * 1.0,
      kvx: (st[:kb_vx] || 0.0) * 1.0,
      kvy: (st[:kb_vy] || 0.0) * 1.0,
      jumps_left: st[:jumps_left] || 0,
      up_b: Map.get(st, :up_b, true),
      side_b: Map.get(st, :side_b, true),
      air_dodge: Map.get(st, :air_dodge, true),
      wall_jump: Map.get(st, :wall_jump, true),
      facing: st[:facing]
    }
  end

  @doc """
  Stage geometry the search uses: the static collision shell from
  `ExPhil.StageCollision` (base group, non-platform lines as
  `{x1, y1, x2, y2, class}`), grabbable ledges `{x, y, side}`, platforms,
  blast zones. Cached per stage.
  """
  def geometry(stage) do
    case :persistent_term.get({__MODULE__, :geo, stage}, nil) do
      nil -> geo = build_geometry(stage); :persistent_term.put({__MODULE__, :geo, stage}, geo); geo
      geo -> geo
    end
  end

  defp build_geometry(stage) do
    geo = Situations.geometry(stage)
    {bl, br, bt, bb} = geo.blast || {-246, 246, 188, -140}
    data = ExPhil.StageCollision.data(stage)
    base = ExPhil.StageCollision.base_group(data.lines)
    lines = Enum.filter(data.lines, &(Map.get(&1, "group", 0) == base and not &1["drop_through"]))
    pt = fn i -> [x, y] = elem(data.vertices, i); {x * 1.0, y * 1.0} end
    shell = for l <- lines, {x1, y1} = pt.(l["v1"]), {x2, y2} = pt.(l["v2"]), do: {x1, y1, x2, y2, l["class"]}
    ledges =
      for {x, y} <- ExPhil.StageCollision.ledge_positions(stage), abs(x) >= geo.edge - 1.0, uniq: true, do: {x * 1.0, y * 1.0, sign(x)}
    %{edge: geo.edge, platforms: geo.platforms, blast_l: bl, blast_r: br, blast_t: bt, blast_b: bb, shell: shell, ledges: ledges, loops: loops(lines, pt)}
  end

  # Ordered outlines of the shell (for drawing): rewind each unvisited line
  # to the start of its chain, then follow v2 -> v1 forward.
  defp loops(lines, pt) do
    by_v1 = Map.new(lines, &{&1["v1"], &1})
    by_v2 = Map.new(lines, &{&1["v2"], &1})
    n = length(lines)

    {loops, _} =
      Enum.reduce(lines, {[], MapSet.new()}, fn l, {acc, seen} ->
        if MapSet.member?(seen, l["index"]) do
          {acc, seen}
        else
          head =
            Enum.reduce_while(1..n, l, fn _, cur ->
              pred = by_v2[cur["v1"]]
              if pred == nil or pred["index"] == l["index"] or MapSet.member?(seen, pred["index"]), do: {:halt, cur}, else: {:cont, pred}
            end)

          {pts, seen} = walk(head, by_v1, seen, [])
          {[Enum.map(pts, fn i -> Tuple.to_list(pt.(i)) end) | acc], seen}
        end
      end)

    Enum.reverse(loops)
  end

  defp walk(l, by_v1, seen, acc) do
    acc = [l["v1"] | acc]
    seen = MapSet.put(seen, l["index"])
    next = by_v1[l["v2"]]
    if next == nil or MapSet.member?(seen, next["index"]), do: {Enum.reverse([l["v2"] | acc]), seen}, else: walk(next, by_v1, seen, acc)
  end

  @doc "True when (x, y) is inside the stage's solid shell (even-odd rule)."
  def solid?(stage, x, y) do
    geometry(stage).shell
    |> Enum.count(fn {x1, y1, x2, y2, _} -> (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1) end)
    |> rem(2) == 1
  end

  # ---------------------------------------------------------------- plans

  @delays [0, 4, 8, 16, 30, 45, 60]
  @angles Enum.map(0..15, &(&1 * 22.5))
  @dodge_angles Enum.map(0..7, &(&1 * 45.0))

  # Illusion, Fire Fox and the air dodge all end helpless, so a plan holds at
  # most one of them, after an optional double jump. Cheapest plans first.
  # `:delays` / `:angles` / `:dodge_angles` override the grid (maps use a coarser one).
  defp plans(s, opts) do
    delays = Keyword.get(opts, :delays, @delays)
    angles = Keyword.get(opts, :angles, @angles)
    dodge_angles = Keyword.get(opts, :dodge_angles, @dodge_angles)

    finals =
      [nil] ++
        if(s.side_b, do: for(t <- delays, len <- @illusion_lengths, do: {:side_b, t, len}), else: []) ++
        if(s.up_b, do: for(t <- delays, a <- angles, do: {:up_b, t, a}), else: []) ++
        if(s.air_dodge, do: for(t <- delays, a <- dodge_angles, do: {:air_dodge, t, a}), else: [])

    jumps = if s.jumps_left > 0, do: [nil | Enum.map(delays, &{:jump, &1})], else: [nil]

    for(j <- jumps, f <- finals, do: Enum.reject([j, f], &is_nil/1))
    |> Enum.reject(fn
      [{:jump, j}, {_, t, _}] -> t <= j
      _ -> false
    end)
    |> Enum.sort_by(&length/1)
  end

  # ---------------------------------------------------------------- simulation

  defp simulate(s, geo, plan) do
    f = %{x: s.x, y: s.y, vx: s.vx, vy: s.vy, kvx: s.kvx, kvy: s.kvy, helpless: false, wall_jump: s.wall_jump, busy: nil, wall: nil, landed: false, anim: {:fall, 30}}
    f = Map.put(f, :facing, s.facing || if(s.x > 0, do: -1.0, else: 1.0))
    if dead?(f, geo), do: nil, else: step(f, geo, Enum.sort_by(plan, &elem(&1, 1)), plan, 0)
  end

  # Drift toward the stage centre, except under the stage, where the way out
  # is the nearer ledge.
  defp drift_dir(f, geo) do
    cond do
      f.y < 0.0 and abs(f.x) < geo.edge -> sign(f.x)
      f.x > 0 -> -1.0
      true -> 1.0
    end
  end

  defp step(_f, _geo, _queue, _plan, t) when t > @horizon, do: nil

  defp step(f, geo, queue, plan, t) do
    dir = drift_dir(f, geo)
    {f, queue} = start_actions(f, queue, dir, t)
    prev = f
    f = f |> self_velocity(dir) |> move(geo) |> advance_anim()

    cond do
      dead?(f, geo) -> nil
      outcome = arrived(prev, f, geo) -> {plan, outcome, t + 1}
      true -> step(maybe_wall_jump(f, geo), geo, queue, plan, t + 1)
    end
  end

  # A queued action starts on its frame if the fighter is free; otherwise it
  # is dropped, so impossible plans degrade to what is possible.
  defp start_actions(f, [{_, at, _} = a | rest], dir, t) when at <= t, do: start_actions(begin(f, a, dir), rest, dir, t)
  defp start_actions(f, [{_, at} = a | rest], dir, t) when at <= t, do: start_actions(begin(f, a, dir), rest, dir, t)
  defp start_actions(f, queue, _dir, _t), do: {f, queue}

  defp begin(%{busy: nil, helpless: false} = f, {:jump, _}, _dir), do: %{f | vy: @jump_vy, anim: {:jump, 0}}
  defp begin(%{busy: nil, helpless: false} = f, {:side_b, _, len}, dir),
    do: %{f | vx: 0.0, vy: 0.0, busy: {:side_b, @illusion_startup + @illusion_dash, dir, len}, facing: dir, anim: {:side_b, 0}}
  defp begin(%{busy: nil, helpless: false} = f, {:up_b, _, angle}, _dir), do: %{f | vy: 0.0, busy: {:charge, @firefox_charge, angle}, anim: {:charge, 0}}
  defp begin(%{busy: nil, helpless: false} = f, {:air_dodge, _, angle}, _dir), do: %{f | busy: {:air_dodge, @airdodge_frames, angle}, anim: {:air_dodge, 0}}
  defp begin(f, _a, _dir), do: f

  # Per-phase animation clock for the ECB lookup; phase changes are set where
  # the busy state changes (tick callbacks below) and in begin/3.
  defp advance_anim(%{anim: {k, t}} = f), do: %{f | anim: {k, t + 1}}

  defp self_velocity(%{busy: {:side_b, n, dir, len}} = f, _) do
    vx = if n > @illusion_dash, do: 0.0, else: dir * len * @illusion_dist / @illusion_dash
    %{f | vx: vx, vy: 0.0} |> tick(fn -> {:side_b, n - 1, dir, len} end, n, fn f -> %{f | vx: 0.0, helpless: true, anim: {:helpless, 0}} end)
  end

  defp self_velocity(%{busy: {:charge, n, angle}} = f, _) do
    elapsed = @firefox_charge - n
    vy = if elapsed < @firefox_hover, do: 0.0, else: f.vy - @firefox_charge_fall
    %{f | vx: toward_zero(f.vx, @friction), vy: vy}
    |> tick(fn -> {:charge, n - 1, angle} end, n, fn f ->
      c = :math.cos(angle * :math.pi() / 180.0)
      facing = if abs(c) > 0.1, do: sign(c), else: f.facing
      %{f | busy: {:travel, @firefox_frames, angle}, facing: facing, anim: {:"travel_#{FoxEcb.travel_bucket(angle)}", 0}}
    end)
  end

  defp self_velocity(%{busy: {:travel, n, angle}} = f, _) do
    r = angle * :math.pi() / 180.0
    speed = @firefox_speed - @firefox_decel * (@firefox_frames - n)
    %{f | vx: speed * :math.cos(r), vy: speed * :math.sin(r)}
    |> tick(fn -> {:travel, n - 1, angle} end, n, fn f -> %{f | busy: {:ff_end, @firefox_end}, anim: {:"end_#{FoxEcb.travel_bucket(angle)}", 0}} end)
  end

  defp self_velocity(%{busy: {:ff_end, n}} = f, _) do
    %{f | vx: toward_zero(f.vx, @friction), vy: max(-@terminal, f.vy - @grav)}
    |> tick(fn -> {:ff_end, n - 1} end, n, fn f -> %{f | helpless: true, anim: {:helpless, 0}} end)
  end

  defp self_velocity(%{busy: {:air_dodge, n, angle}} = f, _) do
    elapsed = @airdodge_frames - n

    f =
      if elapsed < @airdodge_slide do
        r = angle * :math.pi() / 180.0
        v = @airdodge_v0 * :math.pow(@airdodge_decay, elapsed)
        %{f | vx: v * :math.cos(r), vy: v * :math.sin(r)}
      else
        %{f | vx: 0.0, vy: if(elapsed == @airdodge_slide, do: 0.0, else: max(-@terminal, f.vy - @grav))}
      end

    tick(f, fn -> {:air_dodge, n - 1, angle} end, n, fn f -> %{f | vx: 0.0, helpless: true, anim: {:helpless, 0}} end)
  end

  defp self_velocity(f, dir) do
    accel = if f.helpless, do: @drift_accel * @helpless_mobility, else: @drift_accel
    vx = max(-@drift_max, min(@drift_max, f.vx + dir * accel))
    %{f | vx: vx, vy: max(-@terminal, f.vy - @grav)}
  end

  # Count a busy state down; on its last frame apply `done`.
  defp tick(f, next, n, done), do: if(n <= 1, do: done.(%{f | busy: nil}), else: %{f | busy: next.()})

  # Move by self + knockback velocity against the solid shell: crossing a
  # ground line downward lands; a ceiling stops the rise; a wall stops the
  # horizontal motion (and marks the frame for a wall jump).
  defp move(f, geo) do
    mag = :math.sqrt(f.kvx * f.kvx + f.kvy * f.kvy)
    {kvx, kvy} = if mag <= @kb_decay, do: {0.0, 0.0}, else: {f.kvx * (mag - @kb_decay) / mag, f.kvy * (mag - @kb_decay) / mag}
    dx = f.vx + f.kvx
    dy = f.vy + f.kvy
    moved = %{f | x: f.x + dx, y: f.y + dy, kvx: kvx, kvy: kvy, wall: nil, landed: false}

    case first_hit(f.x, f.y, dx, dy, geo.shell) do
      nil -> moved
      {t, "ground"} when dy < 0 -> %{moved | x: f.x + dx * t, y: f.y + dy * t, landed: true}
      {t, "ceiling"} when dy > 0 -> %{moved | y: f.y + dy * t - 0.01, vy: 0.0, kvy: 0.0}
      {t, wall} when wall in ["wall_left", "wall_right"] -> %{moved | x: f.x + dx * t - 0.01 * sign(dx), vx: 0.0, kvx: 0.0, wall: sign(dx)}
      _ -> moved
    end
  end

  defp first_hit(x, y, dx, dy, shell) do
    shell
    |> Enum.flat_map(fn {x1, y1, x2, y2, class} ->
      ex = x2 - x1
      ey = y2 - y1
      den = dx * ey - dy * ex
      if abs(den) < 1.0e-9 do
        []
      else
        t = ((x1 - x) * ey - (y1 - y) * ex) / den
        u = ((x1 - x) * dy - (y1 - y) * dx) / den
        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0, do: [{t, class}], else: []
      end
    end)
    |> Enum.min_by(&elem(&1, 0), fn -> nil end)
  end

  defp toward_zero(v, d) when v > d, do: v - d
  defp toward_zero(v, d) when v < -d, do: v + d
  defp toward_zero(_, _), do: 0.0

  # Wall jump: touching a wall this frame, free and not helpless, jump away.
  defp maybe_wall_jump(%{wall_jump: true, busy: nil, helpless: false, wall: w} = f, _geo) when w != nil do
    %{f | vx: -@walljump_vx * w, vy: @walljump_vy, kvx: 0.0, kvy: 0.0, wall_jump: false, anim: {:jump, 0}}
  end

  defp maybe_wall_jump(f, _), do: f

  # ftCo_800D3158: side and bottom zones always kill; the top one only while
  # upward knockback exceeds ftCommonData.x4F0 (2.4) for an airborne fighter
  # (the other clauses are grounded, shield-break flight and freeze).
  @top_kill_kb 2.4
  defp dead?(f, geo), do: f.y < geo.blast_b or f.x < geo.blast_l or f.x > geo.blast_r or (f.y > geo.blast_t and f.kvy > @top_kill_kb)

  defp arrived(prev, f, geo) do
    dy = f.y - prev.y

    cond do
      f.landed -> :stage
      grabs_ledge?(prev, f, geo) -> :ledge
      Enum.any?(geo.platforms, fn {h, l, r} -> dy <= 0 and f.x >= l and f.x <= r and prev.y >= h and f.y <= h end) -> :platform
      true -> nil
    end
  end

  # mpColl_80044164 (left ledge, fighter facing +1) and its mirror
  # mpColl_800443C4 (right ledge, facing -1), for a static corner.
  defp grabs_ledge?(prev, f, geo) do
    {snap_x, snap_y, snap_h} = FoxEcb.snap()
    air_dodging = match?({:air_dodge, _, _}, f.busy)

    if air_dodging or f.y >= prev.y do
      false
    else
      {kind, t} = f.anim
      kind = if kind == :air_dodge, do: :helpless, else: kind
      {bottom_y, right_x, top_y} = FoxEcb.lookup(kind, t)
      bottom = min(prev.y, f.y) + snap_y - snap_h / 2
      top = max(prev.y, f.y) + snap_y + snap_h / 2

      Enum.any?(geo.ledges, fn {lx, ly, side} ->
        faces = f.facing == -side
        # side -1 = left ledge: the box runs from the fighter rightward.
        reach_ok =
          if side < 0,
            do: lx <= snap_x + max(prev.x, f.x) + right_x and f.x < lx,
            else: lx >= -snap_x + min(prev.x, f.x) - right_x and f.x > lx

        faces and reach_ok and ly >= bottom and ly <= top and f.y + bottom_y < ly and
          clear_line?(f.x, f.y + top_y, lx, ly, geo) and clear_line?(f.x, f.y + bottom_y - 2.0, lx, ly, geo)
      end)
    end
  end

  # No shell line crosses the segment, ignoring lines that end at the corner.
  defp clear_line?(x1, y1, x2, y2, geo) do
    dx = x2 - x1
    dy = y2 - y1

    not Enum.any?(geo.shell, fn {ax, ay, bx, by, _} ->
      touches = (abs(ax - x2) < 0.01 and abs(ay - y2) < 0.01) or (abs(bx - x2) < 0.01 and abs(by - y2) < 0.01)
      ex = bx - ax
      ey = by - ay
      den = dx * ey - dy * ex

      not touches and abs(den) > 1.0e-9 and
        (fn ->
           t = ((ax - x1) * ey - (ay - y1) * ex) / den
           u = ((ax - x1) * dy - (ay - y1) * dx) / den
           t > 0.0 and t < 1.0 and u >= 0.0 and u <= 1.0
         end).()
    end)
  end
  defp sign(v) when v < 0, do: -1.0
  defp sign(_), do: 1.0
end
