# Fit the PS water-transformation windmill's rotation rate (task #5).
#
# RESOLVED 2026-08-12: the exact rate is the DECOMP CONSTANT — melee
# decomp grpstadium.c:579, HSD_JObjAddRotationZ(windmill_jobj,
# -0.5F * deg_to_rad) per frame => -0.5 deg/frame CW, 720 frames
# (12.000 s) per revolution. This script remains as the empirical
# cross-check: its estimators bracketed the truth (-0.73 raw median /
# -0.37 slide-corrected deg/frame vs -0.5 true) and confirmed CW; the
# residual bias is blade-slide contamination (see below).
#
# Hub (-36.64, 38.8) and CW direction were fitted 2026-08-11 (bisector
# method over riding chords), but the rate resisted chord-slope fitting:
# riders SLIDE on the wheel (grounded players are carried 1:1 by the
# surface, conveyor-style, and walk against it), so a rider's angle
# about the hub is not the wheel's angle. First fit attempt (per-chord
# theta(t) OLS, 2026-08-12) confirmed: chord slopes scatter -0.005..
# -0.042 rad/frame with a pile of static-floor standers at 0.
#
# The estimator that survives contact physics: a grounded player is
# displaced each frame by the LOCAL SURFACE velocity (plus their own
# walking). At contact point r (hub-relative), rigid rotation gives
# v_surface = omega x r, so
#
#   omega_i = (r_x*v_y - r_y*v_x) / |r|^2        (per frame-pair)
#
# exactly, independent of where on the wheel they stand. Contamination
# and its filters:
#   - static geometry standers: positively excluded — samples within
#     2.5u of any KNOWN static segment (base shell + pinned water
#     structures from pokemon_stadium_types.json) are dropped
#   - walking adds tangential velocity: require near-neutral main stick
#   - knockback/landing frames: require grounded on BOTH frames and
#     sane per-frame speed (< 5 u/frame)
# Report the median + quartiles over all surviving frame-pairs.
#
#   mix run scripts/fit_windmill_rate.exs --replays "~/Slippi/2026-08-Mainline/*.slp"
#     [--r-min 6] [--r-max 45] [--stick 0.25] [--json OUT.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.StageCollision
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, r_min: :float, r_max: :float, stick: :float, json: :string]
  )

replays =
  (opts[:replays] || raise("--replays is required"))
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.sort()

if replays == [], do: raise("no replays matched")

{hub_x, hub_y} = {-36.64, 38.8}
r_min = opts[:r_min] || 6.0
r_max = opts[:r_max] || 45.0
stick_tol = opts[:stick] || 0.25

Output.banner("Windmill rate fit v2")
Output.config([
  {"Replays", length(replays)},
  {"Radius band", "#{r_min}..#{r_max}"},
  {"Stick tolerance", stick_tol}
])

# ---------------------------------------------------------------------------
# Known static geometry near the wheel: base shell + pinned water
# structures. Riding samples must NOT be explained by any of these.
# ---------------------------------------------------------------------------

base = File.read!("priv/stage_collision/pokemon_stadium.json") |> Jason.decode!()
varr = List.to_tuple(base["vertices"])

base_segs =
  for l <- base["lines"], Map.get(l, "group", 0) == StageCollision.base_group(base["lines"]) do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])
    {x1, y1, x2, y2}
  end

water_segs =
  case File.read("priv/stage_collision/pokemon_stadium_types.json") do
    {:ok, raw} ->
      for %{"seg" => [x1, y1, x2, y2]} <- get_in(Jason.decode!(raw), ["9", "segments"]) || [],
          do: {x1, y1, x2, y2}

    _ ->
      []
  end

static_segs = base_segs ++ water_segs

on_static? = fn x, y ->
  Enum.any?(static_segs, &(StageCollision.point_segment_distance(x, y, &1) < 2.5))
end

# Raw pre-frame stick on the player frame, normalized 0..1 (0.5 center)
neutral_stick? = fn pl ->
  mx = Map.get(pl, :main_stick_x)
  my = Map.get(pl, :main_stick_y)

  if is_number(mx) and is_number(my) do
    abs(mx - 0.5) < stick_tol / 2 and abs(my - 0.5) < stick_tol / 2
  else
    # No controller data: keep the sample (velocity filters still apply)
    true
  end
end

# ---------------------------------------------------------------------------
# Per-frame-pair angular velocity samples during stable water phases
# ---------------------------------------------------------------------------

samples =
  Enum.flat_map(replays, fn path ->
    case Peppi.parse(path, player_port: 1) do
      {:ok, r} ->
        {tagged, _} =
          Enum.map_reduce(r.frames, {nil, nil}, fn f, {ev, ty} ->
            ev = f.stadium_event || ev
            ty = f.stadium_type || ty
            {{ev, ty, f}, {ev, ty}}
          end)

        tagged
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.with_index()
        |> Enum.flat_map(fn {[{ev, ty, fa}, {ev2, ty2, fb}], idx} ->
          if ev == 0 and ty == 9 and ev2 == 0 and ty2 == 9 do
            for {port, pa} <- fa.players || %{},
                pb = (fb.players || %{})[port],
                pb != nil,
                pa.on_ground and pb.on_ground,
                is_number(pa.x) and is_number(pa.y) and is_number(pb.x) and is_number(pb.y),
                rx = pa.x - hub_x,
                ry = pa.y - hub_y,
                s2 = rx * rx + ry * ry,
                s = :math.sqrt(s2),
                s >= r_min and s <= r_max,
                not on_static?.(pa.x, pa.y),
                neutral_stick?.(pa),
                # Not self-propelled: walking/dashing shows up in the
                # self ground-speed channel; carried motion does not
                abs(Map.get(pa, :speed_ground_x_self) || 0.0) < 0.05,
                abs(Map.get(pb, :speed_ground_x_self) || 0.0) < 0.05,
                vx = pb.x - pa.x,
                vy = pb.y - pa.y,
                :math.sqrt(vx * vx + vy * vy) < 5.0 do
              %{
                omega: (rx * vy - ry * vx) / s2,
                s: s,
                rx: rx,
                ry: ry,
                vx: vx,
                vy: vy,
                # Radial slide rate over s^2: the slide-contamination
                # regressor (contact line offset w from the wheel center
                # biases omega_obs by -/+ w * sdot / s^2)
                z: ((pb.x - hub_x) * (pb.x - hub_x) + (pb.y - hub_y) * (pb.y - hub_y) - s2) /
                     (2 * s * s2),
                idx: idx,
                file: Path.basename(path),
                port: port
              }
            end
          else
            []
          end
        end)

      _ ->
        []
    end
  end)

n = length(samples)
Output.puts("#{n} rider frame-pair sample(s)")
if n < 30, do: (Output.error("too few samples — need more water-phase riding footage"); System.halt(2))

omegas = samples |> Enum.map(& &1.omega) |> Enum.sort()
q = fn p -> Enum.at(omegas, min(round(p * (n - 1)), n - 1)) end
med = q.(0.5)

# Zero-velocity standers that slipped the static filter show up as an
# omega spike at 0; report the share so contamination is visible.
still = Enum.count(omegas, &(abs(&1) < 1.0e-4))

# ---------------------------------------------------------------------------
# Slide-corrected fit: omega_obs_i = omega + c_k * z_i, with c_k (=-/+w,
# the blade-face offset) per riding STREAK (same file/port, consecutive
# frames — the rider stays on one blade face within a streak, so the
# sign of the offset is constant). Alternate closed-form updates of c_k
# and omega; converges in a few rounds.
# ---------------------------------------------------------------------------

streaks =
  samples
  |> Enum.group_by(&{&1.file, &1.port})
  |> Enum.flat_map(fn {_key, ss} ->
    ss
    |> Enum.sort_by(& &1.idx)
    |> Enum.chunk_while([], fn s, acc ->
      case acc do
        [prev | _] when s.idx > prev.idx + 1 -> {:cont, Enum.reverse(acc), [s]}
        _ -> {:cont, [s | acc]}
      end
    end, fn acc -> {:cont, Enum.reverse(acc), []} end)
  end)
  |> Enum.filter(&(length(&1) >= 5))

corrected_omega =
  Enum.reduce(1..30, med, fn _, om ->
    cs =
      Enum.map(streaks, fn ss ->
        num = Enum.sum(Enum.map(ss, &(&1.z * (&1.omega - om))))
        den = Enum.sum(Enum.map(ss, &(&1.z * &1.z)))
        if den > 1.0e-12, do: num / den, else: 0.0
      end)

    resids =
      streaks
      |> Enum.zip(cs)
      |> Enum.flat_map(fn {ss, c} -> Enum.map(ss, &(&1.omega - c * &1.z)) end)

    Enum.sum(resids) / length(resids)
  end)

deg_per_frame = med * 180.0 / :math.pi()
sec_per_rev = if med != 0.0, do: Float.round(abs(2 * :math.pi() / med) / 60.0, 3), else: :infinity

Output.puts("")
Output.puts("omega (median): #{Float.round(med, 6)} rad/frame")
Output.puts("  = #{Float.round(deg_per_frame, 4)} deg/frame")
Output.puts("  = #{sec_per_rev} s/revolution")
Output.puts("  direction: #{if med < 0, do: "CW", else: "CCW"} (math convention, +y up)")
Output.puts("  quartiles: #{Enum.map([0.1, 0.25, 0.5, 0.75, 0.9], &Float.round(q.(&1), 5)) |> Enum.join(" / ")}")
Output.puts("  near-zero samples (likely static leakage): #{still}/#{n}")
Output.puts("")
Output.puts("slide-corrected omega: #{Float.round(corrected_omega, 6)} rad/frame")
Output.puts("  = #{Float.round(corrected_omega * 180 / :math.pi(), 4)} deg/frame")
Output.puts("  = #{Float.round(abs(2 * :math.pi() / corrected_omega) / 60.0, 3)} s/revolution")
Output.puts("  (#{length(streaks)} streaks >= 5 samples)")

if opts[:json] do
  File.write!(
    opts[:json],
    Jason.encode!(
      %{
        hub: [hub_x, hub_y],
        omega_rad_per_frame: med,
        deg_per_frame: deg_per_frame,
        sec_per_rev: sec_per_rev,
        n_samples: n,
        near_zero: still,
        quantiles: Map.new([0.1, 0.25, 0.5, 0.75, 0.9], &{to_string(&1), q.(&1)}),
        samples:
          Enum.map(samples, &Map.take(&1, [:omega, :s, :rx, :ry, :vx, :vy, :port])),
        by_file:
          samples
          |> Enum.group_by(& &1.file)
          |> Map.new(fn {f, ss} ->
            os = ss |> Enum.map(& &1.omega) |> Enum.sort()
            {f, %{n: length(os), median: Enum.at(os, div(length(os), 2))}}
          end)
      },
      pretty: true
    )
  )

  Output.success("wrote #{opts[:json]}")
end
