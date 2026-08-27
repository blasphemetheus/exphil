# Fit the PS windmill's blade PHASE at water-transformation start
# (task #5 follow-up; the RATE is exact: decomp grpstadium.c:579,
# -0.5 deg/frame CW, 720 frames/rev).
#
# VERDICT (2026-08-12, 12 water phases): the phase is NOT deterministic
# from any replay-visible anchor — same-game phase pairs disagree under
# every --anchor tested, and the decomp's water init never sets the
# jobj rotation (AddRotationZ only accumulates). Within-phase
# concentration is 0.95-0.99 though, so the phase is a per-water-phase
# LATENT, cleanly fittable whenever riders touch the wheel. The rewind
# viewer now runs this exact fit per session (fitWindmillPhases).
#
# Model: with t_rel = frames since the water phase's first STABLE frame
# (stadium_event 0, type 9 — the morph animation before it is
# fixed-length, so any anchor inside the sequence is equivalent up to a
# constant), the 4-blade family angle is
#
#   blade(t_rel) = phi0 - 0.5 * t_rel   (degrees, mod 90)
#
# Riders stand ON a blade, so their hub angle theta satisfies
# theta ~ blade + atan(w/s) (small blade-face offset). Therefore
# phi_i = (theta_i + 0.5 * t_rel_i) mod 90 concentrates at phi0.
# Circular mean over the 90-degree period (work at 4x angle). Riders can
# slide/walk freely — the phase estimate doesn't care (they stay on the
# blade) — so no velocity/stick filters here, just geometry:
# grounded, in the radius band, not on known static segments.
#
# If SEPARATE water phases yield the same phi0, the phase is
# deterministic at water-start and the viewer can draw the true blades.
#
#   mix run scripts/fit_windmill_phase.exs --replays "~/Slippi/**/*.slp"
#     [--r-min 6] [--r-max 45] [--json OUT.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.StageCollision
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, r_min: :float, r_max: :float, json: :string, anchor: :string]
  )

# --anchor water (default): t_rel from each water phase's first stable
#   frame — tests "phase resets at water start".
# --anchor game: t_rel = global frame index — tests "windmill runs on a
#   continuous game clock (rotating even while hidden)".
# --anchor cumwater: t_rel = cumulative STABLE water frames so far this
#   game — tests "wheel advances only while water is active (stable)".
# --anchor cumwater-all: same but counts every type-9 frame including
#   the morph animation.
anchor = opts[:anchor] || "water"

replays =
  (opts[:replays] || raise("--replays is required"))
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.sort()

if replays == [], do: raise("no replays matched")

{hub_x, hub_y} = {-36.64, 38.8}
r_min = opts[:r_min] || 6.0
r_max = opts[:r_max] || 45.0
rate_deg = -0.5

Output.banner("Windmill phase fit")
Output.config([{"Replays", length(replays)}, {"Radius band", "#{r_min}..#{r_max}"}])

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

# phase-id: one water phase = one contiguous stable-water stretch.
# Samples: %{phase: {file, k}, t_rel: n, theta_deg: d}
collect = fn path ->
    case Peppi.parse(path, player_port: 1) do
      {:ok, r} ->
        {tagged, _} =
          Enum.map_reduce(r.frames, {nil, nil}, fn f, {ev, ty} ->
            ev = f.stadium_event || ev
            ty = f.stadium_type || ty
            {{ev, ty, f}, {ev, ty}}
          end)

        {phase_samples, _k, _t0, _cw, _cwa} =
          tagged
          |> Enum.with_index()
          |> Enum.reduce({[], 0, nil, 0, 0}, fn {{ev, ty, f}, idx}, {acc, k, t0, cw, cwa} ->
            stable_water = ev == 0 and ty == 9

            {k, t0} =
              cond do
                stable_water and t0 == nil -> {k + 1, idx}
                not stable_water and t0 != nil -> {k, nil}
                true -> {k, t0}
              end

            cw = if stable_water, do: cw + 1, else: cw
            cwa = if ty == 9, do: cwa + 1, else: cwa

            acc =
              if stable_water do
                pts =
                  for {port, pl} <- f.players || %{},
                      pl.on_ground,
                      is_number(pl.x) and is_number(pl.y),
                      rx = pl.x - hub_x,
                      ry = pl.y - hub_y,
                      s = :math.sqrt(rx * rx + ry * ry),
                      s >= r_min and s <= r_max,
                      not on_static?.(pl.x, pl.y) do
                    %{
                      phase: {Path.basename(path), k},
                      t_rel:
                        case anchor do
                          "game" -> idx
                          "cumwater" -> cw
                          "cumwater-all" -> cwa
                          _ -> idx - t0
                        end,
                      idx: idx,
                      port: port,
                      s: s,
                      theta_deg: :math.atan2(ry, rx) * 180 / :math.pi()
                    }
                  end

                pts ++ acc
              else
                acc
              end

            {acc, k, t0, cw, cwa}
          end)

        phase_samples

      _ ->
        []
    end
end

samples =
  replays
  |> Task.async_stream(collect, max_concurrency: 8, timeout: 120_000, on_timeout: :kill_task)
  |> Enum.flat_map(fn
    {:ok, ss} -> ss
    _ -> []
  end)

n = length(samples)
by_phase = Enum.group_by(samples, & &1.phase)
Output.puts("#{n} rider sample(s) across #{map_size(by_phase)} water phase(s)")
if n == 0, do: (Output.error("no samples"); System.halt(2))

# Circular mean over the 90-degree blade period: lift to 4x angle.
circ = fn ss ->
  # blade(t) = phi0 + rate*t  =>  phi0 = theta - rate*t = theta + 0.5*t
  phis = Enum.map(ss, &(&1.theta_deg - rate_deg * &1.t_rel))
  rads = Enum.map(phis, &(4 * &1 * :math.pi() / 180))
  c = Enum.sum(Enum.map(rads, &:math.cos/1)) / length(rads)
  s = Enum.sum(Enum.map(rads, &:math.sin/1)) / length(rads)
  mean = :math.atan2(s, c) * 180 / :math.pi() / 4
  conc = :math.sqrt(c * c + s * s)
  {Float.round(mean, 2), Float.round(conc, 3)}
end

# -- Face-offset-corrected per-phase fit ------------------------------------
# A rider at radius s on a blade face of half-width w reads
# atan(w/s) off the true blade angle (~20 deg at s=10!) with a sign set
# by WHICH face — constant within a riding streak. Fit per phase:
#   phi_i = phi0 + f_k * atan(w/s_i),  f_k in {+1,-1} per streak
# by grid over w with alternating sign-assignment / circular refit.

norm90 = fn d ->
  d = :math.fmod(d, 90.0)
  cond do
    d > 45.0 -> d - 90.0
    d < -45.0 -> d + 90.0
    true -> d
  end
end

circ_mean90 = fn phis ->
  rads = Enum.map(phis, &(4 * &1 * :math.pi() / 180))
  c = Enum.sum(Enum.map(rads, &:math.cos/1)) / length(rads)
  s = Enum.sum(Enum.map(rads, &:math.sin/1)) / length(rads)
  {:math.atan2(s, c) * 180 / :math.pi() / 4, :math.sqrt(c * c + s * s)}
end

corrected_fit = fn ss ->
  streaks =
    ss
    |> Enum.group_by(& &1.port)
    |> Enum.flat_map(fn {_p, ps} ->
      ps
      |> Enum.sort_by(& &1.idx)
      |> Enum.chunk_while([], fn x, acc ->
        case acc do
          [prev | _] when x.idx > prev.idx + 1 -> {:cont, Enum.reverse(acc), [x]}
          _ -> {:cont, [x | acc]}
        end
      end, fn acc -> {:cont, Enum.reverse(acc), []} end)
    end)

  phi_raw = fn x -> x.theta_deg - rate_deg * x.t_rel end

  for w <- Enum.map(0..24, &(&1 * 0.25)) do
    # iterate sign assignment <-> phi0
    {phi0, conc, signs} =
      Enum.reduce(1..6, {elem(circ_mean90.(Enum.map(ss, phi_raw)), 0), 0.0, %{}}, fn _,
                                                                                     {p0, _, _} ->
        signs =
          Map.new(Enum.with_index(streaks), fn {st, i} ->
            err = fn f ->
              st
              |> Enum.map(fn x ->
                off = :math.atan(w / x.s) * 180 / :math.pi()
                abs(norm90.(phi_raw.(x) - f * off - p0))
              end)
              |> Enum.sum()
            end

            {i, if(err.(1) <= err.(-1), do: 1, else: -1)}
          end)

        phis =
          streaks
          |> Enum.with_index()
          |> Enum.flat_map(fn {st, i} ->
            f = signs[i]
            Enum.map(st, fn x -> phi_raw.(x) - f * :math.atan(w / x.s) * 180 / :math.pi() end)
          end)

        {p0n, conc} = circ_mean90.(phis)
        {p0n, conc, signs}
      end)

    {conc, w, phi0, signs}
  end
  |> Enum.max_by(&elem(&1, 0))
end

Output.puts("")
Output.puts("phi0 per water phase ((theta + 0.5*t_rel) mod 90, circular):")

phase_results =
  by_phase
  |> Enum.sort_by(fn {{file, k}, _} -> {file, k} end)
  |> Enum.map(fn {{file, k}, ss} ->
    {phi0, conc} = circ.(ss)
    {cconc, w, cphi0, _signs} = corrected_fit.(ss)

    Output.puts(
      "  #{file} phase #{k}: raw #{phi0} (c #{conc})  " <>
        "face-corrected #{Float.round(cphi0, 2)} (c #{Float.round(cconc, 3)}, w #{w}, n #{length(ss)})"
    )

    %{
      file: file,
      phase: k,
      phi0: phi0,
      concentration: conc,
      phi0_corrected: Float.round(cphi0, 2),
      concentration_corrected: Float.round(cconc, 3),
      blade_half_width: w,
      n: length(ss)
    }
  end)

{phi_all, conc_all} = circ.(samples)
Output.puts("")
Output.puts("pooled phi0: #{phi_all} deg (mod 90), concentration #{conc_all}")
Output.puts("(concentration ~1.0 = tight fit; ~0 = no blade-phase signal)")

if opts[:json] do
  File.write!(
    opts[:json],
    Jason.encode!(
      %{
        rate_deg_per_frame: rate_deg,
        anchor: "first stable water frame (stadium_event 0, type 9)",
        pooled_phi0_mod90: phi_all,
        pooled_concentration: conc_all,
        phases: phase_results
      },
      pretty: true
    )
  )

  Output.success("wrote #{opts[:json]}")
end
