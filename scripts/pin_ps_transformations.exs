# Pin Pokemon Stadium transformation collision subsets (task #4).
#
# All GrPs collision files carry an IDENTICAL union of segments; the
# active transformation's subset is runtime-selected, so static
# extraction cannot attribute segments to types. This script pins them
# OBSERVATIONALLY from event-bearing replays (Slippi spec >= 3.18):
#
#   held (event, type) per frame  ->  frames where event == 0
#   ("finished": stable geometry) and type != normal  ->  grounded
#   player points NOT explained by the neutral layout  ->  union
#   segments within tolerance of >= @min_hits distinct points belong to
#   that type.
#
# Accumulates into priv/stage_collision/pokemon_stadium_types.json
# across runs (coverage grows as more types are observed). The viewer
# draws the pinned subset for the active type; unpinned types keep the
# badge-only rendering.
#
#   mix run scripts/pin_ps_transformations.exs \
#     --replays "eval_runs/0811_ps_pin/r*/*.slp" \
#     [--out priv/stage_collision/pokemon_stadium_types.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.StageCollision
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [replays: :string, out: :string])

replays =
  (opts[:replays] || raise("--replays is required"))
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.sort()

if replays == [], do: raise("no replays matched")

out_path = opts[:out] || "priv/stage_collision/pokemon_stadium_types.json"

min_hits = 3
tolerance = 2.0
type_names = %{3 => "fire", 4 => "grass", 5 => "normal", 6 => "rock", 9 => "water"}

Output.banner("PS transformation pinning")

# The union: every line in the base file (all groups — structures live
# outside the primary group), with coordinates resolved
base = File.read!("priv/stage_collision/pokemon_stadium.json") |> Jason.decode!()
varr = List.to_tuple(base["vertices"])

union =
  for l <- base["lines"] do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])
    %{index: l["index"], class: l["class"], seg: {x1, y1, x2, y2}, ledge: l["ledge_grab"], drop: l["drop_through"]}
  end

# Neutral layout = primary group's segments (what the viewer draws as
# base). Points near these are explained by the flat stage and carry no
# transformation information.
# Collection pre-filter: only drop points explained by the BASE SHELL
# (ledge-grab rule). A stale largest-group version of this filter used
# G1 — the fire tree — whose static footprint shadowed the midfield and
# silently starved fire (and others) of observations.
neutral_segs =
  for l <- base["lines"],
      Map.get(l, "group", 0) == StageCollision.base_group(base["lines"]) do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])
    {x1, y1, x2, y2}
  end

explained_by_neutral? = fn {x, y} ->
  Enum.any?(neutral_segs, &(StageCollision.point_segment_distance(x, y, &1) < tolerance))
end

# A line-GROUP is one structure (windmill, mound, mountain...): touching
# any of its lines claims the whole group, which is what recovers the
# walls/ceilings under a structure that players only ever stand ON TOP
# of. The primary (main-stage) group is never claimable.
# Ledge-grab rule, NOT largest — on PS the largest group is a structure
primary_group = StageCollision.base_group(base["lines"])

group_of_index =
  Map.new(base["lines"], fn l -> {l["index"], Map.get(l, "group", 0)} end)

lines_of_group =
  base["lines"] |> Enum.group_by(&Map.get(&1, "group", 0))

# type -> list of {x, y} observation points
points_by_type =
  Enum.reduce(replays, %{}, fn path, acc ->
    case Peppi.parse(path, player_port: 1) do
      {:ok, r} ->
        {samples, _} =
          Enum.map_reduce(r.frames, {nil, nil}, fn f, {ev, ty} ->
            ev = f.stadium_event || ev
            ty = f.stadium_type || ty
            pts = for {_p, pl} <- f.players, pl.on_ground, do: {pl.x, pl.y}
            {{ev, ty, pts}, {ev, ty}}
          end)

        Enum.reduce(samples, acc, fn
          # NORMAL (5) is a layout like any other: the middle floor +
          # real side platforms are its exclusive groups (the base shell
          # holds only the ledge slivers — transformations replace the
          # whole middle)
          {0, type, pts}, a when type in [3, 4, 5, 6, 9] ->
            fresh = Enum.reject(pts, explained_by_neutral?)
            Map.update(a, type, fresh, &(fresh ++ &1))

          _, a ->
            a
        end)

      _ ->
        acc
    end
  end)

# Attribute union segments per type. Two-stage: first collect each
# type's touched groups, then split PERSISTENT geometry (groups touched
# during EVERY observed type — the two platforms survive all
# transformations) from per-type structures (claims minus shared).
# Without the subtraction every type claims the platforms too and the
# per-type overlays draw as "a bunch of transforms at once".
existing =
  case File.read(out_path) do
    {:ok, raw} -> Jason.decode!(raw)
    _ -> %{}
  end

# Ground truth from Bradley (2026-08-11): the two normal platforms
# DISAPPEAR during transformations — nothing but the base shell
# persists across states. So: platforms are identified GEOMETRICALLY
# (the group made entirely of drop-through ground lines — drawn only in
# the normal state), and every remaining group is assigned EXCLUSIVELY
# to the single type with the most observation hits on it. Exclusive
# majority assignment is what prevents overlay soup: a group can only
# ever belong to one transformation.
# (A geometric "all drop-through = platforms" rule misfired here — it
# caught the fire tree's branch platforms. Normal's real geometry is
# pinned observationally like every other type.)
platform_groups = []

# hits per {group, type}
group_hits =
  for {type, pts} <- points_by_type,
      pts = Enum.uniq_by(pts, fn {x, y} -> {Float.round(x, 1), Float.round(y, 1)} end),
      %{seg: seg, index: idx, class: class} <- union,
      class != "dynamic",
      g = group_of_index[idx],
      g != primary_group,
      g not in platform_groups,
      hits = Enum.count(pts, fn {x, y} -> StageCollision.point_segment_distance(x, y, seg) < tolerance end),
      hits > 0,
      reduce: %{} do
    acc -> Map.update(acc, {g, type}, hits, &(&1 + hits))
  end

# group -> its majority type (needs at least min_hits total)
group_owner =
  group_hits
  |> Enum.group_by(fn {{g, _t}, _h} -> g end)
  |> Map.new(fn {g, entries} ->
    {{_g, best_type}, best_hits} = Enum.max_by(entries, &elem(&1, 1))
    total = entries |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    {g, if(best_hits >= min_hits, do: best_type, else: nil) |> then(&{&1, best_hits, total})}
  end)

touched_by_type =
  Map.new(points_by_type, fn {type, pts} ->
    groups =
      for {g, {owner, _bh, _tot}} <- group_owner, owner == type, uniq: true, do: g

    {type, {pts, MapSet.new(groups)}}
  end)

shared_groups = MapSet.new(platform_groups)

emit_groups = fn groups ->
  for g <- groups, l <- lines_of_group[g] || [], l["class"] != "dynamic" do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])

    %{
      "index" => l["index"],
      "class" => l["class"],
      "group" => g,
      "seg" => [x1, y1, x2, y2],
      "ledge_grab" => l["ledge_grab"],
      "drop_through" => l["drop_through"]
    }
  end
end

pinned =
  Enum.reduce(touched_by_type, existing, fn {type, {pts, groups}}, acc ->
    pts = Enum.uniq_by(pts, fn {x, y} -> {Float.round(x, 1), Float.round(y, 1)} end)

    # Exclusive: this type's structures are only its majority-owned groups
    matched = emit_groups.(groups)

    key = to_string(type)
    old = Map.get(acc, key, %{"segments" => [], "points_seen" => 0})

    merged_segments =
      (old["segments"] ++ matched)
      |> Enum.uniq_by(& &1["index"])

    Map.put(acc, key, %{
      "name" => type_names[type],
      "segments" => merged_segments,
      "points_seen" => old["points_seen"] + length(pts)
    })
  end)

# Runtime-positioned geometry (normal side platforms, fire's floor and
# trunk, ...) matches NO static segment — synthesize per type from the
# standing points its owned+base geometry can't explain: bin by height
# (2 units), split x-clusters at gaps > 6, keep well-supported clusters,
# emit observed segments. Normal-state clusters get symmetric mirroring
# (the stage is symmetric; some sides are under-sampled).
base_segs =
  for l <- lines_of_group[primary_group] || [] do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])
    {x1, y1, x2, y2}
  end

synthesize = fn type, owned_groups ->
  pts = points_by_type[type] || []

  explained =
    base_segs ++
      (for g <- owned_groups, l <- lines_of_group[g] || [], l["class"] != "dynamic" do
         [x1, y1] = elem(varr, l["v1"])
         [x2, y2] = elem(varr, l["v2"])
         {x1, y1, x2, y2}
       end)

  clusters =
    pts
    |> Enum.filter(fn {x, y} ->
      Enum.all?(explained, &(StageCollision.point_segment_distance(x, y, &1) > 2.5))
    end)
    |> Enum.group_by(fn {_x, y} -> round(y / 2) * 2 end)
    |> Enum.flat_map(fn {_yb, ps} ->
      ps
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.chunk_while([], fn {x, y}, acc ->
        case acc do
          [] -> {:cont, [{x, y}]}
          [{px, _} | _] when x - px <= 6.0 -> {:cont, [{x, y} | acc]}
          _ -> {:cont, Enum.reverse(acc), [{x, y}]}
        end
      end, fn acc -> {:cont, Enum.reverse(acc), []} end)
      |> Enum.filter(&(length(&1) >= 15))
    end)
    |> Enum.map(fn ps ->
      ys = Enum.map(ps, &elem(&1, 1))
      y = Float.round(Enum.sum(ys) / length(ys), 2)
      xs = Enum.map(ps, &elem(&1, 0))
      {y, Float.round(Enum.min(xs) - 2, 2), Float.round(Enum.max(xs) + 2, 2)}
    end)
    # A real new surface is SEPARATED from known geometry; clusters
    # hovering within 8 units of an explained segment are measurement
    # noise (hitlag/slope/ECB standing offsets around the floor), not
    # platforms — they were rendering as junk strips along the stage
    |> Enum.reject(fn {y, x1, x2} ->
      mid = (x1 + x2) / 2
      Enum.any?(explained, &(StageCollision.point_segment_distance(mid, y, &1) < 8.0))
    end)
    # Water: the windmill's sweep disc (hub -36.64, 38.8, fitted) is
    # ROTATING geometry — riding samples cluster at frozen angles and
    # would render as a stack of static strips. The spinner indicator
    # owns that region; static synthesis stays out of it.
    |> Enum.reject(fn {y, x1, x2} ->
      type == 9 and
        (fn ->
           mid = (x1 + x2) / 2
           dx = mid - -36.64
           dy = y - 38.8
           :math.sqrt(dx * dx + dy * dy) < 50.0
         end).()
    end)

  clusters =
    if type == 5 do
      mirrored =
        for {y, x1, x2} <- clusters,
            x2 < 0 or x1 > 0,
            not Enum.any?(clusters, fn {y2, mx1, mx2} ->
              abs(y2 - y) < 3.0 and mx1 <= -x2 + 3 and mx2 >= -x1 - 3
            end),
            do: {y, -x2, -x1}

      clusters ++ mirrored
    else
      clusters
    end

  for {y, x1, x2} <- clusters do
    %{
      "index" => -1,
      "class" => "ground",
      "group" => -1,
      "seg" => [x1, y, x2, y],
      "ledge_grab" => false,
      "drop_through" => true,
      "observed" => true
    }
  end
end

owned_by_type =
  group_owner
  |> Enum.filter(fn {_g, {o, _, _}} -> o != nil end)
  |> Enum.group_by(fn {_g, {o, _, _}} -> o end, fn {g, _} -> g end)

pinned =
  Enum.reduce(Map.keys(points_by_type), pinned |> Map.delete("_shared") |> Map.delete("_platforms"), fn type, acc ->
    synth = synthesize.(type, Map.get(owned_by_type, type, []))
    key = to_string(type)

    Map.update(acc, key, %{"name" => type_names[type], "segments" => synth, "points_seen" => 0}, fn e ->
      Map.update(e, "segments", synth, &(&1 ++ synth))
    end)
  end)

File.write!(out_path, Jason.encode!(pinned, pretty: true))

Output.puts("  observed types this run: #{inspect(Map.new(points_by_type, fn {t, p} -> {type_names[t], length(p)} end))}")

for {type, %{"name" => name, "segments" => segs, "points_seen" => n}} <- Enum.sort(pinned),
    type != "_shared" do
  Output.puts("  type #{type} (#{name}): #{length(segs)} pinned segments from #{n} points")
end

Output.puts("  group ownership: #{inspect(Map.new(group_owner, fn {g, {o, bh, tot}} -> {g, {o, bh, tot}} end))}")

covered = map_size(pinned)
Output.success("#{covered}/4 transformation types have pinned segments -> #{out_path}")
