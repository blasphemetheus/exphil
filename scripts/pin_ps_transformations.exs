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

@min_hits = min_hits = 3
tolerance = 2.0
type_names = %{3 => "fire", 4 => "grass", 6 => "rock", 9 => "water"}

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
primary =
  base["lines"] |> Enum.map(&Map.get(&1, "group", 0)) |> Enum.frequencies() |> Enum.max_by(&elem(&1, 1)) |> elem(0)

neutral_segs =
  for l <- base["lines"], Map.get(l, "group", primary) == primary do
    [x1, y1] = elem(varr, l["v1"])
    [x2, y2] = elem(varr, l["v2"])
    {x1, y1, x2, y2}
  end

explained_by_neutral? = fn {x, y} ->
  Enum.any?(neutral_segs, &(StageCollision.point_segment_distance(x, y, &1) < tolerance))
end

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
          {0, type, pts}, a when type in [3, 4, 6, 9] ->
            fresh = Enum.reject(pts, explained_by_neutral?)
            Map.update(a, type, fresh, &(fresh ++ &1))

          _, a ->
            a
        end)

      _ ->
        acc
    end
  end)

# Attribute union segments per type
existing =
  case File.read(out_path) do
    {:ok, raw} -> Jason.decode!(raw)
    _ -> %{}
  end

pinned =
  Enum.reduce(points_by_type, existing, fn {type, pts}, acc ->
    pts = Enum.uniq_by(pts, fn {x, y} -> {Float.round(x, 1), Float.round(y, 1)} end)

    matched =
      for %{seg: seg} = line <- union,
          hits = Enum.count(pts, fn {x, y} -> StageCollision.point_segment_distance(x, y, seg) < tolerance end),
          hits >= min_hits do
        {x1, y1, x2, y2} = line.seg

        %{
          "index" => line.index,
          "class" => line.class,
          "seg" => [x1, y1, x2, y2],
          "ledge_grab" => line.ledge,
          "drop_through" => line.drop,
          "hits" => hits
        }
      end

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

File.write!(out_path, Jason.encode!(pinned, pretty: true))

Output.puts("  observed types this run: #{inspect(Map.new(points_by_type, fn {t, p} -> {type_names[t], length(p)} end))}")

for {type, %{"name" => name, "segments" => segs, "points_seen" => n}} <- Enum.sort(pinned) do
  Output.puts("  type #{type} (#{name}): #{length(segs)} pinned segments from #{n} points")
end

covered = map_size(pinned)
Output.success("#{covered}/4 transformation types have pinned segments -> #{out_path}")
