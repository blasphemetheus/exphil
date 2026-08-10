# Stage collision extraction (task #1): HSD .dat -> per-stage JSON.
#
# Parses the HSD archive header, finds the `coll_data` root node, and
# decodes SBM_Coll_Data per the documented layout (HSDLib
# SBM_Coll_Data.cs; docs/planning/STAGE_COLLISION_EXTRACTION.md):
#   vertices: 8 bytes  {f32 x, f32 y} (big-endian — GameCube)
#   lines:   16 bytes  {s16 v1, s16 v2, s16 next, s16 prev,
#                       s16 alt_next, s16 alt_prev,
#                       s16 physics, u8 property, u8 material}
#   physics: 1 Top = GROUND (a surface you stand on top of),
#            2 Bottom = CEILING (hit from below), 4 Right wall,
#            8 Left wall, 16 Disabled. Verified against FD raw data:
#            flag-1 lines sit at y=0 out to |x|=85.57 with LedgeGrab on
#            the outer two — grounds beyond doubt.
#   property bit 2 = LedgeGrab, bit 1 = DropThrough
#
# Validation: the max |x| over ground segments must agree with
# Melee.Stages.edge_ground_position to ~0.01 — a full-pipeline check.
#
#   mix run scripts/extract_stage_collision.exs \
#     [--in cache/stage_dat] [--out priv/stage_collision]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [in: :string, out: :string])
in_dir = opts[:in] || "cache/stage_dat"
out_dir = opts[:out] || "priv/stage_collision"

stages = %{
  "GrNBa.dat" => :battlefield,
  "GrNLa.dat" => :final_destination,
  "GrSt.dat" => :yoshis_story,
  "GrIz.dat" => :fountain_of_dreams,
  "GrOp.dat" => :dreamland,
  "GrPs.dat" => :pokemon_stadium,
  # Stadium transformation overlays. Numeric suffix = the file's index;
  # the mapping to Slippi transformation TYPES (3 fire / 4 grass /
  # 6 rock / 9 water) gets pinned when task #3's events land — the
  # geometry itself identifies them (rock = mountain, water = windmill).
  "GrPs1.dat" => :pokemon_stadium_1,
  "GrPs2.dat" => :pokemon_stadium_2,
  "GrPs3.dat" => :pokemon_stadium_3,
  "GrPs4.dat" => :pokemon_stadium_4
}

File.mkdir_p!(out_dir)
Output.banner("Stage collision extraction")

read_cstring = fn bin, off ->
  bin |> binary_part(off, min(64, byte_size(bin) - off)) |> :binary.split(<<0>>) |> hd()
end

parse_dat = fn path ->
  bin = File.read!(path)

  <<_file_size::32-big, data_size::32-big, reloc_count::32-big, root_count::32-big,
    ref_count::32-big, _::binary>> = bin

  data = binary_part(bin, 0x20, data_size)
  roots_at = 0x20 + data_size + reloc_count * 4
  strtab_at = roots_at + (root_count + ref_count) * 8

  roots =
    for i <- 0..(root_count - 1) do
      <<data_off::32-big, str_off::32-big>> = binary_part(bin, roots_at + i * 8, 8)
      {read_cstring.(bin, strtab_at + str_off), data_off}
    end

  {_name, coll_off} =
    Enum.find(roots, fn {name, _} -> String.contains?(name, "coll_data") end) ||
      raise "no coll_data root in #{path} (roots: #{inspect(Enum.map(roots, &elem(&1, 0)))})"

  # Stage scale: collision is authored unscaled; the engine applies
  # grGroundParam's leading float at load (YS 0.7, BF 0.8, FoD 0.75,
  # FD/DL/PS 1.0 — discovered via edge validation failures: YS raw
  # slant-end x=80 vs teeter 56 = exactly 0.7)
  scale =
    case Enum.find(roots, fn {name, _} -> String.contains?(name, "grGroundParam") end) do
      nil ->
        1.0

      {_n, goff2} ->
        <<s::float-32-big, _::binary>> = binary_part(data, goff2, 4)
        if s > 0.01 and s < 100.0, do: s, else: 1.0
    end

  # 0x10..0x23: {u16 first_line_idx, u16 count} for the ACTIVE groups
  # top(ground)/bottom(ceiling)/right/left/dynamic — the line array also
  # holds dynamic/inactive entries (Randall, PS transformation pieces),
  # so the groups are the authority on what is static stage collision
  # (found via BF/YS/FoD edge-validation failures: ungrouped lines put
  # phantom grounds out at |x|~85 on every stage).
  <<vtx_off::32-big, vtx_count::32-big, line_off::32-big, line_count::32-big,
    top_i::16-big, top_n::16-big, bot_i::16-big, bot_n::16-big, right_i::16-big,
    right_n::16-big, left_i::16-big, left_n::16-big, dyn_i::16-big, dyn_n::16-big,
    _::binary>> = binary_part(data, coll_off, 36)

  groups = [
    {"ground", top_i, top_n},
    {"ceiling", bot_i, bot_n},
    {"wall_right", right_i, right_n},
    {"wall_left", left_i, left_n},
    {"dynamic", dyn_i, dyn_n}
  ]

  vertices =
    for i <- 0..(vtx_count - 1) do
      <<x::float-32-big, y::float-32-big>> = binary_part(data, vtx_off + i * 8, 8)
      [Float.round(x * scale, 4), Float.round(y * scale, 4)]
    end

  line_at = fn i ->
    <<v1::16-signed-big, v2::16-signed-big, _next::16-signed-big, _prev::16-signed-big,
      _an::16-signed-big, _ap::16-signed-big, physics::16-big, prop::8,
      mat::8>> = binary_part(data, line_off + i * 16, 16)

    %{
      v1: v1,
      v2: v2,
      physics: physics,
      ledge_grab: Bitwise.band(prop, 2) != 0,
      drop_through: Bitwise.band(prop, 1) != 0,
      material: mat
    }
  end

  lines =
    for {class, first, n} <- groups, n > 0, i <- first..(first + n - 1), i < line_count do
      line_at.(i) |> Map.put(:class, class) |> Map.put(:index, i)
    end

  # Line-GROUP membership (SBM_CollLineGroup at 0x24: each group's own
  # top/bottom/right/left index+count pairs in its first 16 bytes).
  # Moving objects live in their own groups — YS's Randall is group 0's
  # single line, parked at its authoring position; FoD's moving side
  # platforms likewise. Tag lines so consumers can keep only the
  # primary (static-stage) group.
  <<goff::32-big, gcount::32-big, _::binary>> = binary_part(data, coll_off + 0x24, 8)

  group_of =
    if gcount > 1 do
      ranges =
        for g <- 0..(gcount - 1) do
          base = goff + g * 0x28

          pairs =
            for k <- 0..3 do
              <<idx::16-big, n::16-big>> = binary_part(data, base + k * 4, 4)
              {idx, n}
            end

          {g, pairs}
        end

      fn i ->
        Enum.find_value(ranges, 0, fn {g, pairs} ->
          if Enum.any?(pairs, fn {idx, n} -> n > 0 and i >= idx and i < idx + n end), do: g
        end)
      end
    else
      fn _i -> 0 end
    end

  lines = Enum.map(lines, &Map.put(&1, :group, group_of.(&1.index)))
  {vertices, lines}
end

results =
  for {file, stage} <- Enum.sort(stages), path = Path.join(in_dir, file), File.exists?(path) do
    {vertices, lines} = parse_dat.(path)

    varr = List.to_tuple(vertices)
    active = Enum.reject(lines, &(&1.class == "disabled"))

    class_counts = active |> Enum.map(& &1.class) |> Enum.frequencies()

    # Solid grounds only: platforms are drop-through lines in the same
    # group, and e.g. Yoshi's platforms extend past its teeter edge
    ground_max_x =
      active
      |> Enum.filter(&(&1.class == "ground" and not &1.drop_through))
      |> Enum.flat_map(fn l -> [elem(varr, l.v1), elem(varr, l.v2)] end)
      |> Enum.map(fn [x, _y] -> abs(x) end)
      |> Enum.max(fn -> 0.0 end)

    # Transformation overlays keep Stadium's main-ground edges — the
    # structures sit inside them — so they validate against PS's edge
    expected =
      Melee.Stages.edge_ground_position(stage) ||
        if String.starts_with?(to_string(stage), "pokemon_stadium_"),
          do: Melee.Stages.edge_ground_position(:pokemon_stadium)
    delta = if expected, do: Float.round(abs(ground_max_x - expected), 4)

    File.write!(
      Path.join(out_dir, "#{stage}.json"),
      Jason.encode!(
        %{
          stage: stage,
          source: file,
          vertices: vertices,
          lines: active,
          summary: Map.merge(class_counts, %{ground_max_x: ground_max_x})
        },
        pretty: true
      )
    )

    status = cond do
      expected == nil -> "no reference edge"
      delta < 0.05 -> "EDGE MATCH (Δ#{delta})"
      true -> "EDGE MISMATCH: parsed #{ground_max_x} vs #{expected}"
    end

    Output.puts(
      "  #{stage}: #{length(vertices)} vertices, #{length(active)} lines " <>
        "#{inspect(class_counts)} — #{status}"
    )

    {stage, delta}
  end

bad = Enum.filter(results, fn {_s, d} -> d == nil or d > 0.05 end)

if bad == [] do
  Output.success("all #{length(results)} stages extracted and edge-validated -> #{out_dir}")
else
  Output.warning("validation issues: #{inspect(bad)}")
end
