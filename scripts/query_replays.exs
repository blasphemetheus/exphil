#!/usr/bin/env elixir
# Query Replay Registry: Filter replays by matchup, stage, and character
#
# Loads a registry JSON (built by build_replay_registry.exs) and filters
# replays for focused training sets.
#
# Usage:
#   # Show available matchups and stages
#   mix run scripts/query_replays.exs --registry registry.json --stats
#
#   # List matching replays
#   mix run scripts/query_replays.exs --registry registry.json --matchup fox,fox --stage bf
#
#   # Create symlinked directory for training
#   mix run scripts/query_replays.exs --registry registry.json --matchup fox,fox --stage bf --output training_set/
#
#   # Filter by character (any matchup containing this character)
#   mix run scripts/query_replays.exs --registry registry.json --character zelda --stage fd
#
#   # Just file paths (for piping)
#   mix run scripts/query_replays.exs --registry registry.json --matchup mewtwo,fox --paths-only

alias ExPhil.Training.Output

require Output

# Suppress Logger noise when piping output (must come after app start)
if "--paths-only" in System.argv() or "--quiet" in System.argv() do
  Logger.configure(level: :error)
end

# ============================================================================
# Character alias -> ID mapping
# Mirrors @character_map from Config but maps to integer IDs for registry lookup
# ============================================================================

character_to_id = %{
  "captain_falcon" => 0, "falcon" => 0,
  "donkey_kong" => 1, "dk" => 1,
  "fox" => 2,
  "game_and_watch" => 3, "gnw" => 3, "gameandwatch" => 3,
  "kirby" => 4,
  "bowser" => 5,
  "link" => 6,
  "luigi" => 7,
  "mario" => 8,
  "marth" => 9,
  "mewtwo" => 10,
  "ness" => 11,
  "peach" => 12,
  "pikachu" => 13, "pika" => 13,
  "ice_climbers" => 14, "ics" => 14, "icies" => 14,
  "jigglypuff" => 15, "puff" => 15, "jiggs" => 15,
  "samus" => 16,
  "yoshi" => 17,
  "zelda" => 18,
  "sheik" => 19,
  "falco" => 20,
  "young_link" => 21, "ylink" => 21,
  "dr_mario" => 22, "doc" => 22,
  "roy" => 23,
  "pichu" => 24,
  "ganondorf" => 25, "ganon" => 25
}

character_names = %{
  0 => "Captain Falcon", 1 => "Donkey Kong", 2 => "Fox",
  3 => "Game & Watch", 4 => "Kirby", 5 => "Bowser",
  6 => "Link", 7 => "Luigi", 8 => "Mario",
  9 => "Marth", 10 => "Mewtwo", 11 => "Ness",
  12 => "Peach", 13 => "Pikachu", 14 => "Ice Climbers",
  15 => "Jigglypuff", 16 => "Samus", 17 => "Yoshi",
  18 => "Zelda", 19 => "Sheik", 20 => "Falco",
  21 => "Young Link", 22 => "Dr. Mario", 23 => "Roy",
  24 => "Pichu", 25 => "Ganondorf"
}

# Stage alias -> ID (mirrors @stage_map from Config)
stage_to_id = %{
  "fountain_of_dreams" => 2, "fod" => 2, "fountain" => 2,
  "pokemon_stadium" => 3, "ps" => 3, "stadium" => 3,
  "yoshis_story" => 8, "yoshis" => 8, "ys" => 8,
  "dream_land" => 28, "dreamland" => 28, "dl" => 28,
  "battlefield" => 31, "bf" => 31,
  "final_destination" => 32, "fd" => 32
}

stage_names = %{
  2 => "Fountain of Dreams", 3 => "Pokemon Stadium",
  8 => "Yoshi's Story", 28 => "Dream Land",
  31 => "Battlefield", 32 => "Final Destination"
}

# ============================================================================
# Argument parsing
# ============================================================================

args = System.argv()

parse_string = fn args, flag ->
  case Enum.find_index(args, &(&1 == flag)) do
    nil -> nil
    idx -> Enum.at(args, idx + 1)
  end
end

registry_path = parse_string.(args, "--registry")
matchup_str = parse_string.(args, "--matchup")
character_str = parse_string.(args, "--character")
stage_str = parse_string.(args, "--stage")
output_dir = parse_string.(args, "--output")
show_stats = "--stats" in args
paths_only = "--paths-only" in args
show_help = "--help" in args or "-h" in args
quiet = "--quiet" in args or "-q" in args

if quiet, do: Output.set_verbosity(:quiet)

if show_help do
  Output.banner("ExPhil Replay Registry Query")
  Output.puts_raw("Usage: mix run scripts/query_replays.exs --registry FILE [options]")
  Output.puts_raw("")
  Output.puts_raw("Options:")
  Output.puts_raw("  --registry FILE    Path to registry JSON (required)")
  Output.puts_raw("  --matchup C1,C2    Filter by matchup (e.g., fox,fox or mewtwo,zelda)")
  Output.puts_raw("  --character CHAR   Filter by character (any matchup containing this char)")
  Output.puts_raw("  --stage STAGE      Filter by stage (bf, fd, ps, ys, dl, fod)")
  Output.puts_raw("  --output DIR       Create symlinked directory with matching replays")
  Output.puts_raw("  --stats            Show registry statistics")
  Output.puts_raw("  --paths-only       Output only file paths (pipe with: | grep .slp$)")
  Output.puts_raw("  --quiet            Suppress non-essential output")
  Output.puts_raw("")
  Output.puts_raw("Character aliases: fox, falco, marth, sheik, falcon, puff/jiggs,")
  Output.puts_raw("  peach, ics/icies, pikachu/pika, dk, gnw, ganon, doc, ylink, etc.")
  Output.puts_raw("")
  Output.puts_raw("Stage aliases: bf (Battlefield), fd (Final Destination),")
  Output.puts_raw("  ps (Pokemon Stadium), ys (Yoshi's Story), dl (Dream Land), fod (Fountain)")
  System.halt(0)
end

unless registry_path do
  Output.error("--registry is required. Use --help for usage.")
  System.halt(1)
end

unless File.exists?(registry_path) do
  Output.error("Registry file not found: #{registry_path}")
  System.halt(1)
end

# ============================================================================
# Load registry
# ============================================================================

registry =
  registry_path
  |> File.read!()
  |> Jason.decode!()

base_path = registry["base"]
replays = registry["replays"]
total = registry["count"]

unless paths_only do
  Output.puts("Loaded registry: #{total} replays from #{base_path}")
end

# ============================================================================
# Stats mode
# ============================================================================

if show_stats do
  summary = registry["summary"]

  Output.section("Registry Statistics")
  Output.kv("Total replays", "#{total}")
  Output.kv("Base path", base_path)
  Output.kv("Created", registry["created"])

  # Stage breakdown
  if stages = summary["stages"] do
    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Stages:", :bold))

    stages
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.each(fn {name, count} ->
      pct = Float.round(count / total * 100, 1)
      bar_width = min(round(pct / 2), 25)
      bar = String.duplicate("~", bar_width)
      Output.puts_raw("    #{String.pad_trailing(name, 22)} #{Output.colorize(bar, :cyan)} #{count} (#{pct}%)")
    end)
  end

  # Matchup breakdown
  if matchups = summary["matchups"] do
    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Top Matchups:", :bold))

    matchups
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.take(20)
    |> Enum.each(fn {name, count} ->
      pct = Float.round(count / total * 100, 1)
      Output.puts_raw("    #{String.pad_trailing(name, 30)} #{count} (#{pct}%)")
    end)
  end

  # Character breakdown
  if characters = summary["characters"] do
    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Characters:", :bold))

    characters
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.each(fn {name, count} ->
      pct = Float.round(count / total * 100, 1)
      bar_width = min(round(pct / 2), 25)
      bar = String.duplicate("~", bar_width)
      Output.puts_raw("    #{String.pad_trailing(name, 18)} #{Output.colorize(bar, :green)} #{count} (#{pct}%)")
    end)
  end

  Output.divider()

  # If no other filters, exit after stats
  unless matchup_str || character_str || stage_str do
    System.halt(0)
  end
end

# ============================================================================
# Parse filters
# ============================================================================

# Resolve matchup to sorted character ID pair
matchup_ids =
  if matchup_str do
    ids =
      matchup_str
      |> String.downcase()
      |> String.split(",")
      |> Enum.map(fn name ->
        name = String.trim(name)
        case Map.get(character_to_id, name) do
          nil ->
            Output.error("Unknown character: '#{name}'. Use --help for aliases.")
            System.halt(1)
          id -> id
        end
      end)
      |> Enum.sort()

    unless length(ids) == 2 do
      Output.error("--matchup requires exactly 2 characters (e.g., --matchup fox,marth)")
      System.halt(1)
    end

    ids
  end

# Resolve single character filter
character_id =
  if character_str do
    name = character_str |> String.downcase() |> String.trim()
    case Map.get(character_to_id, name) do
      nil ->
        Output.error("Unknown character: '#{name}'. Use --help for aliases.")
        System.halt(1)
      id -> id
    end
  end

# Resolve stage filter
stage_id =
  if stage_str do
    name = stage_str |> String.downcase() |> String.trim()
    case Map.get(stage_to_id, name) do
      nil ->
        Output.error("Unknown stage: '#{name}'. Use --help for aliases.")
        System.halt(1)
      id -> id
    end
  end

# Need at least one filter (unless --stats already handled)
unless matchup_ids || character_id || stage_id do
  Output.error("Specify at least --matchup, --character, or --stage to filter. Use --stats to browse.")
  System.halt(1)
end

# ============================================================================
# Filter replays
# ============================================================================

char_name = fn id -> Map.get(character_names, id, "Unknown (#{id})") end
stage_name_fn = fn id -> Map.get(stage_names, id, "Stage #{id}") end

# Describe the filter for output
filter_parts = []
filter_parts = if matchup_ids, do: filter_parts ++ ["matchup: #{Enum.map(matchup_ids, char_name) |> Enum.join(" vs ")}"], else: filter_parts
filter_parts = if character_id, do: filter_parts ++ ["character: #{char_name.(character_id)}"], else: filter_parts
filter_parts = if stage_id, do: filter_parts ++ ["stage: #{stage_name_fn.(stage_id)}"], else: filter_parts

unless paths_only do
  Output.puts("Filtering: #{Enum.join(filter_parts, ", ")}")
end

matches =
  replays
  |> Enum.filter(fn r ->
    # Matchup filter: sorted char IDs must match exactly
    matchup_ok =
      if matchup_ids do
        r_chars = r["c"] |> Enum.sort()
        r_chars == matchup_ids
      else
        true
      end

    # Character filter: at least one player is this character
    char_ok =
      if character_id do
        character_id in r["c"]
      else
        true
      end

    # Stage filter
    stage_ok =
      if stage_id do
        r["s"] == stage_id
      else
        true
      end

    matchup_ok and char_ok and stage_ok
  end)

match_count = length(matches)

# Calculate total duration
total_frames = Enum.reduce(matches, 0, fn r, acc -> acc + (r["d"] || 0) end)
total_minutes = Float.round(total_frames / 60 / 60, 1)

# ============================================================================
# Output results
# ============================================================================

if paths_only do
  # Just print absolute paths, one per line
  Enum.each(matches, fn r ->
    IO.puts(Path.join(base_path, r["p"]))
  end)
else
  Output.success("#{match_count} replays matched (#{total_minutes} minutes of gameplay)")

  if match_count == 0 do
    Output.puts("Try --stats to see available matchups and stages.")
    System.halt(0)
  end

  # Show a sample of matches
  if match_count > 0 and not paths_only do
    sample_count = min(match_count, 5)
    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Sample (#{sample_count}/#{match_count}):", :dim))

    matches
    |> Enum.take(sample_count)
    |> Enum.each(fn r ->
      chars = r["c"] |> Enum.map(char_name) |> Enum.join(" vs ")
      stage = stage_name_fn.(r["s"])
      duration_sec = div(r["d"] || 0, 60)
      Output.puts_raw("    #{r["p"]}  [#{chars} on #{stage}, #{duration_sec}s]")
    end)

    if match_count > sample_count do
      Output.puts_raw("    ... and #{match_count - sample_count} more")
    end
  end

  # Create symlinked output directory if requested
  if output_dir do
    Output.puts_raw("")
    Output.puts("Creating symlinked directory: #{output_dir}")

    File.mkdir_p!(output_dir)

    # Track naming collisions
    name_counts = %{}

    {_, collisions} =
      Enum.reduce(matches, {name_counts, 0}, fn r, {counts, cols} ->
        source = Path.join(base_path, r["p"])
        basename = Path.basename(r["p"])

        # Handle duplicate basenames by adding a counter
        {link_name, new_counts} =
          case Map.get(counts, basename) do
            nil ->
              {basename, Map.put(counts, basename, 1)}
            n ->
              ext = Path.extname(basename)
              base = Path.basename(basename, ext)
              {base <> "_#{n}" <> ext, Map.put(counts, basename, n + 1)}
          end

        target = Path.join(output_dir, link_name)

        case File.ln_s(source, target) do
          :ok -> {new_counts, cols}
          {:error, :eexist} -> {new_counts, cols}
          {:error, reason} ->
            Output.warning("Failed to symlink #{link_name}: #{reason}")
            {new_counts, cols + 1}
        end
      end)

    Output.success("Created #{match_count} symlinks in #{output_dir}")

    if collisions > 0 do
      Output.warning("#{collisions} symlinks failed")
    end

    Output.puts_raw("")
    Output.puts_raw("  Use with training:")
    Output.puts_raw("    mix run scripts/train_from_replays.exs --replays #{output_dir} --preset quick")
  end
end
