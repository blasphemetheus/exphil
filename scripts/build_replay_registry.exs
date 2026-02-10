#!/usr/bin/env elixir
# Build Replay Registry: Index .slp files by stage, character matchup, and duration
#
# Scans a directory of Slippi replay files using fast Rust NIF metadata extraction
# (no frame parsing) and outputs a compact JSON registry for filtering/querying.
#
# Usage:
#   mix run scripts/build_replay_registry.exs --replays /path/to/slp --output registry.json
#   mix run scripts/build_replay_registry.exs --replays /path/to/slp --output registry.json --max-files 50000
#
# The output JSON can be queried with scripts/query_replays.exs to filter by
# matchup + stage for focused training sets.

if "--quiet" in System.argv() or "-q" in System.argv() do
  System.put_env("TF_CPP_MIN_LOG_LEVEL", "3")
end

alias ExPhil.CLI
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

require Output

# Character ID -> display name (matches Rust NIF character_name)
character_names = %{
  0 => "Captain Falcon",
  1 => "Donkey Kong",
  2 => "Fox",
  3 => "Game & Watch",
  4 => "Kirby",
  5 => "Bowser",
  6 => "Link",
  7 => "Luigi",
  8 => "Mario",
  9 => "Marth",
  10 => "Mewtwo",
  11 => "Ness",
  12 => "Peach",
  13 => "Pikachu",
  14 => "Ice Climbers",
  15 => "Jigglypuff",
  16 => "Samus",
  17 => "Yoshi",
  18 => "Zelda",
  19 => "Sheik",
  20 => "Falco",
  21 => "Young Link",
  22 => "Dr. Mario",
  23 => "Roy",
  24 => "Pichu",
  25 => "Ganondorf"
}

stage_names = %{
  2 => "Fountain of Dreams",
  3 => "Pokemon Stadium",
  8 => "Yoshi's Story",
  28 => "Dream Land",
  31 => "Battlefield",
  32 => "Final Destination"
}

# Parse args
opts = CLI.parse_args(System.argv(),
  flags: [:verbosity, :replay, :common],
  extra: [
    output: :string
  ],
  defaults: [
    output: "registry.json",
    replays: "./replays"
  ]
)

CLI.setup_verbosity(opts)

if opts[:help] do
  Output.banner("ExPhil Replay Registry Builder")
  Output.puts_raw("Usage: mix run scripts/build_replay_registry.exs [options]")
  Output.puts_raw("")
  Output.puts_raw("Options:")
  Output.puts_raw("  --replays DIR     Directory containing .slp files (default: ./replays)")
  Output.puts_raw("  --output FILE     Output JSON file (default: registry.json)")
  Output.puts_raw("  --max-files N     Maximum files to scan")
  Output.puts_raw("  --quiet           Suppress non-essential output")
  Output.puts_raw("  --verbose         Show debug output")
  System.halt(0)
end

replays_dir = opts[:replays]
output_path = opts[:output]
max_files = opts[:max_files]

Output.banner("ExPhil Replay Registry Builder")

# Validate input directory
unless File.dir?(replays_dir) do
  Output.error("Replays directory not found: #{replays_dir}")
  System.halt(1)
end

# Step 1: Find all .slp files
Output.step(1, 3, "Scanning for .slp files")

slp_files =
  replays_dir
  |> Path.join("**/*.slp")
  |> Path.wildcard()
  |> Enum.sort()

slp_files = if max_files, do: Enum.take(slp_files, max_files), else: slp_files
total = length(slp_files)

Output.puts("Found #{total} .slp files")

if total == 0 do
  Output.warning("No .slp files found in #{replays_dir}")
  System.halt(0)
end

# Step 2: Extract metadata from each file using parallel Task.async_stream
Output.step(2, 3, "Extracting metadata (parallel)")

# Track progress with an Agent
{:ok, counter} = Agent.start_link(fn -> 0 end)
errors = :ets.new(:registry_errors, [:set, :public])
progress_interval = max(div(total, 100), 1)

replays =
  slp_files
  |> Task.async_stream(
    fn path ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          # Increment counter for progress
          count = Agent.get_and_update(counter, fn n -> {n + 1, n + 1} end)

          if rem(count, progress_interval) == 0 do
            Output.progress_bar(count, total, label: "  Metadata")
          end

          rel_path = Path.relative_to(path, replays_dir)

          # Sort character IDs ascending for matchup dedup
          char_ids =
            meta.players
            |> Enum.map(& &1.character)
            |> Enum.sort()

          tags =
            meta.players
            |> Enum.map(fn p ->
              case p.tag do
                tag when is_binary(tag) and tag != "" -> tag
                _ -> nil
              end
            end)

          {:ok,
           %{
             p: rel_path,
             s: meta.stage,
             c: char_ids,
             d: meta.duration_frames,
             t: tags
           }}

        {:error, reason} ->
          count = Agent.get_and_update(counter, fn n -> {n + 1, n + 1} end)

          if rem(count, progress_interval) == 0 do
            Output.progress_bar(count, total, label: "  Metadata")
          end

          :ets.insert(errors, {path, reason})
          :error
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: 30_000,
    on_timeout: :kill_task
  )
  |> Enum.reduce([], fn
    {:ok, {:ok, entry}}, acc -> [entry | acc]
    {:ok, :error}, acc -> acc
    {:exit, _reason}, acc -> acc
  end)
  |> Enum.reverse()

Output.progress_bar(total, total, label: "  Metadata")
Output.progress_done()

error_list = :ets.tab2list(errors)
error_count = length(error_list)
:ets.delete(errors)
Agent.stop(counter)

if error_count > 0 do
  Output.warning("#{error_count} file(s) skipped due to errors")

  if Output.verbose?() do
    Enum.take(error_list, 5)
    |> Enum.each(fn {path, reason} ->
      Output.debug("  #{Path.relative_to(path, replays_dir)}: #{inspect(reason)}")
    end)
  end
end

# Step 3: Build summary and write JSON
Output.step(3, 3, "Building registry")

char_name = fn id -> Map.get(character_names, id, "Unknown (#{id})") end
stage_name = fn id -> Map.get(stage_names, id, "Stage #{id}") end

# Build stage counts
stage_counts =
  replays
  |> Enum.frequencies_by(& &1.s)
  |> Enum.map(fn {id, count} -> {stage_name.(id), count} end)
  |> Enum.sort_by(fn {_, count} -> -count end)
  |> Map.new()

# Build matchup counts (sorted char pair for dedup)
matchup_counts =
  replays
  |> Enum.frequencies_by(fn r ->
    case r.c do
      [a, b] -> "#{char_name.(a)} vs #{char_name.(b)}"
      [a] -> "#{char_name.(a)} (solo)"
      ids -> Enum.map(ids, char_name) |> Enum.join(" vs ")
    end
  end)
  |> Enum.sort_by(fn {_, count} -> -count end)
  |> Map.new()

# Build character counts (each character counted once per replay they appear in)
character_counts =
  replays
  |> Enum.flat_map(fn r -> Enum.uniq(r.c) end)
  |> Enum.frequencies()
  |> Enum.map(fn {id, count} -> {char_name.(id), count} end)
  |> Enum.sort_by(fn {_, count} -> -count end)
  |> Map.new()

registry = %{
  base: replays_dir |> Path.expand(),
  created: Date.utc_today() |> Date.to_string(),
  count: length(replays),
  replays: replays,
  summary: %{
    stages: stage_counts,
    matchups: matchup_counts,
    characters: character_counts
  }
}

json = Jason.encode!(registry, pretty: false)
File.write!(output_path, json)

file_size = byte_size(json)
Output.success("Registry written to #{output_path} (#{Output.format_bytes(file_size)})")

# Print summary
Output.section("Registry Summary")
Output.kv("Total replays", "#{length(replays)}")
Output.kv("Errors skipped", "#{error_count}")
Output.kv("Base path", Path.expand(replays_dir))

Output.puts_raw("")
Output.puts_raw("  " <> Output.colorize("Top Stages:", :bold))

stage_counts
|> Enum.sort_by(fn {_, count} -> -count end)
|> Enum.take(6)
|> Enum.each(fn {name, count} ->
  pct = Float.round(count / length(replays) * 100, 1)
  Output.puts_raw("    #{String.pad_trailing(name, 22)} #{count} (#{pct}%)")
end)

Output.puts_raw("")
Output.puts_raw("  " <> Output.colorize("Top Matchups:", :bold))

matchup_counts
|> Enum.sort_by(fn {_, count} -> -count end)
|> Enum.take(10)
|> Enum.each(fn {name, count} ->
  pct = Float.round(count / length(replays) * 100, 1)
  Output.puts_raw("    #{String.pad_trailing(name, 30)} #{count} (#{pct}%)")
end)

Output.puts_raw("")
Output.puts_raw("  " <> Output.colorize("Top Characters:", :bold))

character_counts
|> Enum.sort_by(fn {_, count} -> -count end)
|> Enum.take(10)
|> Enum.each(fn {name, count} ->
  pct = Float.round(count / length(replays) * 100, 1)
  Output.puts_raw("    #{String.pad_trailing(name, 18)} #{count} (#{pct}%)")
end)

Output.divider()
