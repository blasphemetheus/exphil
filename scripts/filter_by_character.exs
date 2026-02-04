#!/usr/bin/env elixir
# Filter SLP replays by character and copy to destination
#
# Usage:
#   mix run scripts/filter_by_character.exs \
#     --source "/run/media/dori/TOSHIBA EXT/slippi" \
#     --dest ~/Downloads/greg_slippi \
#     --characters zelda,iceclimbers,ganondorf,gameandwatch,link,mewtwo

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

defmodule CharacterFilter do
  # Character ID mapping
  @character_ids %{
    "zelda" => 18,
    "iceclimbers" => 14,
    "ics" => 14,
    "ganondorf" => 25,
    "ganon" => 25,
    "gameandwatch" => 3,
    "gnw" => 3,
    "gw" => 3,
    "link" => 6,
    "mewtwo" => 10,
    "mew2" => 10,
    # Common top tiers for reference
    "fox" => 2,
    "falco" => 20,
    "marth" => 9,
    "sheik" => 19,
    "peach" => 12,
    "falcon" => 0,
    "puff" => 15,
    "jiggs" => 15
  }

  @character_names %{
    0 => "Captain Falcon",
    2 => "Fox",
    3 => "Game & Watch",
    6 => "Link",
    9 => "Marth",
    10 => "Mewtwo",
    12 => "Peach",
    14 => "Ice Climbers",
    15 => "Jigglypuff",
    18 => "Zelda",
    19 => "Sheik",
    20 => "Falco",
    25 => "Ganondorf"
  }

  def run(args) do
    # Parse arguments
    {source_dir, dest_dir, target_char_ids} = parse_args(args)

    Output.banner("ExPhil Character Filter")
    Output.config([
      {"Source", source_dir},
      {"Destination", dest_dir},
      {"Target characters", Enum.map(target_char_ids, &char_name/1) |> Enum.join(", ")}
    ])

    # Ensure destination exists
    File.mkdir_p!(dest_dir)

    # Find all SLP files
    Output.step(1, 4, "Finding SLP files...")
    files = Path.wildcard(Path.join(source_dir, "**/*.slp"))
    total = length(files)
    Output.puts("Found #{total} SLP files")

    # Scan files and filter by character
    Output.step(2, 4, "Scanning replays for character data...")

    target_set = MapSet.new(target_char_ids)

    {matching_files, stats} =
      files
      |> Enum.with_index(1)
      |> Enum.reduce({[], %{scanned: 0, matched: 0, errors: 0, char_counts: %{}}}, fn {file, idx}, {matches, stats} ->
        # Progress update every 100 files
        if rem(idx, 100) == 0 do
          IO.write(:stderr, "\r  Progress: #{idx}/#{total} (#{length(matches)} matches)\e[K")
        end

        try do
          case Peppi.metadata(file) do
            {:ok, meta} ->
              # Get character IDs from this replay
              char_ids = Enum.map(meta.players, & &1.character) |> MapSet.new()

              # Update character counts
              char_counts = Enum.reduce(char_ids, stats.char_counts, fn char_id, acc ->
                Map.update(acc, char_id, 1, &(&1 + 1))
              end)

              # Check if any target character is present
              if MapSet.size(MapSet.intersection(char_ids, target_set)) > 0 do
                {[file | matches], %{stats | scanned: stats.scanned + 1, matched: stats.matched + 1, char_counts: char_counts}}
              else
                {matches, %{stats | scanned: stats.scanned + 1, char_counts: char_counts}}
              end

            {:error, _reason} ->
              {matches, %{stats | errors: stats.errors + 1}}
          end
        rescue
          # Handle NIF panics from corrupted/incompatible SLP files
          _e in ErlangError ->
            {matches, %{stats | errors: stats.errors + 1}}
        catch
          # Handle any other exceptions
          _, _ ->
            {matches, %{stats | errors: stats.errors + 1}}
        end
      end)

    IO.write(:stderr, "\r  Progress: #{total}/#{total} (#{length(matching_files)} matches)\e[K\n")

    Output.puts("Scanned: #{stats.scanned}, Errors: #{stats.errors}")
    Output.puts("Matching files: #{length(matching_files)}")

    # Show character distribution in matches
    Output.puts("")
    Output.section("Target character counts:")
    for char_id <- target_char_ids do
      count = Map.get(stats.char_counts, char_id, 0)
      name = char_name(char_id)
      Output.puts("  #{String.pad_trailing(name, 16)}: #{count}")
    end

    # Copy matching files
    Output.step(3, 4, "Copying #{length(matching_files)} files...")

    matching_files
    |> Enum.with_index(1)
    |> Enum.each(fn {src_path, idx} ->
      if rem(idx, 50) == 0 do
        IO.write(:stderr, "\r  Copying: #{idx}/#{length(matching_files)}\e[K")
      end

      # Preserve filename only (flatten directory structure)
      filename = Path.basename(src_path)
      dest_path = Path.join(dest_dir, filename)

      # Handle duplicate filenames by appending number
      dest_path = ensure_unique_path(dest_path)

      case File.cp(src_path, dest_path) do
        :ok -> :ok
        {:error, reason} ->
          IO.write(:stderr, "\n")
          Output.warning("Failed to copy #{filename}: #{reason}")
      end
    end)

    IO.write(:stderr, "\r  Copying: #{length(matching_files)}/#{length(matching_files)}\e[K\n")

    # Final summary
    Output.step(4, 4, "Done!")
    Output.success("Copied #{length(matching_files)} replays to #{dest_dir}")
  end

  defp char_name(char_id) do
    Map.get(@character_names, char_id, "ID #{char_id}")
  end

  defp ensure_unique_path(path) do
    if File.exists?(path) do
      ext = Path.extname(path)
      base = Path.basename(path, ext)
      dir = Path.dirname(path)

      # Find next available number
      Stream.iterate(1, &(&1 + 1))
      |> Enum.find(fn n ->
        candidate = Path.join(dir, "#{base}_#{n}#{ext}")
        not File.exists?(candidate)
      end)
      |> then(fn n -> Path.join(dir, "#{base}_#{n}#{ext}") end)
    else
      path
    end
  end

  defp parse_args(args) do
    source_idx = Enum.find_index(args, &(&1 == "--source"))
    dest_idx = Enum.find_index(args, &(&1 == "--dest"))
    chars_idx = Enum.find_index(args, &(&1 == "--characters"))

    source_dir = if source_idx, do: Enum.at(args, source_idx + 1), else: raise("--source required")
    dest_dir = if dest_idx, do: Enum.at(args, dest_idx + 1), else: raise("--dest required")

    char_ids =
      if chars_idx do
        Enum.at(args, chars_idx + 1)
        |> String.downcase()
        |> String.split(",")
        |> Enum.map(fn name ->
          Map.get(@character_ids, String.trim(name)) ||
            raise "Unknown character: #{name}. Valid: #{Map.keys(@character_ids) |> Enum.join(", ")}"
        end)
      else
        # Default: the low-tier characters Greg wants
        [18, 14, 25, 3, 6, 10]  # Zelda, ICs, Ganon, G&W, Link, Mewtwo
      end

    {source_dir, dest_dir, char_ids}
  end
end

CharacterFilter.run(System.argv())
