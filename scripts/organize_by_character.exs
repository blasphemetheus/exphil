#!/usr/bin/env elixir
# Organize SLP replays into character subdirectories
# Only processes 1v1 games and copies to each matching character's folder
#
# Usage:
#   mix run scripts/organize_by_character.exs --source ~/Downloads/greg_slippi

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

defmodule CharacterOrganizer do
  # Target characters and their folder names
  @target_characters %{
    3 => "game_and_watch",
    6 => "link",
    10 => "mewtwo",
    14 => "ice_climbers",
    18 => "zelda",
    25 => "ganondorf"
  }

  @character_names %{
    3 => "Game & Watch",
    6 => "Link",
    10 => "Mewtwo",
    14 => "Ice Climbers",
    18 => "Zelda",
    25 => "Ganondorf"
  }

  def run(args) do
    source_dir = parse_args(args)

    Output.banner("ExPhil Character Organizer")
    Output.config([
      {"Source", source_dir},
      {"Target characters", @target_characters |> Map.values() |> Enum.join(", ")}
    ])

    # Create subdirectories
    Output.step(1, 4, "Creating character directories...")
    for {_id, folder} <- @target_characters do
      dir = Path.join(source_dir, folder)
      File.mkdir_p!(dir)
      Output.puts("  Created: #{folder}/")
    end

    # Find all SLP files in root (not subdirectories)
    Output.step(2, 4, "Finding SLP files...")
    files = Path.wildcard(Path.join(source_dir, "*.slp"))
    total = length(files)
    Output.puts("Found #{total} SLP files")

    # Process each file
    Output.step(3, 4, "Organizing replays...")

    target_ids = Map.keys(@target_characters) |> MapSet.new()

    stats =
      files
      |> Enum.with_index(1)
      |> Enum.reduce(%{processed: 0, copied: 0, skipped_not_1v1: 0, skipped_no_target: 0, errors: 0, per_char: %{}}, fn {file, idx}, stats ->
        # Progress update every 100 files
        if rem(idx, 100) == 0 do
          IO.write(:stderr, "\r  Progress: #{idx}/#{total} (#{stats.copied} copies)\e[K")
        end

        try do
          case Peppi.metadata(file) do
            {:ok, meta} ->
              # Check if 1v1 (exactly 2 players)
              if length(meta.players) == 2 do
                # Get character IDs
                char_ids = Enum.map(meta.players, & &1.character)
                matching_chars = Enum.filter(char_ids, &MapSet.member?(target_ids, &1))

                if length(matching_chars) > 0 do
                  # Copy to each matching character's folder
                  filename = Path.basename(file)

                  per_char = Enum.reduce(matching_chars, stats.per_char, fn char_id, acc ->
                    folder = Map.get(@target_characters, char_id)
                    dest_path = Path.join([source_dir, folder, filename])
                    dest_path = ensure_unique_path(dest_path)

                    case File.cp(file, dest_path) do
                      :ok -> Map.update(acc, char_id, 1, &(&1 + 1))
                      {:error, _} -> acc
                    end
                  end)

                  %{stats |
                    processed: stats.processed + 1,
                    copied: stats.copied + length(matching_chars),
                    per_char: per_char
                  }
                else
                  %{stats | processed: stats.processed + 1, skipped_no_target: stats.skipped_no_target + 1}
                end
              else
                %{stats | processed: stats.processed + 1, skipped_not_1v1: stats.skipped_not_1v1 + 1}
              end

            {:error, _reason} ->
              %{stats | errors: stats.errors + 1}
          end
        rescue
          _e in ErlangError ->
            %{stats | errors: stats.errors + 1}
        catch
          _, _ ->
            %{stats | errors: stats.errors + 1}
        end
      end)

    IO.write(:stderr, "\r  Progress: #{total}/#{total} (#{stats.copied} copies)\e[K\n")

    # Summary
    Output.step(4, 4, "Done!")
    Output.puts("")
    Output.section("Summary:")
    Output.puts("  Processed: #{stats.processed}")
    Output.puts("  Skipped (not 1v1): #{stats.skipped_not_1v1}")
    Output.puts("  Skipped (no target char): #{stats.skipped_no_target}")
    Output.puts("  Errors: #{stats.errors}")
    Output.puts("  Total copies made: #{stats.copied}")
    Output.puts("")

    Output.section("Files per character:")
    for {char_id, folder} <- @target_characters |> Enum.sort_by(fn {_, v} -> v end) do
      count = Map.get(stats.per_char, char_id, 0)
      name = Map.get(@character_names, char_id, folder)
      Output.puts("  #{String.pad_trailing(name, 16)}: #{count} files -> #{folder}/")
    end

    Output.puts("")
    Output.success("Organization complete!")
  end

  defp ensure_unique_path(path) do
    if File.exists?(path) do
      ext = Path.extname(path)
      base = Path.basename(path, ext)
      dir = Path.dirname(path)

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

    if source_idx do
      Enum.at(args, source_idx + 1)
    else
      raise "--source required"
    end
  end
end

CharacterOrganizer.run(System.argv())
