#!/usr/bin/env elixir
# Scan replay directory and report character statistics
#
# Usage:
#   mix run scripts/scan_replays.exs --replays /path/to/replays
#   mix run scripts/scan_replays.exs --replays /path/to/replays --max-files 1000
#
# This script quickly scans .slp files to show:
# - Total number of replays
# - Games per character
# - Character matchup distribution
# - Recommended characters for training

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule ReplayScanner do
  @moduledoc """
  Fast replay scanner that extracts character info without full parsing.
  Uses py-slippi metadata extraction which is much faster than full parsing.
  """

  @python_script Path.join(:code.priv_dir(:exphil), "python/replay_scanner.py")

  @character_names %{
    0 => "Captain Falcon",
    1 => "Donkey Kong",
    2 => "Fox",
    3 => "Mr. Game & Watch",
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

  @low_tier_ids [3, 6, 10, 18, 25]  # G&W, Link, Mewtwo, Zelda, Ganondorf

  def scan(dir, opts \\ []) do
    max_files = Keyword.get(opts, :max_files)

    files = Path.wildcard(Path.join(dir, "**/*.slp"))
    files = if max_files, do: Enum.take(files, max_files), else: files
    total = length(files)

    IO.puts("\nScanning #{total} replay files...")
    IO.puts("=" |> String.duplicate(60))

    # Try Python scanner first (faster), fall back to Elixir
    case scan_with_python(dir, max_files) do
      {:ok, stats} ->
        display_stats(stats, total)

      {:error, _reason} ->
        IO.puts("Python scanner not available, using Elixir scanner (slower)...")
        stats = scan_with_elixir(files)
        display_stats(stats, total)
    end
  end

  defp scan_with_python(dir, max_files) do
    python = System.find_executable("python3") || System.find_executable("python")

    if is_nil(python) or not File.exists?(@python_script) do
      {:error, "Python or scanner script not found"}
    else
      args = [dir]
      args = if max_files, do: args ++ ["--max-files", to_string(max_files)], else: args

      case System.cmd(python, [@python_script | args], stderr_to_stdout: true) do
        {output, 0} ->
          case Jason.decode(output) do
            {:ok, stats} -> {:ok, stats}
            _ -> {:error, "Failed to parse scanner output"}
          end

        {_, _} ->
          {:error, "Python scanner failed"}
      end
    end
  end

  defp scan_with_elixir(files) do
    # Simple Elixir-based scanning (slower but works without Python)
    # Just counts files for now - full metadata requires py-slippi
    %{
      "total_files" => length(files),
      "character_counts" => %{},
      "matchup_counts" => %{},
      "error" => "Full character analysis requires Python. Install py-slippi: pip install py-slippi"
    }
  end

  defp display_stats(stats, total) do
    IO.puts("")
    IO.puts("REPLAY STATISTICS")
    IO.puts("=" |> String.duplicate(60))
    IO.puts("")

    # Total files
    scanned = stats["scanned_files"] || stats["total_files"] || total
    errors = stats["error_count"] || 0
    IO.puts("Total replays found: #{total}")
    IO.puts("Successfully scanned: #{scanned}")
    if errors > 0, do: IO.puts("Errors: #{errors}")
    IO.puts("")

    # Character counts
    char_counts = stats["character_counts"] || %{}
    if map_size(char_counts) > 0 do
      IO.puts("GAMES BY CHARACTER")
      IO.puts("-" |> String.duplicate(40))

      char_counts
      |> Enum.map(fn {char_id, count} ->
        id = if is_binary(char_id), do: String.to_integer(char_id), else: char_id
        {id, count}
      end)
      |> Enum.sort_by(fn {_, count} -> -count end)
      |> Enum.each(fn {char_id, count} ->
        name = Map.get(@character_names, char_id, "Unknown (#{char_id})")
        pct = Float.round(count / scanned * 100, 1)
        bar_len = round(count / scanned * 30)
        bar = String.duplicate("#", bar_len) <> String.duplicate(" ", 30 - bar_len)

        # Mark low-tier characters
        marker = if char_id in @low_tier_ids, do: " *", else: ""
        IO.puts("  #{String.pad_trailing(name, 18)} #{String.pad_leading(to_string(count), 6)} [#{bar}] #{pct}%#{marker}")
      end)

      IO.puts("")
      IO.puts("  * = Low-tier characters (ExPhil targets)")
      IO.puts("")

      # Low-tier summary
      low_tier_counts = char_counts
      |> Enum.filter(fn {char_id, _} ->
        id = if is_binary(char_id), do: String.to_integer(char_id), else: char_id
        id in @low_tier_ids
      end)
      |> Enum.map(fn {_, count} -> count end)
      |> Enum.sum()

      if low_tier_counts > 0 do
        IO.puts("LOW-TIER SUMMARY")
        IO.puts("-" |> String.duplicate(40))
        IO.puts("  Total low-tier games: #{low_tier_counts}")
        IO.puts("  Percentage: #{Float.round(low_tier_counts / scanned * 100, 1)}%")
        IO.puts("")
      end
    else
      if stats["error"] do
        IO.puts("Note: #{stats["error"]}")
        IO.puts("")
      end
    end

    # Recommendations
    IO.puts("TRAINING RECOMMENDATIONS")
    IO.puts("-" |> String.duplicate(40))

    if map_size(char_counts) > 0 do
      # Find best low-tier to train
      low_tier_data = char_counts
      |> Enum.filter(fn {char_id, _} ->
        id = if is_binary(char_id), do: String.to_integer(char_id), else: char_id
        id in @low_tier_ids
      end)
      |> Enum.map(fn {char_id, count} ->
        id = if is_binary(char_id), do: String.to_integer(char_id), else: char_id
        {id, count}
      end)
      |> Enum.sort_by(fn {_, count} -> -count end)

      if length(low_tier_data) > 0 do
        {best_id, best_count} = hd(low_tier_data)
        best_name = Map.get(@character_names, best_id, "Unknown")

        IO.puts("  Best low-tier for training: #{best_name} (#{best_count} games)")

        # Preset recommendation
        preset = case best_id do
          10 -> "mewtwo"
          25 -> "ganondorf"
          6 -> "link"
          3 -> "gameandwatch"
          18 -> "zelda"
          _ -> "production"
        end
        IO.puts("  Recommended preset: --preset #{preset}")

        # Data sufficiency
        cond do
          best_count >= 10000 ->
            IO.puts("  Data: Excellent - enough for production model")
          best_count >= 1000 ->
            IO.puts("  Data: Good - enough for solid training")
          best_count >= 100 ->
            IO.puts("  Data: Limited - consider augmentation")
          true ->
            IO.puts("  Data: Very limited - need more replays")
        end
      else
        IO.puts("  No low-tier character data found.")
        IO.puts("  Download replays from Slippi Discord or play online!")
      end
    else
      IO.puts("  Install py-slippi for full analysis: pip install py-slippi")
      IO.puts("  Or download replays from the Slippi Discord server")
    end

    IO.puts("")
    IO.puts("=" |> String.duplicate(60))
  end
end

# Parse arguments
args = System.argv()

replays_dir = case Enum.find_index(args, &(&1 == "--replays")) do
  nil -> "/home/dori/git/melee/replays"
  idx -> Enum.at(args, idx + 1)
end

max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> nil
  idx -> String.to_integer(Enum.at(args, idx + 1))
end

if not File.dir?(replays_dir) do
  IO.puts("Error: Replay directory not found: #{replays_dir}")
  IO.puts("")
  IO.puts("Usage: mix run scripts/scan_replays.exs --replays /path/to/replays")
  IO.puts("")
  IO.puts("To get replays:")
  IO.puts("  1. Join the Slippi Discord (via slippi.gg)")
  IO.puts("  2. Look for 'anonymized ranked collections' in resources")
  IO.puts("  3. Or play online - replays save to ~/.slippi-launcher/replays/")
  System.halt(1)
end

ReplayScanner.scan(replays_dir, max_files: max_files)
