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

alias ExPhil.Training.Output

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

  # G&W, Link, Mewtwo, Zelda, Ganondorf
  @low_tier_ids [3, 6, 10, 18, 25]

  def scan(dir, opts \\ []) do
    alias ExPhil.Training.Output

    max_files = Keyword.get(opts, :max_files)

    files = Path.wildcard(Path.join(dir, "**/*.slp"))
    files = if max_files, do: Enum.take(files, max_files), else: files
    total = length(files)

    Output.banner("ExPhil Replay Scanner")

    Output.config([
      {"Directory", dir},
      {"Max files", max_files || "all"},
      {"Total found", total}
    ])

    # Try Python scanner first (faster), fall back to Elixir
    case scan_with_python(dir, max_files) do
      {:ok, stats} ->
        display_stats(stats, total)

      {:error, _reason} ->
        Output.warning("Python scanner not available, using Elixir scanner (slower)...")
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
      "error" =>
        "Full character analysis requires Python. Install py-slippi: pip install py-slippi"
    }
  end

  defp display_stats(stats, total) do
    alias ExPhil.Training.Output

    Output.divider()
    Output.section("REPLAY STATISTICS")

    # Total files
    scanned = stats["scanned_files"] || stats["total_files"] || total
    errors = stats["error_count"] || 0
    Output.puts("Total replays found: #{total}")
    Output.puts("Successfully scanned: #{scanned}")
    if errors > 0, do: Output.warning("Errors: #{errors}")
    Output.puts("")

    # Character counts
    char_counts = stats["character_counts"] || %{}

    if map_size(char_counts) > 0 do
      Output.section("GAMES BY CHARACTER")

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

        Output.puts(
          "  #{String.pad_trailing(name, 18)} #{String.pad_leading(to_string(count), 6)} [#{bar}] #{pct}%#{marker}"
        )
      end)

      Output.puts("")
      Output.puts("  * = Low-tier characters (ExPhil targets)")
      Output.puts("")

      # Low-tier summary
      low_tier_counts =
        char_counts
        |> Enum.filter(fn {char_id, _} ->
          id = if is_binary(char_id), do: String.to_integer(char_id), else: char_id
          id in @low_tier_ids
        end)
        |> Enum.map(fn {_, count} -> count end)
        |> Enum.sum()

      if low_tier_counts > 0 do
        Output.section("LOW-TIER SUMMARY")
        Output.puts("  Total low-tier games: #{low_tier_counts}")
        Output.puts("  Percentage: #{Float.round(low_tier_counts / scanned * 100, 1)}%")
        Output.puts("")
      end
    else
      if stats["error"] do
        Output.warning("Note: #{stats["error"]}")
        Output.puts("")
      end
    end

    # Recommendations
    Output.section("TRAINING RECOMMENDATIONS")

    if map_size(char_counts) > 0 do
      # Find best low-tier to train
      low_tier_data =
        char_counts
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

        Output.success("Best low-tier for training: #{best_name} (#{best_count} games)")

        # Preset recommendation
        preset =
          case best_id do
            10 -> "mewtwo"
            25 -> "ganondorf"
            6 -> "link"
            3 -> "gameandwatch"
            18 -> "zelda"
            _ -> "production"
          end

        Output.puts("  Recommended preset: --preset #{preset}")

        # Data sufficiency
        cond do
          best_count >= 10000 ->
            Output.success("Data: Excellent - enough for production model")

          best_count >= 1000 ->
            Output.puts("  Data: Good - enough for solid training")

          best_count >= 100 ->
            Output.warning("Data: Limited - consider augmentation")

          true ->
            Output.warning("Data: Very limited - need more replays")
        end
      else
        Output.warning("No low-tier character data found.")
        Output.puts("  Download replays from Slippi Discord or play online!")
      end
    else
      Output.puts("  Install py-slippi for full analysis: pip install py-slippi")
      Output.puts("  Or download replays from the Slippi Discord server")
    end

    Output.divider()
  end
end

# Parse arguments
args = System.argv()

replays_dir =
  case Enum.find_index(args, &(&1 == "--replays")) do
    nil -> "./replays"
    idx -> Enum.at(args, idx + 1)
  end

max_files =
  case Enum.find_index(args, &(&1 == "--max-files")) do
    nil -> nil
    idx -> String.to_integer(Enum.at(args, idx + 1))
  end

if not File.dir?(replays_dir) do
  Output.error("Replay directory not found: #{replays_dir}")
  Output.puts("")
  Output.puts("Usage: mix run scripts/scan_replays.exs --replays /path/to/replays")
  Output.puts("")
  Output.puts("To get replays:")
  Output.puts("  1. Join the Slippi Discord (via slippi.gg)")
  Output.puts("  2. Look for 'anonymized ranked collections' in resources")
  Output.puts("  3. Or play online - replays save to ~/.slippi-launcher/replays/")
  System.halt(1)
end

ReplayScanner.scan(replays_dir, max_files: max_files)
