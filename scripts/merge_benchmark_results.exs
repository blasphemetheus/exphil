#!/usr/bin/env elixir
# Merge partial benchmark results from isolated architecture runs.
#
# Usage:
#   mix run scripts/merge_benchmark_results.exs --dir /tmp/exphil_bench_XXXXX
#   mix run scripts/merge_benchmark_results.exs --dir /tmp/partial --output checkpoints/merged.json
#
# Reads per-architecture JSON files from --dir, merges them into a single
# ranked results file with a comparison table.

alias ExPhil.Training.Output
require Output

# Parse args
args = System.argv()

dir =
  case Enum.find_index(args, &(&1 == "--dir")) do
    nil -> raise "Missing --dir argument"
    idx -> Enum.at(args, idx + 1) || raise "Missing value for --dir"
  end

output_path =
  case Enum.find_index(args, &(&1 == "--output")) do
    nil ->
      timestamp = Calendar.strftime(DateTime.utc_now(), "%Y%m%d_%H%M%S")
      "checkpoints/benchmark_results_#{timestamp}.json"
    idx ->
      Enum.at(args, idx + 1) || raise "Missing value for --output"
  end

# Read all partial JSON files
json_files = Path.wildcard(Path.join(dir, "*.json"))

if length(json_files) == 0 do
  Output.error("No JSON files found in #{dir}")
  System.halt(1)
end

Output.banner("Merging Benchmark Results")
Output.puts("Reading #{length(json_files)} partial results from #{dir}")

# Parse and extract results
{all_results, config, machine} =
  Enum.reduce(json_files, {[], nil, nil}, fn file, {results, cfg, mach} ->
    case Jason.decode!(File.read!(file)) do
      %{"results" => file_results} = data ->
        new_cfg = cfg || data["config"]
        new_mach = mach || data["machine"]
        {results ++ file_results, new_cfg, new_mach}

      _ ->
        Output.warning("Skipping #{Path.basename(file)} - unexpected format")
        {results, cfg, mach}
    end
  end)

if length(all_results) == 0 do
  Output.error("No valid results found!")
  System.halt(1)
end

# Safe rounding helper
safe_round = fn
  value, precision when is_float(value) -> Float.round(value, precision)
  value, precision when is_integer(value) -> Float.round(value / 1, precision)
  :nan, _precision -> :nan
  :infinity, _precision -> :infinity
  :neg_infinity, _precision -> :neg_infinity
  nil, _precision -> "N/A"
  value, _precision -> value
end

# Sort by validation loss (NaN goes to end)
sorted = Enum.sort_by(all_results, fn r ->
  val = r["final_val_loss"]
  cond do
    is_number(val) -> val
    true -> 999999.0
  end
end)

# Print ranking table
Output.section("Benchmark Results (#{length(sorted)} architectures)")
Output.puts("Ranked by validation loss (lower is better):\n")
Output.puts("  Rank | Architecture               | Val Loss | Train Loss | Speed (b/s) | Inference | Time")
Output.puts("  -----+----------------------------+----------+------------+-------------+-----------+--------")

sorted
|> Enum.with_index(1)
|> Enum.each(fn {r, rank} ->
  name = String.pad_trailing(r["name"] || r["id"] || "?", 26)
  val = safe_round.(r["final_val_loss"], 4) |> to_string() |> String.pad_leading(8)
  train = safe_round.(r["final_train_loss"], 4) |> to_string() |> String.pad_leading(10)
  speed = safe_round.(r["avg_batches_per_sec"], 1) |> to_string() |> String.pad_leading(11)

  inference =
    case r["inference_us_per_batch"] do
      us when is_number(us) -> "#{safe_round.(us / 1000, 1)}ms" |> String.pad_leading(9)
      _ -> "N/A" |> String.pad_leading(9)
    end

  time =
    case r["total_time_ms"] do
      ms when is_number(ms) -> "#{safe_round.(ms / 1000, 0)}s" |> String.pad_leading(6)
      _ -> "N/A" |> String.pad_leading(6)
    end

  # Mark NaN architectures
  marker = if r["final_val_loss"] in [:nan, "NaN", nil], do: " (NaN)", else: ""

  Output.puts("  #{rank |> to_string() |> String.pad_leading(4)} | #{name} | #{val} | #{train} | #{speed} | #{inference} | #{time}#{marker}")
end)

# Best architecture (excluding NaN)
valid_results = Enum.filter(sorted, fn r ->
  val = r["final_val_loss"]
  is_number(val) and val != :nan
end)

if length(valid_results) > 0 do
  best = List.first(valid_results)
  Output.puts("")
  Output.success("Best: #{best["name"]} (val_loss=#{safe_round.(best["final_val_loss"], 4)})")
end

# NaN summary
nan_results = Enum.filter(sorted, fn r ->
  val = r["final_val_loss"]
  not is_number(val) or val == :nan
end)

if length(nan_results) > 0 do
  nan_names = Enum.map(nan_results, fn r -> r["name"] || r["id"] end)
  Output.warning("Diverged to NaN (#{length(nan_results)}): #{Enum.join(nan_names, ", ")}")
end

# Save merged JSON
File.mkdir_p!(Path.dirname(output_path))

merged = %{
  "timestamp" => DateTime.to_iso8601(DateTime.utc_now()),
  "mode" => "isolated",
  "machine" => machine,
  "config" => config,
  "results" => sorted,
  "best" => if(length(valid_results) > 0, do: List.first(valid_results)["id"], else: nil),
  "num_architectures" => length(sorted),
  "num_diverged" => length(nan_results)
}

File.write!(output_path, Jason.encode!(merged, pretty: true))
Output.success("Merged results saved to #{output_path}")

# Update latest symlink
latest = "checkpoints/benchmark_results_latest.json"
File.rm(latest)
File.ln_s!(Path.basename(output_path), latest)
Output.puts("Symlinked to #{latest}")
