#!/usr/bin/env elixir
# Regenerate HTML visualization from benchmark JSON results
#
# Usage:
#   mix run scripts/regenerate_benchmark_html.exs --json checkpoints/benchmark_results_20260130_002013.json
#

alias ExPhil.Reports.Benchmark

# Parse args
args = System.argv()

json_path = case Enum.find_index(args, &(&1 == "--json")) do
  nil ->
    # Try to find most recent benchmark JSON
    case Path.wildcard("checkpoints/benchmark_results_*.json") |> Enum.sort() |> List.last() do
      nil ->
        IO.puts("Usage: mix run scripts/regenerate_benchmark_html.exs --json <path>")
        System.halt(1)
      path -> path
    end
  idx -> Enum.at(args, idx + 1)
end

IO.puts("Loading results from: #{json_path}")

# Parse and normalize results using the module
results = Benchmark.parse_results(json_path)

IO.puts("Loaded #{length(results)} results")

# Generate HTML
html = Benchmark.generate_html(results)

# Save HTML
output_path = String.replace(json_path, ".json", ".html")
File.write!(output_path, html)
IO.puts("HTML report saved to: #{output_path}")
