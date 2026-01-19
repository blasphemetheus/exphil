#!/usr/bin/env elixir
# Learning Rate Finder Script
#
# Usage:
#   mix run scripts/find_lr.exs --replays /path/to/replays
#   mix run scripts/find_lr.exs --replays /path/to/replays --min-lr 1e-8 --max-lr 10
#
# Options:
#   --replays      Path to replay directory (required)
#   --max-files    Max replay files to use (default: 10)
#   --min-lr       Starting LR (default: 1e-7)
#   --max-lr       Ending LR (default: 1.0)
#   --num-steps    Number of steps (default: 100)
#   --batch-size   Batch size (default: 64)
#   --hidden-sizes Hidden layer sizes (default: 64,64)
#   --player       Player port to learn from (default: 1)

alias ExPhil.Training.{Config, Data, LRFinder}
alias ExPhil.Networks.Policy

# Parse args
args = System.argv()

opts = [
  replays: Config.defaults()[:replays],
  max_files: 10,
  min_lr: 1.0e-7,
  max_lr: 1.0,
  num_steps: 100,
  batch_size: 64,
  hidden_sizes: [64, 64],
  player_port: 1
]

opts = Enum.reduce(args, opts, fn arg, acc ->
  case arg do
    "--replays" -> acc
    "--max-files" -> acc
    "--min-lr" -> acc
    "--max-lr" -> acc
    "--num-steps" -> acc
    "--batch-size" -> acc
    "--hidden-sizes" -> acc
    "--player" -> acc
    value ->
      # Get the previous flag
      idx = Enum.find_index(args, &(&1 == value)) - 1
      if idx >= 0 do
        flag = Enum.at(args, idx)
        case flag do
          "--replays" -> Keyword.put(acc, :replays, value)
          "--max-files" -> Keyword.put(acc, :max_files, String.to_integer(value))
          "--min-lr" -> Keyword.put(acc, :min_lr, String.to_float(value))
          "--max-lr" -> Keyword.put(acc, :max_lr, String.to_float(value))
          "--num-steps" -> Keyword.put(acc, :num_steps, String.to_integer(value))
          "--batch-size" -> Keyword.put(acc, :batch_size, String.to_integer(value))
          "--hidden-sizes" -> Keyword.put(acc, :hidden_sizes, Config.parse_hidden_sizes(value))
          "--player" -> Keyword.put(acc, :player_port, String.to_integer(value))
          _ -> acc
        end
      else
        acc
      end
  end
end)

IO.puts("""

╔══════════════════════════════════════════════════════════════╗
║              Learning Rate Finder                             ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Replays:      #{opts[:replays]}
  Max files:    #{opts[:max_files]}
  LR range:     #{opts[:min_lr]} -> #{opts[:max_lr]}
  Steps:        #{opts[:num_steps]}
  Batch size:   #{opts[:batch_size]}
  Hidden sizes: #{inspect(opts[:hidden_sizes])}

""")

# Check replays directory exists
unless File.dir?(opts[:replays]) do
  IO.puts("Error: Replays directory not found: #{opts[:replays]}")
  System.halt(1)
end

# Load replay data
IO.puts("Loading replay data...")

{frames, _stats} = Data.load_replays(
  opts[:replays],
  max_files: opts[:max_files],
  player_port: opts[:player_port]
)

IO.puts("Loaded #{length(frames)} frames")

# Create batched dataset
dataset = Data.create_batched_dataset(frames, opts[:batch_size])
embed_size = Data.compute_embed_size(frames)

IO.puts("Embed size: #{embed_size}")
IO.puts("")

# Initialize random model params
IO.puts("Initializing model...")

policy_config = %{
  embed_size: embed_size,
  hidden_sizes: opts[:hidden_sizes],
  num_buttons: 8,
  stick_classes: 9,
  shoulder_classes: 4
}

{init_fn, _} = Policy.create(policy_config)

# Create dummy input to initialize
dummy_input = Nx.broadcast(0.0, {1, embed_size})
model_params = init_fn.(dummy_input, Nx.Random.key(42))

IO.puts("Running LR finder (#{opts[:num_steps]} steps)...")
IO.puts("")

# Run LR finder
case LRFinder.find(model_params, dataset,
  min_lr: opts[:min_lr],
  max_lr: opts[:max_lr],
  num_steps: opts[:num_steps],
  hidden_sizes: opts[:hidden_sizes],
  embed_size: embed_size
) do
  {:ok, results} ->
    IO.puts(LRFinder.format_results(results))

    IO.puts("""

    ════════════════════════════════════════════════════════════════

    Recommendation:
      Use --lr #{LRFinder.format_results(results) |> String.split("Suggested LR:") |> Enum.at(1) |> String.split("\n") |> hd() |> String.trim()}

    Example:
      mix run scripts/train_from_replays.exs \\
        --lr #{Float.round(results.suggested_lr || results.min_loss_lr / 10, 6)} \\
        --epochs 10

    """)

  {:error, reason} ->
    IO.puts("Error: #{reason}")
    System.halt(1)
end
