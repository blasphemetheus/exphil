#!/usr/bin/env elixir
# Model evaluation script: Test a trained policy
#
# Usage:
#   mix run scripts/eval_model.exs [options]
#
# Options:
#   --policy PATH     - Path to exported policy file (required)
#   --replays PATH    - Path to replay directory for test data
#   --max-files N     - Max replay files to use (default: 10)
#   --player PORT     - Player port (1-4, default: 1)
#   --deterministic   - Use deterministic action selection
#   --benchmark       - Run inference speed benchmark

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.Data
alias ExPhil.Agents.Agent
alias ExPhil.Embeddings

# Parse command line arguments
args = System.argv()

get_arg = fn flag, default ->
  case Enum.find_index(args, &(&1 == flag)) do
    nil -> default
    idx -> Enum.at(args, idx + 1) || default
  end
end

has_flag = fn flag -> Enum.member?(args, flag) end

opts = [
  policy: get_arg.("--policy", nil),
  replays: get_arg.("--replays", "/home/dori/git/melee/replays"),
  max_files: String.to_integer(get_arg.("--max-files", "10")),
  player_port: String.to_integer(get_arg.("--player", "1")),
  deterministic: has_flag.("--deterministic"),
  benchmark: has_flag.("--benchmark")
]

if opts[:policy] == nil do
  IO.puts("""
  Error: --policy PATH is required

  Usage:
    mix run scripts/eval_model.exs --policy checkpoints/policy.bin [options]

  Options:
    --replays PATH    - Replay directory for test frames
    --max-files N     - Max files (default: 10)
    --deterministic   - Deterministic action selection
    --benchmark       - Run speed benchmark
  """)
  System.halt(1)
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                    ExPhil Model Evaluation                     ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Policy:       #{opts[:policy]}
  Replays:      #{opts[:replays]}
  Max Files:    #{opts[:max_files]}
  Player Port:  #{opts[:player_port]}
  Deterministic: #{opts[:deterministic]}
  Benchmark:    #{opts[:benchmark]}

""")

# Step 1: Load policy
IO.puts("Step 1: Loading policy...")

case ExPhil.Training.load_policy(opts[:policy]) do
  {:ok, policy} ->
    config = policy.config
    IO.puts("  ✓ Policy loaded")
    IO.puts("    Temporal: #{config[:temporal] || false}")
    if config[:temporal] do
      IO.puts("    Backbone: #{config[:backbone]}")
      IO.puts("    Window:   #{config[:window_size]} frames")
    end
    IO.puts("    Embed size: #{config[:embed_size]}")

  {:error, reason} ->
    IO.puts("  ✗ Failed to load policy: #{inspect(reason)}")
    System.halt(1)
end

# Step 2: Start agent
IO.puts("\nStep 2: Starting agent...")

{:ok, agent} = Agent.start_link(
  policy_path: opts[:policy],
  deterministic: opts[:deterministic]
)

agent_config = Agent.get_config(agent)
IO.puts("  ✓ Agent started")
IO.puts("    Temporal: #{agent_config.temporal}")
IO.puts("    Window:   #{agent_config.window_size}")

# Step 3: Load test frames
IO.puts("\nStep 3: Loading test frames...")

replay_files = Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
|> Enum.take(opts[:max_files])

IO.puts("  Found #{length(replay_files)} replay files")

test_frames = replay_files
|> Enum.flat_map(fn path ->
  case Peppi.parse(path, player_port: opts[:player_port]) do
    {:ok, replay} ->
      Peppi.to_training_frames(replay, player_port: opts[:player_port])
    {:error, _} ->
      []
  end
end)
|> Enum.take(1000)  # Limit to 1000 frames for eval

IO.puts("  Loaded #{length(test_frames)} test frames")

if length(test_frames) == 0 do
  IO.puts("\n❌ No test frames found. Check replay path and player port.")
  System.halt(1)
end

# Step 4: Run inference
IO.puts("\nStep 4: Running inference...")

# Test on sample frames
sample_frames = Enum.take(test_frames, 10)

IO.puts("\n  Sample actions:")

# Helper to convert button tensor to human-readable format
button_names = [:a, :b, :x, :y, :z, :l, :r, :d_up]
format_buttons = fn buttons ->
  cond do
    is_struct(buttons, Nx.Tensor) ->
      # Convert tensor to list of active button names
      buttons
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.filter(fn {val, _idx} -> val == 1 or val == true end)
      |> Enum.map(fn {_val, idx} -> Enum.at(button_names, idx) end)
      |> Enum.join(", ")

    is_map(buttons) ->
      # Already a map, filter active buttons
      buttons
      |> Enum.filter(fn {_k, v} -> v end)
      |> Enum.map(fn {k, _} -> k end)
      |> Enum.join(", ")

    true ->
      inspect(buttons)
  end
end

for {frame, idx} <- Enum.with_index(sample_frames) do
  case Agent.get_action(agent, frame.game_state, player_port: opts[:player_port]) do
    {:ok, action} ->
      buttons_str = format_buttons.(action.buttons)
      buttons_str = if buttons_str == "", do: "none", else: buttons_str

      # Handle tensor or integer for stick values
      main_x = if is_struct(action.main_x, Nx.Tensor), do: Nx.to_number(Nx.squeeze(action.main_x)), else: action.main_x
      main_y = if is_struct(action.main_y, Nx.Tensor), do: Nx.to_number(Nx.squeeze(action.main_y)), else: action.main_y

      IO.puts("    Frame #{idx}: buttons=[#{buttons_str}] stick=(#{main_x}, #{main_y})")

    {:error, reason} ->
      IO.puts("    Frame #{idx}: ERROR - #{inspect(reason)}")
  end
end

# Step 5: Benchmark (optional)
if opts[:benchmark] do
  IO.puts("\nStep 5: Running inference benchmark...")

  # Warmup
  IO.puts("  Warming up (100 frames)...")
  for frame <- Enum.take(test_frames, 100) do
    Agent.get_action(agent, frame.game_state, player_port: opts[:player_port])
  end

  # Reset buffer for temporal models
  Agent.reset_buffer(agent)

  # Benchmark
  bench_frames = Enum.take(test_frames, 500)
  IO.puts("  Benchmarking #{length(bench_frames)} frames...")

  {time_us, _} = :timer.tc(fn ->
    for frame <- bench_frames do
      Agent.get_action(agent, frame.game_state, player_port: opts[:player_port])
    end
  end)

  total_ms = time_us / 1000
  per_frame_ms = total_ms / length(bench_frames)
  fps = 1000 / per_frame_ms

  IO.puts("""

  Benchmark Results:
    Total time:    #{Float.round(total_ms, 1)} ms
    Per frame:     #{Float.round(per_frame_ms, 3)} ms
    Throughput:    #{Float.round(fps, 1)} FPS
    60 FPS target: #{if per_frame_ms < 16.67, do: "✓ PASS", else: "✗ FAIL (need < 16.67ms)"}
  """)
end

# Cleanup
GenServer.stop(agent)

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                     Evaluation Complete!                       ║
╚════════════════════════════════════════════════════════════════╝
""")
