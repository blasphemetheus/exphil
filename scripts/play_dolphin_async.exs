#!/usr/bin/env elixir
# Play against the trained agent in Dolphin via libmelee (ASYNC VERSION)
#
# This version uses separate processes for frame reading and inference,
# allowing the game loop to run at full speed even with slow models.
#
# Usage:
#   mix run scripts/play_dolphin_async.exs [options]
#
# Options:
#   --policy PATH       - Path to policy file (required)
#   --dolphin PATH      - Path to Slippi/Dolphin folder (required)
#   --iso PATH          - Path to Melee 1.02 ISO (required)
#   --port N            - Agent controller port (1-4, default: 1)
#   --opponent-port N   - Your controller port (1-4, default: 2)
#   --character NAME    - Agent character (default: mewtwo)
#   --stage NAME        - Stage (default: final_destination)
#   --frame-delay N     - Simulated online delay (default: 0)
#   --deterministic     - Use deterministic action selection

require Logger

alias ExPhil.Bridge.MeleePort
alias ExPhil.Bridge.AsyncRunner
alias ExPhil.Agents.Agent

# Parse command line arguments
args = System.argv()

get_arg = fn flag, default ->
  case Enum.find_index(args, &(&1 == flag)) do
    nil -> default
    idx -> Enum.at(args, idx + 1) || default
  end
end

has_flag = fn flag -> Enum.member?(args, flag) end

parse_on_game_end = fn str ->
  case str do
    "restart" -> :restart
    "stop" -> :stop
    _ -> :restart
  end
end

opts = [
  policy: get_arg.("--policy", nil),
  dolphin: get_arg.("--dolphin", nil),
  iso: get_arg.("--iso", nil),
  port: String.to_integer(get_arg.("--port", "1")),
  opponent_port: String.to_integer(get_arg.("--opponent-port", "2")),
  character: String.to_atom(get_arg.("--character", "mewtwo")),
  stage: String.to_atom(get_arg.("--stage", "final_destination")),
  frame_delay: String.to_integer(get_arg.("--frame-delay", "0")),
  deterministic: has_flag.("--deterministic"),
  no_auto_menu: has_flag.("--no-auto-menu"),
  on_game_end: parse_on_game_end.(get_arg.("--on-game-end", "restart"))
]

# Validate required args
if opts[:policy] == nil or opts[:dolphin] == nil or opts[:iso] == nil do
  IO.puts("""

  ExPhil Dolphin Play Script (ASYNC VERSION)
  ==========================================

  Play against a trained agent in Dolphin/Slippi!

  This async version separates frame reading from inference,
  allowing the game to run smoothly even with slow LSTM models.

  Usage:
    mix run scripts/play_dolphin_async.exs \\
      --policy checkpoints/policy.bin \\
      --dolphin /path/to/slippi \\
      --iso /path/to/melee.iso

  Required:
    --policy PATH       Path to exported policy file
    --dolphin PATH      Path to Slippi/Dolphin folder
    --iso PATH          Path to Melee 1.02 ISO

  Options:
    --port N            Agent controller port (default: 1)
    --opponent-port N   Your controller port (default: 2)
    --character NAME    Agent character (default: mewtwo)
    --stage NAME        Stage (default: final_destination)
    --frame-delay N     Simulated online delay (default: 0)
    --deterministic     Use deterministic actions (no sampling)
    --on-game-end MODE  What to do after game ends (default: restart)
                        restart = auto-start next game
                        stop    = exit after one game
  """)
  System.halt(1)
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil Dolphin Play (ASYNC)                       ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Policy:       #{opts[:policy]}
  Dolphin:      #{opts[:dolphin]}
  ISO:          #{opts[:iso]}
  Agent Port:   #{opts[:port]}
  Your Port:    #{opts[:opponent_port]}
  Character:    #{opts[:character]}
  Stage:        #{opts[:stage]}
  Frame Delay:  #{opts[:frame_delay]}
  Deterministic: #{opts[:deterministic]}
  On Game End:  #{opts[:on_game_end]}

  Architecture: ASYNC (separate frame reader + inference processes)

""")

# Step 1: Load the agent
IO.puts("Step 1: Loading agent...")

{:ok, agent} = Agent.start_link(
  policy_path: opts[:policy],
  deterministic: opts[:deterministic],
  frame_delay: opts[:frame_delay]
)

config = Agent.get_config(agent)
IO.puts("  ✓ Agent loaded")
IO.puts("    Temporal: #{config.temporal}")
if config.temporal do
  IO.puts("    Backbone: #{config.backbone}")
  IO.puts("    Window:   #{config.window_size} frames")
end

# Step 2: Start the Melee bridge
IO.puts("\nStep 2: Starting Melee bridge...")

{:ok, bridge} = MeleePort.start_link()
IO.puts("  ✓ Bridge process started")

# Step 3: Initialize Dolphin
IO.puts("\nStep 3: Initializing Dolphin...")
IO.puts("  (This will launch Dolphin - make sure to plug in your controller!)")

bridge_config = %{
  dolphin_path: opts[:dolphin],
  iso_path: opts[:iso],
  controller_port: opts[:port],
  opponent_port: opts[:opponent_port],
  character: opts[:character],
  stage: opts[:stage],
  online_delay: opts[:frame_delay]
}

case MeleePort.init_console(bridge, bridge_config, 60_000) do
  {:ok, info} ->
    IO.puts("  ✓ Dolphin initialized and connected!")
    IO.puts("    Controller on port: #{info.controller_port}")

  :ok ->
    IO.puts("  ✓ Dolphin initialized and connected!")

  {:error, reason} ->
    IO.puts("  ✗ Failed to initialize Dolphin: #{inspect(reason)}")
    System.halt(1)
end

# Step 4: JIT Warmup
IO.puts("\nStep 4: JIT Warmup (this may take a minute for temporal models)...")
case Agent.warmup(agent) do
  {:ok, warmup_ms} ->
    IO.puts("  ✓ JIT warmup complete (#{warmup_ms}ms)")
  {:error, reason} ->
    IO.puts("  ⚠ Warmup failed: #{inspect(reason)} (will warmup on first inference)")
end

# Step 5: Start async runner
IO.puts("\nStep 5: Starting async game runner...")

{:ok, runner} = AsyncRunner.start_link(
  agent: agent,
  bridge: bridge,
  player_port: opts[:port],
  auto_menu: not opts[:no_auto_menu],
  on_game_end: opts[:on_game_end]
)

IO.puts("  ✓ Async runner started")

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                   ASYNC Game Loop Running!                     ║
╚════════════════════════════════════════════════════════════════╝

Frame reader and inference are running in separate processes.
The game should respond smoothly even with slow LSTM models.

Press Ctrl+C to stop.

""")

# Stats monitoring loop
defmodule StatsMonitor do
  def run(runner, interval_ms \\ 5000) do
    Process.sleep(interval_ms)

    stats = ExPhil.Bridge.AsyncRunner.get_stats(runner)

    if stats.elapsed_ms > 0 do
      elapsed_s = stats.elapsed_ms / 1000
      games_str = if stats.games_played > 0, do: " | Games: #{stats.games_played}", else: ""
      IO.puts("[Stats] #{Float.round(elapsed_s, 1)}s | Frames: #{stats.frames_read} (#{Float.round(stats.fps, 1)} fps) | Inferences: #{stats.inferences_run} (#{Float.round(stats.inference_rate, 1)}/s)#{games_str}")
    end

    run(runner, interval_ms)
  end
end

# Run stats monitor
try do
  StatsMonitor.run(runner, 5000)
rescue
  e in RuntimeError ->
    IO.puts("\nError: #{Exception.message(e)}")
catch
  :exit, _ ->
    IO.puts("\nExiting...")
end

# Cleanup
IO.puts("\nCleaning up...")
case AsyncRunner.stop(runner) do
  {:ok, final_stats} ->
    IO.puts("  Final stats: #{final_stats.frames} frames, #{final_stats.inferences} inferences, #{final_stats.games} games")
  _ ->
    :ok
end

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> IO.puts("  (cleanup timed out, Dolphin may still be running)")
end
GenServer.stop(agent)

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                        Session Complete!                       ║
╚════════════════════════════════════════════════════════════════╝
""")
