#!/usr/bin/env elixir
# Play against the trained agent in Dolphin via libmelee
#
# Usage:
#   mix run scripts/play_dolphin.exs [options]
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
alias ExPhil.Agents.Agent
alias ExPhil.Bridge.ControllerState

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
  dolphin: get_arg.("--dolphin", nil),
  iso: get_arg.("--iso", nil),
  port: String.to_integer(get_arg.("--port", "1")),
  opponent_port: String.to_integer(get_arg.("--opponent-port", "2")),
  character: String.to_atom(get_arg.("--character", "mewtwo")),
  stage: String.to_atom(get_arg.("--stage", "final_destination")),
  frame_delay: String.to_integer(get_arg.("--frame-delay", "0")),
  deterministic: has_flag.("--deterministic")
]

# Validate required args
if opts[:policy] == nil or opts[:dolphin] == nil or opts[:iso] == nil do
  IO.puts("""

  ExPhil Dolphin Play Script
  ==========================

  Play against a trained agent in Dolphin/Slippi!

  Usage:
    mix run scripts/play_dolphin.exs \\
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

  Example:
    mix run scripts/play_dolphin.exs \\
      --policy checkpoints/imitation_latest_policy.bin \\
      --dolphin ~/.local/share/Slippi\\ Launcher/netplay \\
      --iso ~/Games/SSBM.iso \\
      --character mewtwo \\
      --stage battlefield
  """)
  System.halt(1)
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                    ExPhil Dolphin Play                         ║
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
  :ok ->
    IO.puts("  ✓ Dolphin initialized and connected!")

  {:error, reason} ->
    IO.puts("  ✗ Failed to initialize Dolphin: #{inspect(reason)}")
    System.halt(1)
end

# Step 4: Game loop
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Game Loop Started!                        ║
╚════════════════════════════════════════════════════════════════╝

Press Ctrl+C to stop.

""")

defmodule GameLoop do
  def run(agent, bridge, player_port, stats \\ %{frames: 0, errors: 0}) do
    case MeleePort.step(bridge) do
      {:ok, game_state} ->
        # In game - run agent inference
        case Agent.get_controller(agent, game_state, player_port: player_port) do
          {:ok, controller} ->
            # Send controller input
            input = controller_to_input(controller)
            MeleePort.send_controller(bridge, input)

            # Print stats occasionally
            new_stats = %{stats | frames: stats.frames + 1}
            if rem(new_stats.frames, 600) == 0 do
              IO.puts("[Frame #{new_stats.frames}] Errors: #{new_stats.errors}")
            end

            run(agent, bridge, player_port, new_stats)

          {:error, reason} ->
            Logger.warning("Agent error: #{inspect(reason)}")
            new_stats = %{stats | frames: stats.frames + 1, errors: stats.errors + 1}
            run(agent, bridge, player_port, new_stats)
        end

      {:menu, _game_state} ->
        # In menu - let auto_menu handle it
        run(agent, bridge, player_port, stats)

      {:error, reason} ->
        IO.puts("\nGame ended or error: #{inspect(reason)}")
        IO.puts("Total frames: #{stats.frames}, Errors: #{stats.errors}")
        :ok
    end
  end

  defp controller_to_input(%ControllerState{} = cs) do
    %{
      main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
      c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
      shoulder: cs.l_shoulder + cs.r_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end
end

# Run the game loop
try do
  GameLoop.run(agent, bridge, opts[:port])
rescue
  e in RuntimeError ->
    IO.puts("\nError: #{Exception.message(e)}")
catch
  :exit, _ ->
    IO.puts("\nExiting...")
end

# Cleanup
IO.puts("\nCleaning up...")
MeleePort.stop(bridge)
GenServer.stop(agent)

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                        Session Complete!                       ║
╚════════════════════════════════════════════════════════════════╝
""")
