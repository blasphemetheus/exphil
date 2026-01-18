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
#   --action-repeat N   - Only compute new action every N frames (default: 1)

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
  deterministic: has_flag.("--deterministic"),
  no_auto_menu: has_flag.("--no-auto-menu"),
  action_repeat: String.to_integer(get_arg.("--action-repeat", "1"))
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
  Action Repeat: #{opts[:action_repeat]}

""")

# Step 1: Load the agent
IO.puts("Step 1: Loading agent...")

{:ok, agent} = Agent.start_link(
  policy_path: opts[:policy],
  deterministic: opts[:deterministic],
  frame_delay: opts[:frame_delay],
  action_repeat: opts[:action_repeat]
)

config = Agent.get_config(agent)
IO.puts("  ✓ Agent loaded")
IO.puts("    Temporal: #{config.temporal}")
if config.temporal do
  IO.puts("    Backbone: #{config.backbone}")
  IO.puts("    Window:   #{config.window_size} frames")
end
if opts[:action_repeat] > 1 do
  IO.puts("    Action Repeat: every #{opts[:action_repeat]} frames")
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

# Step 4: JIT Warmup (run dummy inference during menu navigation)
IO.puts("\nStep 4: JIT Warmup (this may take a minute for temporal models)...")
case Agent.warmup(agent) do
  {:ok, warmup_ms} ->
    IO.puts("  ✓ JIT warmup complete (#{warmup_ms}ms)")
  {:error, reason} ->
    IO.puts("  ⚠ Warmup failed: #{inspect(reason)} (will warmup on first game frame)")
end

# Step 5: Game loop
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Game Loop Started!                        ║
╚════════════════════════════════════════════════════════════════╝

Press Ctrl+C to stop.

""")

defmodule GameLoop do
  @moduledoc "Main game loop with input logging and game-end detection."

  defp timestamp do
    Time.utc_now() |> Time.truncate(:second) |> Time.to_string()
  end

  defp elapsed_time(nil), do: "0s"
  defp elapsed_time(start_time) do
    elapsed_ms = System.monotonic_time(:millisecond) - start_time
    elapsed_s = div(elapsed_ms, 1000)
    "#{elapsed_s}s"
  end

  def run(agent, bridge, player_port, opts \\ []) do
    stats = Keyword.get(opts, :stats, %{
      frames: 0,
      errors: 0,
      in_game: false,
      game_ended: false,
      last_stocks: nil,
      start_time: nil
    })
    auto_menu = not Keyword.get(opts, :no_auto_menu, false)

    case MeleePort.step(bridge, auto_menu: auto_menu) do
      {:ok, game_state} ->
        handle_in_game(agent, bridge, player_port, game_state, stats, opts)

      {:postgame, game_state} ->
        elapsed = elapsed_time(stats.start_time)
        IO.puts("\n[#{timestamp()}] 🏆 POSTGAME! Game ended at frame #{game_state.frame} (#{elapsed})")
        IO.puts("   Total agent frames: #{stats.frames}, Errors: #{stats.errors}")
        new_stats = %{stats | in_game: false, game_ended: true}
        run(agent, bridge, player_port, Keyword.put(opts, :stats, new_stats))

      {:menu, game_state} ->
        if stats.in_game do
          elapsed = elapsed_time(stats.start_time)
          IO.puts("\n[#{timestamp()}] 📋 Back to MENU (#{inspect(game_state.menu_state)}) after #{stats.frames} frames (#{elapsed})")
          new_stats = %{stats | in_game: false}
          run(agent, bridge, player_port, Keyword.put(opts, :stats, new_stats))
        else
          run(agent, bridge, player_port, opts)
        end

      {:game_ended, reason} ->
        elapsed = elapsed_time(stats.start_time)
        IO.puts("\n[#{timestamp()}] 🏁 Game ended: #{reason} (#{elapsed})")
        IO.puts("   Total agent frames: #{stats.frames}, Errors: #{stats.errors}")
        {:ok, stats}

      {:error, reason} ->
        IO.puts("\n[#{timestamp()}] Error: #{inspect(reason)}")
        IO.puts("Total frames: #{stats.frames}, Errors: #{stats.errors}")
        {:error, reason}
    end
  end

  defp handle_in_game(agent, bridge, player_port, game_state, stats, opts) do
    # Log game start
    stats = if not stats.in_game do
      IO.puts("\n[#{timestamp()}] 🎮 IN GAME! Starting agent control at frame #{game_state.frame}")
      %{stats | in_game: true, start_time: System.monotonic_time(:millisecond)}
    else
      stats
    end

    # Check for stock changes and game end
    {stats, game_over} = check_stocks(game_state, stats, player_port)

    if game_over do
      elapsed = elapsed_time(stats.start_time)
      IO.puts("\n[#{timestamp()}] 🏆 GAME OVER! Detected via stocks at frame #{game_state.frame} (#{elapsed})")
      stats = %{stats | game_ended: true, in_game: false}
      run(agent, bridge, player_port, Keyword.put(opts, :stats, stats))
    else
      # Run agent inference and send input
      case Agent.get_controller(agent, game_state, player_port: player_port) do
        {:ok, controller} ->
          input = controller_to_input(controller)

          # Log inputs periodically (every 30 agent frames)
          if rem(stats.frames, 30) == 0 do
            log_input(controller, game_state, stats.frames, stats.start_time)
          end

          case MeleePort.send_controller(bridge, input) do
            :ok ->
              stats = if stats.frames == 0 do
                IO.puts("[#{timestamp()}] 🕹️  First input sent at game frame #{game_state.frame}")
                stats
              else
                stats
              end

              stats = %{stats | frames: stats.frames + 1}
              run(agent, bridge, player_port, Keyword.put(opts, :stats, stats))

            {:game_ended, reason} ->
              elapsed = elapsed_time(stats.start_time)
              IO.puts("\n[#{timestamp()}] 🏁 Game ended (controller send): #{reason} (#{elapsed})")
              IO.puts("   Total agent frames: #{stats.frames}")
              {:ok, stats}
          end

        {:error, reason} ->
          Logger.warning("Agent error: #{inspect(reason)}")
          stats = %{stats | frames: stats.frames + 1, errors: stats.errors + 1}
          run(agent, bridge, player_port, Keyword.put(opts, :stats, stats))
      end
    end
  end

  defp check_stocks(game_state, stats, player_port) do
    players = game_state.players || %{}
    agent_player = players[player_port]
    opponent_port = if player_port == 1, do: 2, else: 1
    opponent_player = players[opponent_port]

    current_stocks = %{
      agent: agent_player && agent_player.stock,
      opponent: opponent_player && opponent_player.stock
    }

    # Log stock changes
    stats = if stats.last_stocks && stats.last_stocks != current_stocks do
      ts = Time.utc_now() |> Time.truncate(:second) |> Time.to_string()
      if current_stocks.agent != stats.last_stocks.agent do
        IO.puts("[#{ts}] 💀 Agent lost a stock! (#{current_stocks.agent} remaining)")
      end
      if current_stocks.opponent != stats.last_stocks.opponent do
        IO.puts("[#{ts}] 💥 Opponent lost a stock! (#{current_stocks.opponent} remaining)")
      end
      %{stats | last_stocks: current_stocks}
    else
      %{stats | last_stocks: current_stocks}
    end

    # Check for game over (someone at 0 stocks)
    game_over = (current_stocks.agent == 0) || (current_stocks.opponent == 0)

    {stats, game_over}
  end

  defp log_input(controller, game_state, frame_count, start_time) do
    buttons = []
    buttons = if controller.button_a, do: ["A" | buttons], else: buttons
    buttons = if controller.button_b, do: ["B" | buttons], else: buttons
    buttons = if controller.button_x, do: ["X" | buttons], else: buttons
    buttons = if controller.button_y, do: ["Y" | buttons], else: buttons
    buttons = if controller.button_z, do: ["Z" | buttons], else: buttons
    buttons = if controller.button_l, do: ["L" | buttons], else: buttons
    buttons = if controller.button_r, do: ["R" | buttons], else: buttons

    stick_x = Float.round(controller.main_stick.x, 2)
    stick_y = Float.round(controller.main_stick.y, 2)

    buttons_str = if buttons == [], do: "-", else: Enum.join(Enum.reverse(buttons), "+")

    # Get player info if available
    players = game_state.players || %{}
    p1 = players[1]
    p2 = players[2]
    p1_info = if p1, do: "P1:#{round(p1.percent)}%/#{p1.stock}stk", else: "P1:?"
    p2_info = if p2, do: "P2:#{round(p2.percent)}%/#{p2.stock}stk", else: "P2:?"

    elapsed = elapsed_time(start_time)
    IO.puts("[#{timestamp()} +#{elapsed}] f#{frame_count} Stick:(#{stick_x},#{stick_y}) #{buttons_str} | #{p1_info} #{p2_info}")
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
  GameLoop.run(agent, bridge, opts[:port], no_auto_menu: opts[:no_auto_menu])
rescue
  e in RuntimeError ->
    IO.puts("\nError: #{Exception.message(e)}")
catch
  :exit, _ ->
    IO.puts("\nExiting...")
end

# Cleanup
IO.puts("\nCleaning up...")
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
