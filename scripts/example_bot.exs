#!/usr/bin/env elixir
# Example script showing how to use the ExPhil bridge to run a simple bot.
#
# Usage:
#   mix run scripts/example_bot.exs \
#     --dolphin /path/to/slippi \
#     --iso /path/to/melee.iso \
#     --character mewtwo

defmodule ExampleBot do
  @moduledoc """
  A simple spacing bot that demonstrates the ExPhil bridge.

  This bot:
  1. Moves toward the opponent when far away
  2. Retreats when too close
  3. Attacks at medium range
  """

  alias ExPhil.Bridge
  alias ExPhil.Bridge.{GameState, Player, ControllerInput}

  # Distance thresholds
  @attack_range 15.0
  @comfortable_range 30.0
  @approach_range 60.0

  def run(opts) do
    dolphin_path = Keyword.fetch!(opts, :dolphin_path)
    iso_path = Keyword.fetch!(opts, :iso_path)
    character = Keyword.get(opts, :character, :mewtwo)

    IO.puts("Starting ExPhil bridge...")
    IO.puts("  Dolphin: #{dolphin_path}")
    IO.puts("  ISO: #{iso_path}")
    IO.puts("  Character: #{character}")

    case Bridge.start(
      dolphin_path: dolphin_path,
      iso_path: iso_path,
      character: character,
      stage: :final_destination
    ) do
      {:ok, bridge} ->
        IO.puts("Bridge started successfully!")
        run_loop(bridge)

      {:error, reason} ->
        IO.puts("Failed to start bridge: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp run_loop(bridge) do
    IO.puts("Running bot... Press Ctrl+C to stop.")

    result = Bridge.run_ai(bridge, &decide/3, max_frames: :infinity)

    case result do
      {:ok, reason} ->
        IO.puts("Bot stopped: #{inspect(reason)}")

      {:error, reason} ->
        IO.puts("Bot error: #{inspect(reason)}")
    end

    Bridge.stop(bridge)
  end

  defp decide(own, opp, _game_state) do
    # Calculate horizontal distance to opponent
    dx = opp.x - own.x
    distance = abs(dx)

    # Determine facing direction needed
    should_face_right = dx > 0

    # Simple state machine
    cond do
      # Too close - retreat and shield if in danger
      distance < @attack_range and Player.in_hitstun?(own) ->
        ControllerInput.shield()

      # Attack range - throw out attacks
      distance < @attack_range ->
        pick_attack(own, opp)

      # Comfortable range - space with movement and pokes
      distance < @comfortable_range ->
        if should_face_right do
          ControllerInput.main_stick(0.6, 0.5)  # Walk right
        else
          ControllerInput.main_stick(0.4, 0.5)  # Walk left
        end

      # Far away - approach
      distance < @approach_range ->
        approach(should_face_right, own)

      # Very far - dash
      true ->
        if should_face_right do
          ControllerInput.right()
        else
          ControllerInput.left()
        end
    end
  end

  defp pick_attack(own, opp) do
    # Simple attack selection based on relative positions
    height_diff = opp.y - own.y

    cond do
      # Opponent above - up tilt or up smash
      height_diff > 10 ->
        ControllerInput.up_tilt()

      # Opponent below (we're on platform) - down air
      height_diff < -10 ->
        ControllerInput.combine([
          ControllerInput.main_stick(0.5, 0.0),
          ControllerInput.button(:a)
        ])

      # Same level - forward attack
      opp.x > own.x ->
        ControllerInput.forward_tilt_right()

      true ->
        ControllerInput.forward_tilt_left()
    end
  end

  defp approach(should_face_right, own) do
    if Player.in_hitstun?(own) do
      # DI away if in hitstun
      if should_face_right do
        ControllerInput.left()
      else
        ControllerInput.right()
      end
    else
      # Dash toward opponent
      if should_face_right do
        ControllerInput.right()
      else
        ControllerInput.left()
      end
    end
  end
end

# Parse command line arguments
{opts, _, _} = OptionParser.parse(System.argv(),
  strict: [
    dolphin: :string,
    iso: :string,
    character: :string
  ]
)

dolphin_path = opts[:dolphin] ||
  System.get_env("SLIPPI_PATH") ||
  raise "Missing --dolphin path or SLIPPI_PATH env var"

iso_path = opts[:iso] ||
  System.get_env("MELEE_ISO") ||
  raise "Missing --iso path or MELEE_ISO env var"

character = case opts[:character] do
  nil -> :mewtwo
  str -> String.to_atom(String.downcase(str))
end

ExampleBot.run(
  dolphin_path: dolphin_path,
  iso_path: iso_path,
  character: character
)
