defmodule ExPhil do
  @moduledoc """
  ExPhil - Elixir Phil

  A successor to slippi-ai for training Melee AI agents,
  with focus on lower-tier characters.

  ## Target Characters

  * Mewtwo - Teleport recovery, tail hurtbox, unique physics
  * Mr. Game & Watch - No L-cancel, random hammer, unique shield
  * Link - Projectiles, tether recovery, bomb tech
  * Ganondorf - Slow but powerful, spacing emphasis

  ## Architecture

  ExPhil uses a two-phase training approach:

  1. **Imitation Learning** - Behavioral cloning from human replay data
  2. **Reinforcement Learning** - Self-play refinement with PPO

  ## Quick Start

      # Parse replay data
      mix exphil.parse_replays --input ./replays --output ./parsed

      # Train with imitation learning
      mix exphil.train --mode imitation --character mewtwo

      # Fine-tune with RL
      mix exphil.train --mode rl --checkpoint ./checkpoints/latest.axon

  ## Core Modules

  * `ExPhil.Networks` - Neural network architectures (policy, value)
  * `ExPhil.Embeddings` - Game state to tensor conversion
  * `ExPhil.Training` - Training loops (imitation, PPO)
  * `ExPhil.Agents` - Inference-time agent implementations
  * `ExPhil.Bridge` - Python/libmelee communication

  """

  @doc """
  Returns the ExPhil version.
  """
  def version, do: "0.1.0"
end
