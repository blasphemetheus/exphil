defmodule ExPhil do
  @moduledoc """
  ExPhil - Elixir-based Melee AI for lower-tier characters.

  ExPhil is a successor to [slippi-ai](https://github.com/vladfi1/slippi-ai),
  implementing behavioral cloning and reinforcement learning for Super Smash Bros. Melee
  using Elixir's Nx/Axon ML stack.

  ## Installation

  Add `exphil` to your list of dependencies in `mix.exs`:

      def deps do
        [
          {:exphil, "~> 0.1.0"}
        ]
      end

  ## Target Characters

  ExPhil focuses on lower-tier characters with unique mechanics:

  | Character | Key Mechanics |
  |-----------|---------------|
  | Mewtwo | Teleport recovery, tail hurtbox, unique physics |
  | Mr. Game & Watch | No L-cancel, random hammer, bucket tech |
  | Link | Projectiles, tether recovery, bomb tech |
  | Ganondorf | Slow but powerful, spacing emphasis |
  | Zelda | Transform mechanics, sweetspot spacing |

  ## Quick Start

  ### Training from Replays

      # Create a trainer
      trainer = ExPhil.Training.Imitation.new(
        embed_size: 287,
        hidden_sizes: [512, 512],
        learning_rate: 1.0e-4
      )

      # Load and train on replay data
      dataset = ExPhil.Training.Data.from_replays("./replays", character: :mewtwo)
      {:ok, trained} = ExPhil.Training.Imitation.train(trainer, dataset, epochs: 10)

      # Save checkpoint
      ExPhil.Training.Imitation.save_checkpoint(trained, "checkpoints/mewtwo_v1.axon")

  ### Running an Agent

      # Start an agent with a trained policy
      {:ok, agent} = ExPhil.Agents.Agent.start_link(
        name: :mewtwo_agent,
        policy_path: "checkpoints/mewtwo_v1.axon"
      )

      # Get action for a game state
      {:ok, action} = ExPhil.Agents.Agent.get_action(agent, game_state)

  ### CLI Training

      # Quick test run
      mix run scripts/train_from_replays.exs --preset quick

      # Production training with Mamba backbone
      mix run scripts/train_from_replays.exs \\
        --preset production \\
        --train-character mewtwo \\
        --backbone mamba \\
        --wandb

  ## Architecture Overview

  ```
  Replays (.slp) ──> Parser ──> Embeddings ──> Policy Network ──> Actions
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  Game State (287d)  │
                          │  - Players (2×58d)  │
                          │  - Stage (7d)       │
                          │  - Projectiles      │
                          └─────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  Backbone Network   │
                          │  - MLP / Mamba      │
                          │  - LSTM / Attention │
                          └─────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  6-Head Controller  │
                          │  - Buttons (8)      │
                          │  - Main stick (x,y) │
                          │  - C-stick (x,y)    │
                          │  - Shoulder (1)     │
                          └─────────────────────┘
  ```

  ## Module Overview

  ### Training

  - `ExPhil.Training.Imitation` - Behavioral cloning from replay data
  - `ExPhil.Training.PPO` - Proximal Policy Optimization for self-play
  - `ExPhil.Training.Config` - CLI argument parsing and presets
  - `ExPhil.Training.Data` - Dataset loading and batching

  ### Embeddings

  - `ExPhil.Embeddings.Game` - Full game state embedding (287 dims default)
  - `ExPhil.Embeddings.Player` - Player state embedding
  - `ExPhil.Embeddings.Controller` - Controller input embedding

  ### Networks

  - `ExPhil.Networks.Policy` - 6-head autoregressive policy network
  - `ExPhil.Networks.GatedSSM` - Mamba/S4 backbone for temporal modeling
  - `ExPhil.Networks.Attention` - Transformer attention layers

  ### Bridge

  - `ExPhil.Bridge.MeleePort` - Python/libmelee communication via Port
  - `ExPhil.Bridge.AsyncRunner` - 60fps frame loop with async inference

  ### Agents

  - `ExPhil.Agents.Agent` - GenServer holding trained policy for inference
  - `ExPhil.Agents.Supervisor` - Dynamic supervisor for agent processes

  ## Embedding Dimensions

  ExPhil uses learned embeddings by default (287 dims) for efficiency:

  | Mode | Dimensions | Description |
  |------|------------|-------------|
  | Learned (default) | 287 | Action/character IDs embedded in network |
  | One-hot (legacy) | 1204 | Full one-hot encodings |

  See `ExPhil.Embeddings.Game` for detailed dimension breakdown.

  ## Guides

  - [Architecture](architecture.html) - System design and data flow
  - [Training](training.html) - Training commands and options
  - [Inference](inference.html) - ONNX export and optimization
  - [Dolphin](dolphin.html) - Running agents against Dolphin

  ## Links

  - [GitHub](https://github.com/blasphemetheus/exphil)
  - [slippi-ai Reference](https://github.com/vladfi1/slippi-ai)
  - [libmelee](https://github.com/altf4/libmelee)
  """

  @doc """
  Returns the ExPhil version.

  ## Examples

      iex> ExPhil.version()
      "0.1.0"

  """
  @spec version() :: String.t()
  def version, do: "0.1.0"

  @doc """
  Returns the default embedding configuration.

  This is a convenience function that delegates to `ExPhil.Embeddings.Game.default_config/0`.

  ## Examples

      iex> config = ExPhil.default_embed_config()
      iex> config.stage_mode
      :one_hot_compact

  """
  @spec default_embed_config() :: ExPhil.Embeddings.Game.config()
  def default_embed_config do
    ExPhil.Embeddings.Game.default_config()
  end

  @doc """
  Returns the embedding size for the given configuration.

  ## Examples

      iex> ExPhil.embedding_size()
      288

      iex> config = %{ExPhil.default_embed_config() | stage_mode: :one_hot_full}
      iex> ExPhil.embedding_size(config)
      344

  """
  @spec embedding_size(ExPhil.Embeddings.Game.config() | nil) :: non_neg_integer()
  def embedding_size(config \\ nil) do
    config = config || default_embed_config()
    ExPhil.Embeddings.Game.embedding_size(config)
  end
end
