defmodule ExPhil.Constants do
  @moduledoc """
  Centralized game constants for Super Smash Bros. Melee.

  This module provides a single source of truth for magic numbers used
  throughout the codebase. All game-specific values should be defined here
  rather than scattered across multiple files.

  ## Usage

      alias ExPhil.Constants

      # Frame timing
      max_frame = Constants.max_game_frames()

      # Action states
      num_actions = Constants.num_actions()

  ## Categories

  - **Frame Timing**: FPS, game duration, timeouts
  - **Action States**: Melee action state counts
  - **Characters**: Character count and IDs
  - **Stages**: Stage IDs for competitive stages
  - **Combat**: Hitstun, shieldstun, ledge timers
  - **Items**: Projectile and item timers
  """

  # ===========================================================================
  # Frame Timing
  # ===========================================================================

  @doc """
  Frames per second in Melee.

  Melee runs at a fixed 60 FPS.
  """
  @spec fps() :: 60
  def fps, do: 60

  @doc """
  Maximum game frames before timeout (8 minutes).

  Standard tournament match time: 8 minutes × 60 seconds × 60 FPS = 28,800 frames.
  Used for frame count normalization and timeout detection.
  """
  @spec max_game_frames() :: 28_800
  def max_game_frames, do: 28_800

  @doc """
  Maximum game duration in seconds.
  """
  @spec max_game_seconds() :: 480
  def max_game_seconds, do: 480

  @doc """
  Typical Slippi online frame delay.

  Online play adds ~18 frames of input delay due to rollback netcode.
  Used for frame delay augmentation during training.
  """
  @spec online_frame_delay() :: 18
  def online_frame_delay, do: 18

  # ===========================================================================
  # Action States
  # ===========================================================================

  @doc """
  Total number of Melee action states.

  Melee has 399 distinct action states that a character can be in.
  This is used for action embedding dimensions and one-hot encoding.
  """
  @spec num_actions() :: 399
  def num_actions, do: 399

  @doc """
  Default action embedding dimension for learned embeddings.
  """
  @spec default_action_embed_dim() :: 64
  def default_action_embed_dim, do: 64

  # ===========================================================================
  # Characters
  # ===========================================================================

  @doc """
  Total number of playable characters.

  Melee has 26 unique characters, but some share internal IDs with variants
  (e.g., Sheik/Zelda), resulting in 33 distinct character IDs in the engine.
  """
  @spec num_characters() :: 33
  def num_characters, do: 33

  @doc """
  Default character embedding dimension for learned embeddings.
  """
  @spec default_char_embed_dim() :: 64
  def default_char_embed_dim, do: 64

  # ===========================================================================
  # Stages
  # ===========================================================================

  @doc """
  Number of stages in the game (for one-hot encoding).
  """
  @spec num_stages() :: 64
  def num_stages, do: 64

  @doc """
  Number of competitive stages (for compact encoding).

  The 6 tournament-legal stages: FoD, PS, YS, DL, BF, FD.
  Plus 1 for "other" = 7 total in compact mode.
  """
  @spec num_competitive_stages() :: 6
  def num_competitive_stages, do: 6

  @doc """
  Stage IDs for competitive (tournament-legal) stages.

  Returns a map of stage atom to internal stage ID.
  """
  @spec competitive_stage_ids() :: %{atom() => non_neg_integer()}
  def competitive_stage_ids do
    %{
      fountain_of_dreams: 2,
      pokemon_stadium: 3,
      yoshis_story: 8,
      dream_land: 28,
      battlefield: 31,
      final_destination: 32
    }
  end

  # ===========================================================================
  # Combat Mechanics
  # ===========================================================================

  @doc """
  Maximum hitstun frames.

  The longest possible hitstun in Melee is approximately 120 frames,
  used for hitstun normalization in embeddings.
  """
  @spec max_hitstun_frames() :: 120
  def max_hitstun_frames, do: 120

  @doc """
  Maximum shieldstun frames.

  Maximum shieldstun is approximately 25 frames.
  """
  @spec max_shieldstun_frames() :: 25
  def max_shieldstun_frames, do: 25

  @doc """
  Standard action duration for normalization.

  Many actions complete within 60 frames; used for action progress normalization.
  """
  @spec standard_action_frames() :: 60
  def standard_action_frames, do: 60

  @doc """
  Ledge invincibility timer (frames).

  Characters get approximately 100 frames of ledge invincibility.
  """
  @spec ledge_invincibility_frames() :: 100
  def ledge_invincibility_frames, do: 100

  @doc """
  Maximum jumps remaining for most characters.

  Most characters have 2 jumps; Kirby/Jigglypuff have more (up to 6).
  """
  @spec max_jumps() :: 6
  def max_jumps, do: 6

  # ===========================================================================
  # Items & Projectiles
  # ===========================================================================

  @doc """
  Link bomb timer in frames.

  Link's bombs explode after approximately 180 frames (3 seconds).
  """
  @spec link_bomb_timer() :: 180
  def link_bomb_timer, do: 180

  @doc """
  Maximum projectile lifetime for normalization.

  Used for normalizing projectile frame counts.
  """
  @spec max_projectile_frames() :: 300
  def max_projectile_frames, do: 300

  # ===========================================================================
  # Controller Input
  # ===========================================================================

  @doc """
  Number of discrete stick positions per axis.

  Default: 17 buckets for main stick discretization.
  """
  @spec default_stick_buckets() :: 17
  def default_stick_buckets, do: 17

  @doc """
  Number of shoulder trigger positions.

  Shoulder buttons are discretized into 4 positions: off, light, medium, hard.
  """
  @spec shoulder_positions() :: 4
  def shoulder_positions, do: 4

  @doc """
  Number of physical buttons.

  A, B, X, Y, Z, L, R, Start = 8 buttons.
  """
  @spec num_buttons() :: 8
  def num_buttons, do: 8

  # ===========================================================================
  # Neural Network Defaults
  # ===========================================================================

  @doc """
  Default hidden layer sizes for MLP networks.

  Used in config.ex as the authoritative default for training CLI.
  """
  @spec default_hidden_sizes() :: [pos_integer()]
  def default_hidden_sizes, do: [512, 256]

  @doc """
  Tensor alignment for GPU tensor cores.

  Dimensions should be multiples of 8 for efficient GPU computation.
  """
  @spec tensor_alignment() :: 8
  def tensor_alignment, do: 8

  # ===========================================================================
  # Normalization Helpers
  # ===========================================================================

  @doc """
  Normalize a frame count to [0, 1] based on max game duration.

  ## Examples

      iex> ExPhil.Constants.normalize_frame(14400)
      0.5

      iex> ExPhil.Constants.normalize_frame(28800)
      1.0

      iex> ExPhil.Constants.normalize_frame(30000)
      1.0

  """
  @spec normalize_frame(number()) :: float()
  def normalize_frame(frame) when is_number(frame) do
    min(frame / max_game_frames(), 1.0)
  end

  @doc """
  Normalize hitstun frames to [0, 1].
  """
  @spec normalize_hitstun(number()) :: float()
  def normalize_hitstun(frames) when is_number(frames) do
    min(frames / max_hitstun_frames(), 1.0)
  end

  @doc """
  Normalize action progress to [0, 1] based on standard action duration.
  """
  @spec normalize_action_progress(number()) :: float()
  def normalize_action_progress(frames) when is_number(frames) do
    min(frames / standard_action_frames(), 1.0)
  end
end
