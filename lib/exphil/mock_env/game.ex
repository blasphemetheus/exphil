defmodule ExPhil.MockEnv.Game do
  @moduledoc """
  Mock Melee environment with simplified physics.

  Provides a fast simulation for testing self-play infrastructure
  without requiring Dolphin. Physics are approximate but capture
  key gameplay dynamics: gravity, jumping, movement, blast zones.

  ## Usage

      game = Game.new()
      game = Game.step(game, p1_action, p2_action)

      # Check game state
      game.p1.x, game.p1.y, game.p1.percent, game.p1.stock

  ## Physics Constants

  Values are approximations of Melee's physics:
  - Gravity: 0.095 units/frame²
  - Terminal velocity: -2.5 units/frame
  - Ground friction: 0.06
  - Air friction: 0.02
  """

  alias ExPhil.MockEnv.Player

  # Stage geometry (Final Destination approximation)
  @stage %{
    # Main platform
    ground_y: 0.0,
    left_edge: -85.0,
    right_edge: 85.0,
    # Blast zones
    blast_left: -224.0,
    blast_right: 224.0,
    blast_top: 200.0,
    blast_bottom: -140.0
  }

  defstruct [
    :p1,
    :p2,
    :frame,
    :done,
    :winner,
    :stage
  ]

  @type t :: %__MODULE__{
    p1: Player.t(),
    p2: Player.t(),
    frame: non_neg_integer(),
    done: boolean(),
    winner: nil | 1 | 2,
    stage: map()
  }

  @doc """
  Create a new game with both players at starting positions.
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    p1_char = Keyword.get(opts, :p1_character, :fox)
    p2_char = Keyword.get(opts, :p2_character, :fox)

    %__MODULE__{
      p1: Player.new(1, p1_char, starting_x: -40.0),
      p2: Player.new(2, p2_char, starting_x: 40.0),
      frame: 0,
      done: false,
      winner: nil,
      stage: @stage
    }
  end

  @doc """
  Step the game forward one frame.

  Actions are maps with:
  - `:stick_x` - Main stick X (-1.0 to 1.0)
  - `:stick_y` - Main stick Y (-1.0 to 1.0)
  - `:button_a` - Attack button
  - `:button_b` - Special button
  - `:button_x` or `:button_y` - Jump button
  - `:button_l` or `:button_r` - Shield button
  """
  @spec step(t(), map() | nil, map() | nil) :: t()
  def step(%__MODULE__{done: true} = game, _p1_action, _p2_action), do: game

  def step(%__MODULE__{} = game, p1_action, p2_action) do
    game
    |> process_inputs(p1_action, p2_action)
    |> apply_physics()
    |> check_stage_collision()
    |> check_blast_zones()
    |> advance_frame()
    |> check_game_end()
  end

  @doc """
  Reset game for new round (after stock loss).
  """
  @spec respawn(t(), 1 | 2) :: t()
  def respawn(game, player_num) do
    case player_num do
      1 -> %{game | p1: Player.respawn(game.p1, -40.0)}
      2 -> %{game | p2: Player.respawn(game.p2, 40.0)}
    end
  end

  # ===========================================================================
  # Input Processing
  # ===========================================================================

  defp process_inputs(game, p1_action, p2_action) do
    %{game |
      p1: Player.process_input(game.p1, p1_action || neutral_action()),
      p2: Player.process_input(game.p2, p2_action || neutral_action())
    }
  end

  defp neutral_action do
    %{stick_x: 0.0, stick_y: 0.0, button_a: false, button_b: false,
      button_x: false, button_y: false, button_l: false, button_r: false}
  end

  # ===========================================================================
  # Physics
  # ===========================================================================

  defp apply_physics(game) do
    %{game |
      p1: Player.apply_physics(game.p1),
      p2: Player.apply_physics(game.p2)
    }
  end

  # ===========================================================================
  # Stage Collision
  # ===========================================================================

  defp check_stage_collision(game) do
    %{game |
      p1: check_player_stage_collision(game.p1, game.stage),
      p2: check_player_stage_collision(game.p2, game.stage)
    }
  end

  defp check_player_stage_collision(player, stage) do
    player
    |> check_ground_collision(stage)
    |> check_edge_collision(stage)
  end

  defp check_ground_collision(player, stage) do
    cond do
      # Already on ground, nothing to do
      player.on_ground ->
        player

      # Falling through ground level (must be close to ground, not in blast zone)
      player.y <= stage.ground_y and player.y > stage.blast_bottom and
      player.vel_y <= 0 and
      player.x >= stage.left_edge and player.x <= stage.right_edge ->
        Player.land(player, stage.ground_y)

      true ->
        player
    end
  end

  defp check_edge_collision(%{on_ground: false} = player, _stage), do: player

  defp check_edge_collision(player, stage) do
    cond do
      # Walked off edge
      player.x < stage.left_edge or player.x > stage.right_edge ->
        Player.fall_off(player)

      true ->
        player
    end
  end

  # ===========================================================================
  # Blast Zones
  # ===========================================================================

  defp check_blast_zones(game) do
    p1_dead = in_blast_zone?(game.p1, game.stage)
    p2_dead = in_blast_zone?(game.p2, game.stage)

    game
    |> handle_blast_zone(:p1, p1_dead)
    |> handle_blast_zone(:p2, p2_dead)
  end

  defp in_blast_zone?(player, stage) do
    player.x < stage.blast_left or
    player.x > stage.blast_right or
    player.y > stage.blast_top or
    player.y < stage.blast_bottom
  end

  defp handle_blast_zone(game, _player_key, false), do: game

  defp handle_blast_zone(game, :p1, true) do
    new_p1 = Player.lose_stock(game.p1)
    game = %{game | p1: new_p1}

    if new_p1.stock > 0 do
      respawn(game, 1)
    else
      game
    end
  end

  defp handle_blast_zone(game, :p2, true) do
    new_p2 = Player.lose_stock(game.p2)
    game = %{game | p2: new_p2}

    if new_p2.stock > 0 do
      respawn(game, 2)
    else
      game
    end
  end

  # ===========================================================================
  # Frame Advance & Game End
  # ===========================================================================

  defp advance_frame(game) do
    %{game | frame: game.frame + 1}
  end

  defp check_game_end(game) do
    cond do
      game.p1.stock <= 0 ->
        %{game | done: true, winner: 2}

      game.p2.stock <= 0 ->
        %{game | done: true, winner: 1}

      # Timeout at 8 minutes (28800 frames)
      game.frame >= 28800 ->
        # Winner by percent (lower wins)
        winner = if game.p1.percent <= game.p2.percent, do: 1, else: 2
        %{game | done: true, winner: winner}

      true ->
        game
    end
  end

  # ===========================================================================
  # Conversion to GameState (for embedding compatibility)
  # ===========================================================================

  @doc """
  Convert mock game to GameState struct for embedding.
  """
  def to_game_state(%__MODULE__{} = game) do
    %ExPhil.Bridge.GameState{
      frame: game.frame,
      stage: 2,  # Final Destination
      menu_state: if(game.done, do: 0, else: 2),
      players: %{
        1 => Player.to_bridge_player(game.p1),
        2 => Player.to_bridge_player(game.p2)
      },
      projectiles: []
    }
  end

  # ===========================================================================
  # Helper Functions (for League compatibility)
  # ===========================================================================

  @doc """
  Get the current frame number.
  """
  @spec get_frame(t()) :: non_neg_integer()
  def get_frame(%__MODULE__{frame: frame}), do: frame

  @doc """
  Check if the game is over.
  """
  @spec is_over?(t()) :: boolean()
  def is_over?(%__MODULE__{done: done}), do: done

  @doc """
  Get the stock count for a player.
  """
  @spec get_stocks(t(), :p1 | :p2) :: non_neg_integer()
  def get_stocks(%__MODULE__{p1: p1}, :p1), do: p1.stock
  def get_stocks(%__MODULE__{p2: p2}, :p2), do: p2.stock

  @doc """
  Get the game state for embedding (alias for to_game_state).
  """
  @spec get_state(t()) :: ExPhil.Bridge.GameState.t()
  def get_state(%__MODULE__{} = game), do: to_game_state(game)
end
