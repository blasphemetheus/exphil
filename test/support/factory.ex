defmodule ExPhil.Test.Factory do
  @moduledoc """
  Test factory for generating mock datasets and game states.

  These factories create minimal but realistic data structures for unit testing
  without needing real replay files.
  """

  alias ExPhil.Bridge.{GameState, Player, ControllerState}
  alias ExPhil.Training.Data
  alias ExPhil.Embeddings.Game, as: GameEmbed

  # ============================================================================
  # Dataset Factories
  # ============================================================================

  @doc """
  Create a mock dataset with sequences for temporal model testing.

  This creates a dataset where each "frame" is actually a sequence structure
  with a :sequence key containing window_size frames. This is the format
  expected by `Data.precompute_embeddings/2`.

  ## Options
    - `:num_sequences` - Number of sequences (default: 10)
    - `:seq_len` - Length of each sequence/window (default: 30)
    - `:embed_config` - Embedding config (default: GameEmbed.default_config())
  """
  def sequence_dataset(opts \\ []) do
    num_sequences = Keyword.get(opts, :num_sequences, 10)
    seq_len = Keyword.get(opts, :seq_len, 30)
    embed_config = Keyword.get(opts, :embed_config, GameEmbed.default_config())

    # Generate sequences - each "frame" in the dataset is a sequence structure
    sequences =
      for seq_idx <- 0..(num_sequences - 1) do
        # Build window_size frames for this sequence
        window_frames =
          for frame_offset <- 0..(seq_len - 1) do
            frame_num = seq_idx * seq_len + frame_offset
            build_frame(frame_num)
          end

        last_frame = List.last(window_frames)

        # This is the sequence structure expected by precompute_embeddings
        %{
          sequence: window_frames,
          game_state: last_frame.game_state,
          controller: last_frame.controller,
          action: get_action(last_frame)
        }
      end

    %Data{
      frames: sequences,
      metadata: %{
        source: "test_factory",
        num_sequences: num_sequences,
        seq_len: seq_len
      },
      embed_config: embed_config,
      size: num_sequences,
      embedded_sequences: nil,
      embedded_frames: nil,
      player_registry: nil
    }
  end

  @doc """
  Create a mock dataset with individual frames for MLP training.

  ## Options
    - `:num_frames` - Number of frames (default: 100)
    - `:embed_config` - Embedding config (default: GameEmbed.default_config())
  """
  def frame_dataset(opts \\ []) do
    num_frames = Keyword.get(opts, :num_frames, 100)
    embed_config = Keyword.get(opts, :embed_config, GameEmbed.default_config())

    frames = for i <- 0..(num_frames - 1), do: build_frame(i)

    %Data{
      frames: frames,
      metadata: %{source: "test_factory", num_frames: num_frames},
      embed_config: embed_config,
      size: num_frames,
      embedded_sequences: nil,
      embedded_frames: nil,
      player_registry: nil
    }
  end

  # ============================================================================
  # Game State Factories
  # ============================================================================

  @doc """
  Build a single frame with game state and controller.

  Note: We deliberately omit the `:action` key so that `Data.get_action/1`
  will derive it from the controller using `controller_to_action/1`.
  """
  def build_frame(frame_num \\ 0, opts \\ []) do
    frame = %{
      game_state: build_game_state(frame_num, opts),
      controller: build_controller(opts),
      player_tag: Keyword.get(opts, :player_tag),
      name_id: Keyword.get(opts, :name_id)
    }

    # Only add action if explicitly requested
    if Keyword.get(opts, :include_action, false) do
      Map.put(frame, :action, build_action(frame.controller))
    else
      frame
    end
  end

  @doc """
  Build an action map from a controller state.
  """
  def build_action(controller) do
    %{
      buttons: %{
        a: controller.button_a,
        b: controller.button_b,
        x: controller.button_x,
        y: controller.button_y,
        z: controller.button_z,
        l: controller.button_l,
        r: controller.button_r,
        d_up: controller.button_d_up
      },
      main_x: discretize_stick(controller.main_stick.x),
      main_y: discretize_stick(controller.main_stick.y),
      c_x: discretize_stick(controller.c_stick.x),
      c_y: discretize_stick(controller.c_stick.y),
      shoulder: discretize_shoulder(controller.l_shoulder)
    }
  end

  @doc """
  Build a mock game state.

  ## Options
    - `:stage` - Stage ID (default: 31 for Battlefield)
    - `:p1_character` - Player 1 character ID (default: 24 for Mewtwo)
    - `:p2_character` - Player 2 character ID (default: 2 for Fox)
  """
  def build_game_state(frame_num \\ 0, opts \\ []) do
    stage = Keyword.get(opts, :stage, 31)
    p1_char = Keyword.get(opts, :p1_character, 24)
    p2_char = Keyword.get(opts, :p2_character, 2)

    %GameState{
      frame: frame_num,
      stage: stage,
      menu_state: 2,
      players: %{
        1 => build_player(1, p1_char, frame_num),
        2 => build_player(2, p2_char, frame_num)
      },
      projectiles: [],
      items: [],
      distance: 50.0 + :rand.uniform() * 20.0
    }
  end

  @doc """
  Build a mock player state.
  """
  def build_player(port, character \\ 24, frame_num \\ 0) do
    # Vary positions slightly based on port and frame for realistic data
    x_offset = if port == 1, do: -30.0, else: 30.0
    x_noise = :rand.uniform() * 10.0 - 5.0
    y_noise = :rand.uniform() * 5.0

    %Player{
      character: character,
      x: x_offset + x_noise,
      y: y_noise,
      percent: :rand.uniform() * 100.0,
      stock: 4 - rem(div(frame_num, 1000), 4),
      facing: if(port == 1, do: 1, else: -1),
      action: Enum.random([14, 15, 20, 21, 44, 45, 46]),
      action_frame: :rand.uniform(30),
      invulnerable: false,
      jumps_left: :rand.uniform(2),
      on_ground: :rand.uniform() > 0.3,
      shield_strength: 60.0,
      hitstun_frames_left: 0,
      speed_air_x_self: :rand.uniform() * 2.0 - 1.0,
      speed_ground_x_self: :rand.uniform() * 2.0 - 1.0,
      speed_y_self: :rand.uniform() * 2.0 - 1.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  @doc """
  Build a mock controller state.
  """
  def build_controller(_opts \\ []) do
    # Mostly neutral with occasional button presses
    button_pressed = :rand.uniform() > 0.7

    %ControllerState{
      main_stick: %{x: 0.5 + :rand.uniform() * 0.4 - 0.2, y: 0.5 + :rand.uniform() * 0.4 - 0.2},
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: if(:rand.uniform() > 0.9, do: 1.0, else: 0.0),
      r_shoulder: 0.0,
      button_a: button_pressed and :rand.uniform() > 0.5,
      button_b: button_pressed and :rand.uniform() > 0.7,
      button_x: false,
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  # ============================================================================
  # Embedding Factories (for pre-embedded data)
  # ============================================================================

  @doc """
  Create mock embedded tensors for a sequence.

  Returns a tensor of shape {seq_len, embedding_size}.
  """
  def build_embedded_sequence(seq_len \\ 30, embedding_size \\ 287) do
    key = Nx.Random.key(System.system_time(:nanosecond))
    {tensor, _key} = Nx.Random.uniform(key, shape: {seq_len, embedding_size}, type: :f32)
    tensor
  end

  @doc """
  Create mock embedded tensor for a single frame.

  Returns a tensor of shape {embedding_size}.
  """
  def build_embedded_frame(embedding_size \\ 287) do
    key = Nx.Random.key(System.system_time(:nanosecond))
    {tensor, _key} = Nx.Random.uniform(key, shape: {embedding_size}, type: :f32)
    tensor
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Extract action from frame for sequence structure
  # Format must match what actions_to_tensors expects
  defp get_action(frame) do
    controller = frame.controller

    if controller do
      %{
        main_x: discretize_stick(controller.main_stick.x),
        main_y: discretize_stick(controller.main_stick.y),
        c_x: discretize_stick(controller.c_stick.x),
        c_y: discretize_stick(controller.c_stick.y),
        shoulder: discretize_shoulder(controller.l_shoulder),
        # buttons must be a map with individual keys, not a list
        buttons: %{
          a: controller.button_a,
          b: controller.button_b,
          x: controller.button_x,
          y: controller.button_y,
          z: controller.button_z,
          l: controller.button_l,
          r: controller.button_r,
          d_up: controller.button_d_up
        }
      }
    else
      nil
    end
  end

  defp discretize_stick(value), do: trunc(value * 16) |> min(15) |> max(0)
  defp discretize_shoulder(value), do: trunc(value * 4) |> min(3) |> max(0)
end
