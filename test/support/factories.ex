defmodule ExPhil.Test.Factories do
  @moduledoc """
  Test factories for generating test data.

  Usage:
    import ExPhil.Test.Factories

    # Generate a random batch for training
    batch = build_batch(batch_size: 8, embed_size: 64)

    # Generate a game state
    game_state = build_game_state()

    # Generate a player
    player = build_player(damage: 50.0, stocks: 3)
  """

  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  @default_embed_size 64
  @default_batch_size 4

  # ============================================================================
  # Tensor Factories
  # ============================================================================

  @doc """
  Build a random tensor with the given shape.
  """
  def random_tensor(shape, opts \\ []) do
    type = Keyword.get(opts, :type, :f32)
    key = Nx.Random.key(System.system_time())
    {tensor, _} = Nx.Random.uniform(key, shape: shape, type: type)
    tensor
  end

  @doc """
  Build a mock training batch with embedded states and targets.

  ## Options
    - `:batch_size` - Number of samples (default: 4)
    - `:embed_size` - Embedding dimension (default: 64)
    - `:seq_len` - Sequence length for temporal batches (default: nil = single frame)
  """
  def build_batch(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    embed_size = Keyword.get(opts, :embed_size, @default_embed_size)
    seq_len = Keyword.get(opts, :seq_len, nil)

    # Build state tensor
    state = if seq_len do
      random_tensor({batch_size, seq_len, embed_size})
    else
      random_tensor({batch_size, embed_size})
    end

    # Build targets
    targets = build_targets(batch_size)

    Map.put(targets, :state, state)
  end

  @doc """
  Build target tensors for controller outputs.
  """
  def build_targets(batch_size) do
    key = Nx.Random.key(System.system_time())

    # Buttons: 8 binary values
    {buttons, key} = Nx.Random.uniform(key, shape: {batch_size, 8})
    buttons = Nx.greater(buttons, 0.5) |> Nx.as_type(:f32)

    # Stick positions: categorical indices (0-16)
    {main_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size})
    {main_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size})
    {c_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size})
    {c_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size})

    # Shoulder: categorical index (0-4)
    {shoulder, _key} = Nx.Random.randint(key, 0, 5, shape: {batch_size})

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder
    }
  end

  @doc """
  Build a mock PPO rollout.

  ## Options
    - `:num_steps` - Number of timesteps (default: 16)
    - `:embed_size` - Embedding dimension (default: 64)
  """
  def build_rollout(opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, 16)
    embed_size = Keyword.get(opts, :embed_size, @default_embed_size)

    key = Nx.Random.key(System.system_time())

    {states, key} = Nx.Random.uniform(key, shape: {num_steps, embed_size})
    {actions, key} = Nx.Random.uniform(key, shape: {num_steps, 8})
    {log_probs, key} = Nx.Random.uniform(key, shape: {num_steps}, min: -5.0, max: 0.0)
    {values, key} = Nx.Random.uniform(key, shape: {num_steps}, min: -1.0, max: 1.0)
    {rewards, _key} = Nx.Random.uniform(key, shape: {num_steps}, min: -0.1, max: 0.1)

    %{
      states: states,
      actions: actions,
      log_probs: log_probs,
      values: values,
      rewards: rewards,
      dones: Nx.broadcast(0, {num_steps})
    }
  end

  # ============================================================================
  # Game State Factories
  # ============================================================================

  @doc """
  Build a game state with default or custom values.

  ## Options
    - `:frame` - Frame number (default: 0)
    - `:stage` - Stage ID (default: 2 = FoD)
    - `:players` - Map of player port => Player struct
  """
  def build_game_state(opts \\ []) do
    frame = Keyword.get(opts, :frame, 0)
    stage = Keyword.get(opts, :stage, 2)

    players = Keyword.get(opts, :players, %{
      1 => build_player(),
      2 => build_player(x: 20.0)
    })

    %GameState{
      frame: frame,
      stage: stage,
      players: players,
      projectiles: []
    }
  end

  @doc """
  Build a player with default or custom values.

  ## Options
    - `:character` - Character ID (default: 0 = Mario)
    - `:x`, `:y` - Position (default: 0.0, 0.0)
    - `:percent` - Damage percent (default: 0.0)
    - `:stock` - Stock count (default: 4)
    - `:facing` - Facing direction 1 or -1 (default: 1)
    - `:action` - Action state ID (default: 14 = Wait)
    - `:action_frame` - Frame within action (default: 0)
    - `:shield_strength` - Shield health (default: 60.0)
    - `:jumps_left` - Jumps remaining (default: 2)
    - `:on_ground` - Grounded state (default: true)
    - `:invulnerable` - Invulnerable state (default: false)
  """
  def build_player(opts \\ []) do
    %Player{
      character: Keyword.get(opts, :character, 0),
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      action: Keyword.get(opts, :action, 14),
      action_frame: Keyword.get(opts, :action_frame, 0),
      shield_strength: Keyword.get(opts, :shield_strength, 60.0),
      jumps_left: Keyword.get(opts, :jumps_left, 2),
      on_ground: Keyword.get(opts, :on_ground, true),
      invulnerable: Keyword.get(opts, :invulnerable, false),
      hitstun_frames_left: Keyword.get(opts, :hitstun_frames_left, 0),
      speed_air_x_self: Keyword.get(opts, :speed_air_x_self, 0.0),
      speed_y_self: Keyword.get(opts, :speed_y_self, 0.0),
      speed_x_attack: Keyword.get(opts, :speed_x_attack, 0.0),
      speed_y_attack: Keyword.get(opts, :speed_y_attack, 0.0),
      speed_ground_x_self: Keyword.get(opts, :speed_ground_x_self, 0.0),
      nana: nil,
      controller_state: nil
    }
  end

  @doc """
  Build a controller state with default or custom values.
  """
  def build_controller_state(opts \\ []) do
    %ControllerState{
      main_stick: Keyword.get(opts, :main_stick, %{x: 0.0, y: 0.0}),
      c_stick: Keyword.get(opts, :c_stick, %{x: 0.0, y: 0.0}),
      l_shoulder: Keyword.get(opts, :l_shoulder, 0.0),
      r_shoulder: Keyword.get(opts, :r_shoulder, 0.0),
      button_a: Keyword.get(opts, :button_a, false),
      button_b: Keyword.get(opts, :button_b, false),
      button_x: Keyword.get(opts, :button_x, false),
      button_y: Keyword.get(opts, :button_y, false),
      button_z: Keyword.get(opts, :button_z, false),
      button_l: Keyword.get(opts, :button_l, false),
      button_r: Keyword.get(opts, :button_r, false),
      button_d_up: Keyword.get(opts, :button_d_up, false)
    }
  end

  # ============================================================================
  # Training Data Factories
  # ============================================================================

  @doc """
  Build a training frame (game state + controller action pair).
  """
  def build_training_frame(opts \\ []) do
    %{
      game_state: Keyword.get_lazy(opts, :game_state, fn -> build_game_state() end),
      controller: Keyword.get_lazy(opts, :controller, fn -> build_controller_state() end),
      player_port: Keyword.get(opts, :player_port, 1)
    }
  end

  @doc """
  Build a list of training frames.
  """
  def build_training_frames(count, opts \\ []) do
    Enum.map(0..(count - 1), fn i ->
      frame_opts = Keyword.put(opts, :game_state,
        build_game_state(frame: i)
      )
      build_training_frame(frame_opts)
    end)
  end

  # ============================================================================
  # Sequence Helpers
  # ============================================================================

  @doc """
  Build a sequence of game states for temporal training.
  """
  def build_game_state_sequence(length, opts \\ []) do
    base_state = Keyword.get_lazy(opts, :base_state, fn -> build_game_state() end)

    Enum.map(0..(length - 1), fn i ->
      # Slightly modify position over time
      players = Map.new(base_state.players, fn {port, player} ->
        {port, %{player | x: player.x + i * 0.5, action_frame: player.action_frame + i}}
      end)

      %{base_state | frame: base_state.frame + i, players: players}
    end)
  end
end
