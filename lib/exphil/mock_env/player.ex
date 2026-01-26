defmodule ExPhil.MockEnv.Player do
  @moduledoc """
  Mock player with simplified Melee physics.

  Handles:
  - Gravity and terminal velocity
  - Ground/air movement with friction
  - Jumping (short hop, full hop, double jump)
  - Basic action states
  """

  # Physics constants (Melee approximations)
  @gravity 0.095
  @terminal_velocity -2.5
  @ground_friction 0.06
  @air_friction 0.02

  # Movement constants (Fox-like)
  @ground_speed 2.2
  @air_accel 0.06
  @jump_velocity 3.68
  @short_hop_velocity 2.1
  @double_jump_velocity 3.23
  @max_air_speed 2.2
  @fast_fall_velocity -3.4

  # Action states (simplified)
  # Standing
  @action_wait 14
  # Walking
  @action_walk 15
  # Dashing
  @action_dash 20
  # Jump startup (3 frames)
  @action_jump_squat 24
  # Rising
  @action_jump 25
  # Falling
  @action_fall 26
  # Landing lag (4 frames)
  @action_land 27
  # In hitstun
  @action_hitstun 75

  @jump_squat_frames 3
  @land_lag_frames 4

  defstruct [
    :port,
    :character,
    :x,
    :y,
    :vel_x,
    :vel_y,
    :percent,
    :stock,
    # 1 = right, -1 = left
    :facing,
    :on_ground,
    # Air jumps remaining
    :jumps_left,
    :action,
    :action_frame,
    :hitstun_left,
    :fastfalling
  ]

  @type t :: %__MODULE__{}

  @doc """
  Create a new player.
  """
  @spec new(1 | 2, atom(), keyword()) :: t()
  def new(port, character, opts \\ []) do
    %__MODULE__{
      port: port,
      character: character,
      x: Keyword.get(opts, :starting_x, 0.0),
      y: Keyword.get(opts, :starting_y, 0.0),
      vel_x: 0.0,
      vel_y: 0.0,
      percent: 0.0,
      stock: Keyword.get(opts, :stocks, 4),
      facing: if(port == 1, do: 1, else: -1),
      on_ground: true,
      # Double jump available
      jumps_left: 1,
      action: @action_wait,
      action_frame: 0,
      hitstun_left: 0,
      fastfalling: false
    }
  end

  @doc """
  Respawn player after stock loss.
  """
  @spec respawn(t(), float()) :: t()
  def respawn(player, spawn_x) do
    %{
      player
      | x: spawn_x,
        # Spawn above stage
        y: 50.0,
        vel_x: 0.0,
        vel_y: 0.0,
        percent: 0.0,
        facing: if(spawn_x < 0, do: 1, else: -1),
        on_ground: false,
        jumps_left: 1,
        action: @action_fall,
        action_frame: 0,
        hitstun_left: 0,
        fastfalling: false
    }
  end

  @doc """
  Lose a stock.
  """
  @spec lose_stock(t()) :: t()
  def lose_stock(player) do
    %{player | stock: player.stock - 1}
  end

  @doc """
  Process input and update action state.
  """
  @spec process_input(t(), map()) :: t()
  def process_input(player, action) do
    player
    |> handle_hitstun()
    |> handle_action_state(action)
  end

  defp handle_hitstun(%{hitstun_left: h} = player) when h > 0 do
    %{player | hitstun_left: h - 1}
  end

  defp handle_hitstun(player), do: player

  defp handle_action_state(%{hitstun_left: h} = player, _action) when h > 0 do
    # In hitstun, can't act
    player
  end

  defp handle_action_state(%{action: @action_jump_squat, action_frame: f} = player, action)
       when f >= @jump_squat_frames do
    # Jump squat complete, perform jump
    jump_vel = if held_long?(action), do: @jump_velocity, else: @short_hop_velocity

    %{
      player
      | vel_y: jump_vel,
        on_ground: false,
        action: @action_jump,
        action_frame: 0,
        fastfalling: false
    }
  end

  defp handle_action_state(%{action: @action_jump_squat} = player, _action) do
    # Still in jump squat
    %{player | action_frame: player.action_frame + 1}
  end

  defp handle_action_state(%{action: @action_land, action_frame: f} = player, _action)
       when f < @land_lag_frames do
    # Landing lag
    %{player | action_frame: f + 1}
  end

  defp handle_action_state(%{action: @action_land} = player, action) do
    # Landing lag complete
    handle_grounded(player, action)
  end

  defp handle_action_state(%{on_ground: true} = player, action) do
    handle_grounded(player, action)
  end

  defp handle_action_state(%{on_ground: false} = player, action) do
    handle_airborne(player, action)
  end

  defp handle_grounded(player, action) do
    jump_pressed = action.button_x or action.button_y

    cond do
      jump_pressed ->
        # Start jump squat
        %{
          player
          | action: @action_jump_squat,
            action_frame: 0,
            # Stop horizontal movement during jump squat
            vel_x: 0.0
        }

      abs(action.stick_x) > 0.3 ->
        # Walking/dashing
        %{
          player
          | vel_x: action.stick_x * @ground_speed,
            facing: if(action.stick_x > 0, do: 1, else: -1),
            action: @action_walk,
            action_frame: 0
        }

      true ->
        # Standing
        %{player | action: @action_wait, action_frame: 0}
    end
  end

  defp handle_airborne(player, action) do
    player = apply_air_drift(player, action.stick_x)

    cond do
      # Double jump
      (action.button_x or action.button_y) and player.jumps_left > 0 ->
        %{
          player
          | vel_y: @double_jump_velocity,
            jumps_left: player.jumps_left - 1,
            action: @action_jump,
            action_frame: 0,
            fastfalling: false
        }

      # Fast fall (tap down while falling)
      action.stick_y < -0.65 and player.vel_y < 0 and not player.fastfalling ->
        %{
          player
          | vel_y: @fast_fall_velocity,
            fastfalling: true,
            action: @action_fall,
            action_frame: player.action_frame + 1
        }

      # Normal airborne
      player.vel_y > 0 ->
        %{player | action: @action_jump, action_frame: player.action_frame + 1}

      true ->
        %{player | action: @action_fall, action_frame: player.action_frame + 1}
    end
  end

  defp apply_air_drift(player, stick_x) do
    # Air acceleration toward stick direction
    target_vel = stick_x * @max_air_speed
    current_vel = player.vel_x

    new_vel =
      cond do
        stick_x > 0.3 and current_vel < target_vel ->
          min(current_vel + @air_accel, target_vel)

        stick_x < -0.3 and current_vel > target_vel ->
          max(current_vel - @air_accel, target_vel)

        true ->
          # Apply air friction
          apply_friction(current_vel, @air_friction)
      end

    # Update facing based on drift
    facing =
      cond do
        stick_x > 0.3 -> 1
        stick_x < -0.3 -> -1
        true -> player.facing
      end

    %{player | vel_x: new_vel, facing: facing}
  end

  # Helper to check if button held long enough for full hop
  # Simplified: always full hop
  defp held_long?(_action), do: true

  @doc """
  Apply physics (gravity, friction, velocity).
  """
  @spec apply_physics(t()) :: t()
  def apply_physics(player) do
    player
    |> apply_gravity()
    |> apply_ground_friction()
    |> apply_velocity()
  end

  defp apply_gravity(%{on_ground: true} = player), do: player

  defp apply_gravity(%{fastfalling: true} = player) do
    # Fast falling: maintain fast fall velocity, don't apply more gravity
    %{player | vel_y: @fast_fall_velocity}
  end

  defp apply_gravity(player) do
    new_vel_y = max(player.vel_y - @gravity, @terminal_velocity)
    %{player | vel_y: new_vel_y}
  end

  defp apply_ground_friction(%{on_ground: false} = player), do: player

  defp apply_ground_friction(%{action: action} = player)
       when action in [@action_walk, @action_dash] do
    # No friction while actively moving
    player
  end

  defp apply_ground_friction(player) do
    %{player | vel_x: apply_friction(player.vel_x, @ground_friction)}
  end

  defp apply_friction(vel, friction) do
    cond do
      vel > friction -> vel - friction
      vel < -friction -> vel + friction
      true -> 0.0
    end
  end

  defp apply_velocity(player) do
    %{player | x: player.x + player.vel_x, y: player.y + player.vel_y}
  end

  @doc """
  Land on ground.
  """
  @spec land(t(), float()) :: t()
  def land(player, ground_y) do
    %{
      player
      | y: ground_y,
        vel_y: 0.0,
        on_ground: true,
        # Reset double jump
        jumps_left: 1,
        fastfalling: false,
        action: @action_land,
        action_frame: 0
    }
  end

  @doc """
  Fall off edge.
  """
  @spec fall_off(t()) :: t()
  def fall_off(player) do
    %{player | on_ground: false, action: @action_fall, action_frame: 0}
  end

  @doc """
  Apply knockback from hit.
  """
  @spec apply_knockback(t(), float(), float()) :: t()
  def apply_knockback(player, kb_x, kb_y) do
    hitstun = trunc(abs(kb_x) + abs(kb_y)) * 2

    %{
      player
      | vel_x: kb_x,
        vel_y: kb_y,
        on_ground: false,
        hitstun_left: hitstun,
        action: @action_hitstun,
        action_frame: 0,
        fastfalling: false
    }
  end

  @doc """
  Add damage percent.
  """
  @spec add_percent(t(), float()) :: t()
  def add_percent(player, damage) do
    %{player | percent: player.percent + damage}
  end

  @doc """
  Convert to Bridge.Player struct for embedding compatibility.
  """
  def to_bridge_player(%__MODULE__{} = p) do
    %ExPhil.Bridge.Player{
      x: p.x,
      y: p.y,
      percent: p.percent,
      stock: p.stock,
      facing: p.facing,
      action: p.action,
      action_frame: p.action_frame,
      shield_strength: 60.0,
      character: character_id(p.character),
      invulnerable: false,
      hitstun_frames_left: p.hitstun_left,
      jumps_left: p.jumps_left,
      on_ground: p.on_ground,
      speed_air_x_self: if(p.on_ground, do: 0.0, else: p.vel_x),
      speed_y_self: p.vel_y,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: if(p.on_ground, do: p.vel_x, else: 0.0),
      nana: nil,
      controller_state: nil
    }
  end

  defp character_id(:fox), do: 2
  defp character_id(:mewtwo), do: 16
  defp character_id(:ganondorf), do: 22
  defp character_id(:link), do: 6
  defp character_id(:game_and_watch), do: 24
  defp character_id(:zelda), do: 18
  # Default to Fox
  defp character_id(_), do: 2
end
