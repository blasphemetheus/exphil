defmodule ExPhil.Test.Generators do
  @moduledoc """
  StreamData generators for property-based testing.

  These generators produce random but valid test data for ExPhil types.
  Use with `ExUnitProperties` for property-based tests.

  ## Usage

      defmodule MyPropertyTest do
        use ExUnit.Case, async: true
        use ExUnitProperties
        import ExPhil.Test.Generators

        property "embedding produces valid tensor shape" do
          check all game_state <- game_state_gen() do
            embedded = Embeddings.Game.embed(game_state)
            assert is_struct(embedded, Nx.Tensor)
          end
        end
      end
  """

  use ExUnitProperties

  alias ExPhil.Bridge.{GameState, Player, ControllerState, Projectile, Item}

  # ============================================================================
  # Primitive Generators
  # ============================================================================

  @doc "Generate a valid player port (1-4)"
  def player_port_gen do
    StreamData.integer(1..4)
  end

  @doc "Generate a valid character ID (0-25)"
  def character_gen do
    StreamData.integer(0..25)
  end

  @doc "Generate a valid stage ID"
  def stage_gen do
    # Common stages: FD=32, BF=31, DL=28, YS=6, PS=3, FoD=2
    StreamData.member_of([2, 3, 6, 28, 31, 32])
  end

  @doc "Generate valid X position (-250 to 250)"
  def x_position_gen do
    StreamData.float(min: -250.0, max: 250.0)
  end

  @doc "Generate valid Y position (-150 to 200)"
  def y_position_gen do
    StreamData.float(min: -150.0, max: 200.0)
  end

  @doc "Generate valid damage percent (0 to 999)"
  def percent_gen do
    StreamData.float(min: 0.0, max: 999.0)
  end

  @doc "Generate stock count (0 to 4)"
  def stock_gen do
    StreamData.integer(0..4)
  end

  @doc "Generate facing direction (-1 or 1)"
  def facing_gen do
    StreamData.member_of([-1, 1])
  end

  @doc "Generate action state ID (0-400)"
  def action_gen do
    StreamData.integer(0..400)
  end

  @doc "Generate action frame (0.0 to 100.0)"
  def action_frame_gen do
    StreamData.float(min: 0.0, max: 100.0)
  end

  @doc "Generate shield strength (0.0 to 60.0)"
  def shield_gen do
    StreamData.float(min: 0.0, max: 60.0)
  end

  @doc "Generate jumps remaining (0 to 5)"
  def jumps_gen do
    StreamData.integer(0..5)
  end

  @doc "Generate stick position (0.0 to 1.0)"
  def stick_gen do
    StreamData.float(min: 0.0, max: 1.0)
  end

  @doc "Generate shoulder pressure (0.0 to 1.0)"
  def shoulder_gen do
    StreamData.float(min: 0.0, max: 1.0)
  end

  @doc "Generate frame number (0 to 10000)"
  def frame_gen do
    StreamData.integer(0..10_000)
  end

  # ============================================================================
  # Struct Generators
  # ============================================================================

  @doc "Generate a valid Player struct"
  def player_gen do
    gen all(
          character <- character_gen(),
          x <- x_position_gen(),
          y <- y_position_gen(),
          percent <- percent_gen(),
          stock <- stock_gen(),
          facing <- facing_gen(),
          action <- action_gen(),
          action_frame <- action_frame_gen(),
          invulnerable <- StreamData.boolean(),
          jumps_left <- jumps_gen(),
          on_ground <- StreamData.boolean(),
          shield_strength <- shield_gen(),
          hitstun <- StreamData.float(min: 0.0, max: 50.0),
          speed_air_x <- StreamData.float(min: -10.0, max: 10.0),
          speed_ground_x <- StreamData.float(min: -10.0, max: 10.0),
          speed_y <- StreamData.float(min: -10.0, max: 10.0),
          speed_x_attack <- StreamData.float(min: -10.0, max: 10.0),
          speed_y_attack <- StreamData.float(min: -10.0, max: 10.0)
        ) do
      %Player{
        character: character,
        x: x,
        y: y,
        percent: percent,
        stock: stock,
        facing: facing,
        action: action,
        action_frame: round(action_frame),
        invulnerable: invulnerable,
        jumps_left: jumps_left,
        on_ground: on_ground,
        shield_strength: shield_strength,
        hitstun_frames_left: round(hitstun),
        speed_air_x_self: speed_air_x,
        speed_ground_x_self: speed_ground_x,
        speed_y_self: speed_y,
        speed_x_attack: speed_x_attack,
        speed_y_attack: speed_y_attack,
        nana: nil,
        controller_state: nil
      }
    end
  end

  @doc "Generate a valid ControllerState struct"
  def controller_state_gen do
    gen all(
          main_x <- stick_gen(),
          main_y <- stick_gen(),
          c_x <- stick_gen(),
          c_y <- stick_gen(),
          l_shoulder <- shoulder_gen(),
          r_shoulder <- shoulder_gen(),
          button_a <- StreamData.boolean(),
          button_b <- StreamData.boolean(),
          button_x <- StreamData.boolean(),
          button_y <- StreamData.boolean(),
          button_z <- StreamData.boolean(),
          button_l <- StreamData.boolean(),
          button_r <- StreamData.boolean(),
          button_d_up <- StreamData.boolean()
        ) do
      %ControllerState{
        main_stick: %{x: main_x, y: main_y},
        c_stick: %{x: c_x, y: c_y},
        l_shoulder: l_shoulder,
        r_shoulder: r_shoulder,
        button_a: button_a,
        button_b: button_b,
        button_x: button_x,
        button_y: button_y,
        button_z: button_z,
        button_l: button_l,
        button_r: button_r,
        button_d_up: button_d_up
      }
    end
  end

  @doc "Generate a valid GameState struct"
  def game_state_gen do
    gen all(
          frame <- frame_gen(),
          stage <- stage_gen(),
          player1 <- player_gen(),
          player2 <- player_gen()
        ) do
      %GameState{
        frame: frame,
        stage: stage,
        menu_state: 2,  # IN_GAME
        players: %{1 => player1, 2 => player2},
        projectiles: [],
        items: [],
        distance: abs(player1.x - player2.x)
      }
    end
  end

  @doc "Generate a valid Projectile struct"
  def projectile_gen do
    gen all(
          owner <- player_port_gen(),
          x <- x_position_gen(),
          y <- y_position_gen(),
          type <- StreamData.integer(0..50),
          subtype <- StreamData.integer(0..10),
          speed_x <- StreamData.float(min: -20.0, max: 20.0),
          speed_y <- StreamData.float(min: -20.0, max: 20.0)
        ) do
      %Projectile{
        owner: owner,
        x: x,
        y: y,
        type: type,
        subtype: subtype,
        speed_x: speed_x,
        speed_y: speed_y
      }
    end
  end

  @doc "Generate a valid Item struct"
  def item_gen do
    gen all(
          x <- x_position_gen(),
          y <- y_position_gen(),
          type <- StreamData.integer(0..100),
          facing <- facing_gen(),
          spawn_id <- StreamData.integer(0..255),
          timer <- StreamData.integer(0..1000)
        ) do
      %Item{
        x: x,
        y: y,
        type: type,
        facing: facing,
        owner: nil,
        held_by: nil,
        spawn_id: spawn_id,
        timer: timer
      }
    end
  end

  # ============================================================================
  # Training Data Generators
  # ============================================================================

  @doc "Generate a training frame (game_state + controller pair)"
  def training_frame_gen do
    gen all(
          game_state <- game_state_gen(),
          controller <- controller_state_gen()
        ) do
      %{
        game_state: game_state,
        controller: controller,
        player_port: 1
      }
    end
  end

  @doc "Generate a list of training frames"
  def training_frames_gen(min_count \\ 1, max_count \\ 100) do
    StreamData.list_of(training_frame_gen(), min_length: min_count, max_length: max_count)
  end

  # ============================================================================
  # Tensor Generators
  # ============================================================================

  @doc "Generate a random tensor of given shape"
  def tensor_gen(shape, opts \\ []) do
    min_val = Keyword.get(opts, :min, -1.0)
    max_val = Keyword.get(opts, :max, 1.0)

    size = Tuple.product(shape)

    gen all values <- StreamData.list_of(
      StreamData.float(min: min_val, max: max_val),
      length: size
    ) do
      values
      |> Nx.tensor()
      |> Nx.reshape(shape)
    end
  end

  @doc "Generate a batch of state embeddings"
  def state_batch_gen(batch_size, embed_size) do
    tensor_gen({batch_size, embed_size})
  end
end
