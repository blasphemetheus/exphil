defmodule ExPhil.Stomped.Decoder do
  @moduledoc """
  Decodes the live `.slp` event stream into the game state a policy embeds.

  The contract is parity with training, not completeness: every frame must
  come out exactly as `ExPhil.Data.Peppi.to_training_frames/2` (with
  `remap_ports: true`) builds it from the stored replay of the same game.
  That means following the Peppi NIF's conventions byte for byte —

    * the subject is slot 1 and the opponent slot 2, with the real ports
      stamped in `real_ports`;
    * only each port's leader is read: Nana is `nil` and `projectiles` is
      `[]`, although the stream carries both, because training never saw them;
    * non-finite floats become `0.0`; `action_frame` and hitstun are rounded;
    * the character is the game's internal id (unknowns clamp to 32);
    * Stadium's transformation type is held between events, FoD platform
      heights and Whispy are reported only on the frame of their event.

  Unlike the NIF, a field the stream's version does not carry is not
  defaulted: `game_start/2` refuses streams older than #{inspect({3, 5, 0})},
  because a silently defaulted field is a quietly wrong bot.

  Byte offsets follow the Slippi replay spec, counted from the first payload
  byte (the spec counts the command byte, so these are one less).
  """

  import Bitwise

  alias ExPhil.Bridge.{ControllerState, GameState, Player}
  alias ExPhil.Stomped.Events

  # Post-frame velocities (3.5.0) are the newest fields the policy reads.
  @min_version {3, 5, 0}
  @unknown_character 32
  @menu_in_game 2

  defstruct port: nil,
            opponent_port: nil,
            version: nil,
            stage: nil,
            stadium_type: nil

  @type t :: %__MODULE__{}

  @typedoc """
  One decoded frame: the state as training built it, plus the controller the
  game actually polled for the subject this frame (a late pad shows up here
  as neutral).
  """
  @type decoded :: %{frame: integer(), game_state: GameState.t(), controller: ControllerState.t()}

  @doc """
  Start a decoder for the subject at `port`. Ports are the Slippi player
  indices (0-based), as the arena's `port` field gives them.
  """
  @spec new(0..3, 0..3 | nil) :: t()
  def new(port, opponent_port \\ nil) do
    %__MODULE__{port: port, opponent_port: opponent_port || 1 - port}
  end

  @doc "The minimum stream version the decoder accepts."
  def min_version, do: @min_version

  @doc """
  Read Game Start from the header's events.

  Raises when the stream is older than the decoder supports. Events without a
  Game Start leave the decoder unchanged (the header is one message, but a
  caller may pass any events outside a frame here).
  """
  @spec game_start(t(), [Events.event()]) :: t()
  def game_start(%__MODULE__{} = decoder, events) do
    game_start = Events.game_start()

    case List.keyfind(events, game_start, 0) do
      {^game_start,
       <<major, minor, patch, _build, _info::binary-size(14), stage::big-16, _::binary>>} ->
        version = {major, minor, patch}

        if version < @min_version do
          raise ArgumentError,
                "slp stream version #{inspect(version)} is older than #{inspect(@min_version)}: " <>
                  "the policy's inputs are not all present"
        end

        %{decoder | version: version, stage: stage}

      _ ->
        decoder
    end
  end

  @doc """
  Decode one frame's events.

  Returns `{:error, reason}` when the frame cannot yield a state: Game Start
  has not been read, or the subject has no Post-Frame Update.
  """
  @spec frame(t(), integer(), [Events.event()]) :: {:ok, decoded(), t()} | {:error, term()}
  def frame(%__MODULE__{stage: nil}, _frame, _events), do: {:error, :no_game_start}

  def frame(%__MODULE__{} = decoder, frame, events) do
    acc =
      Enum.reduce(events, %{players: %{}, controller: nil}, fn event, acc ->
        read(event, decoder, acc)
      end)

    stadium_type = Map.get(acc, :stadium_type) || decoder.stadium_type
    decoder = %{decoder | stadium_type: stadium_type}

    case {acc.players[decoder.port], acc.controller} do
      {%Player{} = subject, %ControllerState{} = controller} ->
        opponent = acc.players[decoder.opponent_port]

        players = if opponent, do: %{1 => subject, 2 => opponent}, else: %{1 => subject}

        state = %GameState{
          frame: frame,
          stage: decoder.stage,
          menu_state: @menu_in_game,
          players: players,
          own_port: 1,
          real_ports: {decoder.port + 1, decoder.opponent_port + 1},
          projectiles: [],
          distance: distance(subject, opponent),
          fod_platform_left: Map.get(acc, :fod_platform_left),
          fod_platform_right: Map.get(acc, :fod_platform_right),
          stadium_event: Map.get(acc, :stadium_event),
          stadium_type: stadium_type,
          whispy_direction: Map.get(acc, :whispy_direction)
        }

        {:ok, %{frame: frame, game_state: state, controller: controller}, decoder}

      _ ->
        {:error, :subject_missing}
    end
  end

  # --- events -------------------------------------------------------------

  # Post-Frame Update, leaders only (follower byte 0).
  defp read(
         {0x38,
          <<_frame::32, port, 0, character, action::big-16, x::32, y::32, direction::32,
            percent::32, shield::32, _last_attack, _combo, _last_hit_by, stocks, state_age::32,
            _flags::binary-size(5), misc_as::32, airborne, _last_ground::16, jumps, _l_cancel,
            hurtbox, self_x_air::32, self_y::32, knockback_x::32, knockback_y::32,
            self_x_ground::32, _::binary>>},
         _decoder,
         acc
       ) do
    player = %Player{
      character: if(character <= 0x20, do: character, else: @unknown_character),
      x: f32(x),
      y: f32(y),
      percent: f32(percent),
      stock: stocks,
      # Sign of the raw float, as the NIF reads it: NaN is not "< 0".
      facing: if(negative?(direction), do: -1, else: 1),
      action: action,
      action_frame: round(f32(state_age)),
      invulnerable: hurtbox != 0,
      jumps_left: jumps,
      on_ground: airborne == 0,
      shield_strength: f32(shield),
      hitstun_frames_left: round(f32(misc_as)),
      speed_air_x_self: f32(self_x_air),
      speed_ground_x_self: f32(self_x_ground),
      speed_y_self: f32(self_y),
      speed_x_attack: f32(knockback_x),
      speed_y_attack: f32(knockback_y),
      nana: nil,
      controller_state: nil
    }

    put_in(acc, [:players, port], player)
  end

  # Pre-Frame Update for the subject's leader: the controller the game read.
  defp read(
         {0x37,
          <<_frame::32, port, 0, _seed::32, _action::16, _x::32, _y::32, _direction::32,
            joy_x::32, joy_y::32, c_x::32, c_y::32, _trigger::32, _buttons::32, physical::big-16,
            physical_l::32, physical_r::32, _::binary>>},
         %{port: port},
         acc
       ) do
    controller = %ControllerState{
      main_stick: %{x: unit(joy_x), y: unit(joy_y)},
      c_stick: %{x: unit(c_x), y: unit(c_y)},
      l_shoulder: f32(physical_l),
      r_shoulder: f32(physical_r),
      button_a: pressed?(physical, 0x0100),
      button_b: pressed?(physical, 0x0200),
      button_x: pressed?(physical, 0x0400),
      button_y: pressed?(physical, 0x0800),
      button_z: pressed?(physical, 0x0010),
      button_l: pressed?(physical, 0x0040),
      button_r: pressed?(physical, 0x0020),
      button_d_up: pressed?(physical, 0x0008)
    }

    %{acc | controller: controller}
  end

  # Fountain of Dreams platform height. Raw platform 0 is the RIGHT one.
  defp read({0x3F, <<_frame::32, 0, height::32, _::binary>>}, _decoder, acc),
    do: Map.put(acc, :fod_platform_right, f32(height))

  defp read({0x3F, <<_frame::32, 1, height::32, _::binary>>}, _decoder, acc),
    do: Map.put(acc, :fod_platform_left, f32(height))

  # Dream Land Whispy blow direction; the last event of the frame wins.
  defp read({0x40, <<_frame::32, direction, _::binary>>}, _decoder, acc),
    do: Map.put(acc, :whispy_direction, direction)

  # Pokémon Stadium transformation; the last event of the frame wins.
  defp read({0x41, <<_frame::32, event::big-16, type::big-16, _::binary>>}, _decoder, acc),
    do: acc |> Map.put(:stadium_event, event) |> Map.put(:stadium_type, type)

  defp read(_event, _decoder, acc), do: acc

  # --- values -------------------------------------------------------------

  defp distance(%Player{} = a, %Player{} = b) do
    dx = a.x - b.x
    dy = a.y - b.y
    :math.sqrt(dx * dx + dy * dy)
  end

  defp distance(_subject, _opponent), do: 0.0

  # An f32 as the NIF's `fin/1` reads it: widened to f64, non-finite -> 0.0.
  # (A float pattern does not match NaN or infinity, so the bits come in as
  # an integer and are converted here.)
  defp f32(bits) do
    case <<bits::32>> do
      <<value::big-float-32>> -> value
      _ -> 0.0
    end
  end

  # Stick axis: the game's processed [-1, 1] to the trained [0, 1].
  defp unit(bits), do: (f32(bits) + 1.0) / 2.0

  # Sign bit set, and not NaN (exponent all ones with a non-zero mantissa).
  defp negative?(bits) do
    nan? = (bits &&& 0x7F800000) == 0x7F800000 and (bits &&& 0x007FFFFF) != 0
    not nan? and (bits &&& 0x80000000) != 0 and (bits &&& 0x7FFFFFFF) != 0
  end

  defp pressed?(buttons, bit), do: (buttons &&& bit) != 0
end
