defmodule ExPhil.Stomped.Pad do
  @moduledoc """
  The 12-byte controller pad a bot sends to the stomped.dev arena.

  Layout (little-endian): `i32 frame · u16 buttons · u8 stickX · u8 stickY ·
  u8 cstickX · u8 cstickY · u8 L · u8 R`. Buttons use the GameCube physical
  bit layout (the slp pre-frame `physicalButtons`). Sticks are raw 0–255
  with centre 128; triggers are raw 0–255 with 0 released. The `:l` / `:r`
  button bits are the digital click (a full press whatever the byte says);
  the trigger byte alone is an analog press (light shield).
  """

  import Bitwise

  @buttons %{
    dpad_left: 0x0001,
    dpad_right: 0x0002,
    dpad_down: 0x0004,
    dpad_up: 0x0008,
    z: 0x0010,
    r: 0x0020,
    l: 0x0040,
    a: 0x0100,
    b: 0x0200,
    x: 0x0400,
    y: 0x0800,
    start: 0x1000
  }

  @centre 128

  @type button ::
          :dpad_left
          | :dpad_right
          | :dpad_down
          | :dpad_up
          | :z
          | :r
          | :l
          | :a
          | :b
          | :x
          | :y
          | :start

  @typedoc """
  Pad inputs. Every key is optional; a missing key is neutral.
  `:buttons` is a raw bit mask or a list of button names.
  """
  @type inputs :: %{
          optional(:buttons) => non_neg_integer() | [button()],
          optional(:stick_x) => 0..255,
          optional(:stick_y) => 0..255,
          optional(:c_x) => 0..255,
          optional(:c_y) => 0..255,
          optional(:l) => 0..255,
          optional(:r) => 0..255
        }

  @doc "Button name → bit, for every button the arena reads."
  @spec buttons() :: %{button() => pos_integer()}
  def buttons, do: @buttons

  @doc "Bit mask for a list of button names."
  @spec mask([button()]) :: non_neg_integer()
  def mask(names) when is_list(names) do
    Enum.reduce(names, 0, fn name, acc -> acc ||| Map.fetch!(@buttons, name) end)
  end

  @doc "The neutral pad for `frame`: no buttons, sticks centred, triggers released."
  @spec neutral(integer()) :: <<_::96>>
  def neutral(frame), do: encode(frame, %{})

  @doc "Encode the pad the game will poll at `frame`."
  @spec encode(integer(), inputs() | keyword()) :: <<_::96>>
  def encode(frame, inputs) when is_integer(frame) do
    inputs = Map.new(inputs)

    buttons =
      case Map.get(inputs, :buttons, 0) do
        names when is_list(names) -> mask(names)
        bits when is_integer(bits) -> bits
      end

    <<frame::little-signed-32, buttons::little-16, byte(inputs, :stick_x, @centre),
      byte(inputs, :stick_y, @centre), byte(inputs, :c_x, @centre), byte(inputs, :c_y, @centre),
      byte(inputs, :l, 0), byte(inputs, :r, 0)>>
  end

  # The game reads a stick as raw offset / 80 (clamped to the unit circle) and
  # a trigger as raw / 140 (clamped to 1).
  @stick_scale 80
  @trigger_scale 140

  @doc """
  Pad inputs for a policy's controller output.

  A `ControllerState` is in training units: stick axes in `[0, 1]` with 0.5
  neutral, which is the game's *processed* stick value `[-1, 1]` rescaled,
  and triggers as the processed `[0, 1]` press. This is the inverse of the
  game's own processing, so the value the game reads back is the value the
  policy meant:

      byte = 128 + round((axis * 2 - 1) * 80)      # sticks
      byte = round(trigger * 140)                  # triggers

  Scaling to the full 0–255 byte range instead would turn a half tilt
  (0.75) into a full press: the game saturates at an offset of 80, not 127.
  """
  @spec from_controller(ExPhil.Bridge.ControllerState.t()) :: inputs()
  def from_controller(%ExPhil.Bridge.ControllerState{} = c) do
    names =
      for {field, name} <- [
            button_a: :a,
            button_b: :b,
            button_x: :x,
            button_y: :y,
            button_z: :z,
            button_l: :l,
            button_r: :r,
            button_d_up: :dpad_up
          ],
          Map.get(c, field) == true,
          do: name

    %{
      buttons: mask(names),
      stick_x: stick_byte(c.main_stick && c.main_stick.x),
      stick_y: stick_byte(c.main_stick && c.main_stick.y),
      c_x: stick_byte(c.c_stick && c.c_stick.x),
      c_y: stick_byte(c.c_stick && c.c_stick.y),
      l: trigger_byte(c.l_shoulder),
      r: trigger_byte(c.r_shoulder)
    }
  end

  @doc "The raw stick byte for a trained axis value in `[0, 1]` (0.5 neutral)."
  @spec stick_byte(number() | nil) :: 0..255
  def stick_byte(nil), do: @centre

  def stick_byte(axis) when is_number(axis) do
    offset = round((axis * 2 - 1) * @stick_scale)
    @centre + (offset |> max(-@stick_scale) |> min(@stick_scale))
  end

  @doc "The raw trigger byte for a trained press in `[0, 1]`."
  @spec trigger_byte(number() | nil) :: 0..255
  def trigger_byte(nil), do: 0

  def trigger_byte(press) when is_number(press) do
    round(press * @trigger_scale) |> max(0) |> min(@trigger_scale)
  end

  @doc "Decode a 12-byte pad (the arena's side of the wire; used by tests and tooling)."
  @spec decode(binary()) :: {:ok, map()} | :error
  def decode(<<frame::little-signed-32, buttons::little-16, stick_x, stick_y, c_x, c_y, l, r>>) do
    {:ok,
     %{
       frame: frame,
       buttons: buttons,
       stick_x: stick_x,
       stick_y: stick_y,
       c_x: c_x,
       c_y: c_y,
       l: l,
       r: r
     }}
  end

  def decode(_), do: :error

  # Clamp instead of raising: one out-of-range value must not cost a frame.
  defp byte(inputs, key, default) do
    inputs |> Map.get(key, default) |> round() |> max(0) |> min(255)
  end
end
