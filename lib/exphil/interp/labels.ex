defmodule ExPhil.Interp.Labels do
  @moduledoc """
  The ONE definition of "the input a player issued from frame i".

  INVARIANTS.md item 9 / GOTCHA #113: Slippi records each frame's
  controller on the frame whose post-update state it PRODUCED. So the
  input that ends state S at frame i is recorded on frame i+1. Two
  instruments (09-08) labelled with the same-frame controller and read
  "calibrated" through a 1000x miscalibration. Every probe, scan, and
  test that asks "what did the player do from this state" must go
  through `issued_input/2,3` — never `frames[i].controller` directly.

  Works on both frame shapes:

    * training frames (`Streaming.parse_chunk` / `Peppi.to_training_frames`):
      `%{game_state: ..., controller: %ControllerState{}}` — the subject is
      already port 1 after remap, so no port argument.
    * raw NIF frames (`Peppi.parse`): `%{players: %{port => %PlayerFrame{controller: ...}}}`
      — pass the subject port.

  Controller predicates accept both controller shapes (bridge
  `ControllerState` with `main_stick: %{x, y}`; Peppi `Controller` with
  `main_stick_x/main_stick_y`). Sticks are in [0, 1], center 0.5.
  """

  @type frames :: :array.array() | list()

  @doc """
  The controller the subject issued FROM frame `i` (recorded on frame
  i+1). Returns nil past the end of the replay. Training-frame shape.
  """
  @spec issued_input(frames(), non_neg_integer()) :: map() | nil
  def issued_input(frames, i) do
    case at(frames, i + 1) do
      %{controller: c} -> c
      _ -> nil
    end
  end

  @doc "Raw-frame shape: the subject's controller recorded on frame i+1."
  @spec issued_input(frames(), non_neg_integer(), pos_integer()) :: map() | nil
  def issued_input(frames, i, port) do
    case at(frames, i + 1) do
      %{players: players} when is_map(players) ->
        case Map.get(players, port) do
          %{controller: c} -> c
          _ -> nil
        end

      _ ->
        nil
    end
  end

  @doc """
  The controller recorded ON frame i — the input that produced frame i's
  state. Legitimate as a prev-action INPUT (known at decision time);
  NEVER a target for state i (that is the leak).
  """
  @spec producing_input(frames(), non_neg_integer()) :: map() | nil
  def producing_input(frames, i) do
    case at(frames, i) do
      %{controller: c} -> c
      _ -> nil
    end
  end

  # ---- controller predicates (shape-agnostic) ----------------------------

  @doc "Main-stick deflection magnitude in [0, 1] (0 = centered)."
  @spec stick_magnitude(map() | nil) :: float()
  def stick_magnitude(nil), do: 0.0

  def stick_magnitude(c) do
    {x, y} = stick_xy(c)
    max(abs(x - 0.5), abs(y - 0.5)) * 2.0
  end

  @doc "Main-stick X deflection magnitude in [0, 1]."
  @spec stick_x_magnitude(map() | nil) :: float()
  def stick_x_magnitude(nil), do: 0.0

  def stick_x_magnitude(c) do
    {x, _} = stick_xy(c)
    abs(x - 0.5) * 2.0
  end

  @doc "Full horizontal deflection (>= 0.75 of range) — a dash/run input."
  @spec full_x?(map() | nil) :: boolean()
  def full_x?(c), do: stick_x_magnitude(c) >= 0.75

  @doc "Any of A/B/X/Y/Z/L/R (d-up excluded by default; pass `d_up: true`)."
  @spec any_button?(map() | nil, keyword()) :: boolean()
  def any_button?(nil, _opts), do: false

  def any_button?(c, opts \\ []) do
    keys = [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]
    keys = if Keyword.get(opts, :d_up, false), do: [:button_d_up | keys], else: keys
    Enum.any?(keys, fn k -> Map.get(c, k) == true end)
  end

  @doc "Non-neutral: any button, or stick beyond `threshold` (default 0.15)."
  @spec non_neutral?(map() | nil, keyword()) :: boolean()
  def non_neutral?(c, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.15)
    any_button?(c, opts) or stick_magnitude(c) > threshold
  end

  @doc """
  Input kind for exit/attempt scans: `:none | :stick_dead | :stick_mid |
  :stick_full | :button | :both`. `:stick_dead` = 0.15..0.30 (inside
  Melee's ~0.29 analog deadzone, a no-op).
  """
  @spec kind(map() | nil) :: atom()
  def kind(nil), do: :none

  def kind(c) do
    mag = stick_magnitude(c)
    button = any_button?(c, d_up: true)

    stick =
      cond do
        mag < 0.15 -> :none
        mag < 0.30 -> :stick_dead
        mag < 0.70 -> :stick_mid
        true -> :stick_full
      end

    cond do
      button and stick != :none -> :both
      button -> :button
      true -> stick
    end
  end

  # ---- internals ---------------------------------------------------------

  defp at(frames, i) when is_list(frames), do: Enum.at(frames, i)

  defp at(frames, i) do
    if i >= 0 and i < :array.size(frames), do: :array.get(i, frames), else: nil
  end

  defp stick_xy(%{main_stick: %{x: x, y: y}}) when is_number(x) and is_number(y), do: {x, y}
  defp stick_xy(%{main_stick_x: x, main_stick_y: y}) when is_number(x) and is_number(y), do: {x, y}
  defp stick_xy(_), do: {0.5, 0.5}
end
