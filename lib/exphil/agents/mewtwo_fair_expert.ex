defmodule ExPhil.Agents.MewtwoFairExpert do
  @moduledoc """
  Scripted Mewtwo fair-chain expert for DAgger-style relabeling.

  Drill: short-hop fair -> L-cancel -> repeat, with full-hop fair and
  double-jump fair variants (fixture: test/fixtures/replays/mewtwo_fair_chains.slp,
  a human recording vs a stationary Fox on FD).

  Same labeling protocol as `ExPhil.Agents.MultishineExpert` (replay landing
  convention, composes with `Data.shift_actions/2`, recovery taps keyed on the
  previously-landed input), with one structural upgrade the multishine drill
  didn't need: a **physics-keyed table**. SH fair, FH fair, and DJ fair all
  occupy action 66 (ATTACK_AIR_F) with the same action_frame values — what
  distinguishes the correct input (most importantly the L-cancel frame) is
  remaining airtime. So the table is hierarchical:

  1. **Fine key** `{action, af, grounded, jumps_left, height_bucket}` —
     disambiguates the jump variants by double-jump availability and height
     above the stage (FD ground is y=0; the drill assumes FD).
  2. **Coarse key** `{action, af, grounded}` — fallback where the fine table
     is sparse.
  3. **Recovery rules** — grounded: tap jump (Y — the recorder's jump button)
     to restart the drill; airborne falling near the ground: tap L (L-cancel
     insurance); airborne rising without an attack out: c-stick fair toward
     facing; otherwise neutral.

  Death/respawn states return `:skip`.
  """

  alias ExPhil.Bridge.ControllerState

  defstruct [:fine, :coarse]

  @type t :: %__MODULE__{fine: map(), coarse: map()}

  @fixture_path "test/fixtures/replays/mewtwo_fair_chains.slp"

  # Universal Melee action-state IDs
  @jump_states 25..28
  @first_actionable 14

  # Height (y) below which a falling aerial should be L-cancelling: an input
  # landing within the next few frames will coincide with touchdown
  @lcancel_height 8.0

  @doc """
  Build the expert from a fair-chain recording (defaults to the canonical
  fixture). Options: `:player_port` (default 1), `:min_count` (default 3
  fine / 4 coarse — drops one-off noise cells).
  """
  @spec from_fixture(String.t(), keyword()) :: t()
  def from_fixture(path \\ @fixture_path, opts \\ []) do
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)

    replay
    |> ExPhil.Data.Peppi.to_training_frames(
      player_port: Keyword.get(opts, :player_port, 1),
      opponent_port: Keyword.get(opts, :opponent_port, 2)
    )
    |> from_frames(opts)
  end

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)
    min_count = Keyword.get(opts, :min_count, 3)

    usable =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.reject(fn f -> trunc(f.game_state.players[port].action) < @first_actionable end)

    %__MODULE__{
      fine: build_table(usable, port, &fine_key/1, min_count),
      coarse: build_table(usable, port, &coarse_key/1, max(min_count, 4))
    }
  end

  @doc """
  Label a player state with the expert's input (landing convention).

  `prev` is the input that actually landed on the previous frame — recovery
  taps alternate against it so press EDGES are learnable from the
  prev-action channel.
  """
  @spec label(t(), map(), ControllerState.t() | nil) :: {:ok, ControllerState.t()} | :skip
  def label(%__MODULE__{fine: fine, coarse: coarse}, player, prev \\ nil) do
    cond do
      trunc(player.action) < @first_actionable ->
        :skip

      controller = fine[fine_key(player)] ->
        {:ok, controller}

      controller = coarse[coarse_key(player)] ->
        {:ok, controller}

      true ->
        {:ok, recovery(player, prev)}
    end
  end

  @doc """
  Fraction of fixture frames where the expert's label matches the recorded
  buttons exactly (coverage diagnostic on expert-quality data).
  """
  @spec button_agreement(t(), [map()], keyword()) :: float()
  def button_agreement(%__MODULE__{} = expert, frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)

    scored =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.flat_map(fn f ->
        case label(expert, f.game_state.players[port]) do
          {:ok, c} -> [{c, f.controller}]
          :skip -> []
        end
      end)

    case scored do
      [] -> 0.0
      pairs -> Enum.count(pairs, fn {a, b} -> buttons(a) == buttons(b) end) / length(pairs)
    end
  end

  # -- Keys -------------------------------------------------------------------

  defp fine_key(player) do
    {trunc(player.action), trunc(player.action_frame), player.on_ground,
     player.jumps_left || 0, height_bucket(player.y)}
  end

  defp coarse_key(player) do
    {trunc(player.action), trunc(player.action_frame), player.on_ground}
  end

  # 6-unit buckets, capped: 0 = at/below stage level, 9 = 54+ units up
  defp height_bucket(y) do
    y
    |> max(0.0)
    |> then(&trunc(&1 / 6))
    |> min(9)
  end

  # -- Recovery rules -----------------------------------------------------------

  defp recovery(player, prev) do
    action = trunc(player.action)
    falling? = (player.speed_y_self || 0.0) < 0.0

    cond do
      # Grounded off-script: restart the drill with a jump (recorder uses Y).
      player.on_ground ->
        if held?(prev, :button_y), do: neutral(), else: tap(:button_y)

      # Falling close to the stage: L-cancel insurance — an L landing in the
      # next couple of frames covers the touchdown window.
      falling? and (player.y || 0.0) < @lcancel_height ->
        if held?(prev, :button_l), do: neutral(), else: tap(:button_l)

      # Rising in a jump without an attack out: c-stick fair toward facing
      # (c-stick avoids main-stick drift side effects).
      action in @jump_states ->
        cstick_fair(player, prev)

      # Anything else airborne (riding out the fair, tumble, specials): let
      # the table/state evolve; neutral is safe.
      true ->
        neutral()
    end
  end

  defp cstick_fair(player, prev) do
    if cstick_deflected?(prev) do
      neutral()
    else
      cx = if (player.facing || 1) > 0, do: 1.0, else: 0.0
      %{neutral() | c_stick: %{x: cx, y: 0.5}}
    end
  end

  defp cstick_deflected?(nil), do: false

  defp cstick_deflected?(prev) do
    case prev.c_stick do
      %{x: x} -> abs(x - 0.5) > 0.2
      _ -> false
    end
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp tap(button), do: neutral() |> Map.put(button, true)
  defp neutral, do: ControllerState.neutral()

  # -- Table construction -------------------------------------------------------

  defp build_table(frames, port, key_fn, min_count) do
    frames
    |> Enum.group_by(fn f -> key_fn.(f.game_state.players[port]) end)
    |> Enum.filter(fn {_key, group} -> length(group) >= min_count end)
    |> Map.new(fn {key, group} -> {key, modal_controller(group)} end)
  end

  defp modal_controller(group) do
    modal_sig =
      group
      |> Enum.frequencies_by(&signature(&1.controller))
      |> Enum.max_by(fn {_sig, count} -> count end)
      |> elem(0)

    Enum.find(group, &(signature(&1.controller) == modal_sig)).controller
  end

  defp signature(c) do
    {buttons(c), grid(c.main_stick.x), grid(c.main_stick.y), grid(c.c_stick.x), grid(c.c_stick.y)}
  end

  defp buttons(c) do
    {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r}
  end

  defp grid(v), do: round(v * 20)
end
