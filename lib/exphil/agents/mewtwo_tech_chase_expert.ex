defmodule ExPhil.Agents.MewtwoTechChaseExpert do
  @moduledoc """
  Rules-only tech-chase expert — the drill system's first REACTION drill.

  Every prior drill taught rhythm (multishine, fair chains) or geometry
  (recovery). This one teaches reading the opponent: when they get knocked
  down, their choice — tech in place, tech roll, or missed tech — is visible
  in their action state for ~19 frames before they are actionable again.
  The chase is mechanical once you can read: move to where they will be,
  punish the getup. The tech_random dummy (`--dummy tech_random`) provides
  the randomized choices that make one canned answer unlearnable.

  Labels ONLY the chase moments (opponent in the knockdown lifecycle while
  we are grounded-actionable); everything else is `:skip` — composable with
  the fair-drill data via curriculum mixing (the fair knocks them down, the
  chase punishes the getup).

  Chase policy (per frame, keyed on the OPPONENT):
  - Opponent bouncing/lying/teching/rolling -> dash toward their CURRENT x.
    Rolls move them frame by frame, so per-frame pursuit converges on the
    roll destination without predicting it.
  - In punish range (|dx| < 15) with them still locked -> dash-dance in
    place (hold position, stick toward them) and GRAB (tap Z) as they
    become actionable (getup/roll endings, late action frames).

  Same protocol as the other experts: landing convention, tap alternation
  against the previously-landed input, label/4 with the opponent.
  """

  alias ExPhil.Bridge.ControllerState

  defstruct table: %{}

  @type t :: %__MODULE__{}

  # Universal action IDs
  @first_actionable 14
  # Our grounded-actionable states (wait/walk/turn/dash/run/jumpsquat/land)
  @our_free_states 14..42

  # Opponent knockdown lifecycle
  # Down-bounce (not actionable): face-up 183, face-down 191
  @opp_bounce [183, 191]
  # Lying, actionable-for-getup: 184 (DownWaitU), 192 (DownWaitD)
  @opp_wait [184, 192]
  # Getups: stand 186/194, attack 187/195, rolls 188/189/196/197
  @opp_getup 186..189
  @opp_getup_d 194..197
  # Techs: 199 in place, 200 roll toward facing, 201 roll away
  @opp_tech 199..201

  # Punish window: grab when inside this gap while they finish their option
  @punish_range 15.0

  @doc "Rules-only: no fixture required."
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(_frames, _opts \\ []), do: new()

  @spec from_fixture(String.t() | nil, keyword()) :: t()
  def from_fixture(_path \\ nil, _opts \\ []), do: new()

  @doc """
  Label our player's input given the opponent's knockdown state (landing
  convention). `:skip` unless (a) the opponent is in the knockdown
  lifecycle and (b) we are grounded and free to act.
  """
  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(expert, player, prev \\ nil, opponent \\ nil)

  def label(%__MODULE__{}, _player, _prev, nil), do: :skip

  def label(%__MODULE__{}, player, prev, opponent) do
    our_action = trunc(player.action || 0)
    opp_action = trunc(opponent.action || 0)

    chase_state? =
      opp_action in @opp_bounce or opp_action in @opp_wait or
        opp_action in @opp_getup or opp_action in @opp_getup_d or
        opp_action in @opp_tech

    cond do
      our_action < @first_actionable ->
        :skip

      not chase_state? ->
        :skip

      not (player.on_ground and our_action in @our_free_states) ->
        # Mid-air or mid-move: other subsystems' business
        :skip

      true ->
        {:ok, chase(player, opponent, opp_action, prev)}
    end
  end

  # -- Chase rules ---------------------------------------------------------------

  defp chase(player, opponent, opp_action, prev) do
    dx = (opponent.x || 0.0) - (player.x || 0.0)

    punishable? =
      opp_action in @opp_wait or opp_action in @opp_getup or
        opp_action in @opp_getup_d or
        (opp_action in @opp_tech and trunc(opponent.action_frame || 0) > 10)

    cond do
      abs(dx) > @punish_range ->
        # Pursue their current position (per-frame pursuit converges on
        # roll destinations without predicting them)
        run_toward(dx)

      punishable? ->
        # In range as their option resolves: grab (tap against prev)
        if held?(prev, :button_z) do
          face(dx)
        else
          %{face(dx) | button_z: true}
        end

      true ->
        # In range but they're still locked (bounce/early tech): hold
        # position facing them, ready to react
        face(dx)
    end
  end

  defp run_toward(dx) do
    %{neutral() | main_stick: %{x: (if dx > 0, do: 1.0, else: 0.0), y: 0.5}}
  end

  # Slight tilt toward them (keeps facing without full dash)
  defp face(dx) do
    %{neutral() | main_stick: %{x: (if dx > 0, do: 0.65, else: 0.35), y: 0.5}}
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp neutral, do: ControllerState.neutral()
end
