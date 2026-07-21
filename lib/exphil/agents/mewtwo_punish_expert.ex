defmodule ExPhil.Agents.MewtwoPunishExpert do
  @moduledoc """
  Rules-only WHIFF-PUNISH expert — the neutral-opening branch (2026-07-20,
  from the second human demo: the policy shields and reacts OOS but never
  opens the opponent up; "attacking, winning neutral, not in its
  vocabulary").

  The diagnosis: every teacher so far demonstrates REACTIONS (fair
  mechanics, tech chase) and rollouts run vs a passive dummy, so "create
  an opening" never appears in the corpus. This expert supplies the
  opening: when the opponent has COMMITTED to something laggy and we are
  free, go in and take the punish. DAgger then relabels every
  "stood there while the opponent was punishable" rollout frame.

  Labels ONLY punish moments (opponent in recovery/endlag while we are
  grounded-free and within reach); everything else is `:skip` —
  composable in the combo cascade: chase (knockdown) > punish (lag) >
  fair (default spacing/approach).

  Punish policy (per frame, keyed on the OPPONENT):
  - Opponent in punishable lag (aerial landing lag 70-74, special-fall
    landing 43, whiffed grab 212/214, or a ground attack 44-64 past its
    dangerous early frames) and within reach:
    - out of range (|dx| > close) -> dash toward them (the "go in")
    - in range, we're standing/walking -> crouch + A tap = DTILT (the
      spacing punish Bradley recorded fixtures for)
    - in range, we're dashing/running -> Z tap = dash grab (dtilt is not
      available out of dash)
  - Beyond @max_reach or lag nearly over -> :skip (fair expert's
    spacing game resumes).

  Same protocol as every expert: label/4 landing convention, taps
  alternate against the previously-landed input.
  """

  alias ExPhil.Bridge.ControllerState

  defstruct []

  @type t :: %__MODULE__{}

  # Universal action IDs (see tech-chase expert for the state map)
  @first_actionable 14
  @our_free_states 14..42
  # Our movement states where dtilt is unavailable -> dash grab instead
  @our_dash_states [20, 21]

  # Opponent lag states
  # Aerial-attack landing lag (nair/fair/bair/uair/dair)
  @opp_aerial_landing 70..74
  # Special-fall landing (post-helpless: teleport/firefox landings etc.)
  @opp_special_landing [43]
  # Grab + dash grab (a whiffed catch holds its state through long endlag)
  @opp_grab [212, 214]
  # Ground attacks: jabs through smashes. Early frames are the dangerous
  # active window — only late frames (endlag) are a go signal.
  @opp_ground_attack 44..64
  @ground_attack_safe_frame 12
  # Grab whiff is committal almost immediately
  @grab_safe_frame 8

  # Reach model: Mewtwo dash ~1.5 units/f; typical punishable lag 20-40f;
  # dtilt/grab land inside ~18 units. Beyond max reach the lag will expire
  # before we arrive — that's the fair expert's spacing game, not ours.
  @close_range 18.0
  @max_reach 60.0

  @doc "Rules-only: no fixture required."
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(_frames, _opts \\ []), do: new()

  @spec from_fixture(String.t() | nil, keyword()) :: t()
  def from_fixture(_path \\ nil, _opts \\ []), do: new()

  @doc """
  Label our player's input given the opponent's lag state (landing
  convention). `:skip` unless the opponent is punishable and we are
  grounded-free within reach.
  """
  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(expert, player, prev \\ nil, opponent \\ nil)

  def label(%__MODULE__{}, _player, _prev, nil), do: :skip

  def label(%__MODULE__{}, player, prev, opponent) do
    our_action = trunc(player.action || 0)

    cond do
      our_action < @first_actionable ->
        :skip

      not punishable?(opponent) ->
        :skip

      not (player.on_ground and our_action in @our_free_states) ->
        :skip

      abs((opponent.x || 0.0) - (player.x || 0.0)) > @max_reach ->
        :skip

      true ->
        {:ok, punish(player, opponent, our_action, prev)}
    end
  end

  @doc "Is the opponent in committed, punishable lag right now?"
  @spec punishable?(map() | nil) :: boolean()
  def punishable?(nil), do: false

  def punishable?(opponent) do
    action = trunc(opponent.action || 0)
    frame = trunc(opponent.action_frame || 0)

    cond do
      action in @opp_aerial_landing -> true
      action in @opp_special_landing -> true
      action in @opp_grab -> frame >= @grab_safe_frame
      action in @opp_ground_attack -> frame >= @ground_attack_safe_frame
      true -> false
    end
  end

  # -- Punish rules ----------------------------------------------------------

  defp punish(player, opponent, our_action, prev) do
    dx = (opponent.x || 0.0) - (player.x || 0.0)

    cond do
      abs(dx) > @close_range ->
        # The "go in": dash at their position while the lag lasts
        run_toward(dx)

      our_action in @our_dash_states ->
        # Out of dash: dtilt is unavailable -> dash grab (tap Z)
        if held?(prev, :button_z) do
          run_toward(dx)
        else
          %{run_toward(dx) | button_z: true}
        end

      true ->
        # Standing/walking in range: crouch + A tap = dtilt. Holding the
        # stick low across frames keeps us crouched (tilt, not dsmash);
        # the A tap alternates against the previously-landed input.
        if held?(prev, :button_a) do
          crouch()
        else
          %{crouch() | button_a: true}
        end
    end
  end

  defp run_toward(dx) do
    %{neutral() | main_stick: %{x: (if dx > 0, do: 1.0, else: 0.0), y: 0.5}}
  end

  defp crouch do
    %{neutral() | main_stick: %{x: 0.5, y: 0.2}}
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp neutral, do: ControllerState.neutral()
end
