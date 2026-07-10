defmodule ExPhil.Agents.MultishineExpert do
  @moduledoc """
  Scripted multishine expert for DAgger-style relabeling.

  Labels any Fox state with the controller input a perfect multishiner would
  have recorded on that frame, under the replay convention (inputs recorded on
  the frame they LAND) so labels compose with `Data.shift_actions/2` exactly
  like human replay data.

  Two layers:

  1. **Table** — modal controller per `{action, action_frame, on_ground}` key,
     extracted from the canonical multishine fixture. Covers the happy path
     (the 9-frame loop: ground reflector → jump-cancel → jumpsquat → aerial
     shine → land) with the recorder's exact stick values.
  2. **Recovery rules** — hand-written fallbacks for states the fixture never
     visits (where a live policy drifts off-distribution): missed jump-cancel
     (grounded reflector af≥3 → press jump), empty hop (airborne non-reflector
     → aerial shine), grounded neutral (→ start shine), aerial reflector
     ride-down (→ neutral, land into the loop).

  Death/respawn states return `:skip` — they carry no learnable signal.

  ## Usage

      expert = MultishineExpert.from_fixture()
      case MultishineExpert.label(expert, player) do
        {:ok, controller} -> # DAgger target for this frame
        :skip -> # drop frame (dead/respawn)
      end
  """

  alias ExPhil.Bridge.ControllerState

  defstruct [:table]

  @type t :: %__MODULE__{table: %{optional({integer(), integer(), boolean()}) => ControllerState.t()}}

  @fixture_path "test/fixtures/replays/fox_multishine_closed.slp"

  # Fox action-state IDs (Melee internal)
  @jumpsquat 24
  @reflector_ground 360..363
  @reflector_air 365..368
  # Everything below Wait (14) is death/respawn/sleep — no learnable signal
  @first_actionable 14

  @doc """
  Build an expert from a multishine replay (defaults to the canonical fixture).

  Options:
    - `:player_port` - port of the multishining player (default 1)
    - `:min_count` - drop table keys seen fewer times (default 4; filters
      warm-up noise while keeping every state of the core loop)
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

  @doc """
  Build an expert from training frames (as produced by `Peppi.to_training_frames/2`).
  """
  @spec from_frames([map()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)
    min_count = Keyword.get(opts, :min_count, 4)

    table =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.reject(&sd_tail?(&1.controller))
      |> Enum.group_by(fn f ->
        p = f.game_state.players[port]
        {trunc(p.action), trunc(p.action_frame), p.on_ground}
      end)
      |> Enum.filter(fn {{action, _af, _g}, group} ->
        action >= @first_actionable and length(group) >= min_count
      end)
      |> Map.new(fn {key, group} -> {key, modal_controller(group)} end)

    %__MODULE__{table: table}
  end

  @doc """
  Label a player state with the expert's controller input (landing convention).

  `prev` is the input that actually landed on the previous frame (for DAgger
  rollouts: the policy's own press, the same value fed to the prev-action
  channel). Recovery taps key off it — press when the button was up, release
  when it was down — so the alternation is a function of inputs the model can
  observe. (Keying on action_frame parity failed: af freezes on repeated
  frames and its embedding saturates at af=120, making parity invisible —
  the labels looked like coin flips and training plateaued then diverged.)

  Returns `{:ok, %ControllerState{}}` or `:skip` for dead/respawn states.
  """
  @spec label(t(), map(), ControllerState.t() | nil) :: {:ok, ControllerState.t()} | :skip
  def label(%__MODULE__{table: table}, player, prev \\ nil, _opponent \\ nil) do
    action = trunc(player.action)
    af = trunc(player.action_frame)
    grounded = player.on_ground

    cond do
      action < @first_actionable ->
        :skip

      controller = table[{action, af, grounded}] ->
        {:ok, controller}

      true ->
        {:ok, recovery(action, grounded, prev)}
    end
  end

  @doc """
  Fraction of frames in a replay where the expert's label matches the recorded
  buttons exactly (diagnostic for table coverage on expert-quality data).
  """
  @spec button_agreement(t(), [map()], keyword()) :: float()
  def button_agreement(%__MODULE__{} = expert, frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)

    scored =
      frames
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.reject(&sd_tail?(&1.controller))
      |> Enum.flat_map(fn f ->
        case label(expert, f.game_state.players[port]) do
          {:ok, c} -> [{c, f.controller}]
          :skip -> []
        end
      end)

    case scored do
      [] ->
        0.0

      pairs ->
        matches = Enum.count(pairs, fn {a, b} -> buttons(a) == buttons(b) end)
        matches / length(pairs)
    end
  end

  # -- Recovery rules (states the fixture never visits) ----------------------

  # Melee registers a jump or special only on a press EDGE: a policy taught
  # "button on, every frame" just holds it (observed live: 582 consecutive X
  # frames in reflector, zero jumps). Taps are keyed on the previously-landed
  # input: press when the button was up, release when it was down — a
  # feedback alternation the model can reproduce from its prev-action channel.
  defp recovery(action, grounded, prev) do
    cond do
      # Missed the jump-cancel window: ground reflector is JC-able on any
      # frame from 4 on — tap jump to re-enter the loop
      grounded and action in @reflector_ground ->
        if held?(prev, :button_x), do: stick_down(), else: jump_cancel()

      # Any other grounded state (wait, walk, dash, landing): start a shine
      grounded ->
        if held?(prev, :button_b), do: stick_down(), else: shine()

      # Aerial reflector: ride it down; landing transfers to ground reflector,
      # which the rule above jump-cancels
      action in @reflector_air ->
        neutral()

      # Airborne otherwise (empty hop, falling): aerial shine — lands into
      # ground reflector and the loop resumes
      true ->
        if held?(prev, :button_b), do: stick_down(), else: shine()
    end
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  # Jump-cancel keeps the stick down so the next shine input is already held
  defp jump_cancel, do: %{neutral() | button_x: true, main_stick: %{x: 0.5, y: 0.0}}
  defp shine, do: %{neutral() | button_b: true, main_stick: %{x: 0.5, y: 0.0}}
  defp stick_down, do: %{neutral() | main_stick: %{x: 0.5, y: 0.0}}
  defp neutral, do: ControllerState.neutral()

  # -- Table construction -----------------------------------------------------

  # The fixture recorder ends the game by holding pure left with no buttons
  # (SD off the ledge) — same signature filter as train_multishine_policy.exs
  defp sd_tail?(c) do
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and
      not c.button_b and not c.button_x
  end

  # Most frequent controller signature in the group, returned as an actual
  # observed ControllerState (preserves the recorder's exact analog values)
  defp modal_controller(group) do
    modal_sig =
      group
      |> Enum.frequencies_by(&signature(&1.controller))
      |> Enum.max_by(fn {_sig, count} -> count end)
      |> elem(0)

    Enum.find(group, &(signature(&1.controller) == modal_sig)).controller
  end

  defp signature(c) do
    {buttons(c), grid(c.main_stick.x), grid(c.main_stick.y)}
  end

  defp buttons(c) do
    {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r}
  end

  defp grid(v), do: round(v * 20)
end
