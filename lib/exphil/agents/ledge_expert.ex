defmodule ExPhil.Agents.LedgeExpert do
  @moduledoc """
  Human ledge knowledge as a reusable, strategy-parameterized expert.

  Observed live (2026-07-27, `ms_crouch_a`): with the crouch absorber covered,
  the next basin the closed loop finds is the LEDGE — aerial drift near the
  edge, ledge-grab, hang (docs/planning/EXPOSURE_BIAS.md item 8). No drill
  fixture visits ledge states, so every drill policy is undefined there. This
  module encodes the full option set once, so any bot's training (multishine
  today; Mewtwo/Ganon drills later) can synthesize or DAgger-label ledge
  escapes instead of re-deriving them.

  ## The option table (from CliffWait, action 253)

  | strategy     | input                    | animation (quick/slow)     |
  |--------------|--------------------------|----------------------------|
  | `:getup`     | stick toward stage       | CliffClimb 255 / 254       |
  | `:attack`    | tap A                    | CliffAttack 257 / 256      |
  | `:roll`      | tap L                    | CliffEscape 259 / 258      |
  | `:jump`      | tap X                    | CliffJump 262-263 / 260-261|
  | `:drop_jump` | down to release, then    | Fall -> JumpAerial -> land |
  |              | tap-jump + drift inward  |                            |
  | `:ledgedash` | NOT IMPLEMENTED (raises) | drop, jump, airdodge-in    |

  **The 100% rule:** at percent >= 100 the game substitutes the SLOW variant
  of getup/attack/roll/jump — same input, longer and more punishable
  animation. The expert's input does not change; `slow_getup?/1` exposes the
  branch so synthesis can build the right follow-up states and future
  opponent-aware strategies can prefer `:drop_jump` (percent-independent) at
  high percent.

  **Ledgedash** is deliberately a stub: it is a frame-tight
  drop -> double-jump -> airdodge-in whose timing window depends on character
  (jump startup, airdodge drift) and stage (ledge geometry). The strategy slot
  and this doc mark where those per-character/per-stage timing tables belong;
  `new(strategy: :ledgedash)` raises until they exist.

  ## States handled

  - CliffCatch (252): ~7 frames, no control -> neutral
  - CliffWait (253): the strategy's input, edge-alternated off `prev`
    (Melee registers presses on EDGES — see MultishineExpert)
  - Getup animations (254..263): no control -> neutral
  - `:drop_jump` only: airborne near the edge with jumps left -> tap-jump +
    drift inward (the post-release half of the maneuver)
  - Everything else: `:skip` — the ledge is this module's whole business;
    compose with a drill expert for the rest of the game.
  """

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Constants

  defstruct strategy: :getup

  @type strategy :: :getup | :attack | :roll | :jump | :drop_jump
  @type t :: %__MODULE__{strategy: strategy()}

  @cliff_catch Constants.cliff_catch()
  @cliff_wait Constants.cliff_wait()
  @ledge_getups Constants.ledge_getups()
  @edge_x Constants.fd_edge_x()

  @strategies [:getup, :attack, :roll, :jump, :drop_jump]

  @doc """
  Build a ledge expert. `:strategy` picks the escape (default `:getup`).

  `:ledgedash` raises: its per-character/per-stage timing tables are not
  implemented (see moduledoc).
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    case Keyword.get(opts, :strategy, :getup) do
      :ledgedash ->
        raise ArgumentError,
              "ledgedash needs per-character/per-stage timing tables (not implemented); " <>
                "see LedgeExpert moduledoc for where they belong"

      s when s in @strategies ->
        %__MODULE__{strategy: s}

      other ->
        raise ArgumentError, "unknown ledge strategy #{inspect(other)} (want #{inspect(@strategies)})"
    end
  end

  @doc """
  Label a ledge-family state (landing convention, same shape as the other
  experts). `prev` is the previously-landed input, used for tap alternation.
  `:skip` for anything that is not the ledge game.
  """
  @spec label(t(), map(), ControllerState.t() | nil) :: {:ok, ControllerState.t()} | :skip
  def label(%__MODULE__{strategy: strategy}, player, prev \\ nil, _opponent \\ nil) do
    action = trunc(player.action || 0)
    x = player.x || 0.0

    cond do
      action == @cliff_catch ->
        {:ok, neutral()}

      action == @cliff_wait ->
        {:ok, cliff_wait_input(strategy, x, prev)}

      action in @ledge_getups ->
        {:ok, neutral()}

      strategy == :drop_jump and airborne_near_edge?(player) ->
        {:ok, tap(:button_x, stick_toward_center(x), prev)}

      true ->
        :skip
    end
  end

  @doc """
  True when the game will pick the SLOW getup/attack/roll/jump variant
  (percent >= 100). The input is identical; the animation is longer.
  """
  @spec slow_getup?(map()) :: boolean()
  def slow_getup?(player), do: (player.percent || 0.0) >= 100.0

  # -- CliffWait inputs --------------------------------------------------------

  defp cliff_wait_input(:getup, x, _prev), do: stick_toward_center(x)
  defp cliff_wait_input(:attack, _x, prev), do: tap(:button_a, neutral(), prev)
  defp cliff_wait_input(:roll, _x, prev), do: tap(:button_l, neutral(), prev)
  defp cliff_wait_input(:jump, _x, prev), do: tap(:button_x, neutral(), prev)
  # Release the ledge: full down. The airborne clause handles the jump-in.
  defp cliff_wait_input(:drop_jump, _x, _prev), do: %{neutral() | main_stick: %{x: 0.5, y: 0.0}}

  # Off the ledge, still near it, resources available: burn the jump inward.
  defp airborne_near_edge?(player) do
    not player.on_ground and
      abs(player.x || 0.0) > @edge_x - 10.0 and
      (player.jumps_left || 0) > 0 and
      (player.y || 0.0) < 5.0
  end

  # -- Input helpers (FoxRecoveryExpert conventions) --------------------------

  defp tap(button, base, prev) do
    if held?(prev, button), do: base, else: Map.put(base, button, true)
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp stick_toward_center(x) do
    %{neutral() | main_stick: %{x: if(x > 0, do: 0.0, else: 1.0), y: 0.5}}
  end

  defp neutral, do: ControllerState.neutral()
end
