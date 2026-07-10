defmodule ExPhil.Agents.FoxRecoveryExpert do
  @moduledoc """
  Rules-only Fox recovery expert for DAgger-style relabeling.

  Born from E2's SD post-mortem (2026-07-10): across 10 deaths the policy
  mashed jump for 55-161 frames offstage and pressed B exactly zero times —
  it has no recovery skill because offstage frames are a sliver of the
  corpus. Unlike the multishine/fair experts there is NO fixture table:
  recovery is pure geometry, so every label comes from rules keyed on
  position, jumps, and action state. Onstage grounded play returns `:skip` —
  this expert only teaches the offstage game, which makes it a relabeler for
  ORDINARY replays (every live session with SDs is a rollout).

  Recovery ladder (checked in order):
  1. Ledge hang -> stick toward stage (regular getup).
  2. Airborne over the stage at a safe height -> drift toward center, no
     buttons (just land).
  3. Offstage/below with a double jump and low -> tap jump (X, against prev).
  4. Offstage/below, jumpless or deep -> tap B with the stick aimed at the
     ledge (Firefox; the stick also AIMS the charge, so the aim label holds
     through the charge frames).
  5. Mid-special (Firefox charge/flight, char-specific action range) ->
     keep steering at the ledge.
  6. Hitstun/tumble offstage -> hold inward (survival DI toward stage).

  Same labeling protocol as the other experts (landing convention, tap
  alternation against the previously-landed input) — composes with
  `Data.shift_actions/2` and the `:prev_controller` override.
  """

  alias ExPhil.Bridge.ControllerState

  defstruct table: %{}

  @type t :: %__MODULE__{}

  # FD geometry (drills run on FD)
  @edge_x 85.57
  @ledge_aim_y 5.0

  # Universal action IDs
  @first_actionable 14
  @cliff_catch 252
  @cliff_wait 253
  @tumble 38
  @damage_air 84..91
  # Character-specific specials live at 341+; airborne offstage for Fox that
  # means side-B/up-B startup, charge, or flight — all want ledge steering
  @char_specials 341..399

  @doc "Rules-only: no fixture required."
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @doc "Fixture-API compatibility for dagger_drill (frames are ignored)."
  @spec from_frames([map()], keyword()) :: t()
  def from_frames(_frames, _opts \\ []), do: new()

  @spec from_fixture(String.t() | nil, keyword()) :: t()
  def from_fixture(_path \\ nil, _opts \\ []), do: new()

  @doc """
  Label a player state (landing convention). `:skip` for anything that is
  not the offstage game — dead states, grounded onstage play, high onstage
  aerials. `prev` is the previously-landed input (tap alternation).
  """
  @spec label(t(), map(), ControllerState.t() | nil) :: {:ok, ControllerState.t()} | :skip
  def label(%__MODULE__{}, player, prev \\ nil) do
    action = trunc(player.action || 0)
    x = player.x || 0.0
    y = player.y || 0.0
    grounded = player.on_ground
    jumps = player.jumps_left || 0

    offstage = abs(x) > @edge_x
    below = y < -5.0

    cond do
      action < @first_actionable ->
        :skip

      action in [@cliff_catch, @cliff_wait] ->
        # On the ledge: regular getup toward the stage
        {:ok, stick_toward_center(x)}

      grounded ->
        # Onstage ground game is not recovery's business
        :skip

      not offstage and not below ->
        if y > 20.0 do
          # High over the stage: drift to center and land
          {:ok, stick_toward_center(x)}
        else
          :skip
        end

      # ---- offstage / below from here ----

      action in @char_specials ->
        # Mid-special (Firefox charge aims with the stick; flight too)
        {:ok, aim_at_ledge(x, y)}

      action in @damage_air or action == @tumble ->
        # Can't act (or barely): survival DI toward the stage
        {:ok, stick_toward_center(x)}

      jumps > 0 and y < 0.0 ->
        # Burn the double jump first, drifting inward
        {:ok, tap_jump(x, prev)}

      y < 15.0 ->
        # Jumpless (or still low after the jump): Firefox, aimed at the ledge
        {:ok, tap_upb(x, y, prev)}

      true ->
        # Offstage but high: drift inward, save resources
        {:ok, stick_toward_center(x)}
    end
  end

  # -- Inputs -----------------------------------------------------------------

  defp tap_jump(x, prev) do
    if held?(prev, :button_x) do
      stick_toward_center(x)
    else
      %{stick_toward_center(x) | button_x: true}
    end
  end

  defp tap_upb(x, y, prev) do
    aim = aim_at_ledge(x, y)

    if held?(prev, :button_b) do
      aim
    else
      %{aim | button_b: true}
    end
  end

  # Unit-circle stick aimed from (x, y) at the near ledge, slightly above it
  defp aim_at_ledge(x, y) do
    target_x = if x > 0, do: @edge_x, else: -@edge_x
    dx = target_x - x
    dy = @ledge_aim_y - y
    mag = max(:math.sqrt(dx * dx + dy * dy), 1.0e-6)

    %{neutral() | main_stick: %{x: 0.5 + 0.5 * dx / mag, y: 0.5 + 0.5 * dy / mag}}
  end

  defp stick_toward_center(x) do
    %{neutral() | main_stick: %{x: (if x > 0, do: 0.0, else: 1.0), y: 0.5}}
  end

  defp held?(nil, _button), do: false
  defp held?(prev, button), do: Map.get(prev, button, false)

  defp neutral, do: ControllerState.neutral()
end
