defmodule ExPhil.Agents.EdgeTurnaroundExpert do
  @moduledoc """
  Rules-only pre-SD edge-correction expert for DAgger-style relabeling.

  Born from the generalist edge-SD loop (fox_il_v2 family, 2026-08-09):
  the IL policy dashdances near the edge, keeps dashing, and runs off.
  The `probe_edge_attribution` verdict was that position IS read at
  near-edge dash sites (|Δstick| 0.033 under ±25-unit x patches), so the
  SD loop is a COVERAGE gap — DAgger corrections are the right fix.

  Complement of `FoxRecoveryExpert` (offstage-only, skips grounded
  play): this expert labels ONLY grounded dash/run frames headed toward
  the near edge inside the danger margin, and its label is a hard stick
  reversal toward center (from DASHING that is a dash-back; from RUNNING
  it initiates run-brake + turn). Everything else is `:skip`.

  Jurisdiction is deliberately narrow because the edge miner
  (`scripts/edge_snippet_mine.exs`) cuts windows around ACTUAL dash-off
  deaths — inside those windows every toward-edge dash frame really was
  a mistake. Do not point this expert at whole ordinary replays without
  that windowing: near-edge toward-edge dashing is often legitimate
  (approach, dashdance spacing), and blanket relabeling would teach the
  policy to never contest the corner.

  Stage geometry is a struct parameter (`edge_x`, teeter x from
  `Melee.Stages.edge_ground_position/1`) so the miner can set it
  per-replay; the default is FD, same convention as FoxRecoveryExpert.

  Same labeling protocol as the other experts (landing convention; the
  label is stick-only, so no tap alternation against `prev` is needed).
  """

  alias ExPhil.Bridge.ControllerState

  defstruct edge_x: 85.5656967163, danger_margin: 20.0

  @type t :: %__MODULE__{edge_x: float(), danger_margin: float()}

  # libmelee Action enum: DASHING 0x14, RUNNING 0x15
  @dashing 0x14
  @running 0x15

  @doc "Rules-only: no fixture required. Options: :edge_x, :danger_margin."
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      edge_x: opts[:edge_x] || %__MODULE__{}.edge_x,
      danger_margin: opts[:danger_margin] || %__MODULE__{}.danger_margin
    }
  end

  @doc "Fixture-API compatibility for dagger_drill (frames are ignored)."
  @spec from_frames([map()], keyword()) :: t()
  def from_frames(_frames, opts \\ []), do: new(opts)

  @spec from_fixture(String.t() | nil, keyword()) :: t()
  def from_fixture(_path \\ nil, opts \\ []), do: new(opts)

  @doc """
  Label a player state (landing convention). `{:ok, controller}` only
  for grounded dash/run toward the near edge inside the danger margin;
  `:skip` for everything else.
  """
  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(%__MODULE__{edge_x: edge, danger_margin: margin}, player, _prev \\ nil, _opp \\ nil) do
    action = trunc(player.action || 0)
    x = player.x || 0.0

    if player.on_ground and action in [@dashing, @running] and
         toward_edge?(player, x) and abs(x) > edge - margin do
      {:ok, dash_back(x)}
    else
      :skip
    end
  end

  # Moving toward the near edge: ground speed sign matches the side we're
  # on. A dash-back's first frames still carry outward speed, which is
  # fine — holding inward remains the correct label until it decays.
  # Facing is the fallback when speed is unavailable/zero.
  defp toward_edge?(player, x) do
    case Map.get(player, :speed_ground_x_self) do
      s when is_number(s) and s != 0.0 -> s * x > 0.0
      _ ->
        case Map.get(player, :facing) do
          f when is_number(f) and f != 0 -> f * x > 0.0
          _ -> true
        end
    end
  end

  defp dash_back(x) do
    %{ControllerState.neutral() | main_stick: %{x: (if x > 0, do: 0.0, else: 1.0), y: 0.5}}
  end
end
