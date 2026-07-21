defmodule ExPhil.Agents.MewtwoComboExpert do
  @moduledoc """
  Composite expert: the combo ladder assembled.

  Routes each frame to the right specialist by the OPPONENT's state:
  - opponent in the knockdown lifecycle -> `MewtwoTechChaseExpert`
    (pursue, read the tech, punish the getup)
  - opponent in punishable lag (whiffed attack/grab, landing lag) ->
    `MewtwoPunishExpert` (go in, dtilt/dash-grab — the neutral-opening
    branch, added 2026-07-20 after the human demos showed the policy
    never creates openings)
  - otherwise -> `MewtwoFairExpert` (approach in range, SH fair, L-cancel)

  This exists because a policy trained on chase labels alone is a statue
  that never CREATES knockdowns (the recovery-farm lesson), while the fair
  policy converts only ~10% of the knockdowns it earns. Composed, the drill
  teaches the full cycle: neutral -> opening -> fair -> knockdown -> read
  -> punish.

  Fixture-backed on the fair side (all Mewtwo recordings), rules-only on
  the chase and punish sides. Same label/4 protocol as every expert.
  Cascade order matters: knockdowns outrank lag punishes (a knocked-down
  opponent is also briefly "in lag" — the chase read is the better label).
  """

  alias ExPhil.Agents.{MewtwoFairExpert, MewtwoPunishExpert, MewtwoTechChaseExpert}
  alias ExPhil.Bridge.ControllerState

  defstruct [:fair, :chase, :punish]

  @type t :: %__MODULE__{}

  @spec from_fixture(String.t(), keyword()) :: t()
  def from_fixture(path, opts \\ []) do
    %__MODULE__{
      fair: MewtwoFairExpert.from_fixture(path, opts),
      chase: MewtwoTechChaseExpert.new(),
      punish: MewtwoPunishExpert.new()
    }
  end

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    %__MODULE__{
      fair: MewtwoFairExpert.from_frames(frames, opts),
      chase: MewtwoTechChaseExpert.new(),
      punish: MewtwoPunishExpert.new()
    }
  end

  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(expert, player, prev \\ nil, opponent \\ nil)

  def label(%__MODULE__{fair: fair, chase: chase, punish: punish}, player, prev, opponent) do
    with :skip <- MewtwoTechChaseExpert.label(chase, player, prev, opponent),
         :skip <- MewtwoPunishExpert.label(punish, player, prev, opponent) do
      MewtwoFairExpert.label(fair, player, prev, opponent)
    end
  end
end
