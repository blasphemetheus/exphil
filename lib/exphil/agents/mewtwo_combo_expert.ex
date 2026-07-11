defmodule ExPhil.Agents.MewtwoComboExpert do
  @moduledoc """
  Composite expert: the combo ladder assembled.

  Routes each frame to the right specialist by the OPPONENT's state:
  - opponent in the knockdown lifecycle -> `MewtwoTechChaseExpert`
    (pursue, read the tech, punish the getup)
  - otherwise -> `MewtwoFairExpert` (approach in range, SH fair, L-cancel)

  This exists because a policy trained on chase labels alone is a statue
  that never CREATES knockdowns (the recovery-farm lesson), while the fair
  policy converts only ~10% of the knockdowns it earns. Composed, the drill
  teaches the full cycle: approach -> fair -> knockdown -> read -> punish.

  Fixture-backed on the fair side (all three Mewtwo recordings), rules-only
  on the chase side. Same label/4 protocol as every expert.
  """

  alias ExPhil.Agents.{MewtwoFairExpert, MewtwoTechChaseExpert}
  alias ExPhil.Bridge.ControllerState

  defstruct [:fair, :chase]

  @type t :: %__MODULE__{}

  @spec from_fixture(String.t(), keyword()) :: t()
  def from_fixture(path, opts \\ []) do
    %__MODULE__{
      fair: MewtwoFairExpert.from_fixture(path, opts),
      chase: MewtwoTechChaseExpert.new()
    }
  end

  @spec from_frames([map()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    %__MODULE__{
      fair: MewtwoFairExpert.from_frames(frames, opts),
      chase: MewtwoTechChaseExpert.new()
    }
  end

  @spec label(t(), map(), ControllerState.t() | nil, map() | nil) ::
          {:ok, ControllerState.t()} | :skip
  def label(expert, player, prev \\ nil, opponent \\ nil)

  def label(%__MODULE__{fair: fair, chase: chase}, player, prev, opponent) do
    case MewtwoTechChaseExpert.label(chase, player, prev, opponent) do
      {:ok, controller} -> {:ok, controller}
      :skip -> MewtwoFairExpert.label(fair, player, prev, opponent)
    end
  end
end
