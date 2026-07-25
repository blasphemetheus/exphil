defmodule ExPhil.Data.ActionFrameConvention do
  @moduledoc """
  Converts `action_frame` between the PARSED and LIVE conventions
  (task #8 phase 2 option 1, GOTCHAS #81).

  Peppi (reading a `.slp`) and libmelee (reading live memory through the
  bridge) do not report the same `action_frame` for the same game frame. Every
  policy in this repo is TRAINED on the parsed convention and RUN on the live
  one, and no training loss can reveal the mismatch because the loss is
  computed entirely in parsed space.

  **Parsed is canonical here.** Live values are converted *into* parsed space
  before embedding, not the other way round — every existing checkpoint
  already learned parsed-space features, so normalizing this direction fixes
  them in place instead of invalidating them.

  ## The table

  Measured by `ExPhil.Eval.StateStreamDiff` from the committed pairs in
  `test/fixtures/statestream/` (task #8 phase 1). `delta` is
  `live_af - parsed_af`, constant per action and identical across both runs:

      live af == parsed af      : 360, 365
      live af == parsed af + 1  : 24, 25, 29, 42, 323, 361, 366

  There is no formula behind this — see `ExPhil.Eval.StateStreamDiff` for the
  two plausible rules that both fail.

  ## COVERAGE LIMIT — read before trusting this

  The table covers **#{9} of #{399} action states**, all derived from two Fox
  multishine recordings. Unmeasured actions are passed through UNCHANGED,
  because the measured deltas are not extrapolable: they are mostly 1, but the
  two shine states (360, 365) are 0, and nothing observed so far predicts
  which an unmeasured action will be.

  So this normalization strictly improves the 9 known actions and changes
  nothing else. It is NOT a complete fix for the state-stream shift. Widening
  it requires more recorded `.slp` + trace pairs, which requires Dolphin.
  Use `known?/1` and `unknown_actions/1` to see what a given workload would
  actually have normalized.
  """

  # Measured, not assumed. Pinned against a fresh re-derivation from the
  # fixtures in test/exphil/data/action_frame_convention_test.exs, so this
  # table cannot silently drift away from the recordings it came from.
  @deltas %{
    24 => 1,
    25 => 1,
    29 => 1,
    42 => 1,
    323 => 1,
    360 => 0,
    361 => 1,
    365 => 0,
    366 => 1
  }

  @doc """
  The measured per-action deltas (`live_af - parsed_af`).
  """
  @spec deltas() :: %{integer() => integer()}
  def deltas, do: @deltas

  @doc """
  Whether the convention for this action has actually been measured.
  """
  @spec known?(integer() | nil) :: boolean()
  def known?(action) when is_integer(action), do: Map.has_key?(@deltas, action)
  def known?(_), do: false

  @doc """
  Number of action states covered by the table.
  """
  @spec coverage() :: non_neg_integer()
  def coverage, do: map_size(@deltas)

  @doc """
  Convert a live `action_frame` into the parsed convention.

  Unmeasured actions and non-integer input pass through unchanged. Negative
  values are sentinels ("no action frame") and are never adjusted.

  ## Examples

      iex> ExPhil.Data.ActionFrameConvention.live_to_parsed(24, 1)
      0

      iex> ExPhil.Data.ActionFrameConvention.live_to_parsed(365, 1)
      1

      iex> ExPhil.Data.ActionFrameConvention.live_to_parsed(999, 7)
      7

  """
  @spec live_to_parsed(integer() | nil, number() | nil) :: number() | nil
  def live_to_parsed(action, af), do: shift(action, af, -1)

  @doc """
  Convert a parsed `action_frame` into the live convention (inverse of
  `live_to_parsed/2`).

  ## Examples

      iex> ExPhil.Data.ActionFrameConvention.parsed_to_live(24, 0)
      1

  """
  @spec parsed_to_live(integer() | nil, number() | nil) :: number() | nil
  def parsed_to_live(action, af), do: shift(action, af, 1)

  defp shift(action, af, sign) when is_integer(action) and is_number(af) and af >= 0 do
    case @deltas[action] do
      nil -> af
      delta -> af + sign * delta
    end
  end

  defp shift(_action, af, _sign), do: af

  @doc """
  Action ids present in `actions` that the table does not cover.

  Use this to size the coverage gap for a real workload before assuming the
  normalization did anything useful.
  """
  @spec unknown_actions(Enumerable.t()) :: [integer()]
  def unknown_actions(actions) do
    actions
    |> Enum.reject(&known?/1)
    |> Enum.uniq()
    |> Enum.sort()
  end

  @doc """
  Normalize a player struct's `action_frame` from live into parsed space.

  Accepts anything with `:action` and `:action_frame` keys; returns it
  unchanged when either is missing.
  """
  @spec normalize_player(map() | nil) :: map() | nil
  def normalize_player(nil), do: nil

  def normalize_player(%{action: action, action_frame: af} = player)
      when not is_nil(action) and not is_nil(af) do
    %{player | action_frame: live_to_parsed(trunc(action), af)}
  end

  def normalize_player(player), do: player
end
