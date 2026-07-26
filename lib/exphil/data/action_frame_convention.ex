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

  Measured by `ExPhil.Eval.StateStreamDiff` from recorded pairs. `delta` is
  `live_af - parsed_af` and is a constant per action — every action ever
  measured has exactly one delta, and it is always 0 or 1.

  Two independent sources agree exactly on the 6 actions both cover
  (24, 25, 29, 42, 323, 360), despite one being a Fox TAS multishine loop and
  the other a Mewtwo policy fighting a level-9 CPU. So the convention is a
  property of Peppi-vs-libmelee, not of a character or technique.

  There is no formula behind this — see `ExPhil.Eval.StateStreamDiff` for the
  two plausible rules that both fail.

  ## COVERAGE — good for Mewtwo/G&W play, still not total

  The table covers **75 of 399** action states. Measured share of frames it
  normalizes, before (9 actions, Fox-only) and after adding the Mewtwo game:

  | fixture | 9 actions | 75 actions |
  |---|---|---|
  | mewtwo_behind_response | 9.7% | 95.4% |
  | mewtwo_dtilt_uptilt_dense | 8.4% | 91.0% |
  | fox_multishine_closed | 77.3% | 85.9% |
  | gnw_neutral_dense | 16.5% | 82.6% |
  | gnw_movement_ledge | 25.9% | 74.4% |
  | mewtwo_approach_fair | 12.1% | 54.9% |

  Unmeasured actions still pass through UNCHANGED, deliberately: the deltas
  are not extrapolable (a majority are 1, but many — including the shine
  states 360/365 and most hitstun/tumble states — are 0), so guessing would
  corrupt states that currently agree.

  Widening further is cheap: record another pair (see
  `ExPhil.Eval.StateStreamTrace`) covering whatever the target character
  actually does, and merge. `unknown_actions/1` sizes the remaining gap for a
  specific workload.
  """

  # Measured, not assumed.
  #
  # Provenance — two independent sources, zero conflicts:
  #   * 9 actions from the Fox multishine pairs in test/fixtures/statestream/
  #     (a frame-perfect TAS loop)
  #   * 72 from a Mewtwo-policy-vs-level-9-Fox-CPU game recorded 2026-07-26
  #     (EXPHIL_STATE_TRACE=1, 9189 frames, 100% action/on_ground/y agreement)
  #
  # The 6 actions both measured (24, 25, 29, 42, 323, 360) agree exactly,
  # across different characters, inputs and situations — so the convention is
  # a property of Peppi-vs-libmelee, not of a character or a technique.
  #
  # Pinned by test/exphil/data/action_frame_convention_test.exs against
  # test/fixtures/statestream/action_frame_map.json, and cross-checked against
  # a live re-derivation from the committed pairs.
  @deltas %{
    12 => 1,
    14 => 1,
    15 => 0,
    16 => 0,
    18 => 0,
    20 => 0,
    24 => 1,
    25 => 1,
    26 => 1,
    27 => 1,
    28 => 0,
    29 => 1,
    35 => 1,
    39 => 0,
    40 => 1,
    41 => 1,
    42 => 1,
    43 => 1,
    44 => 0,
    47 => 0,
    48 => 1,
    49 => 1,
    53 => 0,
    56 => 0,
    57 => 0,
    60 => 0,
    63 => 0,
    64 => 0,
    65 => 0,
    66 => 0,
    67 => 0,
    68 => 0,
    69 => 0,
    70 => 0,
    71 => 0,
    72 => 0,
    74 => 0,
    76 => 0,
    78 => 0,
    79 => 0,
    80 => 0,
    84 => 0,
    85 => 0,
    86 => 0,
    88 => 0,
    180 => 0,
    181 => 0,
    212 => 1,
    213 => 0,
    214 => 1,
    216 => 0,
    217 => 0,
    218 => 0,
    219 => 0,
    220 => 0,
    221 => 0,
    222 => 0,
    233 => 0,
    234 => 0,
    235 => 0,
    236 => 0,
    264 => 1,
    323 => 1,
    341 => 0,
    342 => 1,
    345 => 1,
    346 => 0,
    347 => 1,
    350 => 1,
    351 => 0,
    359 => 0,
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
