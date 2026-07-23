defmodule ExPhil.Eval.NeutralScan do
  @moduledoc """
  Neutral-opener taxonomy + diversity metrics (DATA_FLYWHEEL_DESIGN
  2026-07-23, stage C2).

  BC on mixed data *averages* multi-modal neutral into mush. Before training
  toward "does a couple different things in neutral" (the Fox goal), make it
  measurable: classify every neutral→engagement transition by opener
  category and score the distribution's diversity. `ExPhil.Interp.StyleCard`
  turns these into per-character gates (opener entropy, top-opener share).

  ## Opener event

  A run of >= `min_neutral` (default 60) consecutive frames where NEITHER
  player is in a non-neutral state (hitstun / knockdown lifecycle / shield —
  mirrors `ExPhil.Eval.FailureScan`), followed by P1 entering an opener
  category state. The category taxonomy uses Melee's universal action-state
  blocks:

  | category | action states |
  |----------|---------------|
  | `:jab` | 44–49 |
  | `:dash_attack` | 50 |
  | `:tilt` | 51–57 |
  | `:smash` | 58–64 |
  | `:aerial` | 65–69 |
  | `:grab` | 212, 214 |
  | `:special` | >= 341 (char-specific block) |

  Documented deviations from the design doc: `laser`/`projectile` fold into
  `:special` for v1 (char-specific action IDs — Fox blaster, G&W bacon —
  live in the >= 341 block and can't be told apart generically; char packs
  can refine later), and "aerial drift-in" is dropped (slim frames carry no
  velocity).

  Consumes plain per-port action lists (the `ReplayStats.load/1` shape) so
  StyleCard composes directly; a slim-frame wrapper serves scripts.
  """

  # Mirror ReplayStats / FailureScan.
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @shield_states MapSet.new([178, 179, 180])
  @non_neutral MapSet.union(@hitstun, MapSet.union(@lifecycle, @shield_states))

  @char_specific_base 341

  @default_min_neutral 60

  @categories [:jab, :dash_attack, :tilt, :smash, :aerial, :grab, :special]

  def categories, do: @categories

  @doc "Opener category for an action-state ID, or nil if not an opener state."
  def category(a) when a in 44..49, do: :jab
  def category(50), do: :dash_attack
  def category(a) when a in 51..57, do: :tilt
  def category(a) when a in 58..64, do: :smash
  def category(a) when a in 65..69, do: :aerial
  def category(a) when a in [212, 214], do: :grab
  def category(a) when is_integer(a) and a >= @char_specific_base, do: :special
  def category(_), do: nil

  @doc """
  Opener events from per-port action lists (P1 = subject). Returns
  `[%{index:, action:, category:}]` — index into the lists.

  ## Options
    - `:min_neutral` — consecutive both-neutral frames required before an
      entry counts as an opener (default #{@default_min_neutral})
  """
  def opener_events(p1_actions, p2_actions, opts \\ []) do
    min_neutral = Keyword.get(opts, :min_neutral, @default_min_neutral)

    Enum.zip(p1_actions, p2_actions)
    |> Enum.with_index()
    |> Enum.reduce({0, nil, []}, fn {{a1, a2}, i}, {streak, prev_a1, events} ->
      cat = category(a1)
      entered = cat != nil and category(prev_a1) != cat

      events =
        if entered and streak >= min_neutral do
          [%{index: i, action: a1, category: cat} | events]
        else
          events
        end

      both_neutral =
        not MapSet.member?(@non_neutral, a1) and not MapSet.member?(@non_neutral, a2)

      # An opener state itself ends the neutral run (the engagement started);
      # any non-neutral state also resets.
      streak = if both_neutral and cat == nil, do: streak + 1, else: 0

      {streak, a1, events}
    end)
    |> elem(2)
    |> Enum.reverse()
  end

  @doc "Opener events over ScenarioScan slim frames. Returns events with `:frame`."
  def opener_events_frames(frames, opts \\ []) do
    p1 = Enum.map(frames, & &1.p1.action)
    p2 = Enum.map(frames, & &1.p2.action)
    frame_nos = Enum.map(frames, & &1.frame)
    idx = :array.from_list(frame_nos)

    opener_events(p1, p2, opts)
    |> Enum.map(fn e -> Map.put(e, :frame, :array.get(e.index, idx)) end)
  end

  @doc "Category distribution `%{category => fraction}` over events ([] -> %{})."
  def distribution(events) do
    n = length(events)

    if n == 0 do
      %{}
    else
      events
      |> Enum.frequencies_by(& &1.category)
      |> Map.new(fn {cat, c} -> {cat, c / n} end)
    end
  end

  @doc "Shannon entropy of a distribution, in bits (%{} -> 0.0)."
  def entropy_bits(dist) do
    dist
    |> Map.values()
    |> Enum.filter(&(&1 > 0))
    |> Enum.reduce(0.0, fn p, acc -> acc - p * :math.log2(p) end)
  end

  @doc "Largest single-category share of a distribution (%{} -> nil)."
  def top_share(dist) when map_size(dist) == 0, do: nil
  def top_share(dist), do: dist |> Map.values() |> Enum.max()

  @doc """
  One-call summary for gating: `%{openers:, distribution:, entropy_bits:,
  top_share:}`. `entropy_bits`/`top_share` are nil below `min_events`
  (default 8) so gates can no-evidence-pass, mirroring
  `ReplayStats.percentile_or_nil/2`.
  """
  def summary(p1_actions, p2_actions, opts \\ []) do
    min_events = Keyword.get(opts, :min_events, 8)
    events = opener_events(p1_actions, p2_actions, opts)
    dist = distribution(events)

    if length(events) >= min_events do
      %{
        openers: length(events),
        distribution: dist,
        entropy_bits: Float.round(entropy_bits(dist), 3),
        top_share: Float.round(top_share(dist) * 1.0, 3)
      }
    else
      %{openers: length(events), distribution: dist, entropy_bits: nil, top_share: nil}
    end
  end
end
