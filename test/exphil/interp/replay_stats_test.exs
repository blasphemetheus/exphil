defmodule ExPhil.Interp.ReplayStatsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.ReplayStats

  @fixture "test/fixtures/replays/mewtwo_fair_chains.slp"

  # Action-state constants mirrored from the module under test. These pin
  # the gate semantics: if the sets in ReplayStats drift, expectations here
  # break loudly instead of silently re-deriving from the same constants.
  @shield_guard 178
  @shield_hold 179
  @shield_release 180
  @wait 14
  @dash 20

  # Knockdown lifecycle
  @down_bound_d 183
  @down_bound_u 191
  @tech_in_place 199
  @tech_roll_f 200
  @tech_roll_b 201
  @down_wait_d 184

  # Hitstun band edges
  @hitstun_lo 75
  @hitstun_hi 91
  @thrown_lo 223

  describe "run_lengths/2" do
    test "finds consecutive runs of states in the set" do
      actions = [@wait, @shield_hold, @shield_hold, @shield_hold, @wait, @shield_guard, @wait]
      assert ReplayStats.run_lengths(actions, [@shield_guard, @shield_hold, @shield_release]) == [3, 1]
    end

    test "counts a run that starts at index 0 and one that ends at the last frame" do
      actions = [@shield_hold, @shield_hold, @wait, @shield_hold]
      assert ReplayStats.run_lengths(actions, [@shield_hold]) == [2, 1]
    end

    test "transitions between different states inside the set stay one run" do
      # guard -> hold -> release is one continuous shield, not three runs
      actions = [@wait, @shield_guard, @shield_hold, @shield_release, @wait]
      assert ReplayStats.run_lengths(actions, [@shield_guard, @shield_hold, @shield_release]) == [3]
    end

    test "empty input and no-match input give no runs" do
      assert ReplayStats.run_lengths([], [@shield_hold]) == []
      assert ReplayStats.run_lengths([@wait, @dash], [@shield_hold]) == []
    end
  end

  describe "shield_stats/1" do
    test "computes fraction, run count, percentiles, and max" do
      # Two shield runs: 4 frames and 1 frame, in 10 frames total
      actions =
        [@wait, @shield_hold, @shield_hold, @shield_hold, @shield_hold, @wait, @wait, @shield_guard, @wait, @wait]

      stats = ReplayStats.shield_stats(actions)

      assert stats.shield_frac == 0.5
      assert stats.runs == 2
      # percentile impl: sorted runs [1, 4]; p50 -> index trunc(0.5*2)=1 -> 4
      assert stats.p50 == 4
      assert stats.p95 == 4
      assert stats.max == 4
      assert stats.breaks == 0
    end

    test "counts OBSERVED shield breaks via break-family successors" do
      # 215f hold ending in ShieldBreakFly (205) = one break; a second
      # shield ending in plain wait = not a break
      actions =
        List.duplicate(@shield_hold, 215) ++ [205, 206] ++
          List.duplicate(@wait, 5) ++ [@shield_hold, @shield_hold, @wait]

      stats = ReplayStats.shield_stats(actions)
      assert stats.breaks == 1
      assert stats.max == 215
    end

    test "dizzy successor (FuraFura 211) also counts as a break" do
      actions = [@wait] ++ List.duplicate(@shield_hold, 10) ++ [211, @wait]
      assert ReplayStats.count_shield_breaks(actions) == 1
    end

    test "no shielding at all" do
      stats = ReplayStats.shield_stats([@wait, @dash, @wait])
      assert stats == %{shield_frac: 0.0, runs: 0, p50: 0, p95: 0, max: 0, breaks: 0}
    end

    test "p95 picks the top run among many" do
      # 20 runs of length 1..20, each separated by a wait frame.
      # sorted lengths [1..20]; p95 -> index trunc(0.95*20)=19 -> 20
      actions =
        Enum.flat_map(1..20, fn len -> List.duplicate(@shield_hold, len) ++ [@wait] end)

      stats = ReplayStats.shield_stats(actions)
      assert stats.runs == 20
      assert stats.p95 == 20
      assert stats.p50 == 11
    end
  end

  describe "knockdown_episodes/1" do
    test "classifies entry states and computes tech rate" do
      # Four separate knockdown entries from neutral:
      # missed (183), tech in place (199), roll F (200), roll B (201)
      actions =
        [@wait, @down_bound_d, @down_wait_d] ++
          [@wait, @tech_in_place] ++
          [@dash, @tech_roll_f] ++
          [@wait, @tech_roll_b, @wait]

      kd = ReplayStats.knockdown_episodes(actions)

      assert kd.episodes == 4
      assert kd.by_class == %{0 => 1, 1 => 1, 2 => 1, 3 => 1}
      assert kd.tech_rate == 0.75
    end

    test "transitions inside the lifecycle do not start new episodes" do
      # 183 (down bound) -> 184 (down wait) -> 191 (bound up) never leaves
      # the lifecycle, so only the initial entry counts
      actions = [@wait, @down_bound_d, @down_wait_d, @down_bound_u, @down_wait_d, @wait]
      kd = ReplayStats.knockdown_episodes(actions)
      assert kd.episodes == 1
      assert kd.by_class == %{0 => 1}
      assert kd.tech_rate == 0.0
    end

    test "missed-tech-only game has tech_rate 0.0, all-tech game 1.0" do
      missed = [@wait, @down_bound_d, @wait, @down_bound_u, @wait]
      assert ReplayStats.knockdown_episodes(missed).tech_rate == 0.0

      teched = [@wait, @tech_in_place, @wait, @tech_roll_f, @wait]
      assert ReplayStats.knockdown_episodes(teched).tech_rate == 1.0
    end

    test "no knockdowns yields zero episodes and 0.0 rate (not division error)" do
      kd = ReplayStats.knockdown_episodes([@wait, @dash, @wait])
      assert kd == %{episodes: 0, by_class: %{}, tech_rate: 0.0}
    end
  end

  describe "hit_events/1" do
    test "counts transitions into hitstun, not frames in hitstun" do
      actions = [@wait, @hitstun_lo, @hitstun_lo + 1, @hitstun_hi, @wait, @thrown_lo, @thrown_lo, @wait]
      # One entry at index 1 (wait -> 75), staying in the band doesn't count;
      # one entry at index 5 (wait -> 223)
      assert ReplayStats.hit_events(actions) == 2
    end

    test "zero when never hit" do
      assert ReplayStats.hit_events([@wait, @dash, @shield_hold]) == 0
    end
  end

  describe "trigger_presses/2" do
    test "detects digital presses with onset and duration" do
      controllers =
        [
          %{button_l: false, button_r: false},
          %{button_l: true, button_r: false},
          %{button_l: true, button_r: false},
          %{button_l: false, button_r: false},
          %{button_l: false, button_r: true}
        ]

      assert ReplayStats.trigger_presses(controllers) == [{1, 2}, {4, 1}]
    end

    test "analog past threshold counts; below threshold does not" do
      controllers =
        [
          %{l_shoulder: 0.0},
          %{l_shoulder: 0.29},
          %{l_shoulder: 0.31},
          %{r_shoulder: 0.31},
          %{l_shoulder: 0.0}
        ]

      assert ReplayStats.trigger_presses(controllers) == [{2, 2}]
      # Raising the threshold suppresses the analog press entirely
      assert ReplayStats.trigger_presses(controllers, 0.5) == []
    end
  end

  describe "airborne_frac/1" do
    test "counts explicit on_ground: false only; nil players are not airborne" do
      players = [
        %{on_ground: true},
        %{on_ground: false},
        %{on_ground: false},
        nil
      ]

      assert ReplayStats.airborne_frac(players) == 0.5
    end
  end

  describe "action_histogram/2" do
    test "most common first, truncated to top n" do
      actions = [1, 2, 2, 3, 3, 3]
      assert ReplayStats.action_histogram(actions, 2) == [{3, 3}, {2, 2}]
    end
  end

  describe "approach_stats/2" do
    # Synthetic walk-in: P2 parked at x=0, P1 starts at -60 (dist 60 >=
    # start threshold), walks in 1 unit/frame until dist 15 (< engage 20),
    # then holds. Engagement onset lands mid-walk; P1 covered 100% of the
    # closure.
    defp walk_in(fair_at_engage?) do
      hold = 20
      walk = 45
      tail = 55
      n = hold + walk + tail

      p1x =
        List.duplicate(-60.0, hold) ++
          Enum.map(1..walk, fn i -> -60.0 + i end) ++ List.duplicate(-15.0, tail)

      p1_players = Enum.map(p1x, &%{x: &1})
      p2_players = List.duplicate(%{x: 0.0}, n)

      # dist crosses 20 at x=-20 -> onset index hold + 40
      onset = hold + 40

      actions =
        if fair_at_engage? do
          List.duplicate(@wait, onset + 2) ++ List.duplicate(66, 8) ++
            List.duplicate(@wait, n - onset - 10)
        else
          List.duplicate(@wait, n)
        end

      {%{actions: actions, players: p1_players}, %{players: p2_players}}
    end

    test "a walk-in ending in fair counts as one armed approach" do
      {p1, p2} = walk_in(true)
      s = ReplayStats.approach_stats(p1, p2)

      assert s.approaches == 1
      assert s.armed == 1
      assert s.armed_per_min > 0
    end

    test "a walk-in with no attack is an approach but not armed" do
      {p1, p2} = walk_in(false)
      s = ReplayStats.approach_stats(p1, p2)

      assert s.approaches == 1
      assert s.armed == 0
    end

    test "the opponent walking in is NOT our approach (closure share)" do
      n = 120
      p1_players = List.duplicate(%{x: -60.0}, n)

      p2x =
        List.duplicate(0.0, 20) ++
          Enum.map(1..45, fn i -> 0.0 - i end) ++ List.duplicate(-45.0, 55)

      p2_players = Enum.map(p2x, &%{x: &1})
      p1 = %{actions: List.duplicate(@wait, n), players: p1_players}

      s = ReplayStats.approach_stats(p1, %{players: p2_players})
      assert s.approaches == 0
    end

    test "closing distance while in hitstun (comboed inward) does not count" do
      {p1, p2} = walk_in(true)
      # Hitstun at the closure start (the furthest-apart frame, index 0)
      p1 = %{p1 | actions: List.duplicate(@hitstun_lo, 30) ++ Enum.drop(p1.actions, 30)}

      s = ReplayStats.approach_stats(p1, p2)
      assert s.approaches == 0
    end
  end

  describe "conversion_stats/2" do
    # walk_in/1 geometry: onset at index 60, closure start (furthest-apart
    # frame in the lookback) at index 0, n = 120.
    defp with_p2_actions({p1, p2}, p2_actions), do: {p1, Map.put(p2, :actions, p2_actions)}

    test "P2 entering hitstun after engagement converts; span covers decision through payoff" do
      n = 120
      # Engagement onset is index 59: dist hits exactly 20.0 (<= engage
      # threshold) one frame before the walk_in comment's "onset = 60"
      onset = 59
      # P2 takes the hit ~10 frames after engagement
      p2_actions =
        List.duplicate(@wait, onset + 10) ++
          List.duplicate(@hitstun_lo, 20) ++ List.duplicate(@wait, n - onset - 30)

      {p1, p2} = with_p2_actions(walk_in(true), p2_actions)
      s = ReplayStats.conversion_stats(p1, p2)

      assert s.approaches == 1
      assert s.conversions == 1
      assert s.spans == [{0, onset + 45}]
    end

    test "an approach with no opponent hitstun does not convert" do
      {p1, p2} = with_p2_actions(walk_in(true), List.duplicate(@wait, 120))
      s = ReplayStats.conversion_stats(p1, p2)

      assert s.approaches == 1
      assert s.conversions == 0
      assert s.spans == []
    end

    test "approaching an already-comboed opponent is not a fresh conversion (entry-only)" do
      # P2 is in hitstun from well before the onset with no new entry inside
      # the window
      p2_actions = List.duplicate(@wait, 30) ++ List.duplicate(@hitstun_lo, 90)
      {p1, p2} = with_p2_actions(walk_in(true), p2_actions)
      s = ReplayStats.conversion_stats(p1, p2)

      assert s.approaches == 1
      assert s.conversions == 0
    end

    test "hitstun entry after the window closes does not convert" do
      n = 120
      onset = 60
      # Entry at onset + 50, past the 45-frame window
      p2_actions = List.duplicate(@wait, onset + 50) ++ List.duplicate(@hitstun_lo, n - onset - 50)
      {p1, p2} = with_p2_actions(walk_in(true), p2_actions)
      s = ReplayStats.conversion_stats(p1, p2)

      assert s.conversions == 0
    end
  end

  describe "opening_stats/2" do
    test "a combo is ONE opening but several hit entries" do
      # opening -> 3 re-hits inside the 60f window -> long gap -> fresh opening
      actions =
        List.duplicate(@wait, 30) ++
          List.duplicate(@hitstun_lo, 20) ++
          List.duplicate(@wait, 10) ++
          List.duplicate(@hitstun_lo, 15) ++
          List.duplicate(@wait, 200) ++
          List.duplicate(@hitstun_lo, 10)

      s = ReplayStats.opening_stats(actions)

      assert s.openings == 2
      assert s.hit_entries == 3
      assert s.per_min > 0
    end

    test "hitstun on frame 0 counts as an opening" do
      s = ReplayStats.opening_stats(List.duplicate(@hitstun_lo, 30) ++ List.duplicate(@wait, 30))
      assert s.openings == 1
      assert s.hit_entries == 1
    end

    test "no hitstun means no openings" do
      s = ReplayStats.opening_stats(List.duplicate(@wait, 600))
      assert s == %{openings: 0, hit_entries: 0, per_min: 0.0, entries_per_min: 0.0}
    end

    test "thrown states count as hitstun entries" do
      s = ReplayStats.opening_stats(List.duplicate(@wait, 30) ++ List.duplicate(@thrown_lo, 20))
      assert s.openings == 1
    end
  end

  describe "load/1 + summarize/1 (fixture integration)" do
    @tag :fixture
    test "loads a real replay into the documented shape" do
      d = ReplayStats.load(@fixture)

      assert d.n > 0
      assert length(d.p1.actions) == d.n
      assert length(d.p2.actions) == d.n
      assert length(d.p1.controllers) == d.n
      assert Enum.all?(d.p1.actions, &is_integer/1)
    end

    @tag :fixture
    test "summarize produces the standard profile with sane ranges" do
      s = ReplayStats.summarize(@fixture)

      assert s.path == Path.basename(@fixture)
      assert s.frames > 0
      assert s.shield.shield_frac >= 0.0 and s.shield.shield_frac <= 1.0
      assert s.airborne_frac >= 0.0 and s.airborne_frac <= 1.0
      assert s.opp_knockdowns >= 0
      assert s.opp_tech_rate >= 0.0 and s.opp_tech_rate <= 1.0
      assert s.opp_hit_events >= 0
      assert s.tech_frames >= 0
    end
  end
end
