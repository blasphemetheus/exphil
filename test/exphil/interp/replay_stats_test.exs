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
