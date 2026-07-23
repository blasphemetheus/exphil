defmodule ExPhil.Eval.NeutralScanTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.NeutralScan

  @wait 14
  @jab 44
  @grab 212
  @fsmash 60
  @hitstun 80
  @special 347

  # Action list: n neutral frames, then the given opener held 5 frames.
  defp neutral_then(n, opener) do
    List.duplicate(@wait, n) ++ List.duplicate(opener, 5)
  end

  describe "category/1" do
    test "maps the universal blocks" do
      assert NeutralScan.category(@jab) == :jab
      assert NeutralScan.category(50) == :dash_attack
      assert NeutralScan.category(55) == :tilt
      assert NeutralScan.category(@fsmash) == :smash
      assert NeutralScan.category(67) == :aerial
      assert NeutralScan.category(@grab) == :grab
      assert NeutralScan.category(@special) == :special
      assert NeutralScan.category(@wait) == nil
      assert NeutralScan.category(@hitstun) == nil
    end
  end

  describe "opener_events/3" do
    test "detects an opener after sufficient neutral" do
      p1 = neutral_then(100, @jab)
      p2 = List.duplicate(@wait, length(p1))

      assert [%{index: 100, action: @jab, category: :jab}] = NeutralScan.opener_events(p1, p2)
    end

    test "no event when the neutral lead-in is too short" do
      p1 = neutral_then(30, @jab)
      p2 = List.duplicate(@wait, length(p1))
      assert NeutralScan.opener_events(p1, p2) == []
    end

    test "opponent in hitstun during lead-in resets the streak" do
      p1 = neutral_then(100, @jab)
      p2 = List.duplicate(@hitstun, 80) ++ List.duplicate(@wait, length(p1) - 80)
      # Only 20 both-neutral frames before the jab -> no opener.
      assert NeutralScan.opener_events(p1, p2) == []
    end

    test "held opener state fires once, and follow-ups inside the punish don't count" do
      p1 = neutral_then(100, @jab) ++ List.duplicate(@fsmash, 5) ++ List.duplicate(@wait, 10)
      p2 = List.duplicate(@wait, length(p1))

      # The fsmash right after the jab has no fresh 60f neutral run.
      assert [%{category: :jab}] = NeutralScan.opener_events(p1, p2)
    end

    test "multiple distinct openers across separate neutral periods" do
      p1 = neutral_then(80, @jab) ++ neutral_then(80, @grab) ++ neutral_then(80, @special)
      p2 = List.duplicate(@wait, length(p1))

      cats = NeutralScan.opener_events(p1, p2) |> Enum.map(& &1.category)
      assert cats == [:jab, :grab, :special]
    end
  end

  describe "distribution/entropy/top_share" do
    test "uniform two-category distribution has 1 bit of entropy" do
      events = [%{category: :jab}, %{category: :grab}]
      dist = NeutralScan.distribution(events)
      assert dist == %{jab: 0.5, grab: 0.5}
      assert_in_delta NeutralScan.entropy_bits(dist), 1.0, 1.0e-9
      assert NeutralScan.top_share(dist) == 0.5
    end

    test "single-category distribution has zero entropy and share 1.0" do
      dist = NeutralScan.distribution([%{category: :jab}, %{category: :jab}])
      assert NeutralScan.entropy_bits(dist) == 0.0
      assert NeutralScan.top_share(dist) == 1.0
    end

    test "empty edge cases" do
      assert NeutralScan.distribution([]) == %{}
      assert NeutralScan.entropy_bits(%{}) == 0.0
      assert NeutralScan.top_share(%{}) == nil
    end
  end

  describe "summary/3" do
    test "below min_events the diversity metrics are nil (no-evidence pass)" do
      p1 = neutral_then(100, @jab)
      p2 = List.duplicate(@wait, length(p1))
      s = NeutralScan.summary(p1, p2)
      assert s.openers == 1
      assert s.entropy_bits == nil
      assert s.top_share == nil
    end

    test "at/above min_events the metrics are computed" do
      block = neutral_then(80, @jab) ++ neutral_then(80, @grab)
      p1 = List.duplicate(block, 4) |> List.flatten()
      p2 = List.duplicate(@wait, length(p1))

      s = NeutralScan.summary(p1, p2)
      assert s.openers == 8
      assert_in_delta s.entropy_bits, 1.0, 1.0e-9
      assert s.top_share == 0.5
    end
  end
end
