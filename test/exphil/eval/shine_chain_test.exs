defmodule ExPhil.Eval.ShineChainTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.ShineChain

  @shine 360
  @ground_shine2 363
  @air_shine 365
  @jumpsquat 24
  @aerial_jump 25
  @wait 14

  # Actions persist for several frames; hold each a few to prove the
  # metric is frame-count robust.
  defp hold(a, n \\ 3), do: List.duplicate(a, n)

  describe "family/1" do
    test "splits grounded from aerial reflectors (the distinction that matters)" do
      assert ShineChain.family(360) == :ground_reflect
      assert ShineChain.family(363) == :ground_reflect
      assert ShineChain.family(365) == :air_reflect
      assert ShineChain.family(368) == :air_reflect
      assert ShineChain.family(@jumpsquat) == :jumpsquat
      assert ShineChain.family(@aerial_jump) == :aerial_jump
      assert ShineChain.family(@wait) == :other
    end
  end

  describe "chains/1 — CLEAN grounded multishine only" do
    test "three GROUNDED shines jump-cancelled is one chain of length 3" do
      actions =
        hold(@wait) ++
          hold(@shine) ++ hold(@jumpsquat) ++
          hold(@ground_shine2) ++ hold(@jumpsquat) ++
          hold(@shine) ++ hold(@wait)

      assert ShineChain.chains(actions) == [3]
    end

    test "an AERIAL shine BREAKS the chain (this is the sloppy loop we must catch)" do
      # ground shine -> jumpsquat -> AIR shine -> ... : left the ground.
      actions =
        hold(@shine) ++ hold(@jumpsquat) ++ hold(@air_shine) ++
          hold(@wait) ++ hold(@shine)

      # two length-1 grounded chains, not a length-3 anything
      assert ShineChain.chains(actions) == [1, 1]
    end

    test "a lone grounded shine is a chain of length 1" do
      assert ShineChain.chains(hold(@wait) ++ hold(@shine) ++ hold(@wait)) == [1]
    end

    test "an empty hop breaks the chain" do
      actions =
        hold(@shine) ++ hold(@jumpsquat) ++ hold(@aerial_jump) ++
          hold(@wait) ++ hold(@shine)

      assert ShineChain.chains(actions) == [1, 1]
    end

    test "two separate grounded multishines are two chains" do
      actions =
        hold(@shine) ++ hold(@jumpsquat) ++ hold(@ground_shine2) ++
          hold(@wait) ++
          hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine) ++ hold(@jumpsquat) ++ hold(@ground_shine2)

      assert ShineChain.chains(actions) == [2, 3]
    end

    test "frame-count robust" do
      short = hold(@shine, 1) ++ hold(@jumpsquat, 1) ++ hold(@ground_shine2, 1)
      long = hold(@shine, 20) ++ hold(@jumpsquat, 9) ++ hold(@ground_shine2, 30)
      assert ShineChain.chains(short) == ShineChain.chains(long)
    end

    test "no shines at all yields no chains" do
      assert ShineChain.chains(hold(@wait, 50)) == []
    end
  end

  describe "grounded fraction (the number that catches airborne 'shining')" do
    test "all-grounded reads ~1.0, all-aerial reads ~0.0" do
      clean = hold(@shine) ++ hold(@jumpsquat) ++ hold(@ground_shine2)
      sloppy = hold(@air_shine) ++ hold(@jumpsquat) ++ hold(@air_shine)
      assert ShineChain.summary(clean).grounded_fraction == 1.0
      assert ShineChain.summary(sloppy).grounded_fraction == 0.0
    end
  end

  describe "empty_hops/1" do
    test "counts jumpsquat immediately followed by an aerial jump" do
      actions =
        hold(@shine) ++ hold(@jumpsquat) ++ hold(@aerial_jump) ++
          hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine)

      assert ShineChain.empty_hops(actions) == 1
    end

    test "a jumpsquat that shines is not an empty hop" do
      assert ShineChain.empty_hops(hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine)) == 0
    end
  end

  describe "summary/2" do
    test "separates entry from sustain" do
      # Two short chains (2 and 3) — enters fine, never sustains to 5.
      actions =
        hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine) ++
          hold(@wait) ++
          hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine) ++ hold(@jumpsquat) ++ hold(@shine)

      s = ShineChain.summary(actions)
      assert s.chains == 2
      assert s.shines == 5
      assert s.max_length == 3
      assert s.entries == 2
      assert s.sustained == 0
      assert_in_delta s.mean_length, 2.5, 1.0e-9
    end

    test "a long chain counts as sustained" do
      long =
        hold(@shine) ++
          Enum.flat_map(1..5, fn _ -> hold(@jumpsquat) ++ hold(@ground_shine2) end)

      s = ShineChain.summary(long)
      assert s.max_length == 6
      assert s.sustained == 1
    end

    test "no-evidence reads as nil, not a misleading 0.0" do
      s = ShineChain.summary(hold(@wait, 30))
      assert s.chains == 0
      assert s.mean_length == nil
      assert s.max_length == nil
    end
  end
end
