defmodule ExPhil.Bridge.MeleePortUnitTest do
  # Pure-function pins for the two crown-decider incident classes
  # (2026-08-24): partial LRAS chords and watcher starvation from
  # volatile stage watches over netplay.
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.MeleePort

  describe "force_quit_ops/1 (LRAS drive)" do
    test "tick 0 grounds once then holds the chord — no churn afterwards" do
      assert MeleePort.force_quit_ops(0) ==
               [:ground, {:press, :l}, {:press, :r}, {:press, :a}]

      # No tick after 0 may ground the pad or touch L/R/A: a release ->
      # re-press gap is sub-frame in the pipe, and a pad sample inside
      # it sees a partial chord (shield -> pause -> statue).
      for t <- 1..200 do
        ops = MeleePort.force_quit_ops(t)
        refute :ground in ops, "tick #{t} re-grounds the pad"

        for b <- [:l, :r, :a] do
          refute {:press, b} in ops, "tick #{t} churns #{b}"
          refute {:release, b} in ops, "tick #{t} releases #{b}"
        end
      end
    end

    test "START alternates press/release every pulse, forever" do
      # Wall-clock pulses (~300ms): edges must RECUR indefinitely —
      # early edges are eaten (Melee ignores Start in a game's first
      # moments), one edge may land as PAUSE (freezing the spectator
      # stream — why this can't be frame-driven), and a LATER fresh
      # edge must complete the quit from the pause menu.
      for t <- [1, 3, 5, 999] do
        assert MeleePort.force_quit_ops(t) == [{:press, :start}]
      end

      for t <- [2, 4, 6, 1000] do
        assert MeleePort.force_quit_ops(t) == [{:release, :start}]
      end
    end

    test "every Start press has L+R+A already held (chord completeness)" do
      # Simulate the pad through 100 ticks; at every :start press the
      # chord buttons must be down.
      Enum.reduce(0..100, MapSet.new(), fn t, held ->
        Enum.reduce(MeleePort.force_quit_ops(t), held, fn
          :ground, _acc ->
            MapSet.new()

          {:press, :start}, acc ->
            assert MapSet.subset?(MapSet.new([:l, :r, :a]), acc),
                   "start pressed at tick #{t} without full chord held"

            MapSet.put(acc, :start)

          {:press, b}, acc ->
            MapSet.put(acc, b)

          {:release, b}, acc ->
            MapSet.delete(acc, b)
        end)
      end)
    end
  end

  describe "memory_watch_set/3 (watcher starvation pin)" do
    test "ONLINE sessions never watch stage-internal addresses" do
      # The FoD/PS words sit in volatile stage-allocation heap on other
      # stages — on-change churn starved the netplay frame loop to
      # ~2fps (decider incident #3). The netplay stage merge is gated
      # off, so online stage watches serve nobody.
      for env <- [nil, "1"], stage <- [:final_destination, :fountain_of_dreams, :pokemon_stadium] do
        watches = MeleePort.memory_watch_set(env, true, stage)
        names = Keyword.keys(watches)
        refute :fod_platform_left in names, "online (env=#{inspect(env)}, #{stage}) watches FoD"
        refute :ps_transform_digit in names, "online (env=#{inspect(env)}, #{stage}) watches PS"
        assert :menu_state in names
      end
    end

    test "local FoD/PS sessions get exactly the stage watches by default" do
      watches = MeleePort.memory_watch_set(nil, false, :fountain_of_dreams)
      assert Keyword.keys(watches) |> Enum.sort() ==
               [:fod_platform_left, :fod_platform_right, :ps_transform_digit]

      assert MeleePort.memory_watch_set(nil, false, :pokemon_stadium) == watches
    end

    test "local non-stage sessions default to no watcher; env overrides" do
      assert MeleePort.memory_watch_set(nil, false, :final_destination) == false
      assert MeleePort.memory_watch_set("0", true, :fountain_of_dreams) == false

      forced = MeleePort.memory_watch_set("1", false, :fountain_of_dreams)
      assert :menu_state in Keyword.keys(forced)
      assert :fod_platform_left in Keyword.keys(forced)
    end
  end
end
