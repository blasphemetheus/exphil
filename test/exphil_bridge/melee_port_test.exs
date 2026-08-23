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

  describe "require_stage_internal_id/1 (id-space pin, GOTCHA #96 class)" do
    test "resolves to the INTERNAL id space gamestate.stage carries" do
      # events.ex converts GAME_START's external stage id to internal
      # (from_external |> to_id); the reject comparison must live in
      # the same space. Decider incident #5: the old external table
      # rejected FD itself (external 32 vs live internal 0x19).
      assert MeleePort.require_stage_internal_id(:final_destination) == 0x19
      assert MeleePort.require_stage_internal_id("fd") == 0x19
      assert MeleePort.require_stage_internal_id(:pokemon_stadium) == 0x12
      assert MeleePort.require_stage_internal_id(:fountain_of_dreams) == 0x08
      assert MeleePort.require_stage_internal_id(nil) == nil
    end

    test "agrees with the events.ex GAME_START conversion for every legal stage" do
      for {atom, external} <- [
            fountain_of_dreams: 2,
            pokemon_stadium: 3,
            yoshis_story: 8,
            dreamland: 28,
            battlefield: 31,
            final_destination: 32
          ] do
        live_id = external |> Melee.Enums.Stage.from_external() |> Melee.Enums.Stage.to_id()

        assert MeleePort.require_stage_internal_id(atom) == live_id,
               "#{atom}: require id diverges from the live gamestate.stage space"
      end
    end
  end

  describe "memory_watch_set/3 (watcher starvation pin)" do
    test "stage-internal addresses are NEVER in any watch set" do
      # The FoD/PS words sit in volatile stage-allocation heap on other
      # stages — the watcher's on-change churn starved the netplay
      # frame loop to ~2fps (decider incident #3). They are read via
      # direct pread (stage_ram_apply), never watched.
      for env <- [nil, "1"],
          online <- [true, false],
          stage <- [:final_destination, :fountain_of_dreams, :pokemon_stadium] do
        case MeleePort.memory_watch_set(env, online, stage) do
          false ->
            :ok

          watches ->
            names = Keyword.keys(watches)
            refute :fod_platform_left in names, "(#{inspect(env)},#{online},#{stage}) watches FoD"
            refute :ps_transform_digit in names, "(#{inspect(env)},#{online},#{stage}) watches PS"
            assert :menu_state in names
        end
      end
    end

    test "menu set online by default; env forces on/off; local defaults off" do
      assert :menu_state in Keyword.keys(MeleePort.memory_watch_set(nil, true, :final_destination))
      assert MeleePort.memory_watch_set(nil, false, :fountain_of_dreams) == false
      assert MeleePort.memory_watch_set("0", true, :fountain_of_dreams) == false
      assert :menu_state in Keyword.keys(MeleePort.memory_watch_set("1", false, :final_destination))
    end
  end
end
