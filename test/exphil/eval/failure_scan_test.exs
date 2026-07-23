defmodule ExPhil.Eval.FailureScanTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.FailureScan

  # Action-state IDs (mirror ReplayStats).
  @wait 14
  @attack 44
  @hitstun 80
  @downbound 191

  defp player(overrides) do
    Map.merge(
      %{x: 0.0, y: 0.0, action: @wait, facing: 1, on_ground: true, stock: 4, percent: 0.0},
      overrides
    )
  end

  defp frame(n, p1_over, p2_over) do
    %{frame: n, p1: player(p1_over), p2: player(p2_over)}
  end

  # Build a run of frames [start..stop] with fixed per-player overrides.
  defp run(start, stop, p1_over, p2_over) do
    for n <- start..stop, do: frame(n, p1_over, p2_over)
  end

  describe "neutral_loss/1" do
    test "flags a hit out of clean neutral, handoff backed off 90f" do
      # 400 neutral frames, then P1 enters hitstun at frame 400.
      fs = run(1, 399, %{}, %{x: 20.0}) ++ [frame(400, %{action: @hitstun}, %{x: 20.0, action: @attack})]

      assert [%{type: :neutral_loss, frame: 310, note: note}] = FailureScan.neutral_loss(fs)
      assert note =~ "hit@400"
    end

    test "ignores a hit that follows a non-neutral lead-in (combo continuation)" do
      # P2 in the knockdown lifecycle during the lookback -> not true neutral.
      fs =
        run(1, 350, %{}, %{action: @downbound}) ++
          [frame(351, %{action: @hitstun}, %{action: @downbound})]

      assert FailureScan.neutral_loss(fs) == []
    end
  end

  describe "dropped_punish/1" do
    test "flags an opening from neutral that yields < 15% and lets P2 escape" do
      # 400 neutral frames, P1 opens P2 (P2 -> hitstun) at 400 with a small
      # percent bump, then P2 returns to neutral with no further hit.
      fs =
        run(1, 399, %{}, %{x: 15.0, percent: 30.0}) ++
          run(400, 405, %{action: @attack}, %{x: 15.0, action: @hitstun, percent: 38.0}) ++
          run(406, 540, %{}, %{x: 40.0, percent: 38.0})

      assert [%{type: :dropped_punish, frame: 400}] = FailureScan.dropped_punish(fs)
    end

    test "does NOT flag when the punish keeps going (a follow-up hit lands)" do
      # Opener at 400, P2 briefly out, then a SECOND hit at 411 -> the punish
      # continued, so it is not dropped (the second-hit discriminator).
      fs =
        run(1, 399, %{}, %{x: 15.0, percent: 30.0}) ++
          run(400, 405, %{action: @attack}, %{x: 15.0, action: @hitstun, percent: 38.0}) ++
          run(406, 410, %{}, %{x: 15.0, percent: 38.0}) ++
          run(411, 416, %{action: @attack}, %{x: 15.0, action: @hitstun, percent: 55.0}) ++
          run(417, 540, %{}, %{x: 15.0, percent: 55.0})

      assert FailureScan.dropped_punish(fs) == []
    end
  end

  describe "death_sequence/1" do
    test "backs up from the death to where neutral broke" do
      # Neutral through 430, neutral breaks (P1 in hitstun) 431+, death at 470.
      fs =
        run(1, 430, %{}, %{x: 20.0}) ++
          run(431, 469, %{action: @hitstun}, %{x: 10.0, action: @attack}) ++
          [frame(470, %{stock: 3, action: @hitstun}, %{x: 10.0})]

      assert [%{type: :death_sequence, frame: 430, note: note}] = FailureScan.death_sequence(fs)
      assert note =~ "death@470"
    end
  end

  describe "passivity_window/1" do
    test "flags a long close-range run with no attack" do
      fs = run(1, 400, %{}, %{x: 25.0})
      assert [%{type: :passivity_window, frame: 1, note: note}] = FailureScan.passivity_window(fs)
      assert note =~ "passive_run=400f"
    end

    test "an attack in range breaks the run" do
      fs =
        run(1, 150, %{}, %{x: 25.0}) ++
          [frame(151, %{action: @attack}, %{x: 25.0})] ++
          run(152, 300, %{}, %{x: 25.0})

      # Neither sub-run reaches the 300-frame minimum.
      assert FailureScan.passivity_window(fs) == []
    end

    test "out-of-range frames don't count as passive pressure" do
      fs = run(1, 400, %{}, %{x: 200.0})
      assert FailureScan.passivity_window(fs) == []
    end
  end

  describe "scan/2 curation" do
    test "min_frame filters early handoffs and same-type spacing applies" do
      fs = run(1, 400, %{}, %{x: 25.0})
      # passivity handoff at frame 1 is below the default min_frame (300).
      assert FailureScan.scan(fs) == []
    end

    test "flip/1 swaps the subject (P1<->P2) so a P2 bot can be scanned" do
      fs = run(1, 3, %{x: 25.0}, %{x: 0.0})
      flipped = FailureScan.flip(fs)
      # After flip the old P2 (x=0) is P1 and the old P1 (x=25) is P2.
      assert Enum.all?(flipped, fn f -> f.p1.x == 0.0 and f.p2.x == 25.0 end)
    end
  end
end
