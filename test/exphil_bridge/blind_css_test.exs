defmodule ExPhil.Bridge.BlindCssTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.BlindCss

  # ---------------------------------------------------------------
  # classify/1: one test per Observation -> Progress class.
  # Words are scene-controller packings <<major, pending, prev, minor>>.
  # ---------------------------------------------------------------
  describe "classify/1 — observation classes" do
    test ":unknown observation (no watcher / no change yet)" do
      assert BlindCss.classify(:unknown) == :unknown
    end

    test "settled at the online CSS" do
      assert BlindCss.classify(0x08080800) == :at_css
    end

    test "transition committed (pending != major) — any target" do
      assert BlindCss.classify(0x08020800) == :departing
      # Even a pending UNKNOWN family counts: the scene is moving.
      assert BlindCss.classify(0x08420800) == :departing
    end

    test "settled at a known non-CSS scene" do
      assert BlindCss.classify(0x02020200) == :elsewhere
      assert BlindCss.classify(0x02020202) == :elsewhere
    end

    test "settled at an UNMAPPED scene is :unknown, not :elsewhere" do
      # Online scene minors aren't fully derived; an unrecognized word
      # must not close the loop (would hand back on noise).
      assert BlindCss.classify(0x42424200) == :unknown
    end

    test "the all-zero transient is :unknown (bot14 scene-load flicker)" do
      # Observed twice live: 0x00000000 for ~17ms during online scene
      # churn. Genuine press-start can't occur inside the blind arm —
      # zero is noise, and :elsewhere here caused premature handback.
      assert BlindCss.classify(0x00000000) == :unknown
    end

    test "the online in-game word is :elsewhere — the true departure signal" do
      # 0x0408 = major 8 minor 4 = online in-game (bot14 capture,
      # replay-correlated). Match forming during the START pulse now
      # confirms departure instead of hiding behind settled-unknown.
      assert BlindCss.classify(0x08080104) == :elsewhere
    end
  end

  # ---------------------------------------------------------------
  # step/3 decision table: phases x progress classes.
  # ---------------------------------------------------------------
  describe "step/3 — timed phases ignore scene evidence" do
    test "steer, press, release phases are progress-independent" do
      for progress <- [:at_css, :departing, :elsewhere, :unknown] do
        assert BlindCss.step(0, progress) == :steer
        assert BlindCss.step(479, progress) == :steer
        assert BlindCss.step(480, progress) == :press_a
        assert BlindCss.step(482, progress) == :press_a
        assert BlindCss.step(483, progress) == :release_a
        assert BlindCss.step(599, progress) == :release_a
      end
    end
  end

  describe "step/4 — selection evidence at the press window (2026-08-22c)" do
    test "RAM-confirmed selection skips the A press (A toggles)" do
      for progress <- [:at_css, :departing, :elsewhere, :unknown] do
        assert BlindCss.step(480, progress, 0, {:character, 0x02}) == :release_a
        assert BlindCss.step(482, progress, 0, {:character, 0x14}) == :release_a
      end
    end

    test ":none and :unknown keep the validated open-loop press" do
      assert BlindCss.step(480, :at_css, 0, :none) == :press_a
      assert BlindCss.step(480, :at_css, 0, :unknown) == :press_a
    end

    test "selection is consulted at the press window and the window end only" do
      # Same answers as the selection-blind table everywhere else.
      for selection <- [:none, :unknown, {:character, 0x02}] do
        assert BlindCss.step(0, :at_css, 0, selection) == :steer
        assert BlindCss.step(483, :at_css, 0, selection) == :release_a
        assert BlindCss.step(600, :at_css, 0, selection) == {:pulse_start, true}
        assert BlindCss.step(601, :departing, 0, selection) == :handback
        assert BlindCss.step(900, :at_css, 2, selection) == :handback
      end

      # Window end, retries left: unconfirmed pick retries (legacy)...
      assert BlindCss.step(900, :at_css, 0, :none) == {:retry_a, 1}
      assert BlindCss.step(900, :at_css, 0, :unknown) == {:retry_a, 1}
    end

    test "window end with RAM-confirmed pick: hand back, never burn retry windows" do
      # Measured live 2026-08-23: every retry cycle after the first
      # press ran with the pick already confirmed — ~7s of pure wait
      # each. With the RAM menu merge, the helper can press START
      # itself, so early handback is safe.
      assert BlindCss.step(900, :at_css, 0, {:character, 0x02}) == :handback
      assert BlindCss.step(900, :at_css, 1, {:character, 0x14}) == :handback
    end
  end

  describe "observe_selected/2 — totality" do
    test "no watcher yields :unknown" do
      assert BlindCss.observe_selected(nil, 1) == :unknown
    end
  end

  describe "step/3 — pulse phase consults the scene" do
    test "no signal / still at CSS: pulse with 3-of-60 duty" do
      for progress <- [:unknown, :at_css] do
        assert BlindCss.step(600, progress) == {:pulse_start, true}
        assert BlindCss.step(602, progress) == {:pulse_start, true}
        assert BlindCss.step(603, progress) == {:pulse_start, false}
        assert BlindCss.step(659, progress) == {:pulse_start, false}
        assert BlindCss.step(660, progress) == {:pulse_start, true}
      end
    end

    test "departure confirmed mid-pulse: hand back immediately" do
      assert BlindCss.step(601, :departing) == :handback
      assert BlindCss.step(750, :elsewhere) == :handback
    end
  end

  describe "step/3 — window end: retry vs handback" do
    test "still settled at CSS with retries left: retry the pick" do
      assert BlindCss.step(900, :at_css, 0) == {:retry_a, 1}
      assert BlindCss.step(900, :at_css, 1) == {:retry_a, 2}
    end

    test "retries exhausted: hand back (legacy behavior)" do
      assert BlindCss.step(900, :at_css, 2) == :handback
    end

    test "no signal at window end: hand back (never retry on no evidence)" do
      assert BlindCss.step(900, :unknown, 0) == :handback
    end

    test "departed by window end: hand back" do
      assert BlindCss.step(900, :departing, 0) == :handback
      assert BlindCss.step(900, :elsewhere, 0) == :handback
    end
  end

  describe "warmup overlap — warmup_step/1 + ready_resteer_reset/1" do
    test "warmup_step: steer until the press point, then animate" do
      assert BlindCss.warmup_step(0) == :steer
      assert BlindCss.warmup_step(479) == :steer
      assert BlindCss.warmup_step(480) == :animate
      assert BlindCss.warmup_step(5000) == :animate
    end

    test "ready reset: full steer rewinds to the re-steer window; partial keeps progress" do
      assert BlindCss.ready_resteer_reset(480) == 360
      assert BlindCss.ready_resteer_reset(5000) == 360
      assert BlindCss.ready_resteer_reset(0) == 0
      assert BlindCss.ready_resteer_reset(300) == 300
    end

    test "the re-steer window re-enters the table as steering, then presses" do
      n = BlindCss.ready_resteer_reset(480)
      assert BlindCss.step(n, :unknown) == :steer
      assert BlindCss.step(479, :unknown) == :steer
      assert BlindCss.step(480, :unknown) == :press_a
    end
  end

  # ---------------------------------------------------------------
  # Whole-trace properties: the sequence a session actually produces.
  # ---------------------------------------------------------------
  describe "traces" do
    defp trace(progress_at, retries \\ 0) do
      # Run n = 0..920 with a progress function; stop at terminal action.
      Enum.reduce_while(0..920, [], fn n, acc ->
        action = BlindCss.step(n, progress_at.(n), retries)

        case action do
          :handback -> {:halt, Enum.reverse([{n, action} | acc])}
          {:retry_a, _} -> {:halt, Enum.reverse([{n, action} | acc])}
          _ -> {:cont, [{n, action} | acc]}
        end
      end)
    end

    test "no watcher signal: exactly the legacy 2026-08-22 open-loop sequence" do
      t = trace(fn _ -> :unknown end)
      actions = Enum.map(t, &elem(&1, 1))

      assert Enum.count(actions, &(&1 == :steer)) == 480
      assert Enum.count(actions, &(&1 == :press_a)) == 3
      assert Enum.count(actions, &(&1 == :release_a)) == 117
      assert Enum.count(actions, &match?({:pulse_start, _}, &1)) == 300
      assert List.last(t) == {900, :handback}
    end

    test "departure at frame 700: pulses stop there, not at 900" do
      t = trace(fn n -> if n >= 700, do: :departing, else: :at_css end)
      assert List.last(t) == {700, :handback}
      refute Enum.any?(t, fn {n, a} -> n > 700 and match?({:pulse_start, _}, a) end)
    end

    test "pick never lands: retry issued at window end, reset point re-enters press phase" do
      t = trace(fn _ -> :at_css end, 0)
      assert List.last(t) == {900, {:retry_a, 1}}
      # The caller resets n to a_press_at/0 — from there the table
      # replays press -> release -> pulse without re-steering.
      assert BlindCss.step(BlindCss.a_press_at(), :at_css, 1) == :press_a
    end
  end
end
