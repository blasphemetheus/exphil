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
  # The event-driven phase machine (2026-08-23): each phase exits on
  # evidence, budgets are legacy fallbacks.
  # ---------------------------------------------------------------

  defp run(phase, opts \\ []) do
    BlindCss.step(
      phase,
      Keyword.get(opts, :progress, :unknown),
      Keyword.get(opts, :selection, :unknown),
      Keyword.get(opts, :hover, false),
      Keyword.get(opts, :retries, 0)
    )
  end

  describe "observe totality" do
    test "observe_selected/2 and observe_hover/2 with no watcher" do
      assert BlindCss.observe_selected(nil, 1) == :unknown
      assert BlindCss.observe_hover(nil, 0x0A) == false
    end

    test "snapshot/1 with no watcher is empty" do
      assert BlindCss.snapshot(nil) == %{}
    end
  end

  describe "snapshot derivations (the one-call-per-frame contract)" do
    test "scene_from/1: present, absent" do
      assert BlindCss.scene_from(%{menu_state: 0x08080100}) == 0x08080100
      assert BlindCss.scene_from(%{}) == :unknown
    end

    test "selection_from/2: all observe_selected classes, pure" do
      assert BlindCss.selection_from(%{css_p1_selected: 0x21}, 1) == :none
      assert BlindCss.selection_from(%{css_p1_selected: 0x02}, 1) == {:character, 2}
      # implausible word (garbage beyond a byte)
      assert BlindCss.selection_from(%{css_p1_selected: 0x12345678}, 1) == :unknown
      # unobserved
      assert BlindCss.selection_from(%{}, 1) == :unknown
      # per-port keying
      assert BlindCss.selection_from(%{css_p2_selected: 0x14}, 2) == {:character, 0x14}
      assert BlindCss.selection_from(%{css_p2_selected: 0x14}, 1) == :unknown
    end
  end

  describe "normalize_selection/2 — the entry-garbage guard" do
    test "only the TARGET character counts as locked" do
      assert BlindCss.normalize_selection({:character, 2}, 2) == {:character, 2}
      # The ONLINE CSS's unselected sentinel is 26 (Master Hand — no
      # CSS-roster mapping): a NONE-class value (confirmed by /proc
      # pread 2026-08-24), so probes get their whiff signal from it.
      assert BlindCss.normalize_selection({:character, 26}, 2) == :none
      assert BlindCss.normalize_selection({:character, 30}, 2) == :none
      # A REAL other character stays :unknown — never locked, never a
      # whiff (the g5 press-skip regression class).
      assert BlindCss.normalize_selection({:character, 7}, 2) == :unknown
    end

    test ":none (the whiff signal) and :unknown pass through" do
      assert BlindCss.normalize_selection(:none, 2) == :none
      assert BlindCss.normalize_selection(:unknown, 2) == :unknown
    end
  end

  describe "phase machine — steer" do
    test "no evidence: counts to the 480 budget, then presses" do
      assert run({:steer, 0}) == {:steer, {:steer, 1}, 0}
      assert run({:steer, 478}) == {:steer, {:steer, 479}, 0}
      assert run({:steer, 479}) == {:steer, {:press, 0}, 0}
    end

    test "hover match ends the steer immediately" do
      assert run({:steer, 5}, hover: true) == {:steer, {:press, 0}, 0}
    end

    test "rematch fast path: locked selection skips straight to the pulses" do
      assert run({:steer, 0}, selection: {:character, 2}) == {:steer, {:pulse, 0}, 0}
      assert run({:steer, 300}, selection: {:character, 2}, hover: true) ==
               {:steer, {:pulse, 0}, 0}
    end
  end

  describe "phase machine — press and confirm" do
    test "press holds A for 3 frames, then waits for the readback" do
      assert run({:press, 0}) == {:press_a, {:press, 1}, 0}
      assert run({:press, 2}) == {:press_a, {:press, 3}, 0}
      assert run({:press, 3}) == {:release_a, {:confirm, 0}, 0}
    end

    test "locked selection skips the press (A toggles)" do
      assert run({:press, 0}, selection: {:character, 2}) == {:release_a, {:pulse, 0}, 0}
    end

    test "confirm exits the instant the selection word flips" do
      assert run({:confirm, 4}, selection: {:character, 2}) == {:release_a, {:pulse, 0}, 0}
    end

    test "confirm without a readback runs the legacy settle (pulse starts at frame 600)" do
      assert run({:confirm, 0}) == {:release_a, {:confirm, 1}, 0}
      assert run({:confirm, 114}) == {:release_a, {:confirm, 115}, 0}
      assert run({:confirm, 115}) == {:release_a, {:pulse, 0}, 0}
    end

    test "a positive whiff (:none) at the confirm budget re-presses, bounded" do
      assert run({:confirm, 115}, selection: :none) == {:release_a, {:press, 0}, 1}
      assert run({:confirm, 115}, selection: :none, retries: 2) == {:release_a, {:pulse, 0}, 2}
    end
  end

  describe "phase machine — pulse" do
    test "3-of-60 duty cycle" do
      assert run({:pulse, 0}, progress: :at_css) == {{:pulse_start, true}, {:pulse, 1}, 0}
      assert run({:pulse, 2}, progress: :at_css) == {{:pulse_start, true}, {:pulse, 3}, 0}
      assert run({:pulse, 3}, progress: :at_css) == {{:pulse_start, false}, {:pulse, 4}, 0}
      assert run({:pulse, 60}, progress: :at_css) == {{:pulse_start, true}, {:pulse, 61}, 0}
    end

    test "confirmed departure hands back immediately" do
      assert {:handback, _, 0} = run({:pulse, 7}, progress: :departing)
      assert {:handback, _, 0} = run({:pulse, 250}, progress: :elsewhere)
    end

    test "locked selection + two pulse periods = early handback" do
      assert {{:pulse_start, _}, {:pulse, 120}, 0} =
               run({:pulse, 119}, progress: :at_css, selection: {:character, 2})

      assert {:handback, _, 0} = run({:pulse, 120}, progress: :at_css, selection: {:character, 2})
    end

    test "budget end at the CSS with an unconfirmed pick re-presses, bounded" do
      assert run({:pulse, 300}, progress: :at_css) == {:release_a, {:press, 0}, 1}
      assert run({:pulse, 300}, progress: :at_css, selection: :none, retries: 1) ==
               {:release_a, {:press, 0}, 2}

      assert {:handback, _, 2} = run({:pulse, 300}, progress: :at_css, retries: 2)
    end

    test "budget end with no scene evidence hands back (never retry on noise)" do
      assert {:handback, _, 0} = run({:pulse, 300}, progress: :unknown)
    end
  end

  describe "warmup overlap" do
    test "warmup_step: steer while steering, animate once parked" do
      assert BlindCss.warmup_step({:steer, 10}) == :steer
      assert BlindCss.warmup_step({:press, 0}) == :animate
      assert BlindCss.warmup_step({:pulse, 50}) == :animate
    end

    test "ready reset grants at least the re-steer window" do
      assert BlindCss.ready_resteer_reset({:steer, 100}) == {:steer, 100}
      assert BlindCss.ready_resteer_reset({:steer, 470}) == {:steer, 360}
      assert BlindCss.ready_resteer_reset({:press, 2}) == {:steer, 360}
    end
  end

  describe "traces" do
    test "silent watcher reproduces the legacy 480/483/600/900 timeline exactly" do
      {actions, _phase, _r} =
        Enum.reduce(1..900, {[], BlindCss.new(), 0}, fn _i, {acc, phase, r} ->
          {action, phase2, r2} = BlindCss.step(phase, :unknown, :unknown, false, r)
          {[action | acc], phase2, r2}
        end)

      actions = Enum.reverse(actions)

      assert Enum.take(actions, 480) == List.duplicate(:steer, 480)
      assert Enum.slice(actions, 480, 3) == List.duplicate(:press_a, 3)
      assert Enum.slice(actions, 483, 117) == List.duplicate(:release_a, 117)

      pulses = Enum.slice(actions, 600, 300)
      assert length(pulses) == 300

      for {p, k} <- Enum.with_index(pulses) do
        assert p == {:pulse_start, rem(k, 60) < 3}
      end

      # The 901st step (budget out, no evidence) hands back.
      {final, _, _} =
        Enum.reduce(1..901, {nil, BlindCss.new(), 0}, fn _i, {_a, phase, r} ->
          BlindCss.step(phase, :unknown, :unknown, false, r)
        end)

      assert final == :handback
    end

    test "full-evidence trace reaches handback in a few seconds of frames" do
      # Hover matches at steer frame 10; selection confirms 5 frames
      # after the press; early handback after 120 pulse frames.
      {frames, final} =
        Enum.reduce_while(1..900, {0, BlindCss.new(), 0}, fn i, {_n, phase, r} ->
          selection =
            case phase do
              {:confirm, n} when n >= 5 -> {:character, 2}
              {:pulse, _} -> {:character, 2}
              _ -> :none
            end

          hover = match?({:steer, n} when n >= 10, phase)

          case BlindCss.step(phase, :at_css, selection, hover, r) do
            {:handback, _, _} -> {:halt, {i, :handback}}
            {_a, phase2, r2} -> {:cont, {i, phase2, r2}}
          end
        end)

      assert final == :handback
      # ~10 steer + 4 press + 6 confirm + 121 pulse ≈ 141 frames (2.4s)
      assert frames < 160
    end

    test "rematch trace: retained pick reaches handback in ~2s" do
      {frames, final} =
        Enum.reduce_while(1..900, {0, BlindCss.new(), 0}, fn i, {_n, phase, r} ->
          case BlindCss.step(phase, :at_css, {:character, 2}, false, r) do
            {:handback, _, _} -> {:halt, {i, :handback}}
            {_a, phase2, r2} -> {:cont, {i, phase2, r2}}
          end
        end)

      assert final == :handback
      # 1 steer + 121 pulse frames
      assert frames < 130
    end
  end
end
