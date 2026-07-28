defmodule ExPhil.Data.RecoverySynthTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MultishineExpert
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Constants
  alias ExPhil.Data.RecoverySynth

  @squat Constants.squat()
  @squat_wait Constants.squat_wait()
  @reflector 361

  # Empty table -> every label falls through to the recovery rules, which is
  # the path both synthesis modes rely on for off-fixture states.
  defp expert, do: %MultishineExpert{table: %{}}

  defp player(action, af) do
    %{action: action, action_frame: af, on_ground: true, x: 0.0, y: 0.0, facing: 1, percent: 0.0, jumps_left: 1}
  end

  defp frame(n, action, af) do
    %{
      game_state: %{frame: n, players: %{1 => player(action, af), 2 => player(14, 1)}},
      controller: ControllerState.neutral()
    }
  end

  # Two reflector segments separated by idle, idle lead-ins long enough for
  # any window.
  defp fixture_frames do
    idle1 = Enum.map(0..19, &frame(&1, 14, &1 + 1))
    shine1 = Enum.map(20..22, &frame(&1, @reflector, &1 - 19))
    idle2 = Enum.map(23..42, &frame(&1, 14, &1 - 22))
    shine2 = Enum.map(43..45, &frame(&1, @reflector, &1 - 42))
    idle1 ++ shine1 ++ idle2 ++ shine2
  end

  describe "build_crouch/2" do
    test "manufactures Squat then SquatWait tails after real lead-ins" do
      out =
        RecoverySynth.build_crouch(fixture_frames(),
          expert: expert(),
          lead_in: 4,
          max_af: 20,
          ratio: 1.0
        )

      actions = Enum.map(out, & &1.game_state.players[1].action)
      assert @squat in actions
      assert @squat_wait in actions

      # Tail structure: ... lead(real) ... 39x2 then 40xmax_af
      first_squat = Enum.find_index(actions, &(&1 == @squat))
      assert Enum.slice(actions, first_squat, 22) == [@squat, @squat] ++ List.duplicate(@squat_wait, 20)
    end

    test "SquatWait af runs 1..max_af so windows past the lead are fully crouched" do
      out =
        RecoverySynth.build_crouch(fixture_frames(),
          expert: expert(),
          lead_in: 4,
          max_af: 20,
          ratio: 1.0
        )

      afs =
        out
        |> Enum.filter(&(&1.game_state.players[1].action == @squat_wait))
        |> Enum.map(&trunc(&1.game_state.players[1].action_frame))

      assert Enum.take(afs, 20) == Enum.to_list(1..20)
    end

    test "labels alternate B press/release instead of holding" do
      out =
        RecoverySynth.build_crouch(fixture_frames(),
          expert: expert(),
          lead_in: 4,
          max_af: 20,
          ratio: 1.0
        )

      tail_buttons =
        out
        |> Enum.filter(&(&1.game_state.players[1].action in [@squat, @squat_wait]))
        |> Enum.map(& &1.controller.button_b)

      # Lead controller is neutral (B up) -> first tail label presses.
      assert hd(tail_buttons) == true
      # Threaded prev -> strict alternation, never a held run.
      assert tail_buttons == Enum.map(0..(length(tail_buttons) - 1), &(rem(&1, 2) == 0))
    end

    test "ratio caps output volume" do
      frames = fixture_frames()

      out =
        RecoverySynth.build_crouch(frames,
          expert: expert(),
          lead_in: 4,
          max_af: 20,
          ratio: 0.3
        )

      # One block fits the 0.3 budget for ~46 input frames; never more than
      # budget + one block of slack.
      assert length(out) <= trunc(length(frames) * 0.3) + (4 + 2 + 20)
    end
  end

  describe "build_ledge/2" do
    test "manufactures CliffCatch then CliffWait at the edge, alternating sides" do
      out =
        RecoverySynth.build_ledge(fixture_frames(),
          lead_in: 4,
          max_af: 10,
          ratio: 1.0
        )

      actions = Enum.map(out, & &1.game_state.players[1].action)
      assert Constants.cliff_catch() in actions
      assert Constants.cliff_wait() in actions

      first_catch = Enum.find_index(actions, &(&1 == Constants.cliff_catch()))

      assert Enum.slice(actions, first_catch, 17) ==
               List.duplicate(Constants.cliff_catch(), 7) ++
                 List.duplicate(Constants.cliff_wait(), 10)

      xs =
        out
        |> Enum.filter(&(&1.game_state.players[1].action == Constants.cliff_wait()))
        |> Enum.map(& &1.game_state.players[1].x)

      assert Enum.any?(xs, &(&1 > 0)) and Enum.any?(xs, &(&1 < 0))
    end

    test "getup labels push toward the stage; facing is stageward" do
      out =
        RecoverySynth.build_ledge(fixture_frames(),
          lead_in: 4,
          max_af: 6,
          ratio: 1.0
        )

      for f <- out, f.game_state.players[1].action == Constants.cliff_wait() do
        p = f.game_state.players[1]
        expected_stick = if p.x > 0, do: 0.0, else: 1.0
        assert f.controller.main_stick.x == expected_stick
        assert p.facing == if(p.x > 0, do: -1, else: 1)
      end
    end
  end

  describe "build_opening/2" do
    # An "opening": entry-ish actions over game frames 0..130, then later
    # non-opening frames that must NOT be used as lead-ins.
    defp opening_frames do
      entry = Enum.map(0..89, &frame(&1, 324, &1 + 1))
      descent = Enum.map(90..129, &frame(&1, 29, &1 - 89))
      late = Enum.map(300..340, &frame(&1, 14, &1 - 299))
      entry ++ descent ++ late
    end

    test "grafts crouch tails onto opening lead-ins at the graft points" do
      out =
        RecoverySynth.build_opening(opening_frames(),
          expert: expert(),
          lead_in: 4,
          max_af: 10,
          graft_frames: [95, 110]
        )

      actions = Enum.map(out, & &1.game_state.players[1].action)
      # Two grafts, each: 4 lead + 2 squat + 10 squat_wait
      assert Enum.count(actions, &(&1 == @squat)) == 4
      assert Enum.count(actions, &(&1 == @squat_wait)) == 20
      # Lead-ins come from the opening (entry/descent actions), never the
      # late segment.
      refute 14 in actions
    end

    test "tail frame numbers continue consecutively from the graft point" do
      out =
        RecoverySynth.build_opening(opening_frames(),
          expert: expert(),
          lead_in: 4,
          max_af: 6,
          graft_frames: [100]
        )

      numbers = Enum.map(out, & &1.game_state.frame)
      # Whole block strictly consecutive: lead (real, consecutive) then tail
      # (graft frame + 1, +2, ...) — the prev-action threading contract.
      assert numbers == Enum.to_list(hd(numbers)..(hd(numbers) + length(numbers) - 1))
    end

    test "extra_sources with frame-number resets are split into openings" do
      # Two concatenated "replays", each with its own opening.
      extra = opening_frames() ++ opening_frames()

      out =
        RecoverySynth.build_opening([],
          expert: expert(),
          lead_in: 4,
          max_af: 5,
          graft_frames: [100],
          extra_sources: extra
        )

      # One graft per source opening -> two blocks.
      assert Enum.count(Enum.map(out, & &1.game_state.players[1].action), &(&1 == @squat)) == 4
    end
  end

  describe "build/2 (regression smoke)" do
    test "extends reflector segments past the fixture's af range" do
      out =
        RecoverySynth.build(fixture_frames(),
          expert: expert(),
          actions: [@reflector..@reflector],
          lead_in: 4,
          max_af: 10,
          ratio: 1.0
        )

      afs =
        out
        |> Enum.filter(&(&1.game_state.players[1].action == @reflector))
        |> Enum.map(&trunc(&1.game_state.players[1].action_frame))

      assert Enum.max(afs) == 10
    end
  end
end
