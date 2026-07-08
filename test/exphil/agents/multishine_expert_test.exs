defmodule ExPhil.Agents.MultishineExpertTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.MultishineExpert
  alias ExPhil.Bridge.ControllerState

  @fixture "test/fixtures/replays/fox_multishine_closed.slp"

  setup_all do
    expert = MultishineExpert.from_fixture(@fixture)

    {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)
    frames = ExPhil.Data.Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)

    %{expert: expert, frames: frames}
  end

  defp player(action, action_frame, on_ground) do
    %{action: action * 1.0, action_frame: action_frame * 1.0, on_ground: on_ground}
  end

  describe "table (happy path from fixture)" do
    test "covers the core 9-frame loop states", %{expert: expert} do
      # jumpsquat 3 frames, ground reflector, aerial shine, reflector-open
      for key <- [
            {24, 0, true},
            {24, 1, true},
            {24, 2, true},
            {361, 1, true},
            {361, 2, true},
            {365, 1, false},
            {365, 2, false},
            {366, 0, false}
          ] do
        assert Map.has_key?(expert.table, key), "missing core loop key #{inspect(key)}"
      end
    end

    test "jumpsquat af0 = jump press landing, stick held down", %{expert: expert} do
      {:ok, c} = MultishineExpert.label(expert, player(24, 0, true))
      assert c.button_x
      refute c.button_b
      assert c.main_stick.y < 0.25
    end

    test "aerial shine start = down-B landing", %{expert: expert} do
      {:ok, c} = MultishineExpert.label(expert, player(365, 1, false))
      assert c.button_b
      assert c.main_stick.y < 0.25
    end

    test "aerial shine af2 releases to neutral (fixture nuance)", %{expert: expert} do
      {:ok, c} = MultishineExpert.label(expert, player(365, 2, false))
      refute c.button_b
      refute c.button_x
      assert_in_delta c.main_stick.y, 0.5, 0.1
    end
  end

  describe "recovery rules (states the fixture never visits)" do
    test "missed jump-cancel: grounded reflector af>=3 taps jump against prev press", %{expert: expert} do
      # the live delay-probe freeze state: reflector af8+, grounded.
      # Jump registers on a press EDGE, so the label alternates against the
      # previously-landed input (what the prev-action channel shows)
      x_up = ControllerState.neutral()
      x_down = %{ControllerState.neutral() | button_x: true}

      for af <- [4, 8, 30], prev <- [nil, x_up] do
        {:ok, c} = MultishineExpert.label(expert, player(361, af, true), prev)
        assert c.button_x, "expected jump-cancel press at reflector af#{af}"
        refute c.button_b
      end

      for af <- [4, 8, 30] do
        {:ok, c} = MultishineExpert.label(expert, player(361, af, true), x_down)
        refute c.button_x, "expected release (edge setup) at reflector af#{af}"
        refute c.button_b
        assert c.main_stick.y < 0.25, "stick stays down between taps"
      end
    end

    test "empty hop: airborne non-reflector taps shine against prev press", %{expert: expert} do
      # JumpF at frames the fixture never reaches
      {:ok, c} = MultishineExpert.label(expert, player(25, 10, false))
      assert c.button_b
      assert c.main_stick.y < 0.25

      b_down = %{ControllerState.neutral() | button_b: true}
      {:ok, c} = MultishineExpert.label(expert, player(25, 10, false), b_down)
      refute c.button_b
    end

    test "aerial reflector past the table rides down neutral", %{expert: expert} do
      {:ok, c} = MultishineExpert.label(expert, player(366, 40, false))
      refute c.button_b
      refute c.button_x
      assert c == ControllerState.neutral()
    end

    test "grounded neutral states tap a shine against prev press", %{expert: expert} do
      # Wait (14) and Dash (20) — B pressed unless it was already down
      b_down = %{ControllerState.neutral() | button_b: true}

      for action <- [14, 20] do
        {:ok, c} = MultishineExpert.label(expert, player(action, 1, true))
        assert c.button_b, "expected shine press from grounded action #{action}"
        assert c.main_stick.y < 0.25

        {:ok, c} = MultishineExpert.label(expert, player(action, 1, true), b_down)
        refute c.button_b, "expected release (edge setup) when B was held"
      end
    end

    test "dead/respawn states are skipped", %{expert: expert} do
      assert :skip = MultishineExpert.label(expert, player(0, -1, false))
      assert :skip = MultishineExpert.label(expert, player(12, 3, false))
    end
  end

  describe "fidelity on expert-quality data" do
    test "labels reproduce the fixture's own recorded buttons", %{expert: expert, frames: frames} do
      agreement = MultishineExpert.button_agreement(expert, frames)
      assert agreement > 0.95, "button agreement only #{Float.round(agreement, 3)}"
    end
  end
end
