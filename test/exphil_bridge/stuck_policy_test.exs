defmodule ExPhil.Bridge.StuckPolicyTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.StuckPolicy

  # One test per input class of (SceneEvidence, Traffic, blind_done?).

  defp report(scene, traffic), do: %{ram_scene: scene, ram_traffic_delta: traffic}

  describe "verdict/2 — alarm classes" do
    test "no watcher: legacy alarm path, regardless of anything else" do
      assert StuckPolicy.verdict(report(:no_watcher, nil), true) == :alarm
      assert StuckPolicy.verdict(report(:no_watcher, 100), true) == :alarm
    end

    test "unreadable or zero traffic always alarms — a dead core is never a hold" do
      legit = {:settled, :slippi_online_css}
      assert StuckPolicy.verdict(report(legit, nil), true) == :alarm
      assert StuckPolicy.verdict(report(legit, 0), true) == :alarm
      assert StuckPolicy.verdict(report({:leaving, :a, :b}, 0), true) == :alarm
    end

    test "the bot14 wedge class: online CSS with the fallback NOT done" do
      # The 08-22 real bug (post-game CSS unpicked forever) fired with
      # exactly this evidence — settled online CSS, healthy traffic.
      # Suppression must never eat it.
      assert StuckPolicy.verdict(report({:settled, :slippi_online_css}, 60), false) == :alarm
    end

    test "offline scenes always alarm — local menus have real feedback" do
      assert StuckPolicy.verdict(report({:settled, :character_select}, 60), true) == :alarm
      assert StuckPolicy.verdict(report({:settled, :stage_select}, 60), true) == :alarm
    end

    test "unknown scene evidence never suppresses (matchmaking minors not yet mapped)" do
      assert StuckPolicy.verdict(report(:unknown, 60), true) == :alarm
      assert StuckPolicy.verdict(report({:settled, {:unknown, 0x42}}, 60), true) == :alarm
    end
  end

  describe "verdict/2 — suppress classes (positive evidence on both axes)" do
    test "committed scene transition: the stall is a load screen" do
      assert StuckPolicy.verdict(report({:leaving, :slippi_online_css, :in_game}, 60), false) ==
               {:suppress, :transition_in_flight}
    end

    test "post-pick online hold: fallback done, waiting on the opponent" do
      assert StuckPolicy.verdict(report({:settled, :slippi_online_css}, 60), true) ==
               {:suppress, :online_wait}
    end
  end
end
