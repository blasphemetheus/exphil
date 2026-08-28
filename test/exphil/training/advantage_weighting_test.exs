defmodule ExPhil.Training.AdvantageWeightingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Constants
  alias ExPhil.Bridge.{GameState, Player}
  alias ExPhil.Training.AdvantageWeighting, as: AW

  @ground_shine Constants.reflector_ground() |> Enum.at(0)
  @air_shine Constants.reflector_air() |> Enum.at(0)
  @jumpsquat Constants.jumpsquat()
  @idle 14

  defp frame(action), do: %{game_state: %{players: %{1 => %{action: action}}}}

  # One multishine cycle: 3 grounded shine frames, jumpsquat, aerial shine
  defp cycle, do: [@ground_shine, @ground_shine, @ground_shine, @jumpsquat, @air_shine]

  defp frames(actions), do: Enum.map(actions, &frame/1)

  describe "rewards/3" do
    test "+1 only on entry into the grounded reflector family" do
      actions = cycle() ++ cycle() ++ [@idle, @idle]
      rs = AW.rewards(frames(actions))

      assert Enum.sum(rs) == 2.0
      # entries at index 0 and at the start of the second cycle
      assert Enum.at(rs, 0) == 1.0
      assert Enum.at(rs, 1) == 0.0
      assert Enum.at(rs, length(cycle())) == 1.0
    end

    test "air-shine entries credited only when air_shine_reward is set" do
      actions = [@idle, @air_shine, @air_shine, @idle]
      assert AW.rewards(frames(actions)) |> Enum.sum() == 0.0
      assert AW.rewards(frames(actions), 1, 0.3) |> Enum.sum() == 0.3
    end
  end

  describe "return_to_go/3" do
    test "matches the direct definition on a small case" do
      rewards = [1.0, 0.0, 1.0, 0.0, 0.0]
      gamma = 0.5
      horizon = 2

      direct = fn t ->
        for k <- 0..horizon, t + k < length(rewards), reduce: 0.0 do
          acc -> acc + :math.pow(gamma, k) * Enum.at(rewards, t + k)
        end
      end

      rtg = AW.return_to_go(rewards, gamma, horizon)

      for t <- 0..4 do
        assert_in_delta Enum.at(rtg, t), direct.(t), 1.0e-9
      end
    end

    test "horizon bounds the lookahead" do
      rewards = List.duplicate(0.0, 10) ++ [1.0]
      # horizon 3: frames 0..6 see nothing, 7..10 see the reward
      rtg = AW.return_to_go(rewards, 1.0, 3)
      assert Enum.at(rtg, 6) == 0.0
      assert Enum.at(rtg, 7) == 1.0
    end
  end

  describe "frame_weights/2" do
    # A pool with contrast: one list that multishines, one that idles
    defp contrast_pool do
      chain = frames(Enum.flat_map(1..20, fn _ -> cycle() end) ++ List.duplicate(@idle, 100))
      idle = frames(List.duplicate(@idle, 200))
      [chain, idle]
    end

    test "chain frames outweigh idle-tail frames; mean is 1" do
      {weights, stats} = AW.frame_weights(contrast_pool())

      assert stats.frames == 400
      assert stats.shine_entries == 20
      assert_in_delta Enum.sum(weights) / length(weights), 1.0, 1.0e-6

      # chain list = frames 0..199 (cycles through 99, idle 100..199)
      chain_head = Enum.take(weights, 50)
      idle_tail = Enum.slice(weights, 150..199)
      assert Enum.sum(chain_head) / 50 > Enum.sum(idle_tail) / 50
    end

    test "weights respect the clip band (pre-normalization ratio bounded)" do
      {weights, _} = AW.frame_weights(contrast_pool(), clip: {0.2, 5.0})
      {lo, hi} = Enum.min_max(weights)
      # after mean-normalization the absolute band shifts, but the
      # max/min ratio can never exceed the clip ratio
      assert hi / lo <= 5.0 / 0.2 + 1.0e-6
    end

    test "percentile beta lands the weight ratio near the target" do
      {_weights, stats} = AW.frame_weights(contrast_pool(), weight_ratio: 7.0)
      assert stats.weight_ratio
      # clip can compress it; it must not exceed the target band wildly
      assert stats.weight_ratio <= 25.0
      assert stats.weight_ratio >= 1.5
    end

    test "signal-free pool degenerates to uniform weights" do
      idle_only = [frames(List.duplicate(@idle, 50))]
      {weights, stats} = AW.frame_weights(idle_only)
      assert Enum.all?(weights, &(abs(&1 - 1.0) < 1.0e-9))
      assert stats.flat_lists == 1
    end

    test "shuffle (B3) preserves the per-list weight multiset, deterministically" do
      pool = contrast_pool()
      {plain, _} = AW.frame_weights(pool)
      {shuffled, _} = AW.frame_weights(pool, shuffle: true, seed: 7)
      {shuffled2, _} = AW.frame_weights(pool, shuffle: true, seed: 7)

      assert shuffled == shuffled2
      assert shuffled != plain

      # per-list multiset identical (list 1 = first 200 frames)
      assert Enum.sort(Enum.take(plain, 200)) == Enum.sort(Enum.take(shuffled, 200))
      assert Enum.sort(Enum.drop(plain, 200)) == Enum.sort(Enum.drop(shuffled, 200))
    end
  end

  # --- standard-reward arm (OFFLINE_RL_SPEC generalist) ---

  defp std_player(percent, stock), do: %Player{percent: percent, stock: stock}
  defp std_gs(own, opp), do: %GameState{players: %{1 => own, 2 => opp}}
  defp std_frame(own, opp), do: %{game_state: std_gs(own, opp)}

  describe "standard_rewards/2" do
    test "damage dealt scales by 0.01; last frame is 0.0" do
      f0 = std_frame(std_player(0, 4), std_player(0, 4))
      f1 = std_frame(std_player(0, 4), std_player(50, 4))
      f2 = std_frame(std_player(0, 4), std_player(80, 4))

      rs = AW.standard_rewards([f0, f1, f2])

      assert length(rs) == 3
      assert_in_delta Enum.at(rs, 0), 0.5, 1.0e-6
      assert_in_delta Enum.at(rs, 1), 0.3, 1.0e-6
      assert Enum.at(rs, 2) == 0.0
    end

    test "taking a stock at 0% is +1.0 (no damage credit)" do
      f0 = std_frame(std_player(0, 4), std_player(0, 4))
      f1 = std_frame(std_player(0, 4), std_player(0, 3))

      rs = AW.standard_rewards([f0, f1])
      assert_in_delta hd(rs), 1.0, 1.0e-6
    end

    test "losing a stock at 0% is -1.0" do
      f0 = std_frame(std_player(0, 4), std_player(0, 4))
      f1 = std_frame(std_player(0, 3), std_player(0, 4))

      rs = AW.standard_rewards([f0, f1])
      assert_in_delta hd(rs), -1.0, 1.0e-6
    end
  end

  describe "frame_weights/2 with :standard reward" do
    test "produces mean-1 weights and marks the reward family" do
      good = [
        std_frame(std_player(0, 4), std_player(0, 4)),
        std_frame(std_player(0, 4), std_player(50, 4)),
        std_frame(std_player(0, 4), std_player(0, 3)),
        std_frame(std_player(0, 4), std_player(0, 3))
      ]

      idle = [
        std_frame(std_player(0, 4), std_player(0, 4)),
        std_frame(std_player(0, 4), std_player(0, 4))
      ]

      {weights, stats} = AW.frame_weights([good, idle], reward: :standard)

      assert stats.reward == :standard
      assert stats.frames == 6
      assert_in_delta Enum.sum(weights) / length(weights), 1.0, 1.0e-6
    end
  end
end
