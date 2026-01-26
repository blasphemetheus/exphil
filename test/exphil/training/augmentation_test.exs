defmodule ExPhil.Training.AugmentationTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Augmentation
  alias ExPhil.Bridge.{GameState, Player, ControllerState, Projectile, Item, Nana}

  # ============================================================================
  # Test Data Helpers
  # ============================================================================

  defp sample_player(opts) do
    %Player{
      x: Keyword.get(opts, :x, 10.0),
      y: Keyword.get(opts, :y, 5.0),
      percent: Keyword.get(opts, :percent, 50.0),
      stock: Keyword.get(opts, :stock, 4),
      facing: Keyword.get(opts, :facing, 1),
      invulnerable: Keyword.get(opts, :invulnerable, false),
      # Mewtwo
      character: Keyword.get(opts, :character, 9),
      # Wait
      action: Keyword.get(opts, :action, 14),
      action_frame: Keyword.get(opts, :action_frame, 1),
      jumps_left: Keyword.get(opts, :jumps_left, 2),
      on_ground: Keyword.get(opts, :on_ground, true),
      shield_strength: Keyword.get(opts, :shield_strength, 60.0),
      hitstun_frames_left: Keyword.get(opts, :hitstun_frames_left, 0),
      speed_air_x_self: Keyword.get(opts, :speed_air_x_self, 2.0),
      speed_ground_x_self: Keyword.get(opts, :speed_ground_x_self, 1.5),
      speed_y_self: Keyword.get(opts, :speed_y_self, 0.0),
      speed_x_attack: Keyword.get(opts, :speed_x_attack, 0.5),
      speed_y_attack: Keyword.get(opts, :speed_y_attack, 0.0),
      nana: Keyword.get(opts, :nana, nil)
    }
  end

  defp sample_game_state(opts \\ []) do
    %GameState{
      frame: Keyword.get(opts, :frame, 100),
      stage: Keyword.get(opts, :stage, :final_destination),
      players:
        Keyword.get(opts, :players, %{
          1 => sample_player(x: 10.0, facing: 1),
          2 => sample_player(x: -15.0, facing: -1)
        }),
      projectiles: Keyword.get(opts, :projectiles, []),
      items: Keyword.get(opts, :items, [])
    }
  end

  defp sample_controller(opts \\ []) do
    %ControllerState{
      main_stick: Keyword.get(opts, :main_stick, %{x: 0.7, y: 0.5}),
      c_stick: Keyword.get(opts, :c_stick, %{x: 0.5, y: 0.5}),
      l_shoulder: Keyword.get(opts, :l_shoulder, 0.0),
      r_shoulder: Keyword.get(opts, :r_shoulder, 0.0),
      button_a: Keyword.get(opts, :button_a, false),
      button_b: Keyword.get(opts, :button_b, false),
      button_x: Keyword.get(opts, :button_x, false),
      button_y: Keyword.get(opts, :button_y, false),
      button_z: Keyword.get(opts, :button_z, false),
      button_l: Keyword.get(opts, :button_l, false),
      button_r: Keyword.get(opts, :button_r, false),
      button_d_up: Keyword.get(opts, :button_d_up, false)
    }
  end

  defp sample_frame(opts \\ []) do
    %{
      game_state: Keyword.get(opts, :game_state, sample_game_state()),
      controller: Keyword.get(opts, :controller, sample_controller())
    }
  end

  # ============================================================================
  # Mirror Tests
  # ============================================================================

  describe "mirror/1" do
    test "flips player X positions" do
      game_state =
        sample_game_state(
          players: %{
            1 => sample_player(x: 10.0),
            2 => sample_player(x: -15.0)
          }
        )

      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].x == -10.0
      assert mirrored.game_state.players[2].x == 15.0
    end

    test "flips player facing directions" do
      game_state =
        sample_game_state(
          players: %{
            1 => sample_player(facing: 1),
            2 => sample_player(facing: -1)
          }
        )

      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].facing == -1
      assert mirrored.game_state.players[2].facing == 1
    end

    test "flips player X velocities" do
      game_state =
        sample_game_state(
          players: %{
            1 =>
              sample_player(speed_air_x_self: 5.0, speed_ground_x_self: 3.0, speed_x_attack: 2.0)
          }
        )

      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].speed_air_x_self == -5.0
      assert mirrored.game_state.players[1].speed_ground_x_self == -3.0
      assert mirrored.game_state.players[1].speed_x_attack == -2.0
    end

    test "mirrors controller stick X values" do
      controller = sample_controller(main_stick: %{x: 0.7, y: 0.3}, c_stick: %{x: 0.2, y: 0.8})
      frame = sample_frame(controller: controller)

      mirrored = Augmentation.mirror(frame)

      # Stick X is in [0, 1] range, so mirroring is 1.0 - x
      # Use assert_in_delta for floating-point comparisons
      assert_in_delta mirrored.controller.main_stick.x, 0.3, 0.0001
      # Y unchanged
      assert_in_delta mirrored.controller.main_stick.y, 0.3, 0.0001
      assert_in_delta mirrored.controller.c_stick.x, 0.8, 0.0001
      # Y unchanged
      assert_in_delta mirrored.controller.c_stick.y, 0.8, 0.0001
    end

    test "mirrors Nana position and facing for Ice Climbers" do
      nana = %Nana{x: 5.0, y: 10.0, facing: 1, percent: 30.0, stock: 1}
      player = sample_player(x: 10.0, facing: 1, nana: nana)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].nana.x == -5.0
      assert mirrored.game_state.players[1].nana.facing == -1
      # Y unchanged
      assert mirrored.game_state.players[1].nana.y == 10.0
    end

    test "handles nil Nana gracefully" do
      player = sample_player(nana: nil)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].nana == nil
    end

    test "mirrors projectile positions" do
      projectile = %Projectile{
        x: 20.0,
        y: 5.0,
        speed_x: 10.0,
        speed_y: 2.0,
        owner: 1,
        type: 1,
        subtype: 0
      }

      game_state = sample_game_state(projectiles: [projectile])
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      [mirrored_proj] = mirrored.game_state.projectiles
      assert mirrored_proj.x == -20.0
      # Y unchanged
      assert mirrored_proj.y == 5.0
      assert mirrored_proj.speed_x == -10.0
    end

    test "mirrors item positions" do
      item = %Item{
        x: -8.0,
        y: 3.0,
        type: 1,
        facing: 1,
        owner: 1,
        held_by: nil,
        spawn_id: 0,
        timer: 100
      }

      game_state = sample_game_state(items: [item])
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      [mirrored_item] = mirrored.game_state.items
      assert mirrored_item.x == 8.0
      # Y unchanged
      assert mirrored_item.y == 3.0
    end

    test "handles frame without controller" do
      frame = %{game_state: sample_game_state()}

      mirrored = Augmentation.mirror(frame)

      # Should still mirror game state
      assert mirrored.game_state.players[1].x == -10.0
    end

    test "preserves non-positional data" do
      # 44 = some attack action
      player = sample_player(percent: 75.0, stock: 3, action: 44, on_ground: false)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players[1].percent == 75.0
      assert mirrored.game_state.players[1].stock == 3
      assert mirrored.game_state.players[1].action == 44
      assert mirrored.game_state.players[1].on_ground == false
    end
  end

  describe "maybe_mirror/2" do
    test "always mirrors when probability is 1.0" do
      frame = sample_frame()

      # With probability 1.0, should always mirror
      results =
        for _ <- 1..10 do
          Augmentation.maybe_mirror(frame, probability: 1.0)
        end

      # All results should have flipped X
      assert Enum.all?(results, fn r -> r.game_state.players[1].x == -10.0 end)
    end

    test "never mirrors when probability is 0.0" do
      frame = sample_frame()

      # With probability 0.0, should never mirror
      results =
        for _ <- 1..10 do
          Augmentation.maybe_mirror(frame, probability: 0.0)
        end

      # All results should have original X
      assert Enum.all?(results, fn r -> r.game_state.players[1].x == 10.0 end)
    end

    test "mirrors approximately half the time with probability 0.5" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()

      results =
        for _ <- 1..100 do
          Augmentation.maybe_mirror(frame, probability: 0.5)
        end

      # Count how many were mirrored (x = -10.0)
      mirrored_count = Enum.count(results, fn r -> r.game_state.players[1].x == -10.0 end)

      # Should be roughly 50%, allow for statistical variance (30-70 range)
      assert mirrored_count >= 30 and mirrored_count <= 70
    end

    test "uses default probability of 0.5" do
      # Just verify it doesn't crash with default opts
      frame = sample_frame()
      result = Augmentation.maybe_mirror(frame)
      assert is_map(result)
      assert Map.has_key?(result, :game_state)
    end
  end

  # ============================================================================
  # Noise Tests
  # ============================================================================

  describe "add_noise/2" do
    test "adds noise to player positions" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()
      original_x = frame.game_state.players[1].x
      original_y = frame.game_state.players[1].y

      noisy = Augmentation.add_noise(frame, scale: 0.1)

      # Values should be different (very unlikely to be exactly the same with noise)
      assert noisy.game_state.players[1].x != original_x
      assert noisy.game_state.players[1].y != original_y

      # But should be close (within reasonable range for scale=0.1)
      assert abs(noisy.game_state.players[1].x - original_x) < 1.0
      assert abs(noisy.game_state.players[1].y - original_y) < 1.0
    end

    test "adds noise to player percent" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      player = sample_player(percent: 100.0)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      noisy = Augmentation.add_noise(frame, scale: 0.1)

      # Percent uses scale * 10, so larger variance
      assert noisy.game_state.players[1].percent != 100.0
    end

    test "ensures percent stays non-negative" do
      :rand.seed(:exsss, {99999, 99999, 99999})
      # Low percent
      player = sample_player(percent: 0.5)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      # Large scale noise on low percent
      noisy = Augmentation.add_noise(frame, scale: 1.0)

      # Percent should be clamped to >= 0
      assert noisy.game_state.players[1].percent >= 0.0
    end

    test "adds noise to velocities" do
      :rand.seed(:exsss, {12345, 12345, 12345})

      player =
        sample_player(
          speed_air_x_self: 5.0,
          speed_ground_x_self: 3.0,
          speed_y_self: 2.0
        )

      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      noisy = Augmentation.add_noise(frame, scale: 0.1)

      # Velocities should have noise
      assert noisy.game_state.players[1].speed_air_x_self != 5.0
      assert noisy.game_state.players[1].speed_ground_x_self != 3.0
      assert noisy.game_state.players[1].speed_y_self != 2.0
    end

    test "does NOT add noise to discrete values" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      # 44 = some attack action
      player = sample_player(stock: 3, action: 44, on_ground: true, facing: 1)
      game_state = sample_game_state(players: %{1 => player})
      frame = sample_frame(game_state: game_state)

      # Large scale
      noisy = Augmentation.add_noise(frame, scale: 1.0)

      # Discrete values unchanged
      assert noisy.game_state.players[1].stock == 3
      assert noisy.game_state.players[1].action == 44
      assert noisy.game_state.players[1].on_ground == true
      assert noisy.game_state.players[1].facing == 1
    end

    test "uses default scale of 0.01" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()

      noisy = Augmentation.add_noise(frame)

      # Should add small noise
      assert noisy.game_state.players[1].x != frame.game_state.players[1].x

      # But very small change with default scale
      diff = abs(noisy.game_state.players[1].x - frame.game_state.players[1].x)
      assert diff < 0.1
    end
  end

  describe "maybe_add_noise/2" do
    test "always adds noise when probability is 1.0" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()
      original_x = frame.game_state.players[1].x

      results =
        for _ <- 1..10 do
          Augmentation.maybe_add_noise(frame, probability: 1.0, scale: 0.1)
        end

      # All should have different X values (noise added)
      assert Enum.all?(results, fn r -> r.game_state.players[1].x != original_x end)
    end

    test "never adds noise when probability is 0.0" do
      frame = sample_frame()
      original_x = frame.game_state.players[1].x

      results =
        for _ <- 1..10 do
          Augmentation.maybe_add_noise(frame, probability: 0.0, scale: 1.0)
        end

      # All should have same X value (no noise)
      assert Enum.all?(results, fn r -> r.game_state.players[1].x == original_x end)
    end
  end

  # ============================================================================
  # Combined Augmentation Tests
  # ============================================================================

  describe "augment/2" do
    test "applies both mirror and noise with their probabilities" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()

      # With both at 1.0, both should apply
      augmented =
        Augmentation.augment(frame,
          mirror_prob: 1.0,
          noise_prob: 1.0,
          noise_scale: 0.1
        )

      # Should be mirrored (negative X)
      assert augmented.game_state.players[1].x < 0

      # Should have noise (not exactly -10.0)
      assert augmented.game_state.players[1].x != -10.0
    end

    test "only mirrors when noise_prob is 0" do
      frame = sample_frame()

      augmented =
        Augmentation.augment(frame,
          mirror_prob: 1.0,
          noise_prob: 0.0,
          noise_scale: 0.1
        )

      # Should be mirrored
      # Exactly mirrored, no noise
      assert augmented.game_state.players[1].x == -10.0
    end

    test "only adds noise when mirror_prob is 0" do
      :rand.seed(:exsss, {12345, 12345, 12345})
      frame = sample_frame()

      augmented =
        Augmentation.augment(frame,
          mirror_prob: 0.0,
          noise_prob: 1.0,
          noise_scale: 0.1
        )

      # Should NOT be mirrored (positive X)
      assert augmented.game_state.players[1].x > 0

      # Should have noise (not exactly 10.0)
      assert augmented.game_state.players[1].x != 10.0
    end

    test "returns unchanged when both probabilities are 0" do
      frame = sample_frame()

      augmented =
        Augmentation.augment(frame,
          mirror_prob: 0.0,
          noise_prob: 0.0
        )

      # Should be identical
      assert augmented.game_state.players[1].x == 10.0
    end

    test "uses default probabilities" do
      # Just verify it doesn't crash with defaults
      frame = sample_frame()
      result = Augmentation.augment(frame)
      assert is_map(result)
    end
  end

  # ============================================================================
  # Edge Cases
  # ============================================================================

  describe "edge cases" do
    test "handles empty player map" do
      game_state = sample_game_state(players: %{})
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.players == %{}
    end

    test "handles nil projectiles list" do
      game_state = %{sample_game_state() | projectiles: nil}
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.projectiles == []
    end

    test "handles nil items list" do
      game_state = %{sample_game_state() | items: nil}
      frame = sample_frame(game_state: game_state)

      mirrored = Augmentation.mirror(frame)

      assert mirrored.game_state.items == []
    end

    test "handles nil controller gracefully" do
      frame = %{game_state: sample_game_state(), controller: nil}

      mirrored = Augmentation.mirror(frame)

      assert mirrored.controller == nil
    end

    test "handles plain map game state (from replays)" do
      # Replay parsing sometimes produces plain maps instead of structs
      game_state = %{
        players: %{
          1 => %{x: 10.0, y: 5.0, facing: 1, percent: 50.0},
          2 => %{x: -5.0, y: 3.0, facing: -1, percent: 30.0}
        },
        frame: 100
      }

      frame = %{game_state: game_state}

      mirrored = Augmentation.mirror(frame)

      # Should still mirror plain maps
      assert mirrored.game_state.players[1].x == -10.0 or
               mirrored.game_state.players[1][:x] == -10.0
    end
  end
end
