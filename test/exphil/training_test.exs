defmodule ExPhil.TrainingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training
  alias ExPhil.Training.{Imitation, PPO, Data}
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  # Helper to create mock game state
  defp mock_game_state(opts) do
    player = %Player{
      x: Keyword.get(opts, :x, 0.0),
      y: Keyword.get(opts, :y, 0.0),
      percent: Keyword.get(opts, :percent, 0.0),
      stock: 4,
      facing: 1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 9,  # Mewtwo
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    opponent = %Player{
      x: 10.0,
      y: 0.0,
      percent: 30.0,
      stock: 4,
      facing: -1,
      action: 14,
      action_frame: 0,
      shield_strength: 60.0,
      character: 2,  # Fox
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }

    %GameState{
      frame: 0,
      stage: 2,
      menu_state: 2,
      players: %{1 => player, 2 => opponent},
      projectiles: []
    }
  end

  defp mock_controller_state(opts) do
    %ControllerState{
      main_stick: %{
        x: Keyword.get(opts, :main_x, 0.5),
        y: Keyword.get(opts, :main_y, 0.5)
      },
      c_stick: %{x: 0.5, y: 0.5},
      l_shoulder: Keyword.get(opts, :l_shoulder, 0.0),
      r_shoulder: 0.0,
      button_a: Keyword.get(opts, :a, false),
      button_b: Keyword.get(opts, :b, false),
      button_x: Keyword.get(opts, :x, false),
      button_y: false,
      button_z: false,
      button_l: false,
      button_r: false,
      button_d_up: false
    }
  end

  defp mock_frames(n) do
    Enum.map(1..n, fn i ->
      %{
        game_state: mock_game_state(x: i * 1.0, percent: i * 0.5),
        controller: mock_controller_state(
          main_x: 0.5 + rem(i, 10) * 0.05,
          a: rem(i, 5) == 0
        )
      }
    end)
  end

  describe "Data module" do
    test "from_frames/2 creates dataset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      assert dataset.size == 100
      assert length(dataset.frames) == 100
    end

    test "batched/2 creates batches of correct size" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      batches = Data.batched(dataset, batch_size: 32, shuffle: false)
      batch_list = Enum.to_list(batches)

      # 100 frames -> 4 batches (32, 32, 32, 4)
      assert length(batch_list) == 4
      assert Nx.axis_size(hd(batch_list).states, 0) == 32
    end

    test "batched/2 with drop_last removes incomplete batch" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      batches = Data.batched(dataset, batch_size: 32, drop_last: true)
      batch_list = Enum.to_list(batches)

      # Only 3 complete batches of 32
      assert length(batch_list) == 3
    end

    test "split/2 splits dataset correctly" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {train, val} = Data.split(dataset, ratio: 0.8, shuffle: false)

      assert train.size == 80
      assert val.size == 20
    end

    test "controller_to_action/2 discretizes correctly" do
      controller = %ControllerState{
        main_stick: %{x: 0.0, y: 1.0},  # Full left, full up
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.5,
        r_shoulder: 0.0,
        button_a: true,
        button_b: false,
        button_x: true,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      }

      action = Data.controller_to_action(controller, axis_buckets: 16, shoulder_buckets: 4)

      assert action.main_x == 0  # 0.0 -> bucket 0
      assert action.main_y == 15  # 1.0 -> bucket 15 (clamped)
      assert action.shoulder == 2  # 0.5 -> bucket 2
      assert action.buttons.a == true
      assert action.buttons.x == true
      assert action.buttons.b == false
    end

    test "actions_to_tensors/1 creates proper tensor shapes" do
      actions = [
        %{buttons: %{a: true, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
          main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0},
        %{buttons: %{a: false, b: true, x: false, y: false, z: false, l: false, r: false, d_up: false},
          main_x: 0, main_y: 15, c_x: 8, c_y: 8, shoulder: 3}
      ]

      tensors = Data.actions_to_tensors(actions)

      assert Nx.shape(tensors.buttons) == {2, 8}
      assert Nx.shape(tensors.main_x) == {2}
      assert Nx.shape(tensors.main_y) == {2}
      assert Nx.shape(tensors.shoulder) == {2}
    end

    test "concat/1 combines datasets" do
      frames1 = mock_frames(50)
      frames2 = mock_frames(30)

      ds1 = Data.from_frames(frames1)
      ds2 = Data.from_frames(frames2)

      combined = Data.concat([ds1, ds2])

      assert combined.size == 80
    end

    test "sample/2 samples subset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      sampled = Data.sample(dataset, 20)

      assert sampled.size == 20
    end

    test "stats/1 computes dataset statistics" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      stats = Data.stats(dataset)

      assert stats.size == 100
      assert is_map(stats.button_rates)
      assert is_list(stats.main_x_distribution)
      assert length(stats.main_x_distribution) == 16
    end
  end

  describe "Imitation module" do
    @tag :slow
    test "new/1 creates trainer with default config" do
      trainer = Imitation.new(embed_size: 512)

      assert trainer.step == 0
      assert trainer.config.batch_size == 64
      assert trainer.config.learning_rate == 1.0e-4
    end

    @tag :slow
    test "new/1 accepts custom config" do
      trainer = Imitation.new(
        embed_size: 1024,
        batch_size: 128,
        learning_rate: 3.0e-4,
        hidden_sizes: [256, 256]
      )

      assert trainer.config.batch_size == 128
      assert trainer.config.learning_rate == 3.0e-4
    end

    @tag :slow
    test "train_step/3 builds loss function" do
      # Note: Full gradient computation requires Axon.Loop for proper
      # ModelState handling in newer Axon versions. This test verifies
      # the loss function can be built and executed.
      trainer = Imitation.new(embed_size: 512)
      {predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      # Verify forward pass works
      key = Nx.Random.key(42)
      {states, _key} = Nx.Random.uniform(key, shape: {4, 512})

      # Forward pass
      {buttons, main_x, _main_y, _c_x, _c_y, shoulder} = predict_fn.(trainer.policy_params, states)

      assert Nx.shape(buttons) == {4, 8}
      # axis_buckets + 1 because we discretize [0, 1] into bucket indices
      assert Nx.shape(main_x) == {4, 17}  # axis_buckets + 1
      assert Nx.shape(shoulder) == {4, 5}  # shoulder_buckets + 1

      # Loss computation
      actions = %{
        buttons: Nx.broadcast(0, {4, 8}),
        main_x: Nx.broadcast(8, {4}),
        main_y: Nx.broadcast(8, {4}),
        c_x: Nx.broadcast(8, {4}),
        c_y: Nx.broadcast(8, {4}),
        shoulder: Nx.broadcast(0, {4})
      }

      loss = loss_fn.(trainer.policy_params, states, actions)
      assert is_struct(loss, Nx.Tensor)
      assert Nx.shape(loss) == {}  # Scalar
    end

    @tag :slow
    test "metrics_summary/1 computes summary" do
      trainer = Imitation.new(embed_size: 512)

      # Add some fake metrics
      trainer = %{trainer |
        step: 100,
        metrics: %{
          loss: [0.5, 0.4, 0.3],
          button_loss: [],
          stick_loss: [],
          learning_rate: []
        }
      }

      summary = Imitation.metrics_summary(trainer)

      assert summary.step == 100
      assert_in_delta summary.avg_loss, 0.4, 0.001
      assert summary.min_loss == 0.3
      assert summary.max_loss == 0.5
    end

    @tag :slow
    test "get_action/3 returns action samples" do
      trainer = Imitation.new(embed_size: 512)

      key = Nx.Random.key(42)
      {state, _key} = Nx.Random.uniform(key, shape: {1, 512})

      action = Imitation.get_action(trainer, state)

      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :main_y)
      assert Map.has_key?(action, :shoulder)
    end

    @tag :slow
    test "get_controller_action/3 returns ControllerState" do
      trainer = Imitation.new(embed_size: 512)

      key = Nx.Random.key(42)
      {state, _key} = Nx.Random.uniform(key, shape: {1, 512})

      controller = Imitation.get_controller_action(trainer, state)

      assert %ControllerState{} = controller
      assert is_map(controller.main_stick)
      assert is_float(controller.main_stick.x)
      assert is_boolean(controller.button_a)
    end
  end

  describe "PPO module" do
    @tag :slow
    test "new/1 creates trainer with default config" do
      trainer = PPO.new(embed_size: 512)

      assert trainer.step == 0
      assert trainer.config.gamma == 0.99
      assert trainer.config.clip_range == 0.2
    end

    @tag :slow
    test "new/1 accepts custom config" do
      trainer = PPO.new(
        embed_size: 1024,
        gamma: 0.995,
        clip_range: 0.1,
        entropy_coef: 0.05
      )

      assert trainer.config.gamma == 0.995
      assert trainer.config.clip_range == 0.1
      assert trainer.config.entropy_coef == 0.05
    end

    @tag :slow
    test "get_action/3 returns action samples" do
      trainer = PPO.new(embed_size: 512)

      key = Nx.Random.key(42)
      {state, _key} = Nx.Random.uniform(key, shape: {1, 512})

      result = PPO.get_action(trainer, state)

      # Check all action components are present
      assert Map.has_key?(result, :buttons)
      assert Map.has_key?(result, :main_x)
      assert Map.has_key?(result, :main_y)
      assert Map.has_key?(result, :c_x)
      assert Map.has_key?(result, :c_y)
      assert Map.has_key?(result, :shoulder)
      assert Map.has_key?(result, :value)
    end

    @tag :slow
    test "get_value/2 returns value estimate" do
      trainer = PPO.new(embed_size: 512)

      key = Nx.Random.key(42)
      {state, _key} = Nx.Random.uniform(key, shape: {1, 512})

      value = PPO.get_value(trainer, state)

      assert is_float(value)
    end

    @tag :slow
    test "metrics_summary/1 computes summary" do
      trainer = PPO.new(embed_size: 512)

      trainer = %{trainer |
        step: 1000,
        timesteps: 10_000,
        metrics: %{
          losses: [0.5, 0.4, 0.45],
          policy_losses: [0.1, 0.2, 0.15],
          value_losses: [0.5, 0.4, 0.45],
          entropies: [0.3, 0.25, 0.28],
          kls: [0.01, 0.02, 0.015],
          clip_fractions: [0.1, 0.1, 0.1]
        }
      }

      summary = PPO.metrics_summary(trainer)

      assert summary.step == 1000
      assert summary.timesteps == 10_000
      assert_in_delta summary.recent_loss, 0.45, 0.001
      assert_in_delta summary.recent_entropy, 0.277, 0.001
    end
  end

  describe "Training module integration" do
    test "from_frames/2 delegates to Data" do
      frames = mock_frames(50)
      dataset = Training.from_frames(frames)

      assert %Data{} = dataset
      assert dataset.size == 50
    end

    @tag :slow
    test "export_policy/2 and load_policy/1 round-trip for Imitation" do
      trainer = Imitation.new(embed_size: 256)

      # Export
      path = Path.join(System.tmp_dir!(), "test_policy_#{:rand.uniform(10_000)}.bin")

      on_exit(fn ->
        File.rm(path)
      end)

      assert :ok = Training.export_policy(trainer, path)

      # Load
      assert {:ok, loaded} = Training.load_policy(path)
      assert Map.has_key?(loaded, :params)
      assert Map.has_key?(loaded, :config)
    end

    @tag :slow
    test "export_policy/2 and load_policy/1 round-trip for PPO" do
      trainer = PPO.new(embed_size: 256, hidden_sizes: [64])

      path = Path.join(System.tmp_dir!(), "test_ppo_policy_#{:rand.uniform(10_000)}.bin")

      on_exit(fn ->
        File.rm(path)
      end)

      assert :ok = Training.export_policy(trainer, path)

      {:ok, loaded} = Training.load_policy(path)
      assert Map.has_key?(loaded, :params)
    end

    test "load_policy/1 returns error for missing file" do
      result = Training.load_policy("/nonexistent/path.bin")
      assert {:error, _} = result
    end

    test "load_policy/1 returns helpful error for corrupted file" do
      path = Path.join(System.tmp_dir!(), "corrupted_#{:rand.uniform(10_000)}.bin")
      File.write!(path, "not a valid erlang term")

      on_exit(fn -> File.rm(path) end)

      result = Training.load_policy(path)
      assert {:error, msg} = result
      assert msg =~ "corrupted"
    end

    test "validate_policy/1 accepts valid exported policy format" do
      policy = %{
        params: %{"backbone_dense_0" => %{kernel: Nx.broadcast(0, {10, 10})}},
        config: %{temporal: false, hidden_sizes: [64]}
      }

      assert :ok = Training.validate_policy(policy)
    end

    test "validate_policy/1 accepts valid checkpoint format" do
      policy = %{
        policy_params: %{"backbone_dense_0" => %{kernel: Nx.broadcast(0, {10, 10})}},
        config: %{temporal: false, hidden_sizes: [64]}
      }

      assert :ok = Training.validate_policy(policy)
    end

    test "validate_policy/1 returns error for missing params" do
      policy = %{config: %{temporal: false}}

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Missing :params"
    end

    test "validate_policy/1 returns error for missing config" do
      policy = %{params: %{}}

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Missing :config"
    end

    test "validate_policy/1 returns error for unrecognized format" do
      policy = %{something_else: true}

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Unrecognized checkpoint format"
    end

    test "validate_policy/1 detects temporal/MLP mismatch - temporal params but MLP config" do
      # Params have LSTM layers but config says temporal=false
      policy = %{
        params: %{
          "lstm_cell" => %{kernel: Nx.broadcast(0, {10, 10})},
          "backbone_dense_0" => %{kernel: Nx.broadcast(0, {10, 10})}
        },
        config: %{temporal: false, hidden_sizes: [64]}
      }

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Architecture mismatch"
      assert msg =~ "temporal=false"
      assert msg =~ "LSTM"
    end

    test "validate_policy/1 detects temporal/MLP mismatch - MLP params but temporal config" do
      # Params have only MLP layers but config says temporal=true
      policy = %{
        params: %{
          "backbone_dense_0" => %{kernel: Nx.broadcast(0, {10, 10})},
          "buttons_logits" => %{kernel: Nx.broadcast(0, {10, 8})}
        },
        config: %{temporal: true, backbone: :mamba, hidden_sizes: [64]}
      }

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Architecture mismatch"
      assert msg =~ "temporal=true"
    end

    test "validate_policy/1 detects hidden size layer count mismatch" do
      # Config says 3 hidden layers but params only have 2
      policy = %{
        params: %{
          "backbone_dense_0" => %{kernel: Nx.broadcast(0, {10, 10})},
          "backbone_dense_1" => %{kernel: Nx.broadcast(0, {10, 10})}
        },
        config: %{temporal: false, hidden_sizes: [64, 64, 64]}  # 3 layers
      }

      {:error, msg} = Training.validate_policy(policy, "test.bin")
      assert msg =~ "Hidden layer size mismatch"
    end

    test "load_policy/2 with validate: false skips validation" do
      path = Path.join(System.tmp_dir!(), "no_validate_#{:rand.uniform(10_000)}.bin")

      # Write a valid binary term but with missing config
      File.write!(path, :erlang.term_to_binary(%{params: %{}}))

      on_exit(fn -> File.rm(path) end)

      # With validation, should fail
      assert {:error, _} = Training.load_policy(path, validate: true)

      # Without validation, should succeed
      assert {:ok, _} = Training.load_policy(path, validate: false)
    end

    @tag :slow
    test "get_action/3 dispatches to Imitation trainer" do
      trainer = Imitation.new(embed_size: 256, hidden_sizes: [64])
      state = Nx.broadcast(0.5, {1, 256})

      action = Training.get_action(trainer, state)

      assert is_map(action)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :buttons)
    end

    @tag :slow
    test "get_action/3 dispatches to PPO trainer" do
      trainer = PPO.new(embed_size: 256, hidden_sizes: [64])
      state = Nx.broadcast(0.5, {1, 256})

      action = Training.get_action(trainer, state)

      assert is_map(action)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :value)
    end

    @tag :slow
    test "get_controller_action/3 dispatches to Imitation trainer" do
      trainer = Imitation.new(embed_size: 256, hidden_sizes: [64])
      state = Nx.broadcast(0.5, {1, 256})

      cs = Training.get_controller_action(trainer, state)

      assert %ControllerState{} = cs
      assert is_map(cs.main_stick)
    end

    @tag :slow
    test "get_controller_action/3 dispatches to PPO trainer" do
      trainer = PPO.new(embed_size: 256, hidden_sizes: [64])
      state = Nx.broadcast(0.5, {1, 256})

      cs = Training.get_controller_action(trainer, state)

      assert %ControllerState{} = cs
      assert is_map(cs.main_stick)
    end

    @tag :slow
    test "metrics_summary/1 dispatches to Imitation trainer" do
      trainer = Imitation.new(embed_size: 256)

      summary = Training.metrics_summary(trainer)

      assert summary.step == 0
      assert Map.has_key?(summary, :avg_loss)
    end

    @tag :slow
    test "metrics_summary/1 dispatches to PPO trainer" do
      trainer = PPO.new(embed_size: 256)

      summary = Training.metrics_summary(trainer)

      assert summary.step == 0
      assert summary.timesteps == 0
    end
  end

  describe "train_imitation/2" do
    @tag :slow
    test "trains imitation policy on dataset" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)

      {:ok, trainer} = Training.train_imitation(dataset,
        epochs: 1,
        batch_size: 32,
        hidden_sizes: [32],
        validation_split: 0.1
      )

      assert %Imitation{} = trainer
      assert trainer.step > 0
    end

    @tag timeout: 120_000
    @tag :slow
    test "runs callbacks during training" do
      # Use minimal data - just testing callback is called
      frames = mock_frames(20)
      dataset = Data.from_frames(frames)

      test_pid = self()
      callback = fn metrics ->
        send(test_pid, {:callback, metrics})
      end

      {:ok, _trainer} = Training.train_imitation(dataset,
        epochs: 1,
        batch_size: 8,
        hidden_sizes: [16],
        callbacks: [callback],
        validation_split: 0
      )

      # Should have received at least one callback
      assert_received {:callback, metrics}
      assert Map.has_key?(metrics, :step)
    end

    @tag :slow
    test "handles zero validation split" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)

      {:ok, trainer} = Training.train_imitation(dataset,
        epochs: 1,
        batch_size: 16,
        hidden_sizes: [32],
        validation_split: 0
      )

      assert %Imitation{} = trainer
    end

    @tag :slow
    test "saves checkpoints when checkpoint_dir provided" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      checkpoint_dir = Path.join(System.tmp_dir!(), "checkpoints_#{:rand.uniform(10_000)}")
      File.mkdir_p!(checkpoint_dir)

      try do
        {:ok, _trainer} = Training.train_imitation(dataset,
          epochs: 1,
          batch_size: 32,
          hidden_sizes: [32],
          checkpoint_dir: checkpoint_dir,
          checkpoint_interval: 1,
          validation_split: 0
        )

        # Should have saved final checkpoint
        final_path = Path.join(checkpoint_dir, "imitation_final.ckpt")
        assert File.exists?(final_path)
      after
        File.rm_rf(checkpoint_dir)
      end
    end
  end

  describe "resume_imitation/3" do
    @tag :slow
    test "resumes training from checkpoint" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)
      checkpoint_path = Path.join(System.tmp_dir!(), "resume_test_#{:rand.uniform(10_000)}.ckpt")

      try do
        {:ok, trainer1} = Training.train_imitation(dataset,
          epochs: 1,
          batch_size: 16,
          hidden_sizes: [32],
          validation_split: 0
        )

        :ok = Imitation.save_checkpoint(trainer1, checkpoint_path)

        {:ok, trainer2} = Training.resume_imitation(checkpoint_path, dataset,
          epochs: 1,
          batch_size: 16,
          hidden_sizes: [32]
        )

        assert %Imitation{} = trainer2
        assert trainer2.step >= trainer1.step
      after
        File.rm(checkpoint_path)
      end
    end

    test "returns error for missing checkpoint" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)

      result = Training.resume_imitation("/nonexistent.ckpt", dataset, hidden_sizes: [32])

      assert {:error, _} = result
    end
  end
end
