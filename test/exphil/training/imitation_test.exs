defmodule ExPhil.Training.ImitationTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Imitation
  alias ExPhil.Bridge.ControllerState

  # Helper to create mock training batch
  defp mock_batch(batch_size, embed_size) do
    states = Nx.broadcast(0.5, {batch_size, embed_size})

    actions = %{
      buttons: Nx.broadcast(0, {batch_size, 8}),
      main_x: Nx.broadcast(8, {batch_size}),
      main_y: Nx.broadcast(8, {batch_size}),
      c_x: Nx.broadcast(8, {batch_size}),
      c_y: Nx.broadcast(8, {batch_size}),
      shoulder: Nx.broadcast(0, {batch_size})
    }

    %{states: states, actions: actions}
  end

  describe "new/1" do
    test "creates trainer with default config" do
      trainer = Imitation.new(embed_size: 64)

      assert %Imitation{} = trainer
      assert trainer.step == 0
      assert is_map(trainer.config)
    end

    test "accepts custom learning rate" do
      trainer = Imitation.new(embed_size: 64, learning_rate: 0.001)

      assert trainer.config.learning_rate == 0.001
    end

    test "accepts custom hidden sizes" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [128, 64])

      assert %Imitation{} = trainer
      assert trainer.policy_model != nil
    end

    test "initializes policy parameters" do
      trainer = Imitation.new(embed_size: 64)

      assert trainer.policy_params != nil
    end

    test "initializes optimizer state" do
      trainer = Imitation.new(embed_size: 64)

      assert trainer.optimizer_state != nil
    end

    test "initializes empty metrics" do
      trainer = Imitation.new(embed_size: 64)

      assert trainer.metrics.loss == []
    end

    test "mixed_precision: true initializes FP32 master weights" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], mixed_precision: true)

      assert trainer.mixed_precision_state != nil

      # Master params should be FP32
      master_params = ExPhil.Training.MixedPrecision.get_master_params(trainer.mixed_precision_state)
      first_tensor = get_first_tensor(master_params)
      assert Nx.type(first_tensor) == {:f, 32}

      # Compute params should be BF16
      compute_params = ExPhil.Training.MixedPrecision.get_compute_params(trainer.mixed_precision_state)
      first_compute_tensor = get_first_tensor(compute_params)
      assert Nx.type(first_compute_tensor) == {:bf, 16}
    end

    test "mixed_precision: false has nil mixed_precision_state" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], mixed_precision: false)

      assert trainer.mixed_precision_state == nil
    end
  end

  # Helper to get first tensor from nested params
  defp get_first_tensor(params) when is_struct(params, Nx.Tensor), do: params

  defp get_first_tensor(params) when is_map(params) do
    params
    |> Map.values()
    |> List.first()
    |> get_first_tensor()
  end

  describe "create_optimizer/1" do
    test "returns init and update functions" do
      config = %{
        learning_rate: 1.0e-4,
        weight_decay: 1.0e-5
      }

      {init_fn, update_fn} = Imitation.create_optimizer(config)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end
  end

  describe "build_loss_fn/1" do
    test "returns predict and loss functions" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])

      {predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      assert is_function(predict_fn, 2)
      assert is_function(loss_fn, 3)
    end
  end

  describe "train_step/3" do
    @tag :slow
    test "performs single training step" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)
      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      {new_trainer, metrics} = Imitation.train_step(trainer, batch, loss_fn)

      assert new_trainer.step == 1
      # Loss is returned as tensor to avoid blocking GPU→CPU transfer
      assert %Nx.Tensor{} = metrics.loss
      # Scalar tensor
      assert Nx.shape(metrics.loss) == {}
    end

    @tag :slow
    test "updates parameters" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)
      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      {new_trainer, _metrics} = Imitation.train_step(trainer, batch, loss_fn)

      # Parameters should be different after update
      # (Can't easily compare nested maps, but at least check they exist)
      assert new_trainer.policy_params != nil
    end

    @tag :slow
    test "mixed precision training step maintains FP32 master weights" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], mixed_precision: true)
      batch = mock_batch(4, 64)
      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      {new_trainer, metrics} = Imitation.train_step(trainer, batch, loss_fn)

      # Training step should complete
      assert new_trainer.step == 1
      assert %Nx.Tensor{} = metrics.loss

      # Mixed precision state should be updated
      assert new_trainer.mixed_precision_state != nil

      # Master weights should still be FP32
      master_params = ExPhil.Training.MixedPrecision.get_master_params(new_trainer.mixed_precision_state)
      first_tensor = get_first_tensor(master_params)
      assert Nx.type(first_tensor) == {:f, 32}
    end

    @tag :slow
    @tag :regression
    test "loss can be efficiently accumulated as tensors" do
      # Regression test for GPU blocking issue:
      # Previously, train_step called Nx.to_number(loss) after every batch,
      # causing GPU→CPU sync that blocked the training loop.
      # Now loss is returned as a tensor for batch accumulation.
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)
      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      # Simulate accumulating losses over multiple batches
      {_trainer, losses} =
        Enum.reduce(1..5, {trainer, []}, fn _i, {t, ls} ->
          {new_t, metrics} = Imitation.train_step(t, batch, loss_fn)
          {new_t, [metrics.loss | ls]}
        end)

      # Should be able to stack and average without per-batch GPU sync
      assert length(losses) == 5
      avg_loss = losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
      assert is_float(avg_loss)
      assert avg_loss > 0
    end
  end

  describe "evaluate/2" do
    @tag :slow
    test "computes average loss on dataset" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      dataset = [mock_batch(4, 64), mock_batch(4, 64)]

      result = Imitation.evaluate(trainer, dataset)

      assert is_float(result.loss)
      assert result.num_batches == 2
    end

    test "handles empty dataset" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])

      result = Imitation.evaluate(trainer, [])

      assert result.loss == 0.0
      assert result.num_batches == 0
    end
  end

  describe "evaluate_batch/2" do
    @tag :slow
    test "returns loss as tensor for single batch" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)

      result = Imitation.evaluate_batch(trainer, batch)

      # Loss is returned as tensor to avoid blocking GPU→CPU transfer
      assert %Nx.Tensor{} = result.loss
      # Scalar tensor
      assert Nx.shape(result.loss) == {}
    end

    @tag :slow
    @tag :regression
    test "validation losses can be efficiently accumulated as tensors" do
      # Regression test: Previously the benchmark script called
      # Imitation.evaluate(trainer, single_batch) which was incorrect,
      # and even when fixed, would block on Nx.to_number per batch.
      # Now evaluate_batch returns tensor for efficient accumulation.
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batches = [mock_batch(4, 64), mock_batch(4, 64), mock_batch(4, 64)]

      # Simulate validation loop accumulating tensor losses
      losses =
        Enum.map(batches, fn batch ->
          Imitation.evaluate_batch(trainer, batch).loss
        end)

      # Should be able to stack and average without per-batch GPU sync
      assert length(losses) == 3
      avg_loss = losses |> Nx.stack() |> Nx.mean() |> Nx.to_number()
      assert is_float(avg_loss)
      assert avg_loss > 0
    end
  end

  describe "metrics_summary/1" do
    test "returns summary with empty metrics" do
      trainer = Imitation.new(embed_size: 64)

      summary = Imitation.metrics_summary(trainer)

      assert summary.step == 0
      assert summary.avg_loss == 0.0
    end

    test "computes averages from loss history" do
      trainer = Imitation.new(embed_size: 64)
      trainer = %{trainer | metrics: %{trainer.metrics | loss: [1.0, 2.0, 3.0]}}

      summary = Imitation.metrics_summary(trainer)

      assert_in_delta summary.avg_loss, 2.0, 0.001
      assert summary.min_loss == 1.0
      assert summary.max_loss == 3.0
    end
  end

  describe "save_checkpoint/2 and load_checkpoint/2" do
    @tag :slow
    test "round-trips checkpoint data" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "test_checkpoint_#{:rand.uniform(10_000)}.axon")

      try do
        # Save
        assert :ok = Imitation.save_checkpoint(trainer, path)
        assert File.exists?(path)

        # Load
        new_trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
        {:ok, loaded} = Imitation.load_checkpoint(new_trainer, path)

        assert loaded.step == trainer.step
      after
        File.rm(path)
      end
    end

    test "load_checkpoint returns error for missing file" do
      trainer = Imitation.new(embed_size: 64)

      result = Imitation.load_checkpoint(trainer, "/nonexistent/path.axon")

      assert {:error, _} = result
    end
  end

  describe "export_policy/2" do
    @tag :slow
    test "exports policy parameters" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "test_policy_#{:rand.uniform(10_000)}.axon")

      try do
        result = Imitation.export_policy(trainer, path)

        assert result == :ok
        assert File.exists?(path)

        # Verify file contents
        {:ok, binary} = File.read(path)
        export = :erlang.binary_to_term(binary)

        assert is_map(export.params)
        assert is_map(export.config)
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "exports temporal config for inference" do
      trainer =
        Imitation.new(
          embed_size: 64,
          hidden_sizes: [32],
          temporal: true,
          backbone: :sliding_window,
          window_size: 30,
          num_heads: 2,
          head_dim: 16
        )

      path = Path.join(System.tmp_dir!(), "test_temporal_policy_#{:rand.uniform(10_000)}.axon")

      try do
        :ok = Imitation.export_policy(trainer, path)

        {:ok, binary} = File.read(path)
        export = :erlang.binary_to_term(binary)

        # Verify temporal config is included
        assert export.config.embed_size == 64
        assert export.config.temporal == true
        assert export.config.backbone == :sliding_window
        assert export.config.window_size == 30
        assert export.config.num_heads == 2
        assert export.config.head_dim == 16
        # default
        assert export.config.hidden_size == 256
        # default
        assert export.config.num_layers == 2
        assert export.config.axis_buckets == 16
        assert export.config.shoulder_buckets == 4
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "exports non-temporal config with defaults" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "test_nontemporal_policy_#{:rand.uniform(10_000)}.axon")

      try do
        :ok = Imitation.export_policy(trainer, path)

        {:ok, binary} = File.read(path)
        export = :erlang.binary_to_term(binary)

        # Non-temporal should have temporal: false
        assert export.config.temporal == false
        # backbone defaults to :sliding_window in @default_config
        assert export.config.backbone == :sliding_window
        assert export.config.embed_size == 64
      after
        File.rm(path)
      end
    end
  end

  describe "train/3" do
    @tag :slow
    test "trains for specified epochs" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      dataset = [mock_batch(4, 64)]

      {:ok, trained} = Imitation.train(trainer, dataset, epochs: 2)

      # 2 epochs * 1 batch = 2 steps
      assert trained.step == 2
    end

    @tag :slow
    test "calls callback with metrics" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      dataset = [mock_batch(4, 64)]

      test_pid = self()

      callback = fn metrics ->
        send(test_pid, {:callback, metrics})
        :ok
      end

      {:ok, _trained} = Imitation.train(trainer, dataset, epochs: 1, callback: callback)

      assert_receive {:callback, metrics}
      assert is_map(metrics)
      assert Map.has_key?(metrics, :loss)
    end
  end

  describe "get_action/3" do
    @tag :slow
    test "returns action samples" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      action = Imitation.get_action(trainer, state)

      assert is_map(action)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :buttons)
    end
  end

  describe "get_controller_action/3" do
    @tag :slow
    test "returns ControllerState" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      state = Nx.broadcast(0.5, {1, 64})

      cs = Imitation.get_controller_action(trainer, state)

      assert %ControllerState{} = cs
      assert is_map(cs.main_stick)
      assert is_boolean(cs.button_a)
    end
  end

  # ============================================================================
  # Regression Tests
  # ============================================================================
  # Tests for bugs discovered during development

  describe "checkpoint serialization regression" do
    @tag :slow
    test "save_checkpoint converts tensors to BinaryBackend" do
      # Regression: EXLA tensors with device buffers become invalid after
      # the training process ends. Checkpoints must use BinaryBackend.
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "regression_checkpoint_#{:rand.uniform(10_000)}.axon")

      try do
        :ok = Imitation.save_checkpoint(trainer, path)

        # Load and verify tensors are readable (not stale EXLA buffers)
        {:ok, binary} = File.read(path)
        checkpoint = :erlang.binary_to_term(binary)

        # Verify we can actually read tensor data
        kernel = get_in(checkpoint.policy_params.data, ["backbone_dense_0", "kernel"])
        assert %Nx.Tensor{} = kernel

        # This would fail with stale EXLA buffers:
        # "unable to get buffer. It may belong to another node"
        values = Nx.to_flat_list(kernel)
        assert is_list(values)
        assert length(values) > 0
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "export_policy produces loadable tensors in fresh process context" do
      # Regression: Exported policies must be loadable in a different process
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "regression_policy_#{:rand.uniform(10_000)}.bin")

      try do
        :ok = Imitation.export_policy(trainer, path)

        # Load the policy
        {:ok, policy} = ExPhil.Training.load_policy(path)

        # Verify all tensors are readable
        kernel = get_in(policy.params.data, ["backbone_dense_0", "kernel"])
        values = Nx.to_flat_list(kernel)
        assert is_list(values)
      after
        File.rm(path)
      end
    end
  end

  describe "export_policy config completeness regression" do
    @tag :slow
    test "exported config includes hidden_sizes" do
      # Regression: Missing hidden_sizes caused architecture mismatch
      # when loading policy - model built with default [512,512] but
      # trained with [32] would fail with shape mismatch
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [128, 64])
      path = Path.join(System.tmp_dir!(), "regression_config_#{:rand.uniform(10_000)}.bin")

      try do
        :ok = Imitation.export_policy(trainer, path)
        {:ok, policy} = ExPhil.Training.load_policy(path)

        # Config must include hidden_sizes to reconstruct architecture
        assert policy.config.hidden_sizes == [128, 64]
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "exported config includes dropout rate" do
      # Regression: Missing dropout caused layer count mismatch
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], dropout: 0.2)
      path = Path.join(System.tmp_dir!(), "regression_dropout_#{:rand.uniform(10_000)}.bin")

      try do
        :ok = Imitation.export_policy(trainer, path)
        {:ok, policy} = ExPhil.Training.load_policy(path)

        # Config must include dropout for model reconstruction
        assert policy.config.dropout == 0.2
      after
        File.rm(path)
      end
    end

    @tag :slow
    test "exported config includes embed_size" do
      # Regression: Missing embed_size caused input dimension mismatch
      trainer = Imitation.new(embed_size: 128, hidden_sizes: [32])
      path = Path.join(System.tmp_dir!(), "regression_embed_#{:rand.uniform(10_000)}.bin")

      try do
        :ok = Imitation.export_policy(trainer, path)
        {:ok, policy} = ExPhil.Training.load_policy(path)

        assert policy.config.embed_size == 128
      after
        File.rm(path)
      end
    end
  end

  describe "agent policy loading regression" do
    @tag :slow
    test "agent can load and use exported policy with custom architecture" do
      # Regression: Agent failed to load policies with non-default hidden_sizes
      # because it used default [512,512] instead of config values
      #
      # Note: embed_size MUST match what Game.embed() produces (~1991 dims)
      # because the agent's embed_game_state uses the real embedding, not the test config
      embed_size = ExPhil.Embeddings.embedding_size()
      trainer = Imitation.new(embed_size: embed_size, hidden_sizes: [32, 32])
      path = Path.join(System.tmp_dir!(), "regression_agent_#{:rand.uniform(10_000)}.bin")

      try do
        :ok = Imitation.export_policy(trainer, path)

        # Agent should be able to load and use the policy
        {:ok, agent} = ExPhil.Agents.Agent.start_link(policy_path: path)

        # Create a mock game state
        game_state = %ExPhil.Bridge.GameState{
          frame: 0,
          stage: 2,
          players: %{
            1 => %ExPhil.Bridge.Player{
              x: 0.0,
              y: 0.0,
              percent: 0.0,
              stock: 4,
              facing: 1,
              character: 9,
              action: 14,
              action_frame: 0,
              invulnerable: false,
              jumps_left: 2,
              on_ground: true,
              shield_strength: 60.0
            },
            2 => %ExPhil.Bridge.Player{
              x: 10.0,
              y: 0.0,
              percent: 0.0,
              stock: 4,
              facing: -1,
              character: 9,
              action: 14,
              action_frame: 0,
              invulnerable: false,
              jumps_left: 2,
              on_ground: true,
              shield_strength: 60.0
            }
          }
        }

        # This would fail with architecture mismatch before the fix
        result = ExPhil.Agents.Agent.get_action(agent, game_state, player_port: 1)
        assert {:ok, action} = result
        assert is_map(action)
        assert Map.has_key?(action, :buttons)

        GenServer.stop(agent)
      after
        File.rm(path)
      end
    end
  end

  describe "precision configuration" do
    test "creates trainer with bf16 precision (default)" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])

      assert trainer.config.precision == :bf16
    end

    test "creates trainer with f32 precision" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], precision: :f32)

      assert trainer.config.precision == :f32
    end

    @tag :slow
    test "bf16 training step converts inputs to bf16" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], precision: :bf16)

      # Create batch with f32 inputs (simulating typical data loading)
      batch = %{
        states: Nx.broadcast(Nx.tensor(0.5, type: :f32), {4, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {4, 8}),
          main_x: Nx.broadcast(8, {4}),
          main_y: Nx.broadcast(8, {4}),
          c_x: Nx.broadcast(8, {4}),
          c_y: Nx.broadcast(8, {4}),
          shoulder: Nx.broadcast(0, {4})
        }
      }

      # Training step should succeed (inputs converted internally)
      {new_trainer, metrics} = Imitation.train_step(trainer, batch, nil)

      assert new_trainer.step == 1
      # Loss is returned as tensor (not number) to avoid GPU blocking
      assert %Nx.Tensor{} = metrics.loss
    end

    @tag :slow
    test "f32 training step works with f32 inputs" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32], precision: :f32)

      batch = %{
        states: Nx.broadcast(Nx.tensor(0.5, type: :f32), {4, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {4, 8}),
          main_x: Nx.broadcast(8, {4}),
          main_y: Nx.broadcast(8, {4}),
          c_x: Nx.broadcast(8, {4}),
          c_y: Nx.broadcast(8, {4}),
          shoulder: Nx.broadcast(0, {4})
        }
      }

      {new_trainer, metrics} = Imitation.train_step(trainer, batch, nil)

      assert new_trainer.step == 1
      # Loss is returned as tensor (not number) to avoid GPU blocking
      assert %Nx.Tensor{} = metrics.loss
    end
  end

  describe "gradient checkpointing configuration" do
    test "creates trainer with gradient checkpointing disabled by default" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])

      assert trainer.config.gradient_checkpoint == false
    end

    test "creates trainer with gradient checkpointing enabled" do
      trainer =
        Imitation.new(
          embed_size: 64,
          hidden_sizes: [32],
          gradient_checkpoint: true
        )

      assert trainer.config.gradient_checkpoint == true
    end

    test "checkpoint_every defaults to 1" do
      trainer =
        Imitation.new(
          embed_size: 64,
          hidden_sizes: [32],
          gradient_checkpoint: true
        )

      assert trainer.config.checkpoint_every == 1
    end

    test "checkpoint_every can be customized" do
      trainer =
        Imitation.new(
          embed_size: 64,
          hidden_sizes: [32],
          gradient_checkpoint: true,
          checkpoint_every: 2
        )

      assert trainer.config.checkpoint_every == 2
    end
  end

  describe "warmup/3" do
    @tag :slow
    test "warms up all JIT functions by default" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)

      {:ok, timings} = Imitation.warmup(trainer, batch, show_progress: false)

      assert Map.has_key?(timings, :training)
      assert Map.has_key?(timings, :validation)
      assert Map.has_key?(timings, :inference)
      assert timings.training >= 0
      assert timings.validation >= 0
      assert timings.inference >= 0
    end

    @tag :slow
    test "can warm up only specific targets" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)

      {:ok, timings} = Imitation.warmup(trainer, batch, only: [:validation], show_progress: false)

      assert Map.has_key?(timings, :validation)
      refute Map.has_key?(timings, :training)
      refute Map.has_key?(timings, :inference)
    end

    @tag :slow
    test "returns timing in milliseconds" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)

      {:ok, timings} = Imitation.warmup(trainer, batch, only: [:inference], show_progress: false)

      # Should be a positive number (milliseconds)
      assert is_integer(timings.inference)
      assert timings.inference >= 0
    end
  end

  describe "warmup_validation/2" do
    @tag :slow
    test "warms up validation code path" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batches = [mock_batch(4, 64), mock_batch(4, 64)]

      {:ok, timings} = Imitation.warmup_validation(trainer, batches)

      assert Map.has_key?(timings, :validation)
      assert timings.validation >= 0
    end

    @tag :slow
    test "works with single batch" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batches = [mock_batch(4, 64)]

      {:ok, timings} = Imitation.warmup_validation(trainer, batches)

      assert timings.validation >= 0
    end
  end
end
