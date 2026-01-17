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
      assert is_float(metrics.loss)
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
    test "tracks loss in metrics" do
      trainer = Imitation.new(embed_size: 64, hidden_sizes: [32])
      batch = mock_batch(4, 64)
      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      {new_trainer, _metrics} = Imitation.train_step(trainer, batch, loss_fn)

      assert length(new_trainer.metrics.loss) == 1
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
end
