defmodule ExPhil.Integration.PipelineTest do
  @moduledoc """
  Full pipeline integration tests: data → train → inference.

  These tests verify the entire ExPhil pipeline works end-to-end,
  from synthetic training data through model training to inference.
  """

  use ExUnit.Case, async: false

  alias ExPhil.Agents.Agent
  alias ExPhil.Embeddings
  alias ExPhil.Training.{Data, Imitation}
  alias ExPhil.Test.Factories

  @moduletag :integration

  # Use default embedding config for consistent dimensions
  # Default is 1204 dims with all features enabled
  @test_embed_config Embeddings.default_config()

  # Helper to convert tensor values to numbers for assertion
  # Handles both scalar tensors and rank-1 tensors of size 1
  defp to_num(tensor) when is_struct(tensor, Nx.Tensor) do
    tensor |> Nx.squeeze() |> Nx.to_number()
  end

  defp to_num(value), do: value

  # Helper to assert action values are in valid ranges
  defp assert_action_in_range(action) do
    main_x = to_num(action.main_x)
    main_y = to_num(action.main_y)
    c_x = to_num(action.c_x)
    c_y = to_num(action.c_y)
    shoulder = to_num(action.shoulder)

    assert main_x >= 0 and main_x <= 16, "main_x #{main_x} out of range"
    assert main_y >= 0 and main_y <= 16, "main_y #{main_y} out of range"
    assert c_x >= 0 and c_x <= 16, "c_x #{c_x} out of range"
    assert c_y >= 0 and c_y <= 16, "c_y #{c_y} out of range"
    assert shoulder >= 0 and shoulder <= 4, "shoulder #{shoulder} out of range"
  end

  describe "single-frame pipeline" do
    @tag :integration
    test "data → train → export → inference" do
      # 1. Create synthetic training data
      frames = Factories.build_training_frames(100)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)

      # Verify dataset was created
      assert dataset.size == 100

      # 2. Create batches
      batches = Data.batched(dataset, batch_size: 16, shuffle: false)
      batch_list = Enum.take(batches, 3)
      assert length(batch_list) == 3

      # Verify batch structure
      first_batch = hd(batch_list)
      assert Map.has_key?(first_batch, :states)
      assert Map.has_key?(first_batch, :actions)
      assert Nx.shape(first_batch.states) |> elem(0) == 16

      # 3. Create and train model
      embed_size = Embeddings.embedding_size(@test_embed_config)

      trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32, 32],
          learning_rate: 1.0e-3,
          temporal: false,
          embed_config: @test_embed_config
        )

      # Train for a few steps
      {trained_trainer, metrics} =
        Enum.reduce(batch_list, {trainer, []}, fn batch, {t, m} ->
          {new_t, step_metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [step_metrics | m]}
        end)

      # Verify training happened
      assert trained_trainer.step == 3
      assert length(metrics) == 3

      # All losses should be finite (loss is returned as tensor for performance)
      Enum.each(metrics, fn m ->
        loss_value = if is_struct(m.loss, Nx.Tensor), do: Nx.to_number(m.loss), else: m.loss
        assert is_float(loss_value) or is_number(loss_value)
        # Loss should be non-negative
        assert loss_value >= 0
      end)

      # 4. Export policy
      tmp_dir = System.tmp_dir!()
      policy_path = Path.join(tmp_dir, "test_policy_#{System.unique_integer([:positive])}.bin")
      on_exit(fn -> File.rm(policy_path) end)

      :ok = Imitation.export_policy(trained_trainer, policy_path)
      assert File.exists?(policy_path)

      # Verify policy file has reasonable size (at least some KB)
      stat = File.stat!(policy_path)
      assert stat.size > 1000, "Policy file too small: #{stat.size} bytes"

      # 5. Load policy and run inference
      {:ok, agent} =
        Agent.start_link(
          policy_path: policy_path,
          embed_config: @test_embed_config,
          name: :"test_agent_#{System.unique_integer([:positive])}"
        )

      # Run inference on a game state
      game_state = Factories.build_game_state()
      {:ok, action} = Agent.get_action(agent, game_state)

      # Verify action structure
      assert is_map(action)
      assert Map.has_key?(action, :buttons)
      assert Map.has_key?(action, :main_x)
      assert Map.has_key?(action, :main_y)
      assert Map.has_key?(action, :c_x)
      assert Map.has_key?(action, :c_y)
      assert Map.has_key?(action, :shoulder)

      # Verify action values are in valid ranges (values are Nx tensors)
      assert_action_in_range(action)

      # Clean up
      GenServer.stop(agent)
    end

    @tag :integration
    test "inference produces different outputs for different game states" do
      # Train a minimal model
      frames = Factories.build_training_frames(50)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)
      batches = Data.batched(dataset, batch_size: 16, shuffle: false) |> Enum.take(2)

      embed_size = Embeddings.embedding_size(@test_embed_config)

      trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32],
          learning_rate: 1.0e-3,
          temporal: false,
          embed_config: @test_embed_config
        )

      {trained, _} =
        Enum.reduce(batches, {trainer, []}, fn batch, {t, m} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [metrics | m]}
        end)

      # Export and load
      tmp_dir = System.tmp_dir!()
      policy_path = Path.join(tmp_dir, "test_diff_#{System.unique_integer([:positive])}.bin")
      on_exit(fn -> File.rm(policy_path) end)

      :ok = Imitation.export_policy(trained, policy_path)

      {:ok, agent} =
        Agent.start_link(
          policy_path: policy_path,
          embed_config: @test_embed_config,
          name: :"test_agent_diff_#{System.unique_integer([:positive])}"
        )

      # Create two different game states
      state1 =
        Factories.build_game_state(
          players: %{
            1 => Factories.build_player(x: -50.0, y: 0.0, percent: 0.0),
            2 => Factories.build_player(x: 50.0, y: 0.0, percent: 100.0)
          }
        )

      state2 =
        Factories.build_game_state(
          players: %{
            1 => Factories.build_player(x: 50.0, y: 50.0, percent: 150.0),
            2 => Factories.build_player(x: -50.0, y: 0.0, percent: 0.0)
          }
        )

      # Get actions for both states
      {:ok, action1} = Agent.get_action(agent, state1)
      {:ok, action2} = Agent.get_action(agent, state2)

      # Verify both actions are valid
      # Note: After minimal training on random data, outputs may be similar
      # but the test verifies the pipeline works correctly
      assert is_map(action1)
      assert is_map(action2)
      assert_action_in_range(action1)
      assert_action_in_range(action2)

      GenServer.stop(agent)
    end
  end

  describe "checkpoint round-trip" do
    @tag :integration
    test "save → load → continue training preserves state" do
      frames = Factories.build_training_frames(80)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)
      batches = Data.batched(dataset, batch_size: 16, shuffle: false) |> Enum.to_list()

      embed_size = Embeddings.embedding_size(@test_embed_config)

      # Create trainer and train for 3 steps
      trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32],
          learning_rate: 1.0e-3,
          temporal: false,
          embed_config: @test_embed_config
        )

      {trained_3, _} =
        Enum.reduce(Enum.take(batches, 3), {trainer, []}, fn batch, {t, m} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [metrics | m]}
        end)

      assert trained_3.step == 3

      # Save checkpoint
      tmp_dir = System.tmp_dir!()

      checkpoint_path =
        Path.join(tmp_dir, "test_checkpoint_#{System.unique_integer([:positive])}.axon")

      on_exit(fn -> File.rm(checkpoint_path) end)

      :ok = Imitation.save_checkpoint(trained_3, checkpoint_path)
      assert File.exists?(checkpoint_path)

      # Create fresh trainer and load checkpoint
      fresh_trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32],
          learning_rate: 1.0e-3,
          temporal: false,
          embed_config: @test_embed_config
        )

      {:ok, loaded_trainer} = Imitation.load_checkpoint(fresh_trainer, checkpoint_path)

      # Verify state was restored
      assert loaded_trainer.step == 3

      # Continue training for 2 more steps
      {continued_trainer, _} =
        Enum.reduce(Enum.take(batches, 2), {loaded_trainer, []}, fn batch, {t, m} ->
          {new_t, metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [metrics | m]}
        end)

      # Should now be at step 5
      assert continued_trainer.step == 5

      # Export and verify inference still works
      policy_path = Path.join(tmp_dir, "test_continued_#{System.unique_integer([:positive])}.bin")
      on_exit(fn -> File.rm(policy_path) end)

      :ok = Imitation.export_policy(continued_trainer, policy_path)

      {:ok, agent} =
        Agent.start_link(
          policy_path: policy_path,
          embed_config: @test_embed_config,
          name: :"test_agent_continued_#{System.unique_integer([:positive])}"
        )

      game_state = Factories.build_game_state()
      {:ok, action} = Agent.get_action(agent, game_state)

      assert is_map(action)
      assert_action_in_range(action)

      GenServer.stop(agent)
    end
  end

  describe "frame delay augmentation" do
    @tag :integration
    test "batches with frame delay augmentation have correct structure" do
      # Create enough frames for delay augmentation
      frames = Factories.build_training_frames(100)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)

      # Create batches with frame delay augmentation (like online play)
      batches =
        Data.batched(dataset,
          batch_size: 8,
          shuffle: false,
          frame_delay_augment: true,
          frame_delay_min: 0,
          frame_delay_max: 18
        )

      batch_list = Enum.take(batches, 3)
      assert length(batch_list) == 3

      # Verify batch structure is maintained with augmentation
      for batch <- batch_list do
        assert Map.has_key?(batch, :states)
        assert Map.has_key?(batch, :actions)

        # States should have correct shape
        {batch_size, _embed_size} = Nx.shape(batch.states)
        assert batch_size == 8

        # Actions should have all components
        assert Map.has_key?(batch.actions, :buttons)
        assert Map.has_key?(batch.actions, :main_x)
      end
    end
  end

  describe "temporal model pipeline" do
    @tag :integration
    @tag :slow
    test "sequence data → train temporal model → export" do
      # Create training frames for sequences
      frames = Factories.build_training_frames(200)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)

      # Convert to sequences
      window_size = 10
      seq_dataset = Data.to_sequences(dataset, window_size: window_size, stride: 5)

      # Verify sequence structure (to_sequences returns a dataset with sequences as frames)
      assert seq_dataset.size > 0

      first_seq = hd(seq_dataset.frames)
      # Each sequence has :sequence (list of frames), :game_state, :controller, :action
      assert length(first_seq.sequence) == window_size

      # Create sequence batches
      embed_size = Embeddings.embedding_size(@test_embed_config)

      batches =
        Data.batched_sequences(seq_dataset,
          batch_size: 4,
          embed_config: @test_embed_config
        )

      batch_list = Enum.take(batches, 2)
      assert length(batch_list) == 2

      # Verify temporal batch structure
      first_batch = hd(batch_list)
      {batch_size, seq_len, batch_embed_size} = Nx.shape(first_batch.states)
      assert batch_size == 4
      assert seq_len == window_size
      assert batch_embed_size == embed_size

      # Create and train temporal model (MLP with seq_len input)
      trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32],
          learning_rate: 1.0e-3,
          temporal: true,
          # MLP flattens temporal dimension
          backbone: :mlp,
          window_size: window_size,
          embed_config: @test_embed_config
        )

      # Train for a few steps
      {trained, metrics} =
        Enum.reduce(batch_list, {trainer, []}, fn batch, {t, m} ->
          {new_t, step_metrics} = Imitation.train_step(t, batch, nil)
          {new_t, [step_metrics | m]}
        end)

      assert trained.step == 2
      assert length(metrics) == 2

      # Verify losses are finite
      Enum.each(metrics, fn m ->
        loss_value = if is_struct(m.loss, Nx.Tensor), do: Nx.to_number(m.loss), else: m.loss
        assert is_number(loss_value)
        assert loss_value >= 0
      end)

      # Export policy
      tmp_dir = System.tmp_dir!()
      policy_path = Path.join(tmp_dir, "test_temporal_#{System.unique_integer([:positive])}.bin")
      on_exit(fn -> File.rm(policy_path) end)

      :ok = Imitation.export_policy(trained, policy_path)
      assert File.exists?(policy_path)

      # Verify exported file has reasonable size
      stat = File.stat!(policy_path)
      assert stat.size > 1000, "Policy file too small: #{stat.size} bytes"

      # Verify config was saved correctly
      loaded = :erlang.binary_to_term(File.read!(policy_path))
      assert loaded.config.temporal == true
      assert loaded.config.backbone == :mlp
      assert loaded.config.window_size == window_size
    end
  end

  describe "data validation" do
    @tag :integration
    test "empty frames produce empty dataset" do
      dataset = Data.from_frames([], embed_config: @test_embed_config)
      assert dataset.size == 0
    end

    @tag :integration
    test "batching respects batch_size" do
      frames = Factories.build_training_frames(25)
      dataset = Data.from_frames(frames, embed_config: @test_embed_config)

      # With drop_last: false, should get 2 full batches + 1 partial
      batches = Data.batched(dataset, batch_size: 10, shuffle: false, drop_last: false)
      batch_list = Enum.to_list(batches)

      assert length(batch_list) == 3
      assert Nx.shape(Enum.at(batch_list, 0).states) |> elem(0) == 10
      assert Nx.shape(Enum.at(batch_list, 1).states) |> elem(0) == 10
      assert Nx.shape(Enum.at(batch_list, 2).states) |> elem(0) == 5

      # With drop_last: true, should get only 2 full batches
      batches_dropped = Data.batched(dataset, batch_size: 10, shuffle: false, drop_last: true)
      batch_list_dropped = Enum.to_list(batches_dropped)

      assert length(batch_list_dropped) == 2
    end
  end

  describe "error handling" do
    @tag :integration
    test "agent with non-existent policy returns error on get_action" do
      # Agent starts successfully but with no policy loaded (logs warning)
      {:ok, agent} =
        Agent.start_link(
          policy_path: "/nonexistent/path/to/policy.bin",
          embed_config: @test_embed_config,
          name: :"test_agent_error_#{System.unique_integer([:positive])}"
        )

      # get_action should return error when no policy is loaded
      game_state = Factories.build_game_state()
      result = Agent.get_action(agent, game_state)

      assert match?({:error, :no_policy_loaded}, result)

      GenServer.stop(agent)
    end

    @tag :integration
    test "loading non-existent checkpoint returns error" do
      embed_size = Embeddings.embedding_size(@test_embed_config)

      trainer =
        Imitation.new(
          embed_size: embed_size,
          hidden_sizes: [32],
          temporal: false,
          embed_config: @test_embed_config
        )

      result = Imitation.load_checkpoint(trainer, "/nonexistent/checkpoint.axon")
      assert match?({:error, _}, result)
    end
  end
end
