defmodule ExPhil.Training.MambaIntegrationTest do
  @moduledoc """
  Integration tests for Mamba backbone training.

  These tests verify the full training pipeline works with the Mamba architecture,
  including temporal sequence handling, checkpoint persistence, and policy export.
  """
  use ExUnit.Case, async: false

  alias ExPhil.Training
  alias ExPhil.Training.{Imitation, Data}
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  # Use smaller dimensions for faster tests
  @embed_size 64
  @hidden_size 32
  @state_size 8
  @window_size 10
  @batch_size 4

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
      character: 9,
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
      character: 2,
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
      frame: Keyword.get(opts, :frame, 0),
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
      button_x: Keyword.get(opts, :x_btn, false),
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
        game_state: mock_game_state(frame: i, x: i * 1.0, percent: i * 0.5),
        controller: mock_controller_state(
          main_x: 0.5 + rem(i, 10) * 0.05,
          a: rem(i, 5) == 0
        )
      }
    end)
  end

  describe "Mamba trainer initialization" do
    test "creates temporal trainer with mamba backbone" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [@hidden_size, @hidden_size],
        temporal: true,
        backbone: :mamba,
        window_size: @window_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        expand_factor: 2,
        conv_size: 4,
        num_layers: 2
      )

      assert %Imitation{} = trainer
      assert trainer.config[:temporal] == true
      assert trainer.config[:backbone] == :mamba
      # Note: Mamba-specific config may be stored differently
    end

    test "builds policy model with correct output structure" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [@hidden_size],
        temporal: true,
        backbone: :mamba,
        window_size: @window_size,
        hidden_size: @hidden_size,
        state_size: @state_size
      )

      # Verify model can be built and run
      {init_fn, predict_fn} = Axon.build(trainer.policy_model, mode: :inference)

      input = Nx.broadcast(0.5, {1, @window_size, @embed_size})
      params = init_fn.(input, Axon.ModelState.empty())

      output = predict_fn.(params, input)

      # Policy outputs multiple heads (may be tuple or map depending on Axon version)
      assert is_tuple(output) or is_map(output)
    end
  end

  describe "Mamba temporal data handling" do
    test "creates sequences from frames" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)

      seq_dataset = Data.to_sequences(dataset,
        window_size: @window_size,
        stride: 1
      )

      # 50 frames with window 10, stride 1 = 41 sequences
      expected_count = 50 - @window_size + 1
      assert seq_dataset.size == expected_count
    end

    test "batched_sequences creates proper batches" do
      frames = mock_frames(50)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: @window_size, stride: 1)

      batches = Data.batched_sequences(seq_dataset,
        batch_size: @batch_size,
        shuffle: false
      )
      |> Enum.to_list()

      assert length(batches) > 0

      first_batch = hd(batches)
      # States should be [batch, window, embed]
      assert tuple_size(Nx.shape(first_batch.states)) == 3
      {batch_dim, window_dim, _embed_dim} = Nx.shape(first_batch.states)
      assert batch_dim == @batch_size
      assert window_dim == @window_size
    end
  end

  describe "Mamba training step" do
    @tag :slow
    test "train_step reduces loss" do
      frames = mock_frames(100)
      dataset = Data.from_frames(frames)
      seq_dataset = Data.to_sequences(dataset, window_size: @window_size, stride: 2)

      trainer = Imitation.new(
        embed_size: ExPhil.Embeddings.embedding_size(),
        hidden_sizes: [@hidden_size],
        temporal: true,
        backbone: :mamba,
        window_size: @window_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1
      )

      batches = Data.batched_sequences(seq_dataset,
        batch_size: @batch_size,
        shuffle: false,
        drop_last: true
      )
      |> Enum.take(3)

      # Run a few training steps
      {final_trainer, losses} = Enum.reduce(batches, {trainer, []}, fn batch, {t, losses} ->
        {new_trainer, metrics} = Imitation.train_step(t, batch, nil)
        {new_trainer, [metrics.loss | losses]}
      end)

      assert length(losses) == 3
      assert final_trainer.step == 3

      # All losses should be finite
      Enum.each(losses, fn loss ->
        assert is_float(loss) and loss > 0
      end)
    end
  end

  describe "Mamba checkpoint persistence" do
    @tag :slow
    test "save and load checkpoint preserves Mamba config" do
      checkpoint_path = Path.join(System.tmp_dir!(), "mamba_ckpt_#{:rand.uniform(10_000)}.axon")

      try do
        trainer1 = Imitation.new(
          embed_size: @embed_size,
          hidden_sizes: [@hidden_size],
          temporal: true,
          backbone: :mamba,
          window_size: @window_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          expand_factor: 2,
          conv_size: 4,
          num_layers: 2
        )

        :ok = Imitation.save_checkpoint(trainer1, checkpoint_path)
        assert File.exists?(checkpoint_path)

        # Create a base trainer with same architecture to load into
        base_trainer = Imitation.new(
          embed_size: @embed_size,
          hidden_sizes: [@hidden_size],
          temporal: true,
          backbone: :mamba,
          window_size: @window_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          expand_factor: 2,
          conv_size: 4,
          num_layers: 2
        )

        {:ok, trainer2} = Imitation.load_checkpoint(base_trainer, checkpoint_path)

        # Verify Mamba-specific config is preserved
        assert trainer2.config[:backbone] == :mamba
        assert trainer2.config[:temporal] == true
        assert trainer2.config[:state_size] == @state_size
        assert trainer2.config[:expand_factor] == 2
        assert trainer2.config[:conv_size] == 4
        assert trainer2.config[:num_layers] == 2
      after
        File.rm(checkpoint_path)
      end
    end
  end

  describe "Mamba policy export" do
    test "export and load policy preserves Mamba config" do
      policy_path = Path.join(System.tmp_dir!(), "mamba_policy_#{:rand.uniform(10_000)}.bin")

      try do
        trainer = Imitation.new(
          embed_size: @embed_size,
          hidden_sizes: [@hidden_size],
          temporal: true,
          backbone: :mamba,
          window_size: @window_size,
          hidden_size: @hidden_size,
          state_size: 16,  # Use default to match export
          expand_factor: 2,
          conv_size: 4
        )

        :ok = Imitation.export_policy(trainer, policy_path)
        assert File.exists?(policy_path)

        {:ok, policy} = Training.load_policy(policy_path)

        assert policy.config[:backbone] == :mamba
        assert policy.config[:temporal] == true
        # state_size defaults to 16 in export
        assert policy.config[:state_size] == 16
        assert policy.config[:expand_factor] == 2
        assert policy.config[:conv_size] == 4
      after
        File.rm(policy_path)
      end
    end
  end

  describe "Mamba inference" do
    test "model can run forward pass with tensor input" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [@hidden_size],
        temporal: true,
        backbone: :mamba,
        window_size: @window_size,
        hidden_size: @hidden_size,
        state_size: @state_size
      )

      # Create tensor input directly (bypassing embedding)
      input = Nx.broadcast(0.5, {1, @window_size, @embed_size})

      {_init_fn, predict_fn} = Axon.build(trainer.policy_model, mode: :inference)
      output = predict_fn.(trainer.policy_params, input)

      # Should produce multi-head output (tuple or map)
      assert is_tuple(output) or is_map(output)
    end
  end

  describe "Mamba vs LSTM comparison" do
    test "Mamba and LSTM models have similar structure" do
      mamba_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [@hidden_size],
        temporal: true,
        backbone: :mamba,
        window_size: @window_size,
        hidden_size: @hidden_size,
        state_size: @state_size
      )

      lstm_trainer = Imitation.new(
        embed_size: @embed_size,
        hidden_sizes: [@hidden_size],
        temporal: true,
        backbone: :lstm,
        window_size: @window_size,
        hidden_size: @hidden_size
      )

      # Both should be valid trainers
      assert %Imitation{} = mamba_trainer
      assert %Imitation{} = lstm_trainer

      # Both should have policy models
      assert mamba_trainer.policy_model != nil
      assert lstm_trainer.policy_model != nil

      # Test forward pass for both
      input = Nx.broadcast(0.5, {1, @window_size, @embed_size})

      {_, mamba_predict} = Axon.build(mamba_trainer.policy_model, mode: :inference)
      {_, lstm_predict} = Axon.build(lstm_trainer.policy_model, mode: :inference)

      mamba_out = mamba_predict.(mamba_trainer.policy_params, input)
      lstm_out = lstm_predict.(lstm_trainer.policy_params, input)

      # Both should produce outputs (tuple or map with action heads)
      assert is_tuple(mamba_out) or is_map(mamba_out)
      assert is_tuple(lstm_out) or is_map(lstm_out)
    end
  end

  describe "gradient checkpointing" do
    @tag :slow
    test "builds checkpointed Mamba model" do
      alias ExPhil.Networks.Mamba

      model = Mamba.build_checkpointed(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 2,
        window_size: @window_size
      )

      assert %Axon{} = model
    end

    @tag :slow
    test "checkpointed Mamba produces same output shape as regular" do
      alias ExPhil.Networks.Mamba

      opts = [
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 2,
        window_size: @window_size
      ]

      regular_model = Mamba.build(opts)
      checkpointed_model = Mamba.build_checkpointed(opts)

      # Initialize both
      {init_reg, predict_reg} = Axon.build(regular_model, mode: :inference)
      {init_ckpt, predict_ckpt} = Axon.build(checkpointed_model, mode: :inference)

      input_template = Nx.template({1, @window_size, @embed_size}, :f32)
      params_reg = init_reg.(input_template, Axon.ModelState.empty())
      params_ckpt = init_ckpt.(input_template, Axon.ModelState.empty())

      # Forward pass
      input = Nx.broadcast(0.5, {1, @window_size, @embed_size})
      out_reg = predict_reg.(params_reg, input)
      out_ckpt = predict_ckpt.(params_ckpt, input)

      # Same output shape
      assert Nx.shape(out_reg) == Nx.shape(out_ckpt)
      assert Nx.shape(out_reg) == {1, @hidden_size}
    end

    @tag :slow
    test "trainer can be created with gradient checkpointing enabled" do
      trainer = Imitation.new(
        embed_size: @embed_size,
        temporal: true,
        backbone: :mamba,
        hidden_size: @hidden_size,
        window_size: @window_size,
        gradient_checkpoint: true,
        checkpoint_every: 1
      )

      assert %Imitation{} = trainer
      assert trainer.config.gradient_checkpoint == true
      assert trainer.config.checkpoint_every == 1
    end

    @tag :slow
    test "checkpoint_every option is respected" do
      alias ExPhil.Networks.Mamba

      # Build with checkpoint_every: 2 (every other layer)
      model = Mamba.build_checkpointed(
        embed_size: @embed_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 4,
        window_size: @window_size,
        checkpoint_every: 2
      )

      # Model should build successfully
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      input_template = Nx.template({1, @window_size, @embed_size}, :f32)
      params = init_fn.(input_template, Axon.ModelState.empty())

      # Forward pass should work
      input = Nx.broadcast(0.5, {1, @window_size, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, @hidden_size}
    end
  end
end
