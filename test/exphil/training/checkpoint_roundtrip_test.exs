defmodule ExPhil.Training.CheckpointRoundtripTest do
  @moduledoc """
  Tests for checkpoint save/load round-tripping.

  Verifies that optimizer state (momentum, velocity, step count) is
  properly preserved across checkpoint save/load cycles.
  """

  use ExUnit.Case, async: true

  alias ExPhil.Training.Imitation

  @moduletag :checkpoint

  describe "optimizer state round-trip" do
    test "optimizer state count is preserved after save/load" do
      # Create a minimal trainer
      trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        temporal: false
      )

      # Create a minimal batch
      batch = %{
        states: Nx.broadcast(0.5, {4, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {4, 8}),
          main_x: Nx.broadcast(8, {4}),
          main_y: Nx.broadcast(8, {4}),
          c_x: Nx.broadcast(8, {4}),
          c_y: Nx.broadcast(8, {4}),
          shoulder: Nx.broadcast(2, {4})
        }
      }

      # Train for a few steps to build up optimizer state
      trained_trainer =
        Enum.reduce(1..10, trainer, fn _, acc ->
          {new_trainer, _metrics} = Imitation.train_step(acc, batch, nil)
          new_trainer
        end)

      # Verify step count advanced
      assert trained_trainer.step == 10

      # Extract optimizer state counts before save
      # Note: Polaris.compose wraps states in an extra tuple: {{clip_state, adam_state}}
      {{clip_state, adam_state}} = trained_trainer.optimizer_state
      clip_count_before = Nx.to_number(clip_state[:count])
      adam_count_before = Nx.to_number(adam_state[:count])

      # Verify counts match trainer step
      assert clip_count_before == 10
      assert adam_count_before == 10

      # Save checkpoint
      path = Path.join(System.tmp_dir!(), "checkpoint_roundtrip_test_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      :ok = Imitation.save_checkpoint(trained_trainer, path)

      # Create a fresh trainer and load checkpoint
      fresh_trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        temporal: false
      )

      {:ok, loaded_trainer} = Imitation.load_checkpoint(fresh_trainer, path)

      # Verify step restored
      assert loaded_trainer.step == 10

      # Extract optimizer state counts after load
      {{loaded_clip_state, loaded_adam_state}} = loaded_trainer.optimizer_state
      clip_count_after = Nx.to_number(loaded_clip_state[:count])
      adam_count_after = Nx.to_number(loaded_adam_state[:count])

      # Verify counts preserved
      assert clip_count_after == 10, "Clip state count not preserved: #{clip_count_after}"
      assert adam_count_after == 10, "Adam state count not preserved: #{adam_count_after}"
    end

    test "momentum and velocity tensors are preserved" do
      trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        temporal: false
      )

      batch = %{
        states: Nx.broadcast(0.5, {4, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {4, 8}),
          main_x: Nx.broadcast(8, {4}),
          main_y: Nx.broadcast(8, {4}),
          c_x: Nx.broadcast(8, {4}),
          c_y: Nx.broadcast(8, {4}),
          shoulder: Nx.broadcast(2, {4})
        }
      }

      # Train to build non-zero momentum/velocity
      trained_trainer =
        Enum.reduce(1..5, trainer, fn _, acc ->
          {new_trainer, _} = Imitation.train_step(acc, batch, nil)
          new_trainer
        end)

      # Get a sample mu/nu tensor before save
      {{_clip_state, adam_state}} = trained_trainer.optimizer_state
      # mu and nu are nested maps matching param structure
      # Get the first value we can find
      first_mu_before = get_first_tensor(adam_state[:mu])
      first_nu_before = get_first_tensor(adam_state[:nu])

      # Verify they're non-zero (training happened)
      assert Nx.to_number(Nx.sum(Nx.abs(first_mu_before))) > 0
      assert Nx.to_number(Nx.sum(Nx.abs(first_nu_before))) > 0

      # Save and load
      path = Path.join(System.tmp_dir!(), "checkpoint_momentum_test_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)

      :ok = Imitation.save_checkpoint(trained_trainer, path)

      fresh_trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        temporal: false
      )

      {:ok, loaded_trainer} = Imitation.load_checkpoint(fresh_trainer, path)

      # Get same tensors after load
      {{_loaded_clip_state, loaded_adam_state}} = loaded_trainer.optimizer_state
      first_mu_after = get_first_tensor(loaded_adam_state[:mu])
      first_nu_after = get_first_tensor(loaded_adam_state[:nu])

      # Verify they match
      assert Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(first_mu_before, first_mu_after)))) < 1.0e-6,
             "Momentum (mu) not preserved"
      assert Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(first_nu_before, first_nu_after)))) < 1.0e-6,
             "Velocity (nu) not preserved"
    end

    test "training continues correctly after resume" do
      trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        lr_schedule: :cosine,
        decay_steps: 100,
        temporal: false
      )

      batch = %{
        states: Nx.broadcast(0.5, {4, 64}),
        actions: %{
          buttons: Nx.broadcast(0, {4, 8}),
          main_x: Nx.broadcast(8, {4}),
          main_y: Nx.broadcast(8, {4}),
          c_x: Nx.broadcast(8, {4}),
          c_y: Nx.broadcast(8, {4}),
          shoulder: Nx.broadcast(2, {4})
        }
      }

      # Train for 20 steps
      trained_20 =
        Enum.reduce(1..20, trainer, fn _, acc ->
          {new_trainer, _} = Imitation.train_step(acc, batch, nil)
          new_trainer
        end)

      # Save at step 20
      path = Path.join(System.tmp_dir!(), "checkpoint_resume_test_#{System.unique_integer()}.axon")
      on_exit(fn -> File.rm(path) end)
      :ok = Imitation.save_checkpoint(trained_20, path)

      # Load and continue training
      fresh_trainer = Imitation.new(
        embed_size: 64,
        hidden_sizes: [32],
        learning_rate: 1.0e-3,
        lr_schedule: :cosine,
        decay_steps: 100,
        temporal: false
      )
      {:ok, resumed} = Imitation.load_checkpoint(fresh_trainer, path)

      # Train 10 more steps
      continued =
        Enum.reduce(1..10, resumed, fn _, acc ->
          {new_trainer, _} = Imitation.train_step(acc, batch, nil)
          new_trainer
        end)

      # Verify total steps
      assert continued.step == 30

      # Verify optimizer state count also at 30
      {{clip_state, adam_state}} = continued.optimizer_state
      assert Nx.to_number(clip_state[:count]) == 30
      assert Nx.to_number(adam_state[:count]) == 30
    end
  end

  # Helper to extract first tensor from nested map
  defp get_first_tensor(map) when is_map(map) do
    case Map.values(map) do
      [%Nx.Tensor{} = tensor | _] -> tensor
      [nested | _] -> get_first_tensor(nested)
      [] -> nil
    end
  end
end
