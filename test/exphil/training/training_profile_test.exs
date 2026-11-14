defmodule ExPhil.Training.TrainingProfileTest do
  @moduledoc """
  The optimizer must decay only what the backbone's Edifice training profile
  allows, and the trainer must record which parameters that was. The checks
  here drive the real optimizer with controlled gradients so the recorded
  groups are tested against what the update actually does.
  """
  use ExUnit.Case, async: true

  alias Edifice.Training.Profile
  alias ExPhil.Training.{Imitation, TrainingProfile}
  alias ExPhil.Training.Imitation.Optimizer

  @moduletag :training

  describe "edifice_arch/1" do
    test "backbones named like the Edifice registry resolve to themselves" do
      for backbone <- [:mamba, :gru, :lstm, :min_gru, :mamba_ssd, :gated_ssm] do
        assert TrainingProfile.edifice_arch(backbone) == backbone
      end
    end

    test "ExPhil-only names map to their Edifice architecture" do
      assert TrainingProfile.edifice_arch(:mamba_2) == :mamba_ssd
      assert TrainingProfile.edifice_arch(:mamba_nif) == :mamba
    end

    test "spec-built backbones resolve through their recipe module" do
      assert TrainingProfile.edifice_arch(:mamba2_full) in [:mamba2_full, Edifice.SSM.Mamba2]
    end

    test "backbones built inside ExPhil have no Edifice architecture" do
      assert TrainingProfile.edifice_arch(:sliding_window) == nil
      assert TrainingProfile.edifice_arch(:mlp) == nil
    end
  end

  describe "for_config/1" do
    test "temporal Mamba gets Edifice's Mamba profile" do
      profile = TrainingProfile.for_config(%{temporal: true, backbone: :mamba, hidden_size: 16})
      assert profile.source == Edifice.SSM.Mamba
      assert profile.arch == :mamba
    end

    test "temporal Mamba-2 gets the profile that protects A_log, dt_bias and D" do
      profile = TrainingProfile.for_config(%{temporal: true, backbone: :mamba2_full})
      assert profile.source == Edifice.SSM.Mamba2

      for leaf <- ["A_log", "dt_bias", "D"] do
        assert {:leaf, leaf} in profile.no_weight_decay
      end
    end

    test "non-temporal and ExPhil-built backbones get the generic profile" do
      assert TrainingProfile.for_config(%{temporal: false, backbone: :mamba}).source == :generic

      assert TrainingProfile.for_config(%{temporal: true, backbone: :sliding_window}).source ==
               :generic

      assert TrainingProfile.for_config(%{}).source == :generic
    end
  end

  describe "the imitation optimizer under a profile" do
    # A tiny parameter tree with the shapes that matter: a kernel (decayed), a
    # bias and a norm gain (never decayed) and a Mamba-2 style A_log (never
    # decayed by its profile, rank 1 anyway).
    defp params do
      %{
        "dense" => %{"kernel" => Nx.broadcast(0.5, {3, 4}), "bias" => Nx.broadcast(0.5, {4})},
        "norm" => %{"gamma" => Nx.broadcast(1.0, {4})},
        "mixer" => %{"A_log" => Nx.broadcast(0.7, {2}), "in_kernel" => Nx.broadcast(0.5, {4, 4})}
      }
    end

    defp zero_grads(p), do: Nx.Defn.Composite.traverse(p, &Nx.multiply(&1, 0.0))

    defp update(optimizer, p, grads) do
      {init_fn, update_fn} = optimizer
      state = init_fn.(p)
      {updates, _state} = update_fn.(grads, state, p)
      updates
    end

    defp max_abs(t), do: Nx.to_number(Nx.reduce_max(Nx.abs(t)))

    test "with zero gradients, only the decay group moves and by exactly -lr * decay * p" do
      config = %{optimizer: :adamw, learning_rate: 0.1, weight_decay: 0.01, max_grad_norm: 0}
      profile = TrainingProfile.for_config(%{temporal: true, backbone: :mamba2_full})
      p = params()
      updates = update(Optimizer.create_optimizer(config, profile), p, zero_grads(p))

      # Decayed: update = -lr * decay * p = -0.1 * 0.01 * 0.5 = -5.0e-4
      assert_in_delta Nx.to_number(updates["dense"]["kernel"][0][0]), -5.0e-4, 1.0e-7
      assert_in_delta Nx.to_number(updates["mixer"]["in_kernel"][0][0]), -5.0e-4, 1.0e-7

      # Protected: no decay, and Adam of a zero gradient is zero
      assert max_abs(updates["dense"]["bias"]) == 0.0
      assert max_abs(updates["norm"]["gamma"]) == 0.0
      assert max_abs(updates["mixer"]["A_log"]) == 0.0
    end

    test "without a profile the plain AdamW decays every parameter" do
      config = %{optimizer: :adamw, learning_rate: 0.1, weight_decay: 0.01, max_grad_norm: 0}
      p = params()
      updates = update(Optimizer.create_optimizer(config), p, zero_grads(p))

      assert max_abs(updates["dense"]["bias"]) > 0.0
      assert max_abs(updates["mixer"]["A_log"]) > 0.0
    end

    test "a non-AdamW optimizer ignores the profile" do
      config = %{optimizer: :adam, learning_rate: 0.1, max_grad_norm: 0}
      profile = Profile.generic()
      p = params()
      updates = update(Optimizer.create_optimizer(config, profile), p, zero_grads(p))
      assert Enum.all?(Nx.Defn.Composite.flatten_list([updates]), &(max_abs(&1) == 0.0))
    end

    test "groups/3 records counts, protected paths and the decay" do
      profile = TrainingProfile.for_config(%{temporal: true, backbone: :mamba2_full})
      g = TrainingProfile.groups(profile, params(), decay: 0.01)

      assert g.profile == ":mamba2_full" and g.weight_decay == 0.01
      assert g.decay_count == 2 and g.no_decay_count == 3 and g.skipped_count == 0
      assert "mixer.A_log" in g.no_decay and "norm.gamma" in g.no_decay
      assert Enum.all?(g.rules, &is_binary/1)
    end
  end

  describe "a full imitation policy update" do
    @tag :slow
    test "trains a temporal Mamba policy with the profile and records its groups" do
      trainer =
        Imitation.new(
          embed_size: 32,
          temporal: true,
          backbone: :mamba,
          hidden_size: 16,
          hidden_sizes: [16],
          num_layers: 1,
          state_size: 4,
          window_size: 5,
          batch_size: 2,
          precision: :f32,
          dropout: 0.0,
          warmup_steps: 0,
          lr_schedule: :constant,
          learning_rate: 0.001,
          weight_decay: 0.01,
          seed: 906
        )

      groups = trainer.config.optimizer_groups
      assert groups.profile == ":mamba" and groups.source == "Edifice.SSM.Mamba"
      assert groups.weight_decay == 0.01
      assert groups.decay_count > 0 and groups.no_decay_count > 0

      # Every layer-norm gain and bias of the policy is protected; every
      # dense kernel is decayed.
      assert Enum.any?(groups.no_decay, &String.ends_with?(&1, "_norm.gamma"))
      assert Enum.any?(groups.no_decay, &String.ends_with?(&1, "in_proj.bias"))
      refute Enum.any?(groups.no_decay, &String.ends_with?(&1, "in_proj.kernel"))

      batch = %{
        states: Nx.broadcast(0.5, {2, 5, 32}),
        actions: %{
          buttons: Nx.broadcast(0, {2, 8}),
          main_x: Nx.broadcast(8, {2}),
          main_y: Nx.broadcast(8, {2}),
          c_x: Nx.broadcast(8, {2}),
          c_y: Nx.broadcast(8, {2}),
          shoulder: Nx.broadcast(0, {2})
        }
      }

      {updated, metrics} = Imitation.train_step(trainer, batch, nil)
      assert updated.step == 1
      loss = Nx.to_number(metrics.loss)
      assert is_number(loss) and loss > 0

      before = Nx.Defn.Composite.flatten_list([trainer.policy_params])
      after_update = Nx.Defn.Composite.flatten_list([updated.policy_params])

      assert Enum.any?(Enum.zip(before, after_update), fn {a, b} ->
               Nx.to_number(Nx.any(Nx.not_equal(a, b))) == 1
             end)

      for tensor <-
            Nx.Defn.Composite.flatten_list([updated.policy_params, updated.optimizer_state]) do
        invalid = Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor))
        assert Nx.to_number(Nx.any(invalid)) == 0
      end
    end
  end
end
