defmodule ExPhil.Networks.ActionChunkingTest do
  @moduledoc """
  Tests for the Action Chunking with Transformers (ACT) implementation.

  Tests cover the CVAE architecture from
  "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., RSS 2023).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.ActionChunking

  @obs_size 64
  @action_dim 32
  @chunk_size 8
  @batch_size 4
  @latent_dim 16

  describe "build/1" do
    test "returns encoder and decoder models" do
      model =
        ActionChunking.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 64,
          num_layers: 2,
          latent_dim: @latent_dim
        )

      assert Map.has_key?(model, :encoder)
      assert Map.has_key?(model, :decoder)
      assert Map.has_key?(model, :config)

      assert model.config.obs_size == @obs_size
      assert model.config.action_dim == @action_dim
      assert model.config.chunk_size == @chunk_size
    end
  end

  describe "build_encoder/1" do
    test "encoder outputs mu and log_var with correct shapes" do
      encoder =
        ActionChunking.build_encoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 64,
          num_layers: 2,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(encoder)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      action_sequence = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "action_sequence" => Nx.template({@batch_size, @chunk_size, @action_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      {mu, log_var} =
        predict_fn.(params, %{
          "observations" => observations,
          "action_sequence" => action_sequence
        })

      assert Nx.shape(mu) == {@batch_size, @latent_dim}
      assert Nx.shape(log_var) == {@batch_size, @latent_dim}
    end

    test "encoder produces finite outputs" do
      encoder =
        ActionChunking.build_encoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 32,
          num_layers: 1,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(encoder)

      key = Nx.Random.key(42)
      {observations, key} = Nx.Random.normal(key, shape: {@batch_size, @obs_size})
      {action_sequence, _} = Nx.Random.normal(key, shape: {@batch_size, @chunk_size, @action_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "action_sequence" => Nx.template({@batch_size, @chunk_size, @action_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      {mu, log_var} =
        predict_fn.(params, %{
          "observations" => observations,
          "action_sequence" => action_sequence
        })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(mu) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(log_var) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(mu) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(log_var) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_decoder/1" do
    test "decoder outputs action chunk with correct shape" do
      decoder =
        ActionChunking.build_decoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 64,
          num_layers: 2,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(decoder)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      latent_z = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "latent_z" => Nx.template({@batch_size, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "observations" => observations,
          "latent_z" => latent_z
        })

      assert Nx.shape(output) == {@batch_size, @chunk_size, @action_dim}
    end

    test "decoder with z=0 (inference mode) produces valid output" do
      decoder =
        ActionChunking.build_decoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 32,
          num_layers: 1,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(decoder)

      observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
      # z = 0 for inference (prior mean)
      latent_z = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "latent_z" => Nx.template({@batch_size, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "observations" => observations,
          "latent_z" => latent_z
        })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles different chunk sizes" do
      for chunk_size <- [4, 8, 16] do
        decoder =
          ActionChunking.build_decoder(
            obs_size: @obs_size,
            action_dim: @action_dim,
            chunk_size: chunk_size,
            hidden_size: 32,
            num_layers: 1,
            latent_dim: @latent_dim
          )

        {init_fn, predict_fn} = Axon.build(decoder)

        observations = Nx.broadcast(0.5, {@batch_size, @obs_size})
        latent_z = Nx.broadcast(0.0, {@batch_size, @latent_dim})

        params =
          init_fn.(
            %{
              "observations" => Nx.template({@batch_size, @obs_size}, :f32),
              "latent_z" => Nx.template({@batch_size, @latent_dim}, :f32)
            },
            Axon.ModelState.empty()
          )

        output =
          predict_fn.(params, %{
            "observations" => observations,
            "latent_z" => latent_z
          })

        assert Nx.shape(output) == {@batch_size, chunk_size, @action_dim},
               "Failed for chunk_size=#{chunk_size}"
      end
    end
  end

  describe "build_inference/1" do
    test "builds decoder-only model" do
      model =
        ActionChunking.build_inference(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 32,
          num_layers: 1,
          latent_dim: @latent_dim
        )

      # Should be an Axon model, not a map
      assert is_struct(model, Axon)
    end
  end

  describe "reparameterize/3" do
    test "returns z with correct shape" do
      mu = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      epsilon = Nx.broadcast(1.0, {@batch_size, @latent_dim})

      z = ActionChunking.reparameterize(mu, log_var, epsilon)

      assert Nx.shape(z) == {@batch_size, @latent_dim}
    end

    test "with zero log_var and unit epsilon, z = mu + epsilon" do
      mu = Nx.broadcast(2.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})  # std = exp(0/2) = 1
      epsilon = Nx.broadcast(1.0, {@batch_size, @latent_dim})

      z = ActionChunking.reparameterize(mu, log_var, epsilon)

      # z = mu + 1 * epsilon = 2 + 1 = 3
      expected = Nx.broadcast(3.0, {@batch_size, @latent_dim})
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(z, expected))))
      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "with zero epsilon, z = mu" do
      mu = Nx.broadcast(5.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(2.0, {@batch_size, @latent_dim})
      epsilon = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      z = ActionChunking.reparameterize(mu, log_var, epsilon)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(z, mu))))
      assert_in_delta diff, 0.0, 1.0e-5
    end
  end

  describe "kl_divergence/2" do
    test "returns zero for standard normal" do
      # For q = N(0, 1), KL(q || p) = 0 when p = N(0, 1)
      mu = Nx.broadcast(0.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      kl = ActionChunking.kl_divergence(mu, log_var)

      assert_in_delta Nx.to_number(kl), 0.0, 1.0e-5
    end

    test "returns positive KL for non-standard normal" do
      mu = Nx.broadcast(1.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.5, {@batch_size, @latent_dim})

      kl = ActionChunking.kl_divergence(mu, log_var)

      assert Nx.to_number(kl) > 0
    end

    test "KL increases with larger mu" do
      log_var = Nx.broadcast(0.0, {@batch_size, @latent_dim})

      mu_small = Nx.broadcast(0.5, {@batch_size, @latent_dim})
      mu_large = Nx.broadcast(2.0, {@batch_size, @latent_dim})

      kl_small = ActionChunking.kl_divergence(mu_small, log_var)
      kl_large = ActionChunking.kl_divergence(mu_large, log_var)

      assert Nx.to_number(kl_large) > Nx.to_number(kl_small)
    end
  end

  describe "reconstruction_loss/2" do
    test "returns zero for identical tensors" do
      actions = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})

      loss = ActionChunking.reconstruction_loss(actions, actions)

      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end

    test "returns positive loss for different tensors" do
      target = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})
      predicted = Nx.broadcast(0.3, {@batch_size, @chunk_size, @action_dim})

      loss = ActionChunking.reconstruction_loss(target, predicted)

      assert Nx.to_number(loss) > 0
    end

    test "loss increases with larger differences" do
      target = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})

      pred_small_diff = Nx.broadcast(0.4, {@batch_size, @chunk_size, @action_dim})
      pred_large_diff = Nx.broadcast(0.1, {@batch_size, @chunk_size, @action_dim})

      loss_small = ActionChunking.reconstruction_loss(target, pred_small_diff)
      loss_large = ActionChunking.reconstruction_loss(target, pred_large_diff)

      assert Nx.to_number(loss_large) > Nx.to_number(loss_small)
    end
  end

  describe "cvae_loss/5" do
    test "combines reconstruction and KL loss" do
      target = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})
      predicted = Nx.broadcast(0.3, {@batch_size, @chunk_size, @action_dim})
      mu = Nx.broadcast(0.5, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.5, {@batch_size, @latent_dim})

      total_loss = ActionChunking.cvae_loss(target, predicted, mu, log_var, 10.0)
      recon_loss = ActionChunking.reconstruction_loss(target, predicted)
      kl_loss = ActionChunking.kl_divergence(mu, log_var)

      expected = Nx.add(recon_loss, Nx.multiply(10.0, kl_loss))
      diff = Nx.to_number(Nx.abs(Nx.subtract(total_loss, expected)))

      assert_in_delta diff, 0.0, 1.0e-5
    end

    test "KL weight affects total loss" do
      target = Nx.broadcast(0.5, {@batch_size, @chunk_size, @action_dim})
      predicted = Nx.broadcast(0.3, {@batch_size, @chunk_size, @action_dim})
      mu = Nx.broadcast(1.0, {@batch_size, @latent_dim})
      log_var = Nx.broadcast(0.5, {@batch_size, @latent_dim})

      loss_low_weight = ActionChunking.cvae_loss(target, predicted, mu, log_var, 1.0)
      loss_high_weight = ActionChunking.cvae_loss(target, predicted, mu, log_var, 100.0)

      assert Nx.to_number(loss_high_weight) > Nx.to_number(loss_low_weight)
    end
  end

  describe "ensemble_weights/2" do
    test "returns normalized weights" do
      weights = ActionChunking.ensemble_weights(8, 0.01)

      assert Nx.shape(weights) == {8}

      # Sum should be ~1 (normalized)
      sum = Nx.to_number(Nx.sum(weights))
      assert_in_delta sum, 1.0, 1.0e-5
    end

    test "weights decrease with index" do
      weights = ActionChunking.ensemble_weights(8, 0.1)

      weight_list = Nx.to_flat_list(weights)
      pairs = Enum.zip(weight_list, tl(weight_list))

      assert Enum.all?(pairs, fn {a, b} -> a >= b end)
    end

    test "higher decay means faster weight decrease" do
      weights_slow = ActionChunking.ensemble_weights(8, 0.01)
      weights_fast = ActionChunking.ensemble_weights(8, 0.5)

      # First weight should be similar, but last weight should be much smaller with fast decay
      first_slow = Nx.to_number(weights_slow[0])
      first_fast = Nx.to_number(weights_fast[0])
      last_slow = Nx.to_number(weights_slow[7])
      last_fast = Nx.to_number(weights_fast[7])

      # Fast decay should have lower last/first ratio
      ratio_slow = last_slow / first_slow
      ratio_fast = last_fast / first_fast

      assert ratio_fast < ratio_slow
    end
  end

  describe "apply_ensemble/2" do
    test "averages overlapping predictions" do
      # Create simple action chunks where we know the answer
      chunk_size = 4
      action_dim = 2

      # 4 overlapping chunks, each [1, chunk_size, action_dim]
      # Chunk 0 predicts current action at position 3
      # Chunk 1 predicts current action at position 2
      # Chunk 2 predicts current action at position 1
      # Chunk 3 predicts current action at position 0

      chunks =
        for i <- 0..3 do
          # Create a chunk where the relevant action is easy to identify
          base = Nx.broadcast(0.0, {1, chunk_size, action_dim})
          # Set the position that represents "current action" to a known value
          pos = 3 - i
          action_val = Nx.broadcast(Float.round(i + 1.0, 1), {1, 1, action_dim})
          Nx.put_slice(base, [0, pos, 0], action_val)
        end

      # Equal weights for simplicity
      weights = Nx.broadcast(0.25, {4})

      result = ActionChunking.apply_ensemble(chunks, weights)

      # Result should be average of [1, 2, 3, 4] = 2.5
      expected = Nx.broadcast(2.5, {1, action_dim})
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(result, expected))))

      assert_in_delta diff, 0.0, 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns chunk_size * action_dim" do
      assert ActionChunking.output_size(action_dim: 32, chunk_size: 8) == 256
      assert ActionChunking.output_size(action_dim: 64, chunk_size: 16) == 1024
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        ActionChunking.param_count(
          obs_size: 287,
          action_dim: 64,
          chunk_size: 8,
          hidden_size: 256,
          num_layers: 4,
          latent_dim: 32
        )

      # Should have significant params (transformer with encoder + decoder)
      assert count > 1_000_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      base_opts = [
        obs_size: 64,
        action_dim: 32,
        chunk_size: 8,
        hidden_size: 128,
        latent_dim: 16
      ]

      count_2 = ActionChunking.param_count(Keyword.put(base_opts, :num_layers, 2))
      count_4 = ActionChunking.param_count(Keyword.put(base_opts, :num_layers, 4))

      assert count_4 > count_2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = ActionChunking.melee_defaults()

      assert Keyword.get(defaults, :action_dim) == 64
      assert Keyword.get(defaults, :chunk_size) == 8
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :kl_weight) == 10.0
    end
  end

  describe "fast_defaults/0" do
    test "returns lighter configuration" do
      defaults = ActionChunking.fast_defaults()

      assert Keyword.get(defaults, :chunk_size) == 4
      assert Keyword.get(defaults, :hidden_size) == 128
      assert Keyword.get(defaults, :num_layers) == 2
    end
  end

  describe "numerical stability" do
    test "encoder produces finite outputs with random input" do
      encoder =
        ActionChunking.build_encoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 32,
          num_layers: 2,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(encoder)

      key = Nx.Random.key(42)
      {observations, key} = Nx.Random.normal(key, shape: {@batch_size, @obs_size})
      {action_sequence, _} = Nx.Random.normal(key, shape: {@batch_size, @chunk_size, @action_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "action_sequence" => Nx.template({@batch_size, @chunk_size, @action_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      {mu, log_var} =
        predict_fn.(params, %{
          "observations" => observations,
          "action_sequence" => action_sequence
        })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(mu) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(log_var) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(mu) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(log_var) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "decoder produces finite outputs with random latent" do
      decoder =
        ActionChunking.build_decoder(
          obs_size: @obs_size,
          action_dim: @action_dim,
          chunk_size: @chunk_size,
          hidden_size: 32,
          num_layers: 2,
          latent_dim: @latent_dim
        )

      {init_fn, predict_fn} = Axon.build(decoder)

      key = Nx.Random.key(42)
      {observations, key} = Nx.Random.normal(key, shape: {@batch_size, @obs_size})
      {latent_z, _} = Nx.Random.normal(key, shape: {@batch_size, @latent_dim})

      params =
        init_fn.(
          %{
            "observations" => Nx.template({@batch_size, @obs_size}, :f32),
            "latent_z" => Nx.template({@batch_size, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "observations" => observations,
          "latent_z" => latent_z
        })

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
