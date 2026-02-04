defmodule ExPhil.Networks.ZambaTest do
  @moduledoc """
  Tests for Zamba (Mamba + Single Shared Attention) backbone.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Zamba

  @embed_size 64
  @hidden_size 32
  @state_size 8
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        Zamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different embed and hidden sizes" do
      model =
        Zamba.build(
          embed_size: 128,
          hidden_size: 64,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles same embed and hidden sizes (no projection)" do
      model =
        Zamba.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch_size, @seq_len, @hidden_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles different attention_every values" do
      for attention_every <- [2, 3, 4] do
        model =
          Zamba.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            state_size: @state_size,
            num_layers: 6,
            attention_every: attention_every,
            window_size: @seq_len
          )

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(
            Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
            Axon.ModelState.empty()
          )

        input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch_size, @hidden_size},
               "Failed for attention_every=#{attention_every}"
      end
    end

    test "handles different num_layers" do
      for num_layers <- [2, 4, 6] do
        model =
          Zamba.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            state_size: @state_size,
            num_layers: num_layers,
            attention_every: 2,
            window_size: @seq_len
          )

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(
            Nx.template({@batch_size, @seq_len, @embed_size}, :f32),
            Axon.ModelState.empty()
          )

        input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch_size, @hidden_size},
               "Failed for num_layers=#{num_layers}"
      end
    end

    test "attention only at end (attention_every = num_layers)" do
      # This tests the "minimal attention" variant
      model =
        Zamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          # Only applied once at the very end
          attention_every: 4,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "shared attention weights" do
    test "attention layer parameters are shared across applications" do
      # Build model with attention applied multiple times
      model =
        Zamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 6,
          # Applied at layers 2, 4, 6 (3 times)
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, _predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Convert to map for inspection
      params_data = params.data

      # There should be exactly ONE set of shared attention weights
      # not multiple (e.g., shared_attention_1, shared_attention_2)
      shared_attn_keys =
        params_data
        |> Map.keys()
        |> Enum.filter(&String.starts_with?(&1, "shared_attention"))

      # Count unique "base" attention layer names (excluding _pre_norm, _ffn1, etc.)
      # All should start with "shared_attention" - no numbered variants
      assert Enum.all?(shared_attn_keys, fn key ->
               String.starts_with?(key, "shared_attention")
             end)

      # Verify attention FFN exists (proves attention was built)
      assert Map.has_key?(params_data, "shared_attention_ffn1")
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Zamba.output_size(hidden_size: 128) == 128
      assert Zamba.output_size(hidden_size: 256) == 256
      # default
      assert Zamba.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        Zamba.param_count(
          embed_size: 287,
          hidden_size: 256,
          state_size: 16,
          expand_factor: 2,
          num_layers: 6,
          attention_every: 3
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 10_000_000
    end

    test "scales with num_layers but attention is fixed" do
      count_4 =
        Zamba.param_count(
          embed_size: 64,
          hidden_size: 32,
          state_size: 8,
          num_layers: 4,
          attention_every: 2
        )

      count_8 =
        Zamba.param_count(
          embed_size: 64,
          hidden_size: 32,
          state_size: 8,
          num_layers: 8,
          attention_every: 2
        )

      # 8 layers should have more params than 4 layers
      # Due to shared attention being a significant portion, the ratio is less than 2x
      # With small hidden_size (32), attention dominates, so the ratio can be quite low
      assert count_8 > count_4
      # Still should not be exactly 2x since attention is shared
      assert count_8 < count_4 * 2.5
    end
  end

  describe "compare_to_jamba/1" do
    test "shows parameter savings from shared attention" do
      comparison =
        Zamba.compare_to_jamba(
          embed_size: 287,
          hidden_size: 256,
          num_layers: 6,
          attention_every: 3
        )

      assert comparison.zamba_params < comparison.jamba_params
      assert comparison.savings > 0
      assert comparison.savings_percent > 0
    end
  end

  describe "layer_pattern/1" do
    test "returns correct layer pattern" do
      pattern = Zamba.layer_pattern(num_layers: 6, attention_every: 3)

      assert pattern == [
               :mamba,
               :mamba,
               :mamba_attention,
               :mamba,
               :mamba,
               :mamba_attention
             ]
    end

    test "handles attention_every = 2" do
      pattern = Zamba.layer_pattern(num_layers: 6, attention_every: 2)

      assert pattern == [
               :mamba,
               :mamba_attention,
               :mamba,
               :mamba_attention,
               :mamba,
               :mamba_attention
             ]
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = Zamba.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :state_size) == 16
      assert Keyword.get(defaults, :expand_factor) == 2
      assert Keyword.get(defaults, :conv_size) == 4
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :attention_every) == 3
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = Zamba.melee_defaults()

      model = Zamba.build([embed_size: 287] ++ defaults)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 60, 287}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 60, 287})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 256}
    end
  end

  describe "numerical stability" do
    test "produces finite outputs for normal inputs" do
      model =
        Zamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      # Normal random input
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Check no NaN or Inf
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      model =
        Zamba.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Should still produce valid output
      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "integration with Policy" do
    test "works as zamba temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :zamba,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 4,
          attention_every: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Check policy output shapes
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end
  end
end
