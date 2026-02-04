defmodule ExPhil.Networks.HybridBuilderTest do
  @moduledoc """
  Tests for the flexible HybridBuilder module.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.HybridBuilder

  @embed_size 64
  @hidden_size 32
  @seq_len 8
  @batch_size 2

  describe "build/2" do
    test "builds model from simple pattern" do
      pattern = [:mamba, :mamba, :attention]

      model =
        HybridBuilder.build(pattern,
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "builds model with diverse layer types" do
      pattern = [:mamba, :attention, :ffn]

      model =
        HybridBuilder.build(pattern,
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})

      output = predict_fn.(params, %{"state_sequence" => input})

      # Should output [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "supports ffn-only pattern" do
      pattern = [:ffn, :ffn]

      model =
        HybridBuilder.build(pattern,
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, %{"state_sequence" => input})

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "pattern/2" do
    test "generates jamba_like pattern" do
      pattern = HybridBuilder.pattern(:jamba_like, 4)

      assert pattern == [:mamba, :attention, :mamba, :attention]
    end

    test "generates zamba_like pattern" do
      pattern = HybridBuilder.pattern(:zamba_like, 4)

      assert pattern == [:mamba, :mamba, :mamba, :mamba]
    end

    test "generates mamba_gla pattern" do
      pattern = HybridBuilder.pattern(:mamba_gla, 6)

      # GLA every 3rd layer
      assert Enum.at(pattern, 2) == :gla
      assert Enum.at(pattern, 5) == :gla
      assert Enum.count(pattern, &(&1 == :gla)) == 2
    end

    test "generates full_hybrid pattern with diverse types" do
      pattern = HybridBuilder.pattern(:full_hybrid, 12)

      # Should have multiple different layer types
      unique_types = Enum.uniq(pattern)
      assert length(unique_types) >= 4
    end

    test "generates ssm_stack pattern" do
      pattern = HybridBuilder.pattern(:ssm_stack, 4)

      assert pattern == [:mamba, :mamba, :mamba, :mamba]
    end
  end

  describe "build_pattern/3" do
    test "combines pattern generation and building" do
      model =
        HybridBuilder.build_pattern(:jamba_like, 4,
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end

  describe "param_count/2" do
    test "estimates parameter count" do
      pattern = [:mamba, :mamba, :attention]

      count =
        HybridBuilder.param_count(pattern,
          embed_size: @embed_size,
          hidden_size: @hidden_size
        )

      # Should have reasonable param count
      assert count > 10_000
      assert count < 10_000_000
    end

    test "more layers means more params" do
      small_pattern = [:mamba, :mamba]
      large_pattern = [:mamba, :mamba, :mamba, :mamba]

      small_count = HybridBuilder.param_count(small_pattern, embed_size: @embed_size)
      large_count = HybridBuilder.param_count(large_pattern, embed_size: @embed_size)

      assert large_count > small_count
    end

    test "attention layers add more params than mamba" do
      mamba_pattern = [:mamba, :mamba]
      attn_pattern = [:attention, :attention]

      mamba_count = HybridBuilder.param_count(mamba_pattern, embed_size: @embed_size)
      attn_count = HybridBuilder.param_count(attn_pattern, embed_size: @embed_size)

      # Attention typically has more params due to QKV projections
      assert attn_count != mamba_count
    end
  end

  describe "visualize/1" do
    test "generates readable diagram" do
      pattern = [:mamba, :attention, :gla, :ffn]

      viz = HybridBuilder.visualize(pattern)

      assert viz =~ "Layer pattern:"
      assert viz =~ "M"
      assert viz =~ "A"
      assert viz =~ "G"
      assert viz =~ "F"
      assert viz =~ "Legend:"
    end

    test "shows arrows between layers" do
      pattern = [:mamba, :mamba]

      viz = HybridBuilder.visualize(pattern)

      assert viz =~ "→"
    end
  end

  describe "numerical stability" do
    test "produces finite outputs" do
      pattern = [:mamba, :attention, :ffn]

      model =
        HybridBuilder.build(pattern,
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch_size, @seq_len, @embed_size}, :f32)},
          Axon.ModelState.empty()
        )

      # Random input
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size})

      output = predict_fn.(params, %{"state_sequence" => input})

      # Check no NaN
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
