defmodule ExPhil.Networks.MambaSSDTest do
  @moduledoc """
  Tests for Mamba SSD (State Space Duality) implementation.

  Tests both inference mode (scan-based) and training mode (matmul-based).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.MambaSSD

  @embed_size 64
  @hidden_size 32
  @state_size 8
  @seq_len 12
  @batch_size 4

  describe "build/1 inference mode (default)" do
    test "builds model with correct output shape" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles chunk_size larger than seq_len" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: @seq_len,
          chunk_size: 32  # Larger than seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles sequences with remainder chunks" do
      # seq_len=14 with chunk_size=4 = 3 full chunks + 2 remainder
      seq_len = 14

      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: seq_len,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build/1 training mode" do
    test "builds model with correct output shape" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len,
          chunk_size: 4,
          training_mode: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles single chunk (matmul formulation)" do
      # Sequence fits in one chunk
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: @seq_len,
          chunk_size: 16,  # Larger than seq_len
          training_mode: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "handles multiple chunks with remainder" do
      seq_len = 15  # 3 chunks of 4 + 3 remainder

      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: seq_len,
          chunk_size: 4,
          training_mode: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MambaSSD.output_size(hidden_size: 128) == 128
      assert MambaSSD.output_size(hidden_size: 256) == 256
      # default
      assert MambaSSD.output_size() == 256
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration with SSD options" do
      defaults = MambaSSD.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :chunk_size) == 16
      assert Keyword.get(defaults, :training_mode) == false
    end
  end

  describe "training_defaults/0" do
    test "returns training-optimized configuration" do
      defaults = MambaSSD.training_defaults()

      assert Keyword.get(defaults, :training_mode) == true
      assert Keyword.get(defaults, :chunk_size) == 32
    end
  end

  describe "numerical stability" do
    test "produces finite outputs for normal inputs (inference)" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "produces finite outputs for normal inputs (training)" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len,
          chunk_size: 4,
          training_mode: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch_size, @seq_len, @embed_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles zero input" do
      model =
        MambaSSD.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 1,
          window_size: @seq_len,
          chunk_size: 4
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "integration with Policy" do
    test "works as mamba_ssd temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :mamba_ssd,
          hidden_size: @hidden_size,
          state_size: @state_size,
          num_layers: 2,
          window_size: @seq_len,
          chunk_size: 4
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
