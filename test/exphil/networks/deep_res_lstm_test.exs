defmodule ExPhil.Networks.DeepResLSTMTest do
  @moduledoc """
  Tests for the DeepResLSTM backbone — slippi-ai's res_lstm architecture.

  Tests both the ExPhil wrapper (ExPhil.Networks.DeepResLSTM) and
  the backbone integration (via ExPhil.Networks.Policy.Backbone).
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.DeepResLSTM

  @embed_size 64
  @hidden_size 32
  @seq_len 12
  @batch_size 4

  describe "build/1" do
    test "builds model with correct output shape" do
      model =
        DeepResLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
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
        DeepResLSTM.build(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles same embed and hidden sizes" do
      model =
        DeepResLSTM.build(
          embed_size: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 2,
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

    test "handles different num_layers" do
      for num_layers <- [1, 2, 3] do
        model =
          DeepResLSTM.build(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: num_layers,
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
  end

  describe "numerical stability" do
    test "produces finite outputs for normal inputs" do
      model =
        DeepResLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          window_size: @seq_len
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
        DeepResLSTM.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          window_size: @seq_len
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

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DeepResLSTM.output_size(hidden_size: 128) == 128
      assert DeepResLSTM.output_size(hidden_size: 256) == 256
    end

    test "returns default when no option given" do
      assert DeepResLSTM.output_size() == 512
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        DeepResLSTM.param_count(
          embed_size: 287,
          hidden_size: 512,
          num_layers: 3
        )

      # Should be substantial for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      count_2 =
        DeepResLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 2
        )

      count_4 =
        DeepResLSTM.param_count(
          embed_size: 64,
          hidden_size: 32,
          num_layers: 4
        )

      assert count_4 > count_2 * 1.8
      assert count_4 < count_2 * 2.2
    end

    test "fewer params than TransformerLike (no FFN blocks)" do
      deep_res_count =
        DeepResLSTM.param_count(
          embed_size: 287,
          hidden_size: 512,
          num_layers: 3
        )

      tx_like_count =
        ExPhil.Networks.TransformerLike.param_count(
          embed_size: 287,
          hidden_size: 512,
          num_layers: 3
        )

      # DeepResLSTM should have fewer params (no FFN blocks)
      assert deep_res_count < tx_like_count
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = DeepResLSTM.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 512
      assert Keyword.get(defaults, :num_layers) == 3
      assert Keyword.get(defaults, :window_size) == 60
    end

    @tag :slow
    test "builds a valid model with defaults" do
      defaults = DeepResLSTM.melee_defaults()

      model = DeepResLSTM.build([embed_size: 287] ++ defaults)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 60, 287}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {1, 60, 287})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 512}
    end
  end

  describe "integration with Policy" do
    test "works as deep_res_lstm temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :deep_res_lstm,
          hidden_size: @hidden_size,
          num_layers: 2,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, input)

      # Check policy output shapes (6-head autoregressive)
      assert Nx.shape(buttons) == {@batch_size, 8}
      assert Nx.shape(main_x) == {@batch_size, 17}
      assert Nx.shape(main_y) == {@batch_size, 17}
      assert Nx.shape(c_x) == {@batch_size, 17}
      assert Nx.shape(c_y) == {@batch_size, 17}
      assert Nx.shape(shoulder) == {@batch_size, 5}
    end
  end
end
