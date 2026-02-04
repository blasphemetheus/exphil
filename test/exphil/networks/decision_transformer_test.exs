defmodule ExPhil.Networks.DecisionTransformerTest do
  @moduledoc """
  Tests for Decision Transformer backbone.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.DecisionTransformer

  @state_size 64
  @action_size 32
  @hidden_size 32
  @context_length 8
  @batch_size 4

  describe "build/1" do
    test "builds full model with correct output shape" do
      model =
        DecisionTransformer.build(
          state_size: @state_size,
          action_size: @action_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
          context_length: @context_length
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "returns" => Nx.template({@batch_size, @context_length}, :f32),
            "states" => Nx.template({@batch_size, @context_length, @state_size}, :f32),
            "actions" => Nx.template({@batch_size, @context_length, @action_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      input = %{
        "returns" => Nx.broadcast(0.5, {@batch_size, @context_length}),
        "states" => Nx.broadcast(0.5, {@batch_size, @context_length, @state_size}),
        "actions" => Nx.broadcast(0.5, {@batch_size, @context_length, @action_size})
      }

      output = predict_fn.(params, input)

      # Output should be [batch, context_length, action_size]
      assert Nx.shape(output) == {@batch_size, @context_length, @action_size}
    end

    test "handles different context lengths" do
      for context_length <- [4, 8, 16] do
        model =
          DecisionTransformer.build(
            state_size: @state_size,
            action_size: @action_size,
            hidden_size: @hidden_size,
            num_layers: 2,
            num_heads: 2,
            head_dim: 16,
            context_length: context_length
          )

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(
            %{
              "returns" => Nx.template({@batch_size, context_length}, :f32),
              "states" => Nx.template({@batch_size, context_length, @state_size}, :f32),
              "actions" => Nx.template({@batch_size, context_length, @action_size}, :f32)
            },
            Axon.ModelState.empty()
          )

        input = %{
          "returns" => Nx.broadcast(0.5, {@batch_size, context_length}),
          "states" => Nx.broadcast(0.5, {@batch_size, context_length, @state_size}),
          "actions" => Nx.broadcast(0.5, {@batch_size, context_length, @action_size})
        }

        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch_size, context_length, @action_size},
               "Failed for context_length=#{context_length}"
      end
    end
  end

  describe "build_simple/1" do
    @embed_size 64
    @seq_len 12

    test "builds simple model with correct output shape" do
      model =
        DecisionTransformer.build_simple(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
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
        DecisionTransformer.build_simple(
          embed_size: 128,
          hidden_size: 64,
          num_layers: 2,
          num_heads: 4,
          head_dim: 16,
          window_size: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, 128}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, 128})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "handles different num_layers" do
      for num_layers <- [1, 2, 4] do
        model =
          DecisionTransformer.build_simple(
            embed_size: @embed_size,
            hidden_size: @hidden_size,
            num_layers: num_layers,
            num_heads: 2,
            head_dim: 16,
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

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DecisionTransformer.output_size(hidden_size: 128) == 128
      assert DecisionTransformer.output_size(hidden_size: 256) == 256
      # default
      assert DecisionTransformer.output_size() == 256
    end
  end

  describe "param_count/1" do
    test "returns reasonable parameter count" do
      count =
        DecisionTransformer.param_count(
          state_size: 287,
          action_size: 64,
          hidden_size: 256,
          num_layers: 6
        )

      # Should be > 100k params for this config
      assert count > 100_000
      # But not unreasonably large
      assert count < 50_000_000
    end

    test "scales with num_layers" do
      count_2 =
        DecisionTransformer.param_count(
          state_size: 64,
          action_size: 32,
          hidden_size: 32,
          num_layers: 2
        )

      count_4 =
        DecisionTransformer.param_count(
          state_size: 64,
          action_size: 32,
          hidden_size: 32,
          num_layers: 4
        )

      # 4 layers should have more params than 2 layers
      assert count_4 > count_2
    end
  end

  describe "melee_defaults/0" do
    test "returns valid default configuration" do
      defaults = DecisionTransformer.melee_defaults()

      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :num_layers) == 6
      assert Keyword.get(defaults, :num_heads) == 8
      assert Keyword.get(defaults, :context_length) == 20
    end
  end

  describe "compute_return_to_go/2" do
    test "computes stock lead correctly" do
      game_state = %{player_stocks: 4, opponent_stocks: 2}
      rtg = DecisionTransformer.compute_return_to_go(game_state, mode: :stock_lead)
      assert rtg == 2

      game_state = %{player_stocks: 1, opponent_stocks: 3}
      rtg = DecisionTransformer.compute_return_to_go(game_state, mode: :stock_lead)
      assert rtg == -2
    end

    test "computes damage lead correctly" do
      game_state = %{player_damage: 50, opponent_damage: 100}
      rtg = DecisionTransformer.compute_return_to_go(game_state, mode: :damage_lead)
      assert rtg == 0.5  # (100 - 50) / 100

      game_state = %{player_damage: 100, opponent_damage: 50}
      rtg = DecisionTransformer.compute_return_to_go(game_state, mode: :damage_lead)
      assert rtg == -0.5
    end

    test "computes game outcome correctly" do
      assert DecisionTransformer.compute_return_to_go(%{game_result: :win}, mode: :game_outcome) == 1.0
      assert DecisionTransformer.compute_return_to_go(%{game_result: :loss}, mode: :game_outcome) == 0.0
      assert DecisionTransformer.compute_return_to_go(%{game_result: :ongoing}, mode: :game_outcome) == 0.5
    end
  end

  describe "numerical stability" do
    test "produces finite outputs for normal inputs" do
      model =
        DecisionTransformer.build_simple(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
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
        DecisionTransformer.build_simple(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
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
    @embed_size 64
    @seq_len 12

    test "works as decision_transformer temporal backbone in Policy" do
      alias ExPhil.Networks.Policy

      model =
        Policy.build_temporal(
          embed_size: @embed_size,
          backbone: :decision_transformer,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
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
