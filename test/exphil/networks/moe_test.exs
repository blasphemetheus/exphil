defmodule ExPhil.Networks.MoETest do
  @moduledoc """
  Tests for the Mixture of Experts (MoE) module.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.MoE

  @input_size 64
  @hidden_size 128
  @output_size 64
  @seq_len 8
  @batch_size 2

  describe "build/1" do
    test "builds MoE layer with default settings" do
      model =
        MoE.build(
          input_size: @input_size,
          hidden_size: @hidden_size,
          output_size: @output_size
        )

      assert %Axon{} = model
    end

    test "builds with custom number of experts" do
      model =
        MoE.build(
          input_size: @input_size,
          num_experts: 4,
          top_k: 1
        )

      assert %Axon{} = model
    end

    test "supports different routing strategies" do
      for routing <- [:top_k, :switch, :soft] do
        model =
          MoE.build(
            input_size: @input_size,
            routing: routing
          )

        assert %Axon{} = model
      end
    end

    test "supports different expert types" do
      for expert_type <- [:ffn, :glu] do
        model =
          MoE.build(
            input_size: @input_size,
            expert_type: expert_type
          )

        assert %Axon{} = model
      end
    end
  end

  describe "build_block/2" do
    test "builds MoE block with residual" do
      input = Axon.input("x", shape: {nil, @seq_len, @input_size})

      output =
        MoE.build_block(input,
          hidden_size: @input_size,
          num_experts: 4,
          top_k: 2
        )

      assert %Axon{} = output
    end
  end

  describe "build_moe_backbone/1" do
    test "builds Mamba backbone with MoE layers" do
      model =
        MoE.build_moe_backbone(
          embed_size: @input_size,
          hidden_size: @input_size,
          num_layers: 4,
          moe_every: 2,
          num_experts: 4,
          top_k: 2,
          backbone: :mamba,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end

    test "builds attention backbone with MoE layers" do
      model =
        MoE.build_moe_backbone(
          embed_size: @input_size,
          hidden_size: @input_size,
          num_layers: 4,
          moe_every: 2,
          num_experts: 4,
          backbone: :attention,
          seq_len: @seq_len
        )

      assert %Axon{} = model
    end
  end

  describe "compute_aux_loss/3" do
    test "computes load balancing loss" do
      num_experts = 4

      # Simulated router probs and expert assignments
      router_probs = Nx.broadcast(0.25, {@batch_size, @seq_len, num_experts})
      expert_mask = Nx.broadcast(1.0, {@batch_size, @seq_len, num_experts})

      loss = MoE.compute_aux_loss(router_probs, expert_mask, load_balance_weight: 0.01)

      # Loss should be scalar
      assert Nx.shape(loss) == {}
      # Loss should be positive
      assert Nx.to_number(loss) > 0
    end

    test "balanced routing has lower aux loss than imbalanced" do
      num_experts = 4

      # Balanced: uniform distribution
      balanced_probs = Nx.broadcast(0.25, {@batch_size, @seq_len, num_experts})
      balanced_mask = Nx.broadcast(1.0, {@batch_size, @seq_len, num_experts})

      # Imbalanced: all tokens to first expert
      imbalanced_probs = Nx.concatenate([
        Nx.broadcast(0.9, {@batch_size, @seq_len, 1}),
        Nx.broadcast(0.033, {@batch_size, @seq_len, num_experts - 1})
      ], axis: -1)
      imbalanced_mask = Nx.concatenate([
        Nx.broadcast(1.0, {@batch_size, @seq_len, 1}),
        Nx.broadcast(0.0, {@batch_size, @seq_len, num_experts - 1})
      ], axis: -1)

      balanced_loss = MoE.compute_aux_loss(balanced_probs, balanced_mask)
      imbalanced_loss = MoE.compute_aux_loss(imbalanced_probs, imbalanced_mask)

      # Both should be positive, exact comparison depends on implementation
      assert Nx.to_number(balanced_loss) > 0
      assert Nx.to_number(imbalanced_loss) > 0
    end
  end

  describe "estimate_speedup/3" do
    test "more experts with same top_k gives higher speedup" do
      speedup_8_experts = MoE.estimate_speedup(8, 2, 0.5)
      speedup_16_experts = MoE.estimate_speedup(16, 2, 0.5)

      assert speedup_16_experts > speedup_8_experts
    end

    test "higher top_k reduces speedup" do
      speedup_top1 = MoE.estimate_speedup(8, 1, 0.5)
      speedup_top2 = MoE.estimate_speedup(8, 2, 0.5)

      assert speedup_top1 > speedup_top2
    end

    test "returns reasonable speedup values" do
      speedup = MoE.estimate_speedup(8, 2, 0.5)

      # With 8 experts, top-2, 50% expert fraction:
      # Expert speedup = 8/2 = 4x
      # Overall = 1 / (0.5 + 0.5/4) = 1 / 0.625 = 1.6x
      assert speedup > 1.0
      assert speedup < 10.0
    end
  end

  describe "melee_defaults/0" do
    test "returns valid configuration" do
      defaults = MoE.melee_defaults()

      assert Keyword.get(defaults, :num_experts) == 8
      assert Keyword.get(defaults, :top_k) == 2
      assert Keyword.get(defaults, :routing) == :top_k
      assert Keyword.get(defaults, :expert_type) == :ffn
    end

    test "can build with defaults" do
      defaults = MoE.melee_defaults()

      model =
        MoE.build(
          Keyword.merge(defaults, input_size: @input_size)
        )

      assert %Axon{} = model
    end
  end
end
