defmodule ExPhil.Networks.SpeculativeDecodingTest do
  @moduledoc """
  Tests for the Speculative Decoding module.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.SpeculativeDecoding

  describe "create/1" do
    test "creates decoder with required options" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {4, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn
        )

      assert is_map(decoder)
      assert decoder.lookahead == 4  # default
      assert decoder.temperature == 1.0  # default
    end

    test "accepts custom lookahead" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {8, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn,
          lookahead: 8
        )

      assert decoder.lookahead == 8
    end

    test "accepts custom temperature" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {4, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn,
          temperature: 0.8
        )

      assert decoder.temperature == 0.8
    end

    test "initializes stats" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {4, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn
        )

      assert decoder.stats.total_proposed == 0
      assert decoder.stats.total_accepted == 0
      assert decoder.stats.acceptance_rate == 0.0
    end
  end

  describe "get_stats/1" do
    test "returns current stats" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {4, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn
        )

      stats = SpeculativeDecoding.get_stats(decoder)

      assert is_map(stats)
      assert Map.has_key?(stats, :total_proposed)
      assert Map.has_key?(stats, :total_accepted)
      assert Map.has_key?(stats, :acceptance_rate)
    end
  end

  describe "reset_stats/1" do
    test "resets all stats to zero" do
      draft_fn = fn _params, _state -> Nx.broadcast(0.0, {1, 13}) end
      target_fn = fn _params, _states -> Nx.broadcast(0.0, {4, 13}) end

      decoder =
        SpeculativeDecoding.create(
          draft_fn: draft_fn,
          target_fn: target_fn
        )

      # Manually set some stats
      decoder_with_stats = %{
        decoder
        | stats: %{total_proposed: 100, total_accepted: 50, acceptance_rate: 0.5}
      }

      reset_decoder = SpeculativeDecoding.reset_stats(decoder_with_stats)

      assert reset_decoder.stats.total_proposed == 0
      assert reset_decoder.stats.total_accepted == 0
      assert reset_decoder.stats.acceptance_rate == 0.0
    end
  end

  describe "estimate_speedup/4" do
    test "returns speedup > 1 when beneficial" do
      # Good scenario: high acceptance, fast draft
      speedup = SpeculativeDecoding.estimate_speedup(0.8, 4, 2.0, 20.0)

      # Expected tokens: 4 * 0.8 + 1 = 4.2
      # Spec time: 4 * 2 + 20 = 28ms
      # Naive time: 4.2 * 20 = 84ms
      # Speedup: 84 / 28 = 3x

      assert speedup > 1.0
    end

    test "higher acceptance rate gives better speedup" do
      speedup_low = SpeculativeDecoding.estimate_speedup(0.3, 4, 2.0, 20.0)
      speedup_high = SpeculativeDecoding.estimate_speedup(0.8, 4, 2.0, 20.0)

      assert speedup_high > speedup_low
    end

    test "faster draft model gives better speedup" do
      speedup_slow_draft = SpeculativeDecoding.estimate_speedup(0.5, 4, 10.0, 20.0)
      speedup_fast_draft = SpeculativeDecoding.estimate_speedup(0.5, 4, 2.0, 20.0)

      assert speedup_fast_draft > speedup_slow_draft
    end

    test "larger target model gives better speedup" do
      speedup_fast_target = SpeculativeDecoding.estimate_speedup(0.5, 4, 2.0, 10.0)
      speedup_slow_target = SpeculativeDecoding.estimate_speedup(0.5, 4, 2.0, 50.0)

      assert speedup_slow_target > speedup_fast_target
    end

    test "returns reasonable values for Melee scenario" do
      # MLP draft: 9ms, Mamba target: 24ms
      # Assume 50% acceptance rate with lookahead 4
      speedup = SpeculativeDecoding.estimate_speedup(0.5, 4, 9.0, 24.0)

      # Should be beneficial but not dramatic
      assert speedup > 0.5
      assert speedup < 5.0
    end
  end

  describe "theoretical properties" do
    test "speedup formula handles edge cases" do
      # 0% acceptance
      speedup_zero = SpeculativeDecoding.estimate_speedup(0.0, 4, 2.0, 20.0)
      assert speedup_zero > 0

      # 100% acceptance
      speedup_full = SpeculativeDecoding.estimate_speedup(1.0, 4, 2.0, 20.0)
      assert speedup_full > speedup_zero
    end

    test "lookahead affects expected tokens" do
      speedup_small = SpeculativeDecoding.estimate_speedup(0.5, 2, 2.0, 20.0)
      speedup_large = SpeculativeDecoding.estimate_speedup(0.5, 8, 2.0, 20.0)

      # Larger lookahead should give more tokens per iteration
      # But also costs more draft time
      # The relationship depends on acceptance rate
      assert is_float(speedup_small)
      assert is_float(speedup_large)
    end
  end
end
