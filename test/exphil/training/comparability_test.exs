defmodule ExPhil.Training.ComparabilityTest do
  @moduledoc "INVARIANTS.md item 11: checkpoints carry a comparability key."
  use ExUnit.Case, async: true

  alias ExPhil.Training.Comparability

  @base [
    frame_delay: 1,
    action_delay: 0,
    embed_canary: [1.0, 2.0, 3.0],
    label_smoothing: 0.0,
    button_pos_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    focal_loss: false,
    focal_gamma: 3.0,
    button_weight: 2.0,
    action_oversample: 1.0,
    entropy_weight: 0.0,
    neutral_weight: 1.0,
    transition_weight: nil,
    offstage_weight: nil,
    stick_edge_weight: 1.0,
    head: :autoregressive,
    train_delays: [1]
  ]

  test "same config => same key; json round-trip (string keys/values) => same key" do
    k = Comparability.key(@base)
    # @base is unstamped (legacy producing convention): delay 1 = reaction 0
    assert k.label_delay == 0
    assert k.train_delays == [0]
    assert is_binary(k.embed_canary) and is_binary(k.loss_recipe)

    json = @base |> Map.new() |> Jason.encode!() |> Jason.decode!()
    assert Comparability.key(json) == k
    assert Comparability.comparable?(@base, json)
  end

  test "legacy delay 0 (leaked) is NOT comparable with legacy delay 1 (causal)" do
    leaky = Keyword.put(@base, :frame_delay, 0)
    refute Comparability.comparable?(@base, leaky)
    assert Comparability.key(leaky).label_delay == -1
    assert Comparability.explain(Comparability.key(@base), Comparability.key(leaky)) =~ "label_delay"
  end

  test "a stamped causal config at delay 0 IS comparable with the legacy delay-1 config" do
    causal = @base |> Keyword.merge(frame_delay: 0, train_delays: [0], label_convention: :causal)
    assert Comparability.comparable?(@base, causal)
  end

  test "loss recipe and canary changes break comparability; irrelevant keys do not" do
    refute Comparability.comparable?(@base, Keyword.put(@base, :neutral_weight, 0.25))
    refute Comparability.comparable?(@base, Keyword.put(@base, :button_pos_weight, :auto))
    refute Comparability.comparable?(@base, Keyword.put(@base, :embed_canary, [1.0, 2.0, 4.0]))
    assert Comparability.comparable?(@base, Keyword.merge(@base, epochs: 99, batch_size: 7, name: "x"))
  end

  test "check/1 groups mixed keys" do
    assert :ok = Comparability.check([{"a", @base}, {"b", @base}])
    assert {:error, groups} = Comparability.check([{"a", @base}, {"b", Keyword.put(@base, :frame_delay, 0)}])
    assert map_size(groups) == 2
  end
end
