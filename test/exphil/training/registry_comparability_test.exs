defmodule ExPhil.Training.RegistryComparabilityTest do
  @moduledoc "INVARIANTS.md item 11: Registry.best/1 refuses to rank across comparability keys."
  use ExUnit.Case, async: false

  alias ExPhil.Training.Registry

  @test_registry "test/fixtures/test_registry_comparability.json"

  setup do
    File.rm(@test_registry)
    original = Application.get_env(:exphil, :registry_path)
    Application.put_env(:exphil, :registry_path, @test_registry)

    on_exit(fn ->
      File.rm(@test_registry)
      if original, do: Application.put_env(:exphil, :registry_path, original), else: Application.delete_env(:exphil, :registry_path)
    end)

    :ok
  end

  # A new-style causal checkpoint (stamped) vs a legacy leaked one (unstamped, delay 0).
  @causal %{frame_delay: 0, label_smoothing: 0.0, neutral_weight: 1.0, train_delays: [0], label_convention: :causal}
  @leaky %{frame_delay: 0, label_smoothing: 0.0, neutral_weight: 1.0, train_delays: [0]}

  test "best/1 refuses a leaked-vs-causal ranking unless allow_incomparable" do
    {:ok, _} = Registry.register(%{checkpoint_path: "c/causal.axon", name: "causal", training_config: @causal, metrics: %{final_loss: 2.4}})
    {:ok, _} = Registry.register(%{checkpoint_path: "c/leaky.axon", name: "leaky", training_config: @leaky, metrics: %{final_loss: 1.8}})

    # The leaked loss is LOWER for a worse policy — exactly the trap.
    assert {:error, err} = Registry.best(metric: :final_loss)
    assert err.reason == :incomparable

    assert {:ok, best} = Registry.best(metric: :final_loss, allow_incomparable: true)
    assert best.name == "leaky"
  end

  test "best/1 ranks freely within one key" do
    {:ok, _} = Registry.register(%{checkpoint_path: "c/a.axon", name: "a", training_config: @causal, metrics: %{final_loss: 2.4}})
    {:ok, _} = Registry.register(%{checkpoint_path: "c/b.axon", name: "b", training_config: Map.put(@causal, :epochs, 9), metrics: %{final_loss: 2.1}})
    assert {:ok, best} = Registry.best(metric: :final_loss)
    assert best.name == "b"
  end
end
