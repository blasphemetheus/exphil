Code.require_file("../../scripts/lib/mamba_choice_metrics.exs", __DIR__)

defmodule MambaChoiceMetricsTest do
  use ExUnit.Case, async: true

  test "detects a button threshold crossing and preserves row and reset counts" do
    logits = %{buttons: Nx.tensor([[0.00001, -2.0], [1.0, -1.0]])}

    logits =
      Enum.reduce([:main_x, :main_y, :c_x, :c_y, :shoulder], logits, fn key, acc ->
        Map.put(acc, key, Nx.tensor([[2.0, 1.5], [3.0, 1.0]]))
      end)

    reference =
      Map.new(logits, fn {k, _} -> {k, Nx.tensor([0, 0])} end)
      |> Map.put(:buttons, Nx.tensor([[1, 0], [1, 0]]))
      |> Map.put(:logits, logits)

    candidate = %{
      reference
      | buttons: Nx.tensor([[0, 0], [1, 0]]),
        logits: %{logits | buttons: Nx.tensor([[-0.00001, -2.0], [1.0, -1.0]])}
    }

    changed = MambaChoiceMetrics.record(candidate, reference, 0)
    same = MambaChoiceMetrics.record(candidate, reference, 1)
    assert changed.changed_components == [:buttons]
    assert same.changed_components == []
    assert changed.changed_details.buttons.candidate_choice == [0, 0]
    assert changed.changed_details.buttons.reference_choice == [1, 0]
    assert_in_delta changed.changed_details.buttons.max_probability_change, 0.000005, 1.0e-9
    assert_in_delta changed.reference_margins.buttons, 0.00001, 1.0e-10
    assert changed.reference_margins.main_x == 0.5
    assert_in_delta changed.max_logit_error, 0.00002, 1.0e-10

    summary =
      MambaChoiceMetrics.summarize([
        Map.put(changed, :resync, true),
        Map.put(same, :resync, false)
      ])

    assert summary.decisions == 2
    assert summary.changed_decisions == 1
    assert summary.resync_changes == 1
    assert summary.ordinary_changes == 0
  end

  test "categorical tie reports zero margin and catches a changed bucket" do
    keys = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

    reference =
      Map.new(keys, &{&1, Nx.tensor([[0]])})
      |> Map.put(:logits, Map.new(keys, &{&1, Nx.tensor([[1.0, 1.0]])}))

    changed = MambaChoiceMetrics.record(%{reference | main_x: Nx.tensor([[1]])}, reference, 0)
    assert changed.changed_components == [:main_x]
    assert changed.reference_margins.main_x == 0.0
  end
end
