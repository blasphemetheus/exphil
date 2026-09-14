defmodule ExPhil.Training.EpochHealthTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.EpochHealth

  test "genuine small and zero losses are accepted without a scale threshold" do
    for loss <- [0.2, 0.000001, 0.00000001, 0.0, -0.01] do
      assert EpochHealth.validate(loss, %{weights: Nx.tensor([1.0, -2.0])}) == :ok
    end
  end

  test "nonfinite losses and poisoned parameters are rejected independently" do
    for loss <- [:nan, :infinity, :neg_infinity] do
      assert EpochHealth.validate(loss, %{weights: Nx.tensor([1.0])}) == {:error, :nonfinite_loss}
    end

    for value <- [:nan, :infinity, :neg_infinity] do
      assert EpochHealth.validate(0.000001, %{nested: {Nx.tensor([value]), []}}) ==
               {:error, :nonfinite_parameters}
    end
  end

  test "empty parameter trees fail closed" do
    assert EpochHealth.validate(0.0, %{}) == {:error, :missing_parameters}
  end

  test "checks an Axon model-state parameter tree" do
    parameters = %Axon.ModelState{data: %{"dense" => %{"kernel" => Nx.tensor([1.0])}}}
    assert EpochHealth.validate(1.0e-8, parameters) == :ok
    poisoned = put_in(parameters.data["dense"]["kernel"], Nx.tensor([:nan]))
    assert EpochHealth.validate(1.0e-8, poisoned) == {:error, :nonfinite_parameters}
  end
end
