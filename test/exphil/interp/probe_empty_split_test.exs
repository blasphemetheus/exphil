defmodule ExPhil.Interp.ProbeEmptySplitTest do
  @moduledoc """
  Regression for the r16 death (2026-07-22): probe-eval at epoch 10 hit a
  feature whose labeled rows all fell on ONE side of the positional 75/25
  split, and the "empty" fallback in mask_rows tried to build a {0}-shaped
  tensor — illegal in Nx — crashing the whole training run.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Interp.Probe

  defp activations(n), do: Nx.iota({n, 4}, type: :f32) |> Nx.divide(10.0)

  defp labels(list), do: Nx.tensor(list, type: :s64)

  test "all eval rows masked (-1) returns nil result instead of crashing" do
    x = activations(8)
    y = labels([0, 1, 0, 1, 0, 1, 0, 1])
    x_eval = activations(4)
    y_eval = labels([-1, -1, -1, -1])

    result = Probe.fit_eval(x, y, x_eval, y_eval, 2, steps: 5)

    assert result.balanced_accuracy == nil
    assert result.accuracy == nil
    assert result.n_train == 0 or result.n_eval == 0
    assert result.params == nil
  end

  test "all train rows masked (-1) returns nil result instead of crashing" do
    x = activations(8)
    y = labels(List.duplicate(-1, 8))
    x_eval = activations(4)
    y_eval = labels([0, 1, 0, 1])

    result = Probe.fit_eval(x, y, x_eval, y_eval, 2, steps: 5)

    assert result.balanced_accuracy == nil
    assert result.params == nil
  end

  test "partially masked rows still fit and evaluate" do
    x = activations(8)
    y = labels([0, 1, 0, 1, -1, -1, 0, 1])
    x_eval = activations(4)
    y_eval = labels([0, 1, -1, 1])

    result = Probe.fit_eval(x, y, x_eval, y_eval, 2, steps: 5)

    assert result.n_train == 6
    assert result.n_eval == 3
    assert is_number(result.accuracy)
  end

  test "single-class eval keeps the degenerate-balanced-accuracy guard" do
    x = activations(8)
    y = labels([0, 1, 0, 1, 0, 1, 0, 1])
    x_eval = activations(4)
    y_eval = labels([1, 1, 1, 1])

    result = Probe.fit_eval(x, y, x_eval, y_eval, 2, steps: 5)

    assert result.balanced_accuracy == nil
    assert is_number(result.accuracy)
  end
end
