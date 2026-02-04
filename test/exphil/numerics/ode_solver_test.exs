defmodule ExPhil.Numerics.ODESolverTest do
  @moduledoc """
  Tests for the ODE Solver module.

  Tests numerical accuracy and correctness of various ODE solvers.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Numerics.ODESolver

  describe "solve/5" do
    test "Euler solver approximates exponential decay" do
      # dx/dt = -x, solution: x(t) = x0 * e^(-t)
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.01)

      # e^(-1) ≈ 0.3679
      expected = :math.exp(-1.0)
      actual = result |> Nx.squeeze() |> Nx.to_number()

      # Euler should be within 5% for small dt
      assert_in_delta actual, expected, 0.05
    end

    test "Midpoint solver is more accurate than Euler" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      euler_result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.1)
      midpoint_result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :midpoint, dt: 0.1)

      expected = :math.exp(-1.0)
      euler_error = abs((euler_result |> Nx.squeeze() |> Nx.to_number()) - expected)
      midpoint_error = abs((midpoint_result |> Nx.squeeze() |> Nx.to_number()) - expected)

      # Midpoint should be more accurate
      assert midpoint_error < euler_error
    end

    test "RK4 solver is highly accurate" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

      expected = :math.exp(-1.0)
      actual = result |> Nx.squeeze() |> Nx.to_number()

      # RK4 should be within 0.1% even with large dt
      assert_in_delta actual, expected, 0.001
    end

    test "DOPRI5 solver with adaptive stepping" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([1.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :dopri5, atol: 1.0e-6, rtol: 1.0e-4)

      expected = :math.exp(-1.0)
      actual = result |> Nx.squeeze() |> Nx.to_number()

      # DOPRI5 should be very accurate
      assert_in_delta actual, expected, 0.0001
    end

    test "handles multi-dimensional state" do
      # System: dx/dt = -x, dy/dt = x - y
      f = fn _t, state ->
        x = Nx.slice_along_axis(state, 0, 1, axis: 0)
        y = Nx.slice_along_axis(state, 1, 1, axis: 0)
        Nx.concatenate([Nx.negate(x), Nx.subtract(x, y)])
      end

      x0 = Nx.tensor([1.0, 0.0])
      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

      # Check dimensions preserved
      assert Nx.shape(result) == {2}

      # Check no NaN
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles batched tensors" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([[1.0], [2.0], [3.0]])  # batch of 3

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

      expected = :math.exp(-1.0)
      assert Nx.shape(result) == {3, 1}

      # Each element should decay proportionally
      for i <- 0..2 do
        actual = Nx.to_number(Nx.slice_along_axis(result, i, 1, axis: 0) |> Nx.squeeze())
        assert_in_delta actual, (i + 1) * expected, 0.01
      end
    end
  end

  describe "solve_ltc/4" do
    test "converges to activation value" do
      # LTC: dx/dt = (activation - x) / tau
      # At equilibrium, x = activation
      x = Nx.tensor([0.0, 0.0])
      activation = Nx.tensor([1.0, 2.0])
      tau = Nx.tensor([0.5, 0.5])  # Fast convergence

      # Use more steps for better convergence
      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 8)

      # Should be close to activation after integration
      # With tau=0.5 and 8 steps over dt=1.0, should converge well
      assert_in_delta Nx.slice_along_axis(result, 0, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number(), 1.0, 0.15
      assert_in_delta Nx.slice_along_axis(result, 1, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number(), 2.0, 0.3
    end

    test "larger tau means slower convergence" do
      x = Nx.tensor([0.0])
      activation = Nx.tensor([1.0])
      # Use tau values that are stable (not too small to avoid stiffness)
      tau_fast = Nx.tensor([0.5])
      tau_slow = Nx.tensor([10.0])

      # Use multiple steps for stable integration
      result_fast = ODESolver.solve_ltc(x, activation, tau_fast, solver: :rk4, steps: 4)
      result_slow = ODESolver.solve_ltc(x, activation, tau_slow, solver: :rk4, steps: 4)

      # Faster tau (smaller value) should be closer to activation
      diff_fast = abs(1.0 - (result_fast |> Nx.squeeze() |> Nx.to_number()))
      diff_slow = abs(1.0 - (result_slow |> Nx.squeeze() |> Nx.to_number()))

      assert diff_fast < diff_slow
    end

    test "handles batched inputs" do
      batch = 4
      hidden = 8

      x = Nx.broadcast(0.0, {batch, hidden})
      activation = Nx.broadcast(1.0, {batch, hidden})
      tau = Nx.broadcast(1.0, {batch, hidden})

      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 2)

      assert Nx.shape(result) == {batch, hidden}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "Euler solver works for LTC" do
      x = Nx.tensor([0.0])
      activation = Nx.tensor([1.0])
      tau = Nx.tensor([1.0])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :euler, steps: 4)

      assert Nx.shape(result) == {1}
      result_val = result |> Nx.squeeze() |> Nx.to_number()
      assert result_val > 0.0
      assert result_val < 1.0
    end

    test "midpoint solver works for LTC" do
      x = Nx.tensor([0.0])
      activation = Nx.tensor([1.0])
      tau = Nx.tensor([1.0])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :midpoint, steps: 2)

      assert Nx.shape(result) == {1}
      assert result |> Nx.squeeze() |> Nx.to_number() > 0.0
    end

    test "DOPRI5 adaptive solver works for LTC" do
      x = Nx.tensor([0.0, 0.0])
      activation = Nx.tensor([1.0, 2.0])
      tau = Nx.tensor([0.5, 0.5])

      result = ODESolver.solve_ltc(x, activation, tau, solver: :dopri5, atol: 1.0e-4, rtol: 1.0e-2)

      assert Nx.shape(result) == {2}
      # Should converge close to activation
      assert Nx.slice_along_axis(result, 0, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number() > 0.5
    end
  end

  describe "numerical stability" do
    test "handles very small time constants" do
      x = Nx.tensor([0.0])
      activation = Nx.tensor([1.0])
      tau = Nx.tensor([0.01])  # Very fast

      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 4)

      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "handles very large time constants" do
      x = Nx.tensor([0.0])
      activation = Nx.tensor([1.0])
      tau = Nx.tensor([1000.0])  # Very slow

      result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 1)

      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      # Should barely move from initial value
      assert result |> Nx.squeeze() |> Nx.to_number() < 0.01
    end

    test "handles zero initial condition" do
      f = fn _t, x -> Nx.negate(x) end
      x0 = Nx.tensor([0.0])

      result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)

      # Should stay at zero
      assert_in_delta result |> Nx.squeeze() |> Nx.to_number(), 0.0, 1.0e-10
    end
  end
end
