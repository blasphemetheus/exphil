# ODE Solver Reference

`ExPhil.Numerics.ODESolver` - Numerical integration for Nx tensors, designed for Liquid Neural Networks.

## Overview

This module provides numerical methods for solving ordinary differential equations (ODEs) of the form:

```
dx/dt = f(t, x)
```

It's primarily used by the [Liquid Neural Networks](architectures/ARCHITECTURE_GUIDE.md#liquid---liquid-neural-networks) backbone, but can be used for any ODE integration with Nx tensors.

## Available Solvers

| Solver | Order | Adaptive | Speed | Accuracy | Best For |
|--------|-------|----------|-------|----------|----------|
| `:euler` | 1 | No | Fastest | Low | Simple problems, prototyping |
| `:midpoint` | 2 | No | Fast | Medium | Better than Euler, still fast |
| `:rk4` | 4 | No | Medium | Good | **Default** - good balance |
| `:dopri5` | 4/5 | Yes | Slower | Best | High accuracy requirements |

### Order Explained

The "order" refers to how the error scales with step size:
- **Order 1 (Euler):** Error ~ O(dt) - halving step size halves error
- **Order 2 (Midpoint):** Error ~ O(dt²) - halving step size quarters error
- **Order 4 (RK4):** Error ~ O(dt⁴) - halving step size reduces error 16x
- **Order 4/5 (DOPRI5):** Uses 4th and 5th order solutions for adaptive stepping

---

## Basic Usage

### General ODE Solving

```elixir
alias ExPhil.Numerics.ODESolver

# Solve dx/dt = -x (exponential decay)
# Analytical solution: x(t) = x₀ × e^(-t)
f = fn _t, x -> Nx.negate(x) end
x0 = Nx.tensor([1.0])

# Solve from t=0 to t=1
result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :rk4, dt: 0.1)
# result ≈ 0.368 (e^(-1))
```

### LTC (Liquid Time-Constant) Solving

For Liquid Neural Networks, use the optimized `solve_ltc/4`:

```elixir
# LTC ODE: dx/dt = (activation - x) / tau
x = Nx.tensor([0.0, 0.0])           # Current hidden state
activation = Nx.tensor([1.0, 2.0])  # Target activation
tau = Nx.tensor([0.5, 0.5])         # Time constants

result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 4)
# State evolves toward activation
```

---

## API Reference

### `solve/5`

Solve a general ODE from t₀ to t₁.

```elixir
@spec solve(
  (float(), Nx.Tensor.t() -> Nx.Tensor.t()),  # f(t, x) -> dx/dt
  float(),                                      # t0
  float(),                                      # t1
  Nx.Tensor.t(),                               # x0
  keyword()                                     # opts
) :: Nx.Tensor.t()
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `:solver` | `:rk4` | Solver type: `:euler`, `:midpoint`, `:rk4`, `:dopri5` |
| `:dt` | `0.01` | Time step for fixed-step methods |
| `:max_steps` | `1000` | Maximum steps for adaptive methods |
| `:atol` | `1.0e-6` | Absolute tolerance (DOPRI5 only) |
| `:rtol` | `1.0e-3` | Relative tolerance (DOPRI5 only) |

**Example:**

```elixir
# Harmonic oscillator: d²x/dt² = -x
# Convert to first-order system: [x, v]' = [v, -x]
f = fn _t, state ->
  x = Nx.slice_along_axis(state, 0, 1, axis: 0)
  v = Nx.slice_along_axis(state, 1, 1, axis: 0)
  Nx.concatenate([v, Nx.negate(x)])
end

x0 = Nx.tensor([1.0, 0.0])  # x=1, v=0
result = ODESolver.solve(f, 0.0, 3.14159, x0, solver: :rk4, dt: 0.01)
# After half period: x ≈ -1, v ≈ 0
```

### `solve_ltc/4`

Optimized solver for the Liquid Time-Constant ODE.

```elixir
@spec solve_ltc(
  Nx.Tensor.t(),  # x: current state [batch, hidden]
  Nx.Tensor.t(),  # activation: target [batch, hidden]
  Nx.Tensor.t(),  # tau: time constants [batch, hidden]
  keyword()       # opts
) :: Nx.Tensor.t()
```

**The LTC ODE:**

```
dx/dt = (activation - x) / tau
```

This describes exponential decay toward `activation` with time constant `tau`:
- Large τ → slow adaptation (state changes slowly)
- Small τ → fast adaptation (state quickly approaches activation)

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `:solver` | `:rk4` | Solver type |
| `:steps` | `1` | Number of sub-steps within one frame |
| `:atol` | `1.0e-4` | Absolute tolerance (DOPRI5 only) |
| `:rtol` | `1.0e-2` | Relative tolerance (DOPRI5 only) |

**Example:**

```elixir
# Batch of 4, hidden size 8
batch = 4
hidden = 8

x = Nx.broadcast(0.0, {batch, hidden})           # Start at zero
activation = Nx.broadcast(1.0, {batch, hidden})  # Target is 1.0
tau = Nx.broadcast(0.5, {batch, hidden})         # Fast time constant

# Integrate with 4 RK4 sub-steps
result = ODESolver.solve_ltc(x, activation, tau, solver: :rk4, steps: 4)
# result approaches 1.0
```

---

## Solver Details

### Euler (1st Order)

The simplest numerical integration method.

```
x_{n+1} = x_n + dt × f(t_n, x_n)
```

**Pros:** Fast, simple
**Cons:** Low accuracy, can be unstable for stiff problems

**Use when:** Speed matters more than accuracy, or for prototyping.

### Midpoint (2nd Order)

Also known as "explicit midpoint" or "modified Euler".

```
k₁ = f(t_n, x_n)
k₂ = f(t_n + dt/2, x_n + dt/2 × k₁)
x_{n+1} = x_n + dt × k₂
```

**Pros:** Better accuracy than Euler, still fast
**Cons:** Not as accurate as RK4

**Use when:** Need better accuracy than Euler but don't want RK4's overhead.

### RK4 (4th Order Runge-Kutta)

The classic "workhorse" of numerical integration.

```
k₁ = f(t_n, x_n)
k₂ = f(t_n + dt/2, x_n + dt/2 × k₁)
k₃ = f(t_n + dt/2, x_n + dt/2 × k₂)
k₄ = f(t_n + dt, x_n + dt × k₃)
x_{n+1} = x_n + (dt/6) × (k₁ + 2k₂ + 2k₃ + k₄)
```

**Pros:** Excellent accuracy for the cost, well-understood
**Cons:** Fixed step size, 4 function evaluations per step

**Use when:** Default choice - good accuracy without adaptive overhead.

### DOPRI5 (Dormand-Prince 4/5)

Adaptive step size method using embedded 4th and 5th order solutions.

**How it works:**
1. Compute both 4th and 5th order solutions
2. Estimate error as difference between them
3. If error > tolerance: reject step, reduce dt
4. If error < tolerance: accept step, possibly increase dt

**Tolerances:**
- `atol` (absolute): Error threshold for small values
- `rtol` (relative): Error threshold relative to solution magnitude
- Combined: `tolerance = atol + rtol × |x|`

**Pros:** Best accuracy, automatically adjusts step size
**Cons:** More computation per step, overhead from adaptivity

**Use when:** High accuracy is critical, or dynamics vary in "stiffness".

---

## Differentiability

All solvers use pure Nx operations and are compatible with automatic differentiation:

```elixir
import Nx.Defn

defn loss_fn(params, x0, target) do
  # ODE with learnable dynamics
  f = fn _t, x -> Nx.multiply(params, x) end
  result = ODESolver.solve(f, 0.0, 1.0, x0, solver: :euler, dt: 0.1)
  Nx.mean(Nx.pow(Nx.subtract(result, target), 2))
end

# Compute gradient
{loss, grad} = Nx.Defn.value_and_grad(params, &loss_fn(&1, x0, target))
```

**Note:** Backpropagation through ODE solvers computes gradients by differentiating through each integration step. For very long integrations, consider adjoint methods (not currently implemented).

---

## Performance Considerations

### Fixed vs Adaptive Step

| Aspect | Fixed (Euler/Midpoint/RK4) | Adaptive (DOPRI5) |
|--------|---------------------------|-------------------|
| Predictable runtime | Yes | No |
| JIT compilation | Simple | More complex |
| Accuracy control | Via dt | Via tolerances |
| Best for training | Yes (fixed graph) | Maybe (variable steps) |
| Best for inference | Either | Either |

### Choosing Step Count for LTC

For `solve_ltc/4`, the `steps` parameter controls accuracy:

```elixir
# Fast, less accurate
ODESolver.solve_ltc(x, act, tau, solver: :rk4, steps: 1)

# More accurate
ODESolver.solve_ltc(x, act, tau, solver: :rk4, steps: 4)

# High accuracy (or use DOPRI5)
ODESolver.solve_ltc(x, act, tau, solver: :rk4, steps: 8)
```

**Rule of thumb:** Start with `steps: 1`, increase if you see numerical issues.

### Stiff Problems

If your ODE has very different time scales (stiff), you may see:
- Euler/Midpoint: Unstable (oscillates or explodes)
- RK4: May need very small dt
- DOPRI5: Handles automatically via adaptive stepping

For Liquid Networks with very small τ (fast adaptation), consider:
- Using more integration steps
- Using DOPRI5 with appropriate tolerances
- Clamping τ to a minimum value (e.g., 0.1)

---

## Integration with Liquid Networks

The ODE solver is used internally by `ExPhil.Networks.Liquid`:

```elixir
# In liquid.ex
defp integrate_ode(h, _x, tau, activation, _dt, steps, solver) do
  alias ExPhil.Numerics.ODESolver
  ODESolver.solve_ltc(h, activation, tau, solver: solver, steps: steps)
end
```

**Configuration via Liquid backbone:**

```bash
# Default (RK4)
mix run scripts/train_from_replays.exs --backbone liquid

# High accuracy (DOPRI5)
mix run scripts/train_from_replays.exs --backbone liquid --solver dopri5

# Fast (Euler with multiple steps)
mix run scripts/train_from_replays.exs --backbone liquid --solver euler --integration-steps 4
```

---

## References

- Dormand, J. R.; Prince, P. J. (1980). "A family of embedded Runge-Kutta formulae"
- Chen, R. T. Q. et al. (2018). "Neural Ordinary Differential Equations" (NeurIPS)
- Hasani, R. et al. (2021). "Liquid Time-constant Networks" (AAAI)

## See Also

- [Liquid Neural Networks](architectures/ARCHITECTURE_GUIDE.md#liquid---liquid-neural-networks) - Architecture using this solver
- [ARCHITECTURE_GUIDE.md](architectures/ARCHITECTURE_GUIDE.md) - All backbone explanations
