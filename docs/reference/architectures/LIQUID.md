# Liquid Neural Networks Architecture

**Type:** Continuous-Time Neural Network (ODE-based)
**Complexity:** O(n × steps) where steps = integration steps
**60 FPS Ready:** Depends on solver (~10-50ms inference)

## Overview

Liquid Neural Networks (LNNs) are inspired by the nervous system of C. elegans (a tiny worm with only 302 neurons). Unlike discrete-time networks that update at fixed intervals, LNNs have continuous-time dynamics governed by ordinary differential equations (ODEs). This gives them remarkable adaptability - they literally "flow" through time.

## Etymology

**Liquid** refers to the fluid, continuous nature of the network's dynamics. The term comes from "Liquid Time-Constant" (LTC) networks, where the time constants that control neural dynamics are themselves adaptive and input-dependent. The name evokes the idea of neural activity "flowing" like liquid rather than jumping in discrete steps.

## Architecture

```
Standard RNN (discrete):
h_t = tanh(W·h_{t-1} + U·x_t + b)

Liquid Neural Network (continuous):
dh/dt = (-h + f(h, x, t)) / τ(h, x)    ← ODE!

Where:
- h: hidden state (continuous function of time)
- x: input
- τ: time constant (adaptive!)
- f: neural nonlinearity
```

The hidden state evolves according to a differential equation, solved numerically.

## When to Use

**Choose Liquid when:**
- Inputs arrive at irregular intervals
- You want adaptive temporal dynamics
- Physical/biological modeling matters (continuous time is more realistic)
- You need very compact models (LNNs are parameter-efficient)

**Avoid Liquid when:**
- Fixed-rate inputs (60 FPS Melee frames)
- Maximum inference speed is critical
- You want deterministic timing (ODE solvers can have variable cost)

## Configuration

```bash
# Basic usage
mix run scripts/train_from_replays.exs --temporal --backbone liquid

# With custom solver settings
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone liquid \
  --solver rk4 \
  --integration-steps 4

# High accuracy (slower)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone liquid \
  --solver dopri5 \
  --integration-steps 8
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_size` | 256 | State dimension |
| `num_layers` | 4 | Number of LTC layers |
| `solver` | `:rk4` | ODE solver (euler, midpoint, rk4, dopri5) |
| `integration_steps` | 4 | Steps per frame |
| `tau_min` | 0.1 | Minimum time constant |
| `tau_max` | 10.0 | Maximum time constant |

### ODE Solvers

| Solver | Accuracy | Speed | Adaptive |
|--------|----------|-------|----------|
| `euler` | Low | Fastest | No |
| `midpoint` | Medium | Fast | No |
| `rk4` | High | Medium | No |
| `dopri5` | Highest | Variable | Yes |

## Implementation

```elixir
# lib/exphil/networks/liquid.ex
defmodule ExPhil.Networks.Liquid do
  @moduledoc """
  Liquid Time-Constant Neural Networks.
  Continuous-time dynamics via ODE integration.
  """

  alias ExPhil.Numerics.ODESolver

  def build(input, opts \\ []) do
    hidden_size = opts[:hidden_size] || 256
    solver = opts[:solver] || :rk4
    steps = opts[:integration_steps] || 4

    # Initial state
    h0 = Axon.dense(input, hidden_size) |> Axon.tanh()

    # Evolve state through time via ODE
    Enum.reduce(1..steps, h0, fn _step, h ->
      # dh/dt = dynamics(h, x)
      # Integrate one step
      ODESolver.step(solver, &ltc_dynamics/3, h, input, dt: 1.0 / steps)
    end)
  end

  defp ltc_dynamics(h, x, _t) do
    # Liquid Time-Constant dynamics:
    # dh/dt = (-h + f(h, x)) / tau(h, x)

    # Compute adaptive time constant
    tau = compute_tau(h, x)

    # Compute activation
    f_hx = Axon.dense(Nx.concatenate([h, x]), hidden_size) |> Axon.tanh()

    # Return derivative
    Nx.divide(Nx.subtract(f_hx, h), tau)
  end

  defp compute_tau(h, x) do
    # Time constant is INPUT-DEPENDENT (the "liquid" part!)
    Axon.dense(Nx.concatenate([h, x]), 1)
    |> Axon.sigmoid()
    |> Nx.multiply(@tau_max - @tau_min)
    |> Nx.add(@tau_min)
  end
end
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training | O(n × steps) | Steps per frame |
| Inference | Variable | Depends on solver |
| Parameters | Very low | Compact representation |

### Benchmark (RTX 4090)

| Solver | Steps | Inference (30 frames) |
|--------|-------|----------------------|
| euler | 1 | 8ms |
| rk4 | 4 | 25ms |
| dopri5 | adaptive | 15-50ms |

## The Liquid Time-Constant

What makes LNNs special is the **adaptive time constant** τ:

```
Fast τ (small): Quick reactions, short memory
Slow τ (large): Slow adaptation, long memory
```

Crucially, τ is computed from the input:
- Important inputs → smaller τ → faster response
- Background inputs → larger τ → slower integration

This is inspired by how real neurons modulate their response dynamics based on input salience.

## ODE Solver Details

ExPhil implements custom ODE solvers in `ExPhil.Numerics.ODESolver`:

```elixir
# Euler (simplest)
h_new = h + dt * f(h, x, t)

# Midpoint (2nd order)
k1 = f(h, x, t)
k2 = f(h + 0.5*dt*k1, x, t + 0.5*dt)
h_new = h + dt * k2

# RK4 (4th order, recommended)
k1 = f(h, x, t)
k2 = f(h + 0.5*dt*k1, x, t + 0.5*dt)
k3 = f(h + 0.5*dt*k2, x, t + 0.5*dt)
k4 = f(h + dt*k3, x, t + dt)
h_new = h + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# DOPRI5 (adaptive, highest accuracy)
# Uses error estimation to adjust step size
```

See [ODE_SOLVER.md](../ODE_SOLVER.md) for full documentation.

## Melee Application

For Melee, Liquid Networks offer interesting properties:

1. **Irregular timing**: Handle frame drops gracefully
2. **Reaction calibration**: Fast τ for tech chasing, slow τ for neutral
3. **Compact models**: Fewer parameters than LSTM/Mamba

However, the variable computation time of adaptive solvers can be problematic for real-time play where consistent frame timing is critical.

## Comparison with Discrete Models

| Feature | Liquid NN | LSTM | Mamba |
|---------|-----------|------|-------|
| Time | Continuous | Discrete | Discrete |
| Dynamics | ODE | Gated | SSM |
| Adaptivity | High (τ) | Fixed | Input-selective |
| Parameters | Low | Medium | Medium |
| Speed | Variable | Fixed | Fixed |

## References

- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) - Original LTC paper
- [Neural Circuit Policies](https://arxiv.org/abs/2006.04439) - Application to control
- [Closed-form Continuous-depth Neural Networks](https://arxiv.org/abs/2106.13898) - CfC (faster variant)
- [C. elegans Connectome](https://www.wormatlas.org/neuronalwiring.html) - Biological inspiration
- [ExPhil Implementation](../../../lib/exphil/networks/liquid.ex)
- [ODE Solver Reference](../ODE_SOLVER.md)

## See Also

- [ODE_SOLVER.md](../ODE_SOLVER.md) - Numerical integration details
- [S5.md](S5.md) - Another continuous-time-inspired model
- [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - All architectures overview
