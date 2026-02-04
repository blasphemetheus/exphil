# Policy Types

ExPhil supports multiple policy architectures beyond the default autoregressive policy.

## Available Policy Types

| Type | CLI Flag | Output Format | Best For |
|------|----------|---------------|----------|
| `autoregressive` | (default) | 6 discrete heads | Fast inference, discrete actions |
| `diffusion` | `--policy-type diffusion` | Continuous actions | Smooth trajectories |
| `act` | `--policy-type act` | Action chunks | Temporal ensembling |
| `flow_matching` | `--policy-type flow_matching` | Continuous actions | ODE-based generation |

---

## Autoregressive (Default)

6-head autoregressive controller prediction.

```
Embedded State → Backbone → Controller Head
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼         ▼         ▼         ▼         ▼         ▼
buttons   main_x   main_y     c_x       c_y    shoulder
[8 Bern]  [17 Cat] [17 Cat] [17 Cat] [17 Cat]  [5 Cat]
```

**Loss:** Cross-entropy per head

**Usage:**
```bash
mix run scripts/train_from_replays.exs --temporal --backbone mamba
```

---

## Diffusion Policy

Denoising diffusion for continuous action generation.

**Key Idea:** Learn to denoise random noise into valid actions by predicting the noise that was added.

```
Noise + Observations → Denoiser → Predicted Noise
                         ↓
         Iterative denoising (20 steps)
                         ↓
                   Clean Actions
```

**Loss:** MSE between true noise and predicted noise

**Usage:**
```bash
mix run scripts/train_from_replays.exs \
  --policy-type diffusion \
  --action-horizon 16 \
  --num-inference-steps 20 \
  --temporal --backbone mamba
```

**Pros:**
- Generates smooth, multi-modal action distributions
- Good for continuous control

**Cons:**
- Slower inference (multiple denoising steps)
- Requires continuous action representation

---

## Action Chunking (ACT)

CVAE-based action chunk prediction with temporal ensembling.

**Key Idea:** Predict k actions at once, then ensemble overlapping predictions for smoother control.

```
Observations → Encoder → Latent z → Decoder → k Actions
     │              ↑
     └──────────────┘ (KL regularization)
```

**Loss:** L1 reconstruction + β·KL divergence (CVAE objective)

**Usage:**
```bash
mix run scripts/train_from_replays.exs \
  --policy-type act \
  --action-horizon 8 \
  --kl-weight 10.0 \
  --temporal --backbone mamba
```

**Temporal Ensembling:**
```
Chunk t:    [a₁, a₂, a₃, a₄]
Chunk t+1:      [a₂, a₃, a₄, a₅]
                 ↓   ↓   ↓
              weighted average
```

**Pros:**
- Smoother trajectories through ensembling
- Better long-horizon planning

**Cons:**
- More complex training (CVAE)
- Needs tuning of KL weight

---

## Flow Matching

ODE-based generative model for continuous actions.

**Key Idea:** Learn a velocity field that transports noise to data via an ODE.

```
Linear interpolation: x_t = (1-t)·noise + t·actions
Target velocity: v* = actions - noise

Velocity Network: observations + x_t + t → v̂
```

**Loss:** MSE between target velocity and predicted velocity

**Usage:**
```bash
mix run scripts/train_from_replays.exs \
  --policy-type flow_matching \
  --action-horizon 8 \
  --num-inference-steps 20 \
  --temporal --backbone mamba
```

**ODE Solvers:**
- `:euler` - Fast, less accurate
- `:midpoint` - Good balance
- `:rk4` - Most accurate, slower

**Pros:**
- Simpler than diffusion (no noise schedule)
- Faster inference (fewer steps needed)

**Cons:**
- Requires continuous action representation
- Still iterative (not single-shot)

---

## Choosing a Policy Type

| Use Case | Recommended |
|----------|-------------|
| Fast inference (60 FPS) | `autoregressive` |
| Smooth trajectories | `act` or `flow_matching` |
| Multi-modal actions | `diffusion` |
| Continuous control | `diffusion` or `flow_matching` |
| Discrete controller | `autoregressive` |

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--policy-type TYPE` | autoregressive | Policy architecture type |
| `--action-horizon N` | 8 | Actions per chunk (ACT/diffusion/flow) |
| `--num-inference-steps N` | 20 | Denoising/ODE steps |
| `--kl-weight N` | 10.0 | KL divergence weight (ACT only) |
