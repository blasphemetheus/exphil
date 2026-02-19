# Benchmark Architecture Fixes TODO

**Created:** 2026-02-19
**Context:** 34-architecture GPU benchmark on RTX 4090, 30 Zelda replays, 3 epochs
**Results:** 30/34 converged. 4 remaining: FNet (complex grad), TTT (NaN), KAN (OOM), H3 (underfitting)

---

## 1. TTT — NaN (Edifice fix applied, needs verification) [P1]

**Status:** Edifice fix pushed (commit 8a67d79). Needs `mix deps.update edifice` on pod + rerun.
**File:** `edifice/lib/edifice/recurrent/ttt.ex`
**Paper:** "Learning to (Learn at Test Time)" (Sun et al., 2024) — https://arxiv.org/abs/2407.04620

The current implementation is missing 5 critical stability mechanisms from the paper. Each fix is independent and can be applied incrementally.

### Fix 1: W_0 initialization — N(0, 0.02) not glorot_uniform

**Location:** `ttt.ex:223-224`
```elixir
# CURRENT (wrong):
w0_param = Axon.param("#{name}_w0", {inner_size, inner_size}, initializer: :glorot_uniform)

# FIX: Use N(0, 0.02) per Section 4.1 of the paper
w0_param = Axon.param("#{name}_w0", {inner_size, inner_size},
  initializer: fn shape, type, _key ->
    Nx.Random.normal(Nx.Random.key(42), shape: shape, type: type)
    |> elem(0)
    |> Nx.multiply(0.02)
  end)
```

**Why:** Glorot for a 64x64 matrix gives stddev ~0.18, which is 9x larger than the paper's 0.02. The inner model's self-supervised gradient update `W -= eta * error * k^T` amplifies any large initial weight through the prediction error. Small init keeps early predictions near zero, preventing gradient explosion in the first few steps.

### Fix 2: Eta scaling by 1/head_dim (CRITICAL)

**Location:** `ttt.ex:265`
```elixir
# CURRENT (wrong):
eta = Nx.sigmoid(eta_pre)

# FIX: Scale by 1/inner_size (= 1/head_dim in paper's multi-head formulation)
eta = Nx.divide(Nx.sigmoid(eta_pre), inner_size)
```

**Why:** This is the single most likely cause of NaN. Without the 1/d scaling, eta values after sigmoid are in [0, 1], but the paper uses eta in [0, 1/d] where d=64. This means our learning rate is 64x too large for the inner model update. The gradient step `W -= eta * error * k^T` produces rank-1 updates of magnitude `eta * ||error|| * ||k||`. With eta=0.5 instead of 0.5/64=0.0078, the weight matrix changes dramatically each step, causing prediction errors to explode.

### Fix 3: Reconstruction target — V-K (residual) not raw V

**Location:** `ttt.ex:309-310` (linear variant) and `ttt.ex:289` (MLP variant)
```elixir
# CURRENT (wrong):
error = Nx.subtract(pred, v_t)

# FIX: Use residual reconstruction target
error = Nx.subtract(pred, Nx.subtract(v_t, k_t))
```

**Why:** The paper trains the inner model to predict the *residual* between V and K, not V directly. This centers the reconstruction target around zero (since K and V are both projections of the same input), making the loss landscape much smoother. Without this, the inner model must learn the full magnitude of V, which means W_0 must already be a good predictor — at odds with near-zero initialization.

### Fix 4: LayerNorm on inner model output before loss

**Location:** `ttt.ex:306-310` — add LayerNorm before computing error

This requires pre-building a learnable LayerNorm (gamma/beta params) and applying it inside the scan. Since Axon.layer_norm can't be used inside a raw Nx computation, implement manually:

```elixir
# Add as Axon.param in build_ttt_layer (alongside w0_param):
ln_gamma = Axon.param("#{name}_inner_ln_gamma", {inner_size},
  initializer: :ones)
ln_beta = Axon.param("#{name}_inner_ln_beta", {inner_size},
  initializer: :zeros)

# In the scan, after computing pred:
pred_normed = manual_layer_norm(pred, ln_gamma, ln_beta)
error = Nx.subtract(pred_normed, Nx.subtract(v_t, k_t))

# Helper function:
defp manual_layer_norm(x, gamma, beta) do
  mean = Nx.mean(x, axes: [-1], keep_axes: true)
  var = Nx.variance(x, axes: [-1], keep_axes: true)
  Nx.add(Nx.multiply(Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, 1.0e-6))), gamma), beta)
end
```

**Why:** LayerNorm on the inner model's output (before computing loss gradient) prevents the prediction magnitudes from drifting as W changes. Without it, the gradient magnitude is proportional to `||pred||` which grows unboundedly as W accumulates updates.

### Fix 5: Learnable ttt_base_lr (optional, lower priority)

**Location:** `ttt.ex:213` (eta_proj)

Currently eta comes from `sigmoid(W_eta @ x)`. The paper uses `sigmoid(W_eta @ x) * ttt_base_lr / head_dim` where `ttt_base_lr` is a learnable scalar initialized to 1.0.

```elixir
# Add as Axon.param:
base_lr = Axon.param("#{name}_base_lr", {1},
  initializer: :ones)

# In scan, after computing eta:
eta = Nx.divide(Nx.multiply(Nx.sigmoid(eta_pre), Nx.abs(base_lr)), inner_size)
```

**Why:** The learnable base_lr lets the model discover its own optimal inner learning rate during training. Less critical than fixes 1-4 but improves convergence.

### Implementation order

1. **Fix 2 (eta scaling)** — most likely NaN cause, 1-line change
2. **Fix 1 (W_0 init)** — 3-line change, independent
3. **Fix 3 (V-K target)** — 1-line change per variant, independent
4. **Fix 4 (LayerNorm)** — most complex, requires new params in Axon.layer
5. **Fix 5 (base_lr)** — optional, try after 1-4

### Testing

Edifice tests all pass (14 correctness + 32 gradient smoke). On next pod run:
```bash
cd /app
git pull origin main
mix deps.update edifice && mix compile --force
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --only ttt --epochs 3 --cache-embeddings
```

Expected: should converge at LR=5e-7 with no NaN. If stable, try relaxing to LR=1e-5.

**Note:** Fix 3 from original TODO (V-K reconstruction target) was WRONG. The official TTT code
uses raw V as the target, not V-K. The "delta rule" equivalence is a property of the gradient
math, not a different target. This fix was NOT applied.

---

## 2. FNet — Complex gradient crash (Edifice fix applied, needs verification) [P1]

**Status:** Fix pushed to Edifice (commit fc5c852) but never successfully run on GPU pod.
**File:** `edifice/lib/edifice/attention/fnet.ex` (already fixed)

### What was done

Replaced `Nx.fft` with real-valued DFT matrix multiply:
- Precompute cosine DFT matrices at build time: `DFT[k,n] = cos(2*pi*k*n/N)`
- Use `Nx.dot(x, dft_matrix)` instead of `Nx.fft(x) |> Nx.real()`
- All operations stay in real f32 space, so gradients never touch complex numbers
- DFT matrices passed as `Axon.constant` nodes

### To verify on next pod run

```bash
cd /app
git pull origin main
mix deps.update edifice && mix compile --force
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --only fnet --epochs 3 --cache-embeddings
```

**Expected:** Should train normally. Quality should be mid-range (3.0-3.2 val loss) since FFT mixing is parameter-free and less expressive than learned attention.

### If still failing

The error would be the same `Nx.less/2 does not support complex inputs` — this would mean the dep didn't update properly. Debug:
```elixir
# In iex, verify the fix is loaded:
Edifice.Attention.FNet.dft_real_matrix(4) |> Nx.shape()
# Should return {4, 4}
```

---

## 3. KAN — OOM (permanent, skip for benchmark) [P3]

**Status:** OOM at batch_size=4, hidden=64, grid=4 on RTX 4090 (24GB). Fundamental memory issue.
**File:** `edifice/lib/edifice/feedforward/kan.ex`

### Root cause

KAN's B-spline basis expansion creates `hidden * grid_size` intermediate tensors per layer in the backward pass. For a 2-layer network with hidden=64, grid=4:
- Forward: each layer expands 64 inputs to 64*4=256 basis functions
- Backward: EXLA materializes gradient for each basis coefficient
- The gradient graph for even this tiny config exceeds 24GB

### Options (in order of preference)

**Option A: Skip permanently in benchmark** (recommended)
KAN is designed for function approximation, not sequence modeling. Its O(hidden * grid * layers) memory scaling in backward pass makes it impractical for GPU training at any reasonable size.

**Option B: CPU-only training (very slow)**
```elixir
# In benchmark config:
{:kan, "KAN (Kolmogorov-Arnold)", [
  backend: :binary,  # Force CPU
  batch_size: 1,
  hidden_sizes: [32, 32],
  grid_size: 3,
  ...
]}
```
Would take hours per epoch. Not worth it for a architecture that's fundamentally mismatched.

**Option C: Gradient checkpointing (Edifice change)**
Implement activation checkpointing in KAN's `basis_expansion` to trade compute for memory. This is a significant engineering effort for marginal benefit.

### Recommendation

Mark KAN as "skipped (OOM)" in benchmark results. It's a fascinating architecture for interpretable function approximation but not viable for sequence modeling at useful scales on consumer GPUs.

---

## 4. H3 — Underfitting at LR=5e-7 (needs LR warmup) [P2]

**Status:** Converges! val=3.6215 after 3 epochs. But severely underfitting (train loss 3.84 > val loss, loss still dropping at epoch 3). LR=5e-7 is too conservative.
**File:** `exphil/scripts/benchmark_architectures.exs:396-409`

### Root cause

H3 uses learnable `a_log` and `dt_log` parameters with exponential parameterization (`exp(a_log)`). The gradient of `exp(x)` is `exp(x)` — a positive feedback loop. At LR=1e-5, this explodes to NaN. At LR=5e-7, it's stable but converges glacially.

### Fix: LR warmup schedule

The ideal approach is linear warmup from 5e-7 to 1e-5 over epoch 1, then constant:

**Option A: In ExPhil benchmark script** (easiest)

Add a `warmup_steps` parameter to the benchmark config:
```elixir
{:h3, "H3 (Hungry Hippos)", [
  temporal: true,
  backbone: :h3,
  window_size: 30,
  num_layers: 2,
  hidden_sizes: [256, 256],
  batch_size: 64,
  dropout: 0.1,
  learning_rate: 1.0e-5,      # Target LR (back to normal)
  max_grad_norm: 0.1,         # Keep tight clipping
  warmup_epochs: 1,           # Warmup over first epoch
  warmup_start_lr: 5.0e-7     # Start from conservative LR
]}
```

This requires the benchmark training loop to support LR warmup. Check if `ExPhil.Training.Imitation.Optimizer` already supports warmup (it likely does via `warmup_steps` or similar).

**Option B: Just run more epochs** (simplest)

The loss was still dropping at epoch 3. Running 10 epochs at LR=5e-7 might get H3 to a competitive val loss (~3.0-3.1). Less optimal but zero code changes:
```bash
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --only h3 --epochs 10 --cache-embeddings
```

**Option C: Intermediate LR** (compromise)

Try LR=1e-6 (5x current, 10x below NaN threshold):
```elixir
learning_rate: 1.0e-6,  # Middle ground between 5e-7 (stable, slow) and 1e-5 (NaN)
```

### Recommendation

Try Option C first (1-line change, quick to verify), then Option B (10 epochs) if stable.

---

## 5. Hopfield — Overfitting (needs dropout) [P2]

**Status:** Converges at val=3.1276, but best was epoch 2 (val=2.9532) before overfitting in epoch 3. Gap of 0.18 between epochs 2-3 is significant.
**File:** `exphil/scripts/benchmark_architectures.exs:528-541`

### Root cause

Hopfield's pattern matrix creates a very expressive model that memorizes training data quickly. At batch_size=4 and hidden=128, there are few regularization constraints.

### Fix: Add dropout

```elixir
{:hopfield, "Hopfield (Associative Memory)", [
  temporal: true,
  backbone: :hopfield,
  window_size: 30,
  num_layers: 2,
  num_heads: 2,
  hidden_sizes: [128, 128],
  hidden_size: 128,
  batch_size: 4,
  dropout: 0.2,          # ADD: regularization to prevent overfitting
  max_grad_norm: 1.0
]}
```

Also consider:
- `learning_rate: 5.0e-4` (slightly lower than default 1e-3)
- Running 5 epochs with early stopping (best-epoch tracking already exists)

### Expected outcome

With dropout=0.2, the epoch 2→3 overfitting gap should narrow. True val loss likely around 2.95-3.05.

---

## 6. Longer runs for top-5 [P3]

Once the above fixes are applied, the next benchmark phase is longer training:

```bash
./scripts/benchmark_isolated.sh \
  --replays /workspace/replays/greg/zelda \
  --only s4,xlstm,zamba,s4d,gru \
  --epochs 10 \
  --cache-embeddings
```

These 5 architectures are within 0.12 of each other (2.82-2.94). 10 epochs will reveal true convergence behavior — some may still be improving, others may plateau.

---

## Summary Table

| Architecture | Issue | Fix Location | Complexity | Priority |
|---|---|---|---|---|
| TTT | NaN | Edifice `ttt.ex` (8a67d79) | DONE (4 fixes) | P1 |
| FNet | Complex grad | Edifice `fnet.ex` (fc5c852) | DONE, verify | P1 |
| KAN | OOM | N/A | Skip | P3 |
| H3 | Underfitting | ExPhil benchmark config | Low (1 line) | P2 |
| Hopfield | Overfitting | ExPhil benchmark config | Low (1 line) | P2 |

**Quick wins (next pod session):**
1. `mix deps.update edifice && mix compile --force` (picks up TTT + FNet fixes)
2. Verify FNet and TTT: `--only fnet,ttt --epochs 3`
3. Try H3 at LR=1e-6
4. Add dropout=0.2 to Hopfield
