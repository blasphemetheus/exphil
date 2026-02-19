# Architecture Benchmark Results

**Date:** 2026-02-19
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB)
**Dataset:** 30 Zelda replays (308K train / 26K val frames), seq_len=30
**Config:** 3 epochs, cached embeddings, 288-dim learned embeddings
**Script:** `benchmark_isolated.sh` (each architecture in its own BEAM process)

## Results (Ranked by Best Validation Loss)

| Rank | Architecture | Val Loss | Train Loss | Speed (b/s) | Inference | Notes |
|------|-------------|----------|------------|-------------|-----------|-------|
| 1 | S4 | 2.8219 | — | — | — | Best overall |
| 2 | xLSTM | 2.8272 | — | — | — | |
| 3 | Zamba | 2.8769 | — | — | — | Fixed from NaN (LR=1e-5, clip=0.5) |
| 4 | S4D | 2.9043 | — | — | — | |
| 5 | GRU | 2.9430 | — | — | — | Classic, very competitive |
| 6 | LSTM | 2.9449 | — | — | — | |
| 7 | LSTM Hybrid | 2.9474 | — | — | — | |
| 8 | RWKV | 2.9475 | — | — | — | |
| 9 | Griffin | 2.9478 | — | — | — | |
| 10 | GLA | 2.9576 | — | — | — | |
| 11 | Mamba SSD | 2.9597 | — | — | — | |
| 12 | HGRN | 2.9631 | — | — | — | |
| 13 | Gated SSM | 2.9633 | — | — | — | |
| 14 | Hawk | 2.9685 | — | — | — | |
| 15 | Mamba | 2.9696 | — | — | — | |
| 16 | S5 | 2.9762 | — | — | — | |
| 17 | Sliding Window | 2.9866 | — | — | — | |
| 18 | DeltaNet | 2.9940 | — | — | — | |
| 19 | Performer | 3.0077 | — | — | — | |
| 20 | RetNet | 3.0102 | — | — | — | |
| 21 | SNN | 3.0144 | 2.4669 | 39.6 | 0.4ms | Non-temporal (last-frame extraction) |
| 22 | NTM | 3.0718 | — | — | — | Non-temporal (controller-style) |
| 23 | Jamba | 3.0623 | — | — | — | Fixed from NaN (LR=5e-6, clip=0.25) |
| 24 | Bayesian | 3.1011 | — | — | — | Non-temporal (last-frame extraction) |
| 25 | Hopfield | 3.1276 | 2.4117 | 116.3 | 0.4ms | Non-temporal, overfitting (best epoch 2: 2.9532) |
| 26 | Perceiver | 3.1471 | — | — | — | |
| 27 | MLP | 3.2082 | — | — | — | Non-temporal baseline |
| 28 | Decision Transformer | 3.2552 | — | — | — | Reduced config (OOM) |
| 29 | Liquid | 3.3048 | — | — | — | Heavily reduced config (OOM) |
| 30 | Reservoir | 3.3283 | — | — | — | Fixed weights (ESN) |
| 31 | H3 | 3.6215 | 3.8436 | 23.1 | 1.1ms | Underfitting at LR=5e-7, still converging |
| — | FNet | pending | — | — | — | Fixed: real DFT matrix multiply (not yet rerun) |
| — | TTT | NaN | — | — | — | Still NaN at LR=5e-7/clip=0.1 — needs Edifice-level init fix |
| — | KAN | OOM | — | — | — | Basis expansion too large even at batch=4/hidden=64/grid=4 |

## Key Findings

### Top Tier (val < 2.95)
S4, xLSTM, Zamba, S4D, GRU, LSTM, LSTM Hybrid, RWKV, Griffin, GLA — all within 0.12 of each other. **SSM architectures dominate** (S4, S4D, Mamba variants), but classical recurrents (GRU, LSTM) are surprisingly competitive. The gap between #1 (S4, 2.82) and #15 (Mamba, 2.97) is only 5%.

### Practical Recommendations
- **Best quality:** S4 (2.82) or xLSTM (2.83) — marginal difference
- **Best 60fps candidate:** Mamba (fast inference, good quality) or GRU (simple, competitive)
- **Best surprise:** Zamba (was NaN, now #3 with conservative hyperparams)
- **Avoid for training:** KAN (OOM), Liquid/Decision Transformer (need tiny configs)

### Architecture Categories
- **Temporal (SSM/recurrent):** 2.82–3.00 val loss — benefit from sequence modeling
- **Non-temporal (last-frame):** 3.01–3.33 val loss — ~0.15 gap from missing temporal context
- **Baseline (MLP):** 3.21 — everything temporal beats it

### Numerical Stability Issues
| Architecture | Issue | Fix | Result |
|-------------|-------|-----|--------|
| Zamba | NaN at default LR | LR=1e-5, clip=0.5 | 2.8769 (#3!) |
| Jamba | NaN at epoch 3 | LR=5e-6, clip=0.25 | 3.0623 |
| H3 | NaN at LR=1e-5 | LR=5e-7, clip=0.1 | 3.6215 (underfitting) |
| TTT | NaN at LR=1e-5 | LR=5e-7, clip=0.1 | Still NaN — init fix needed |
| FNet | Nx.less/2 complex crash | Real DFT matrix multiply | pending |

Root cause: architectures with **exponential parameterization** (H3 `a_log`/`dt_log`, TTT inner-loop) or **shared parameters** (Zamba QKV) have gradient feedback loops. The gradient of `exp(x)` is `exp(x)` — a positive feedback cycle that explodes without tight clipping and low LR.

### GPU Memory Issues
| Architecture | Batch | Hidden | Issue | Fix |
|-------------|-------|--------|-------|-----|
| Hopfield | 64→4 | 256→128 | OOM | Reduced config |
| KAN | 16→4 | 128→64 | Still OOM at batch=4 | Skipped (basis expansion) |
| Liquid | 64→4 | 256→64 | OOM | 1 layer, integration_steps=1 |
| Decision Transformer | 64→16 | 256→128 | OOM | Reduced heads |

KAN's B-spline basis expansion creates `hidden * grid_size` intermediate tensors per layer in the backward pass. Even at hidden=64/grid=4, the gradient graph exceeds 24GB.

### XLA Allocator Gotcha
`System.put_env("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")` in Elixir is **too late** — EXLA reads this at native library load time (before your script's code runs). Must `export` in the shell. BFC allocator pre-grabs ~95% of GPU memory and never releases it.

## Reproduction

```bash
# Full benchmark (all 34 architectures, ~3 hours on RTX 4090)
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --epochs 3 --cache-embeddings

# Specific architectures
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --only s4,xlstm,zamba,gru --epochs 3 --cache-embeddings

# Skip known-broken
./scripts/benchmark_isolated.sh --replays /workspace/replays/greg/zelda --skip kan --epochs 3 --cache-embeddings
```

## Inference Latency (from Edifice GPU bench, seq_len=1, batch=1)

Cross-referencing training quality with inference speed for 60fps viability (< 16ms):

| Architecture | Val Loss | Inference (ms) | 60fps? | Verdict |
|-------------|----------|---------------|--------|---------|
| S4 | 2.82 | 14.7 | YES | Best quality + viable speed |
| xLSTM | 2.83 | 27.1* | NO | Top quality but too slow |
| Zamba | 2.88 | 24.7* | NO | Needs ONNX export |
| S4D | 2.90 | 32.5* | NO | Needs optimization |
| GRU | 2.94 | 162.0* | NO | Classical, very slow |
| Mamba | 2.97 | 10.8 | YES | Great speed/quality |
| Mamba SSD | 2.96 | 9.9 | YES | Fastest good architecture |
| Gated SSM | 2.96 | 13.0 | YES | Simple, viable |
| HGRN | 2.96 | 15.9 | YES | Borderline |
| GLA | 2.96 | 35.1* | NO | Too slow native |

*These latencies are from Edifice microbenchmarks with embed=256, seq_len=1. ExPhil inference with full policy heads adds overhead. ONNX INT8 export reduces latency ~20x.

**Best 60fps candidates (quality + speed):**
1. **S4** — 2.82 val, 14.7ms (best quality at viable speed)
2. **Mamba SSD** — 2.96 val, 9.9ms (fastest among good models)
3. **Mamba** — 2.97 val, 10.8ms (proven architecture)
4. **Gated SSM** — 2.96 val, 13.0ms (simple + fast)

## Next Steps

- [ ] Run FNet and TTT with latest fixes
- [ ] Try H3 with LR warmup (5e-7 → 1e-5 over epoch 1) for better convergence
- [ ] Try Hopfield with dropout (currently overfits by epoch 3)
- [ ] Longer runs (10+ epochs) for top-5 architectures to see true convergence
- [ ] ONNX export for top-5 to measure real inference latency with policy heads
- [ ] Scale experiments: larger hidden sizes for top performers
