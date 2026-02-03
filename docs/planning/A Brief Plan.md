# A Brief Plan: Top 30 Priorities for ExPhil

**Date:** 2026-01-31
**Purpose:** Concise implementation plans for the 30 most impactful tasks for ExPhil.

---

## Priority Tiers

| Tier | Focus | Count |
|------|-------|-------|
| **Critical** | Self-play, production deployment | 5 |
| **High** | Architecture improvements, training speedup | 10 |
| **Medium** | Tooling, evaluation, research | 10 |
| **Low** | Nice-to-have, exploration | 5 |

---

## Critical Priority (Do First)

### 1. Large-Scale Self-Play Training

**Why:** Current next step per GOALS.md. Infrastructure is complete, needs GPU cluster testing.

**Plan:**
1. Deploy to RunPod with multi-GPU setup
2. Run `train_self_play.exs` with 8-16 parallel games
3. Monitor Elo progression and policy diversity
4. Tune: PPO hyperparameters, population size, historical sampling ratio

**Files:** `scripts/train_self_play.exs`, `lib/exphil/self_play/`
**Effort:** 2-3 days
**Dependencies:** GPU cluster access

---

### 2. Production ONNX Deployment Pipeline

**Why:** Current ONNX INT8 inference is 0.55ms (30x under budget). Need end-to-end deployment.

**Plan:**
1. Script to export trained model → ONNX → INT8 quantized
2. Create inference server (GenServer with model loaded)
3. Benchmark end-to-end: game state → embedding → inference → controller output
4. Document deployment for Dolphin integration

**Files:** `scripts/export_onnx.exs`, `lib/exphil/inference/`
**Effort:** 2 days
**Dependencies:** None (ONNX export already works)

---

### 3. Stick Prediction Improvements

**Why:** Eval shows stick accuracy is main weakness (56-68% vs 86-99% for buttons).

**Plan:**
1. Add stick edge weighting loss (`--stick-edge-weight 2.0`)
2. Implement class-balanced sampling for stick buckets
3. Try temporal backbone (Mamba) specifically for sticks
4. Measure improvement on stick confusion matrix

**Files:** `lib/exphil/networks/policy.ex`, `lib/exphil/training/data.ex`
**Effort:** 2-3 days
**Dependencies:** None

---

### 4. Zamba Architecture (Single Shared Attention)

**Why:** Research shows 10x KV cache reduction with single shared attention vs interleaved Jamba.

**Plan:**
1. Modify `lib/exphil/networks/hybrid.ex` to support Zamba mode
2. Add `--backbone zamba` CLI flag
3. Benchmark: Zamba vs Jamba on same dataset
4. Profile: inference latency, memory usage

**Files:** `lib/exphil/networks/hybrid.ex`, `lib/exphil/training/config.ex`
**Effort:** 1-2 days
**Dependencies:** None (build on existing Jamba)

---

### 5. Mamba-2 SSD Training Algorithm

**Why:** 2-8x training speedup via better hardware utilization. Existing partial impl.

**Plan:**
1. Complete `lib/exphil/networks/mamba_ssd.ex`
2. Validate output matches Mamba-1 selective scan
3. Add `--mamba-variant ssd` flag for training
4. Benchmark training throughput improvement

**Files:** `lib/exphil/networks/mamba_ssd.ex`
**Effort:** 2-3 days
**Dependencies:** None

---

## High Priority

### 6. RWKV-7 Architecture

**Why:** O(1) space complexity, proven at 1.5B device scale. Could replace Mamba.

**Plan:**
1. Study RWKV-7 architecture and Generalized Delta Rule
2. Implement `lib/exphil/networks/rwkv.ex`
3. Add `--backbone rwkv` CLI flag
4. Benchmark vs Mamba on inference speed and accuracy

**Files:** `lib/exphil/networks/rwkv.ex`
**Effort:** 3-5 days
**Dependencies:** None

---

### 7. Frame Delay Testing for Online Play

**Why:** Training has frame delay augmentation, but no systematic evaluation at different delays.

**Plan:**
1. Add `--eval-frame-delay N` to `scripts/eval_model.exs`
2. Test accuracy at delays 0, 6, 12, 18 frames
3. Generate robustness curve (accuracy vs delay)
4. Recommend optimal delay for online play

**Files:** `scripts/eval_model.exs`
**Effort:** 1 day
**Dependencies:** None

---

### 8. Multi-Character Model Training

**Why:** Bitter Lesson suggests single model + scale beats specialists. Need to validate.

**Plan:**
1. Collect replays for 5+ characters
2. Train single model with character ID embedding
3. Compare: multi-char model vs char-specific models
4. Measure per-character accuracy differences

**Files:** `scripts/train_from_replays.exs` with `--character all`
**Effort:** 3-4 days
**Dependencies:** Multi-character replay dataset

---

### 9. Decision Transformer Experiment

**Why:** Alternative training paradigm - condition on desired outcomes (stock lead, damage).

**Plan:**
1. Modify training data: add return-to-go per sequence
2. Implement return embedding layer
3. Train DT on replay data with outcome conditioning
4. Evaluate: can model generate actions for different target outcomes?

**Files:** `lib/exphil/networks/decision_transformer.ex`
**Effort:** 3-4 days
**Dependencies:** None

---

### 10. Mamba State Caching for Fast Warmup

**Why:** Pre-compute Mamba states from typical game starts for instant context.

**Plan:**
1. Generate canonical game states (neutral, advantaged, disadvantaged)
2. Compute Mamba hidden states for each scenario
3. Save to disk with `MambaStateCache.save/2`
4. Load at inference time instead of cold start

**Files:** `lib/exphil/inference/state_cache.ex`
**Effort:** 2 days
**Dependencies:** None

---

### 11. INT4 Quantization (AWQ/GPTQ)

**Why:** 2.7x faster inference, 50% memory reduction vs INT8.

**Plan:**
1. Research AWQ/GPTQ implementations for ONNX
2. Add `--quantize int4` to export script
3. Benchmark: INT4 vs INT8 accuracy and speed
4. Document quality/speed tradeoffs

**Files:** `scripts/export_onnx.exs`, `priv/python/quantize.py`
**Effort:** 2-3 days
**Dependencies:** Python quantization libraries

---

### 12. GLA (Gated Linear Attention) Implementation

**Why:** O(L) with attention expressiveness, faster than FlashAttention on short seqs.

**Plan:**
1. Port GLA from flash-linear-attention library
2. Implement `lib/exphil/networks/gla.ex`
3. Add `--backbone gla` CLI flag
4. Benchmark vs Mamba and Attention

**Files:** `lib/exphil/networks/gla.ex`
**Effort:** 3-4 days
**Dependencies:** None

---

### 13. Checkpoint Averaging for Better Evaluation

**Why:** Transformer paper shows averaging last 5-20 checkpoints improves eval.

**Plan:**
1. Add `--checkpoint-avg N` to eval script
2. Load last N checkpoints, average weights
3. Evaluate averaged model vs single checkpoint
4. Add to best practices

**Files:** `scripts/eval_model.exs`
**Effort:** 1 day
**Dependencies:** Multiple checkpoints from training

---

### 14. Sparse Reward Experiment

**Why:** Bitter Lesson suggests shaped rewards may be suboptimal. Test win/loss only.

**Plan:**
1. Implement sparse reward function (±1 for win/loss, 0 otherwise)
2. Run self-play with sparse vs shaped rewards
3. Compare: strategy diversity, win rates, training speed
4. Document findings

**Files:** `lib/exphil/rewards/sparse.ex`
**Effort:** 2 days
**Dependencies:** Self-play infrastructure

---

### 15. Opponent Modeling via Cross-Attention

**Why:** Explicit attention to opponent state may improve prediction.

**Plan:**
1. Add opponent cross-attention layer option
2. Separate query (self) and key/value (opponent) embeddings
3. Benchmark accuracy improvement
4. Profile inference overhead

**Files:** `lib/exphil/networks/attention.ex`
**Effort:** 2-3 days
**Dependencies:** None

---

## Medium Priority

### 16. Live Training Dashboard

**Why:** Better visibility during long training runs.

**Plan:**
1. Create Phoenix LiveView dashboard
2. Display: loss curves, GPU memory, ETA, current config
3. Add WebSocket updates from training process
4. Optional: integrate with W&B

**Files:** `lib/exphil_web/` (new)
**Effort:** 3-4 days
**Dependencies:** Phoenix (already in deps)

---

### 17. Replay Quality Scoring V2

**Why:** Current quality scoring is basic. Filter out more bad replays.

**Plan:**
1. Detect SD chains (3+ SDs in 30 seconds)
2. Detect AFK behavior (no inputs for 10+ seconds)
3. Detect sandbagging (intentional losing)
4. Add `--quality-threshold N` CLI

**Files:** `lib/exphil/training/replay_validation.ex`
**Effort:** 2 days
**Dependencies:** None

---

### 18. Attention Head Visualization

**Why:** Understand what model attends to for interpretability.

**Plan:**
1. Export attention weights during inference
2. Create Livebook notebook for visualization
3. Analyze: what frames get high attention?
4. Document patterns (combo starts, reactions, etc.)

**Files:** `notebooks/attention_analysis.livemd`
**Effort:** 1-2 days
**Dependencies:** Livebook

---

### 19. Per-Character Accuracy Breakdown

**Why:** Current eval is aggregate. Need per-character performance.

**Plan:**
1. Add character filter to eval script
2. Generate accuracy table per character
3. Identify weakest matchups/characters
4. Guide data collection priorities

**Files:** `scripts/eval_model.exs`
**Effort:** 1 day
**Dependencies:** Multi-character dataset

---

### 20. Model Distillation for Faster Inference

**Why:** Train small model to mimic large model, faster deployment.

**Plan:**
1. Train teacher model (large Mamba)
2. Implement distillation loss (KL divergence on logits)
3. Train student model (small MLP or tiny Mamba)
4. Compare: student accuracy vs teacher

**Files:** `lib/exphil/training/distillation.ex`
**Effort:** 3-4 days
**Dependencies:** Trained large model

---

### 21. Curriculum Learning for Self-Play

**Why:** Start with easy opponents, gradually increase difficulty.

**Plan:**
1. Implement curriculum scheduler in self-play
2. Stages: random → level 1-3 CPU → level 5-7 CPU → self
3. Track skill progression per stage
4. Auto-advance when win rate > threshold

**Files:** `lib/exphil/self_play/curriculum.ex`
**Effort:** 2-3 days
**Dependencies:** Self-play infrastructure

---

### 22. Action State Prediction Auxiliary Task

**Why:** Predicting next Melee action state may regularize policy.

**Plan:**
1. Add auxiliary head predicting next action state
2. Multi-task loss: policy loss + λ × action_state_loss
3. Evaluate: does auxiliary task improve main accuracy?
4. Tune λ coefficient

**Files:** `lib/exphil/networks/policy.ex`
**Effort:** 2 days
**Dependencies:** None

---

### 23. Replay Deduplication by Game ID

**Why:** Duplicate replays waste training compute. Current hash detection is slow.

**Plan:**
1. Extract game ID from Slippi metadata (faster than content hash)
2. Build index of seen game IDs
3. Skip duplicates during loading
4. Add `--skip-duplicates` flag

**Files:** `lib/exphil/training/data.ex`
**Effort:** 1 day
**Dependencies:** None

---

### 24. Export Predictions for Livebook Analysis

**Why:** Eval shows metrics but not raw predictions. Need detailed analysis.

**Plan:**
1. Add `--export-predictions PATH` to eval script
2. Save: predicted vs actual for each frame
3. Create Livebook notebook for analysis
4. Patterns: when does model fail? Which situations?

**Files:** `scripts/eval_model.exs`, `notebooks/prediction_analysis.livemd`
**Effort:** 1-2 days
**Dependencies:** None

---

### 25. Online Learning / Adaptation

**Why:** Adapt to specific opponent during play session.

**Plan:**
1. Collect experiences during live play
2. Fine-tune policy on collected data (few-shot)
3. Measure: does adaptation improve win rate?
4. Tune: update frequency, learning rate

**Files:** `lib/exphil/inference/online_learning.ex`
**Effort:** 3-4 days
**Dependencies:** Live Dolphin integration

---

## Low Priority

### 26. Liquid Neural Networks Prototype

**Why:** Continuous adaptation is unique. Exploratory research.

**Plan:**
1. Research ODE solvers in Elixir/Nx
2. Implement simple Liquid cell
3. Test on small sequence task
4. Evaluate feasibility for game AI

**Files:** `lib/exphil/networks/liquid.ex`
**Effort:** 5-7 days
**Dependencies:** ODE solver library

---

### 27. DreamerV3 World Model

**Why:** Generate synthetic training scenarios. Different paradigm.

**Plan:**
1. Study DreamerV3 architecture
2. Implement world model (state prediction)
3. Generate synthetic game states
4. Use for data augmentation

**Files:** `lib/exphil/world_model/`
**Effort:** 2-3 weeks
**Dependencies:** Significant research

---

### 28. League System (AlphaStar-style)

**Why:** More sophisticated self-play with exploiters and main agents.

**Plan:**
1. Implement exploiter agents (find weaknesses)
2. Add league manager (historical + exploiters + main)
3. Track strategy diversity
4. Prevent rock-paper-scissors collapse

**Files:** `lib/exphil/self_play/league.ex`
**Effort:** 5-7 days
**Dependencies:** Basic self-play working

---

### 29. FlashAttention-3 Integration

**Why:** 30-50% memory reduction for attention. H100-specific.

**Plan:**
1. Research CUDA NIF wrapper options
2. Port FlashAttention-3 kernel to Rust NIF
3. Integrate with attention backbone
4. Benchmark on H100

**Files:** `native/flash_attention_nif/`
**Effort:** 1-2 weeks
**Dependencies:** H100 access, CUDA expertise

---

### 30. S5 Simplified State Space

**Why:** Ablation study to understand Mamba's complexity value.

**Plan:**
1. Implement S5 (simpler than Mamba)
2. Benchmark vs Mamba on same task
3. Understand: what does Mamba's complexity buy?
4. Document findings

**Files:** `lib/exphil/networks/s5.ex`
**Effort:** 2-3 days
**Dependencies:** None

---

## Quick Reference: Effort vs Impact Matrix

```
                    Low Effort          High Effort
              ┌─────────────────────┬─────────────────────┐
              │                     │                     │
   High      │  4. Zamba           │  1. Self-Play       │
   Impact    │  7. Frame Delay     │  6. RWKV-7          │
              │ 13. Ckpt Averaging  │  9. Decision Trans  │
              │                     │  5. Mamba-2 SSD     │
              ├─────────────────────┼─────────────────────┤
              │                     │                     │
   Low       │ 19. Per-char Eval   │ 26. Liquid NNs      │
   Impact    │ 23. Deduplication   │ 27. DreamerV3       │
              │                     │ 29. FlashAttn-3     │
              └─────────────────────┴─────────────────────┘
```

## Recommended Sequence

**Week 1:**
1. #4 Zamba architecture (quick win)
2. #7 Frame delay testing
3. #3 Stick prediction improvements

**Week 2:**
1. #1 Large-scale self-play training
2. #5 Mamba-2 SSD algorithm
3. #2 ONNX deployment pipeline

**Week 3-4:**
1. #6 RWKV-7 architecture
2. #9 Decision Transformer experiment
3. #8 Multi-character model

**Ongoing:**
- #10-15 as training runs complete
- #16-25 based on evaluation results

---

**Last Updated:** 2026-01-31
