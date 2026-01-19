# Inference & Deployment

Optimizing models for real-time gameplay (target: <16ms for 60 FPS).

## Architecture Benchmarks

| Architecture | Inference | 60 FPS Ready |
|--------------|-----------|--------------|
| LSTM (Axon) | 220.95 ms | No |
| Mamba (Axon) | 8.93 ms | **Yes** |
| ONNX INT8 | 0.55 ms | **Yes** |

## Optimization Strategies

### 1. Mamba Backbone (Recommended)

24.75x faster than LSTM, achieves <16ms target.

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone mamba \
  --hidden 256 --window-size 60 \
  --state-size 16 --expand-factor 2 --conv-size 4 \
  --num-layers 2 --epochs 5
```

### 2. Knowledge Distillation

Train fast MLP student from slow LSTM/Mamba teacher:

```bash
# Step 1: Generate soft labels
mix run scripts/generate_soft_labels.exs \
  --teacher checkpoints/mamba_policy.bin \
  --replays replays \
  --output soft_labels.bin \
  --temperature 2.0 \
  --max-files 50

# Step 2: Train distilled MLP
mix run scripts/train_distillation.exs \
  --soft-labels soft_labels.bin \
  --hidden 64,64 \
  --epochs 10 \
  --alpha 0.7 \
  --output distilled_policy.bin
```

| Hidden | Parameters | Inference | Accuracy |
|--------|------------|-----------|----------|
| 32,32 | ~70K | ~1ms | Good |
| 64,64 | ~270K | ~2ms | Better |
| 128,64 | ~400K | ~3ms | Best |

### 3. Architecture Simplification

Quick wins for LSTM models:

| Change | Expected Speedup |
|--------|------------------|
| Reduce hidden size (256->64) | 4-16x |
| Reduce window size (60->20) | ~3x |
| Use GRU instead of LSTM | 1.3x |
| Single layer vs stacked | 2x |

```bash
mix run scripts/train_from_replays.exs \
  --temporal --backbone lstm \
  --hidden 64 --window-size 20 \
  --epochs 5 --max-files 20
```

### 4. GPU Acceleration

10-100x speedup with CUDA:

```elixir
# config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.8],
  default: [platform: :host]
config :exla, default_client: :cuda
```

### 5. BF16 Training

2x speedup, minimal accuracy loss (default in ExPhil):

```bash
# BF16 is default
mix run scripts/train_from_replays.exs --temporal --backbone lstm

# Use F32 for full precision
mix run scripts/train_from_replays.exs --precision f32 --temporal --backbone lstm
```

## ONNX Export & Quantization

For production deployment with maximum performance.

### Current Status

- INT8 quantization script via ONNX Runtime
- Direct `axon_onnx` export has runtime API issues
- NumPy export workaround for Python-side ONNX conversion

### Complete Workflow

```bash
# Step 1: Training produces checkpoint and policy files
ls checkpoints/
# imitation_latest.axon (full checkpoint)
# imitation_latest_policy.bin (exported policy)

# Step 2: Export to NumPy format
mix run scripts/export_numpy.exs \
  --policy checkpoints/imitation_latest_policy.bin \
  --output weights.npz

# Step 3: Rebuild model in Python and export to ONNX
python priv/python/convert_to_onnx.py \
  --weights weights.npz \
  --config checkpoints/imitation_latest_policy.bin \
  --output policy.onnx

# Step 4: Quantize to INT8
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx

# Step 5: Benchmark
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('policy_int8.onnx')
inp = sess.get_inputs()[0]
x = np.random.randn(1, 60, 1991).astype(np.float32)

for _ in range(10): sess.run(None, {inp.name: x})
start = time.time()
for _ in range(100): sess.run(None, {inp.name: x})
print(f'ONNX INT8 inference: {(time.time() - start) * 10:.2f} ms')
"
# Expected: ~0.5ms per inference
```

### Direct ONNX Export (Limited)

```bash
# From checkpoint
mix run scripts/export_onnx.exs \
  --checkpoint checkpoints/imitation_latest.axon \
  --output policy.onnx

# From exported policy
mix run scripts/export_onnx.exs \
  --policy checkpoints/imitation_latest_policy.bin \
  --output policy.onnx
```

### Verify ONNX Model

```bash
python -c "import onnxruntime as ort; sess = ort.InferenceSession('policy.onnx'); print('OK:', [i.name for i in sess.get_inputs()])"
```

### Quantization Options

```bash
# Dynamic quantization (no calibration needed, fast)
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx

# Static quantization (more accurate, requires calibration data)
python priv/python/quantize_onnx.py policy.onnx policy_int8.onnx \
  --static --calibration-data calibration.npz
```

### Elixir Inference with ONNX

Add `ortex` dependency:
```elixir
{:ortex, "~> 0.1"}
```

```elixir
{:ok, model} = Ortex.load("policy_int8.onnx")
output = Ortex.run(model, input_tensor)
```

## When to Use What

| Use Case | Recommendation |
|----------|----------------|
| Training | Axon (native Elixir, easy debugging) |
| Prototyping | Axon Mamba (8.93ms, good enough) |
| Production gameplay | ONNX INT8 (0.55ms, maximum performance) |
| Python/Rust integration | ONNX (cross-platform) |

## Priority Order for Optimization

1. **Mamba backbone** - Available now, 24.75x faster than LSTM
2. **ONNX + INT8 quantization** - 0.55ms inference
3. **Train smaller LSTM** (hidden=64, window=20) - Quick experiment
4. **Knowledge distillation to MLP** - Best accuracy/speed tradeoff
5. **GPU acceleration** - If NVIDIA GPU available

## Temporal Inference in Agents

Agents automatically detect temporal policies and handle frame buffering:

```elixir
# Load temporal policy - auto-detects config
{:ok, agent} = ExPhil.Agents.start_with_policy(:my_agent, "checkpoints/temporal_policy.bin")

# Check temporal config
config = ExPhil.Agents.Agent.get_config(agent)
# => %{temporal: true, backbone: :sliding_window, window_size: 60, ...}

# Each get_action call adds to the frame buffer
{:ok, action} = ExPhil.Agents.get_action(:my_agent, game_state)

# Reset buffer when starting a new game
ExPhil.Agents.Agent.reset_buffer(agent)
```

**How temporal inference works:**
1. Each `get_action` embeds the game state and adds to rolling buffer
2. If buffer < `window_size` frames, pad with first frame (warmup)
3. Build sequence tensor `[1, window_size, embed_size]`
4. Feed to temporal policy network
5. Sample action from output logits

## Open Source: axon_onnx

PR submitted for Nx 0.10+ compatibility:
- PR: https://github.com/mortont/axon_onnx/compare/master...blasphemetheus:axon_onnx:support-nx-0.10
- Enables compilation, but runtime API issues remain
- Workaround: `{:axon_onnx, github: "mortont/axon_onnx"}`
- Forum: https://elixirforum.com/t/error-using-axononnx-v0-4-0-undefined-function-transform-2/63326
