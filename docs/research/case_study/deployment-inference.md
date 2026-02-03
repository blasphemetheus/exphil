# Deployment & Inference for Real-Time Game AI

This document covers the technical requirements and optimization strategies for deploying neural network models in real-time game environments, specifically targeting 60 FPS gameplay.

## The Frame Budget Problem

### 60 FPS Requirements

| FPS | Frame Budget | Notes |
|-----|--------------|-------|
| 60 | 16.67ms | Melee, most fighting games |
| 30 | 33.33ms | Some strategy games |
| 120+ | <8.33ms | High refresh rate |

For Melee at 60 FPS, the entire inference pipeline must complete in **<16ms**.

### Budget Breakdown

Typical frame budget allocation:

```
Total Budget: 16.67ms
├── Input polling:     ~1ms
├── State embedding:   ~2ms
├── Model inference:   ~8ms (target)
├── Action decoding:   ~1ms
├── Buffer/overhead:   ~4ms
└── Reserved margin:   ~0.67ms
```

**Key constraint**: Model inference should target **<10ms** to leave headroom.

---

## ExPhil Inference Benchmarks

From ExPhil's current testing (see `docs/INFERENCE.md`):

| Architecture | Inference Time | 60 FPS Ready |
|--------------|----------------|--------------|
| LSTM (Axon) | 220.95ms | No |
| Mamba (Axon) | 8.93ms | **Yes** |
| ONNX INT8 | 0.55ms | **Yes** |

### Why Mamba Works

Mamba's O(n) complexity vs O(n²) attention enables real-time inference:
- No quadratic attention computation
- Efficient state-space model
- Linear scaling with sequence length

---

## ONNX Deployment

### What is ONNX?

**Open Neural Network Exchange** - Framework-agnostic model format.

```
PyTorch/TensorFlow/Axon Model
    │
    ▼ Export
┌─────────────────────────────────┐
│         ONNX Format             │
│  (Directed Acyclic Graph)       │
└─────────────────┬───────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐
│ ONNX   │  │ Tensor │  │ Direct │
│Runtime │  │  RT    │  │   ML   │
└────────┘  └────────┘  └────────┘
```

### Benefits

1. **Framework Independence**: Train in Axon, deploy anywhere
2. **Hardware Optimization**: Runtime handles GPU/CPU dispatch
3. **Quantization**: INT8/FP16 support built-in
4. **Cross-Platform**: Windows, Linux, mobile, web

### ONNX Runtime Optimizations

ONNX Runtime applies automatic optimizations:

| Optimization | Description |
|--------------|-------------|
| **Constant Folding** | Pre-compute static operations |
| **Dead Code Elimination** | Remove unused graph paths |
| **Operator Fusion** | Combine sequential ops |
| **Memory Planning** | Optimize tensor allocation |

### Execution Providers

| Provider | Platform | Use Case |
|----------|----------|----------|
| CUDA | NVIDIA GPU | Training, high-throughput |
| TensorRT | NVIDIA GPU | Optimized inference |
| DirectML | Windows GPU | Cross-vendor GPU |
| CoreML | Apple | Mac/iOS deployment |
| CPU | Any | Fallback, edge devices |
| NNAPI | Android | Mobile deployment |

---

## Quantization

### Precision Levels

| Precision | Bits | Size | Speed | Accuracy |
|-----------|------|------|-------|----------|
| FP32 | 32 | 1x | 1x | Baseline |
| FP16 | 16 | 0.5x | ~2x | ~Same |
| INT8 | 8 | 0.25x | ~4x | Slight loss |
| INT4 | 4 | 0.125x | ~8x | Noticeable loss |

### ExPhil Quantization Results

From ONNX export testing:

| Model | FP32 | INT8 | Speedup |
|-------|------|------|---------|
| Policy | 2.1ms | 0.55ms | 3.8x |

### Quantization Methods

**Post-Training Quantization (PTQ)**:
- No retraining required
- Use calibration dataset
- Some accuracy loss

**Quantization-Aware Training (QAT)**:
- Train with simulated quantization
- Better accuracy retention
- More complex training

### When to Quantize

| Scenario | Recommendation |
|----------|----------------|
| Development | FP32 (accuracy) |
| Testing | FP16 (balance) |
| Deployment | INT8 (speed) |
| Edge/Mobile | INT8 or INT4 |

---

## TensorRT Integration

### What is TensorRT?

NVIDIA's inference optimization SDK. Takes ONNX models and creates GPU-optimized engines.

```
ONNX Model
    │
    ▼ TensorRT Optimization
┌─────────────────────────────────┐
│  • Layer fusion                 │
│  • Kernel auto-tuning           │
│  • Precision calibration        │
│  • Memory optimization          │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│     TensorRT Engine (.plan)     │
│     Hardware-specific binary    │
└─────────────────────────────────┘
```

### TensorRT Benefits

| Optimization | Typical Speedup |
|--------------|-----------------|
| Layer Fusion | 2-3x |
| INT8 Quantization | 2-4x |
| Kernel Selection | 1.5-2x |
| **Combined** | **4-10x** |

### When to Use TensorRT

✅ **Use when**:
- Deploying on NVIDIA hardware
- Need maximum inference speed
- Model architecture is stable

❌ **Avoid when**:
- Cross-platform deployment needed
- Rapid model iteration
- Non-NVIDIA hardware

---

## GPU vs CPU Inference

### GPU Advantages

| Factor | GPU | CPU |
|--------|-----|-----|
| Throughput | High | Low |
| Parallelism | Massive | Limited |
| Batch inference | Excellent | Poor |
| Power | High | Low |

### CPU Advantages

| Factor | CPU | GPU |
|--------|-----|-----|
| Latency (single sample) | Can be lower | Memory transfer overhead |
| Determinism | Better | Variable |
| Availability | Always present | May not have |
| Power | Lower | Higher |

### For Fighting Games

**Single-sample, low-latency** requirements favor:
1. **Small models on CPU**: Avoid GPU transfer overhead
2. **Batched on GPU**: If running multiple agents
3. **Hybrid**: Embedding on CPU, inference on GPU

### Memory Transfer Overhead

```
CPU Memory ─────────────────────────► GPU Memory
            PCIe transfer (~10GB/s)

For small models, transfer time can exceed compute time!
```

**Solution**: Keep entire pipeline on one device.

---

## Zero-Copy Strategies

### The Problem

```
Traditional Pipeline:
CPU (state) → Copy → GPU (inference) → Copy → CPU (action)

Overhead: 2 memory copies per frame
```

### Solutions

**1. GPU-Resident State**
```
Keep game state on GPU
GPU (state) → GPU (inference) → GPU (action)
Only copy action output (small)
```

**2. Unified Memory (CUDA)**
```cpp
cudaMallocManaged(&data, size);
// Automatic migration between CPU/GPU
```

**3. Pinned Memory**
```cpp
cudaHostAlloc(&data, size, cudaHostAllocMapped);
// Direct GPU access to CPU memory
```

### ShaderNN Approach

For graphics integration, [ShaderNN](https://www.sciencedirect.com/science/article/pii/S0925231224013997) provides texture-based I/O for zero-copy with rendering pipelines.

---

## Latency Compensation

### Frame Delay in Online Play

Melee online (Slippi) has 18+ frame delay:
- 18 frames × 16.67ms = **300ms**

### Training for Delay

**slippi-ai approach**:
```python
# Train with delay buffer
delayed_action = action_buffer[t - delay]
env.step(delayed_action)
```

**Predictive compensation** (from [ACM research](https://dl.acm.org/doi/fullHtml/10.1145/3543758.3543767)):
- Neural network predicts future state
- ~1.6ms inference for prediction
- Compensates for network latency

### ExPhil Delay Handling

```elixir
# Frame delay configuration
config = %{
  policy_delay: 18,  # Online play
  # policy_delay: 0,  # Offline training
}
```

---

## Deterministic Execution

### Why It Matters

Fighting games require consistent timing:
- Frame-perfect techniques (wavedash, L-cancel)
- Reaction-based gameplay
- Tournament fairness

### Sources of Non-Determinism

| Source | Impact | Mitigation |
|--------|--------|------------|
| GPU scheduling | Variable latency | Fixed batch size |
| Memory allocation | Spikes | Pre-allocate |
| Thermal throttling | Slowdown | Cooling/limits |
| OS scheduling | Jitter | Real-time priority |

### FPGA Alternative

From [research on deterministic execution](https://liquidinstruments.com/blog/neural-networks-on-gpus-vs-cpus-vs-fpgas/):

> "FPGAs provide deterministic execution by physically structuring the logic for the task. This results in identical latency every time."

**Trade-off**: Development complexity vs timing guarantees.

---

## Practical Recommendations

### For ExPhil Development

**Phase 1: Training**
- Use EXLA/CUDA for training speed
- FP32 precision
- Large batch sizes

**Phase 2: Testing**
- ONNX export
- FP16 quantization
- Validate accuracy

**Phase 3: Deployment**
- ONNX Runtime or TensorRT
- INT8 quantization
- Benchmark on target hardware

### Minimum Viable Deployment

```
Target: <10ms inference
Hardware: Consumer GPU (RTX 3060+) or modern CPU

Recommended Stack:
├── Model: Mamba backbone (O(n) complexity)
├── Format: ONNX
├── Runtime: ONNX Runtime + CUDA EP
├── Precision: INT8
└── Expected: ~1-2ms inference
```

### Hardware Requirements

| Tier | Hardware | Expected Latency |
|------|----------|------------------|
| Minimum | CPU only | 5-15ms |
| Recommended | RTX 3060 | 1-3ms |
| Optimal | RTX 4080+ | <1ms |

---

## Benchmarking Protocol

### What to Measure

1. **Cold start**: First inference (JIT compilation)
2. **Warm inference**: Subsequent inferences
3. **P50/P95/P99 latency**: Percentiles for consistency
4. **Memory usage**: Peak and steady-state

### ExPhil Benchmark Script

```bash
# Run inference benchmark
mix run scripts/benchmark_inference.exs \
  --model checkpoints/policy.onnx \
  --iterations 1000 \
  --warmup 100
```

### Reporting Template

```
Model: policy_mamba_v1
Hardware: RTX 3080
Precision: INT8

Results (n=1000):
  P50: 0.55ms
  P95: 0.62ms
  P99: 0.71ms
  Max: 1.2ms

Memory: 245MB VRAM
```

---

## Resources

### Books
- [Real-Time AI with ONNX Runtime](https://www.amazon.com/REAL-TIME-ONNX-RUNTIME-HIGH-PERFORMANCE-ENGINEERING-ebook/dp/B0G7VFYSBH) - Logan Dolling

### Documentation
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX Specification](https://onnx.ai/)

### Papers
- [Latency Estimation for Mobile GPU](https://www.mdpi.com/2073-431X/10/8/104)
- [ShaderNN: Real-time Mobile GPU Inference](https://www.sciencedirect.com/science/article/pii/S0925231224013997)
- [Using ANNs for Latency Compensation](https://dl.acm.org/doi/fullHtml/10.1145/3543758.3543767)

### ExPhil Docs
- [INFERENCE.md](../INFERENCE.md) - ExPhil-specific inference guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Model architecture details
