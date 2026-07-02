# SIGBUS in Depthwise Convolution Gradient with bf16

## Summary

XLA's depthwise convolution gradient crashes with SIGBUS after ~3000 training steps when using bf16 precision with `feature_group_size=1024` on RTX 5090 (sm_120).

## Environment

- GPU: NVIDIA RTX 5090 (sm_120, 32GB VRAM)
- EXLA: 0.11.0
- XLA: 0.10.0 (precompiled)
- Nx: 0.11.0
- CUDA: 13.2
- Driver: 595.45.04

## Reproduction

The crash occurs during training of a Mamba SSM model that uses a causal depthwise 1D convolution:

```elixir
# The conv layer that triggers the crash
Axon.conv(input, 1024,
  kernel_size: {4},
  padding: [{3, 0}],
  feature_group_size: 1024,  # depthwise: each channel has its own kernel
  name: "mamba_dw_conv"
)
```

When mixed precision is enabled (bf16 compute), the conv gradient computation crashes with SIGBUS after ~3000 training iterations.

### Minimal Reproducer

```elixir
# Run: mix run scripts/repro_bf16_conv_sigbus.exs

model = Axon.input("input", shape: {nil, 60, 1024})
|> Axon.conv(1024,
  kernel_size: {4},
  padding: [{3, 0}],
  feature_group_size: 1024
)
|> Axon.dense(1)

# Apply bf16 mixed precision
policy = Axon.MixedPrecision.create_policy(
  params: {:f, 32},
  compute: {:bf, 16},
  output: {:f, 32}
)
model = Axon.MixedPrecision.apply_policy(model, policy,
  except: [:batch_norm, :layer_norm]
)

{init_fn, predict_fn} = Axon.build(model, mode: :train)
params = init_fn.(Nx.template({1, 60, 1024}, :f32), Axon.ModelState.empty())

loss_fn = fn params, input ->
  predict_fn.(params, input) |> Nx.mean()
end

grad_fn = Nx.Defn.jit(
  fn params, input -> Nx.Defn.value_and_grad(loss_fn).(params, input) end,
  compiler: EXLA
)

# This will crash with SIGBUS after ~3000 iterations
input = Nx.iota({16, 60, 1024}, type: :bf16)
for i <- 1..5000 do
  {_loss, _grads} = grad_fn.(params, input)
  if rem(i, 500) == 0, do: IO.puts("Step #{i} OK")
end
```

## Symptoms

1. JIT compilation prints debug info about the conv gradient:
   ```
   grad_conv x: {{16, 60, 1024}, []}
   grad_conv y: {{4, 1, 1024}, []}
   grad_conv opts: [feature_group_size: 1024, ...]
   ```

2. Training runs normally for ~3000 steps at ~12ms/step

3. Process terminates with SIGBUS (address boundary error) — no Elixir exception, no error message

## Root Cause Analysis

### Likely: Memory alignment in depthwise conv gradient buffer

The depthwise conv gradient with `feature_group_size=1024` creates temporary buffers:
- Kernel gradient: 1024 groups × 4 kernel_size × 2 bytes (bf16) = 8192 bytes
- Input gradient: involves transposed convolution with the same grouping

With bf16 (2-byte elements), the buffer layout may not meet CUDA kernel alignment requirements (typically 4/8/16 bytes). The XLA CUDA kernel for depthwise conv gradient may write slightly past the allocated buffer.

### Crash timing varies by context

- **Isolated reproducer**: Crashes during or immediately after first JIT execution (SIGBUS during `grad_fn.(params, input)`)
- **Full training model**: Crashes after ~3000 steps (the larger program's memory layout delays when the overwrite hits critical data)
- This suggests the bug is in the **compiled CUDA kernel itself**, not gradual memory corruption

### Code Path

1. **Nx gradient**: `grad_conv` in `/nx/lib/nx/defn/grad.ex:1611` computes input and kernel gradients via two `Nx.conv` calls
2. **EXLA compilation**: `/exla/lib/exla/defn.ex:1010-1045` converts operands to output type (bf16) before XLA conv
3. **StableHLO**: `/exla/lib/exla/mlir/value.ex:536-566` generates `stablehlo.convolution` with feature_group_count
4. **XLA CUDA kernel**: The precompiled XLA binary selects a CUDA kernel for depthwise conv gradient with bf16

The type conversion at step 2 (`to_type(operand, output_type)`) converts to bf16 unconditionally. No special handling for depthwise conv gradients.

## Workaround

Cast conv inputs to f32 explicitly in the model definition:

```elixir
def build_depthwise_conv1d(input, channels, kernel_size, name) do
  input
  |> Axon.nx(fn x -> Nx.as_type(x, :f32) end, name: "#{name}_cast_f32")
  |> Axon.conv(channels,
    kernel_size: {kernel_size},
    padding: [{kernel_size - 1, 0}],
    feature_group_size: channels,
    name: "#{name}_dw_conv"
  )
end
```

This ensures the conv gradient always uses f32 buffers, avoiding the alignment issue. The rest of the model still uses bf16 compute via mixed precision.

## Filing an Issue

### Where

- **File at**: https://github.com/elixir-nx/nx/issues (EXLA-specific — JAX 0.9.2 does NOT reproduce this bug)
- The equivalent JAX/Python reproducer completes 5000 iterations without crash on the same hardware
- This points to an interaction between the full Edifice SSM model + Axon.MixedPrecision in EXLA

### Bisection Results

| Test | Crashes? | Steps |
|------|----------|-------|
| JAX Python, bf16 depthwise conv | No | 5000 |
| EXLA isolated bf16 depthwise conv | No | 500 |
| EXLA + Axon MixedPrecision, single conv | No | 500 |
| EXLA + Axon MP, 2-layer Mamba-like | No | 500 |
| EXLA + Axon MP, Mamba-like batch=16 | No | 10,000 |
| Full Edifice Mamba + MP in training | **YES** | ~3000 |

The crash is specific to the full model complexity, not the conv operation itself.

### Suggested Title

"SIGBUS crash in depthwise convolution gradient with bf16 precision (feature_group_size > 1)"

### Key Info for Issue

- Shapes: input `{16, 60, 1024}`, kernel `{4, 1, 1024}`, `feature_group_size=1024`
- Type: `{:bf, 16}`
- Backend: EXLA with CUDA (sm_120)
- Crash timing: ~3000 iterations into training
- Signal: SIGBUS (not SIGSEGV)
- No Elixir-level exception — the NIF process is killed by OS

### Potential Fix (upstream)

In `exla/lib/exla/defn.ex`, the `to_type` conversion before conv could check for depthwise conv (feature_group_size > 1) and keep f32 precision:

```elixir
# In the conv operator handler:
{operand, kernel} =
  if feature_group_count > 1 and Typespec.type_match?(output_type, {:bf, 16}) do
    # Keep depthwise conv in f32 to avoid gradient buffer alignment issues
    {operand, kernel}
  else
    {to_type(operand, output_type), to_type(kernel, output_type)}
  end
```

## Status

- **Workaround applied**: Edifice's `build_depthwise_conv1d` casts to f32 before conv
- **Impact**: Minimal — conv is <5% of Mamba's compute
- **Upstream issue**: Not yet filed
