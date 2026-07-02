"""
Minimal JAX reproducer for SIGBUS in depthwise conv gradient with bfloat16.

XLA crashes with SIGBUS after ~3000 gradient iterations when computing
gradients for a depthwise convolution with:
  - feature_group_count = 1024
  - input shape: (16, 60, 1024)
  - kernel shape: (4, 1024, 1)  [depthwise: kernel_in_channels=1]
  - dtype: bfloat16
  - GPU: RTX 5090 (sm_120 / Blackwell)

Run: python repro_bf16_conv_sigbus.py
Run with f32 (should not crash): python repro_bf16_conv_sigbus.py --f32

Environment:
  - XLA 0.10.0 (via EXLA precompiled binary)
  - CUDA 13.2
  - Driver 595.45.04
  - GPU: NVIDIA RTX 5090 (sm_120)
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import sys

use_f32 = "--f32" in sys.argv
dtype = jnp.float32 if use_f32 else jnp.bfloat16

print(f"=== bf16 Depthwise Conv Gradient SIGBUS Reproducer ===")
print(f"  Precision: {dtype}")
print(f"  Devices: {jax.devices()}")
print()

# Depthwise conv: feature_group_count = in_channels
# JAX uses lax.conv_general_dilated for this
from jax import lax

# Parameters
batch, seq_len, channels = 16, 60, 1024
kernel_size = 4

# Initialize kernel: shape (kernel_size, 1, channels) for depthwise
key = jax.random.PRNGKey(42)
kernel = jax.random.normal(key, (kernel_size, 1, channels), dtype=jnp.float32)

def depthwise_conv_loss(kernel, x):
    """Forward pass: causal depthwise conv1d + mean."""
    # Cast to compute dtype
    k = kernel.astype(dtype)

    # lax.conv_general_dilated with feature_group_count = channels
    # dimension_numbers: (batch, spatial, features) for input and kernel
    out = lax.conv_general_dilated(
        x,                              # (batch, seq_len, channels)
        k,                              # (kernel_size, 1, channels)
        window_strides=(1,),
        padding=[(kernel_size - 1, 0)], # causal padding
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=channels,
    )
    return jnp.mean(out)

# JIT compile the gradient function
grad_fn = jit(grad(depthwise_conv_loss))

# Input tensor
x = jnp.ones((batch, seq_len, channels), dtype=dtype) * 0.1

# Warmup
print("  JIT compiling...")
g = grad_fn(kernel, x)
g.block_until_ready()
print(f"  JIT done, grad norm: {jnp.linalg.norm(g):.6f}")

# Run iterations
print(f"\n  Running 5000 gradient iterations...")
print(f"  (SIGBUS expected around step ~3000 for bf16)")
print()

for i in range(1, 5001):
    g = grad_fn(kernel, x)
    g.block_until_ready()

    if i % 500 == 0:
        print(f"  Step {i}: OK (grad norm: {jnp.linalg.norm(g):.6f})")

print(f"\n  Completed 5000 iterations without crash!")
print(f"  Bug may not reproduce on this hardware/XLA version.")
