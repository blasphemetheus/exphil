# Flash Attention Documentation (Archived)

**Archived:** 2026-02-04

## Why Archived

Flash Attention optimizes transformer attention from O(n²) memory to O(n) memory, but ExPhil now has 15 backbone architectures including several that are fundamentally O(n) in both memory AND compute:

- **Mamba, Mamba-2 SSD** - Linear complexity state space models
- **RWKV-7** - O(1) memory per token
- **GLA** - Gated Linear Attention
- **HGRN-2** - Hierarchical gated recurrence
- **S5** - Simplified state space

These attention-free architectures make Flash Attention optimization less relevant. The standard `attention` backbone (which would benefit from Flash Attention) has 220ms inference vs 8.9ms for Mamba - it's not recommended for production use.

The XLA custom call implementation would require 4-6 weeks of C++/CUDA work for diminishing returns.

## Contents

| File | Original Location | Description |
|------|-------------------|-------------|
| `FLASH_ATTENTION_NIF_RESEARCH.md` | internals/ | Research on NIF approach limitations |
| `FLASH_ATTENTION_XLA_PLAN.md` | planning/ | Detailed XLA custom call implementation plan |
| `FLASH_ATTENTION_BACKWARD.md` | internals/ | Backward pass implementation details |
| `FLASH_ATTENTION_PLAN.md` | planning/ | High-level options comparison |

## If Reviving This Work

If Flash Attention becomes relevant again (e.g., for very long sequence attention layers), start with `FLASH_ATTENTION_XLA_PLAN.md` which has the most actionable implementation plan.

See also: `docs/internals/GPU_OPTIMIZATIONS.md` for current optimization status.
