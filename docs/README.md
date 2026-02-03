# ExPhil Documentation

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| **guides/** | User-facing how-to documentation (training, inference, scripts) |
| **reference/** | Technical specifications (architecture, embedding dimensions) |
| **operations/** | Deployment and infrastructure (Docker, RunPod, cloud storage) |
| **research/** | Research notes, case studies, prior art |
| **planning/** | Project tracking (GOALS, FIXES, roadmaps, plans) |
| **internals/** | Deep implementation details (GPU, kernels, optimizations) |

## Quick Links

### Getting Started
- [Training Guide](guides/TRAINING.md) - Main training documentation
- [Training Cheatsheet](guides/TRAINING_CHEATSHEET.md) - Quick reference
- [Scripts Reference](guides/SCRIPTS.md) - All available scripts

### Reference
- [Architecture](reference/ARCHITECTURE.md) - System design
- [Embedding Dimensions](reference/EMBEDDING_DIMENSIONS.md) - Tensor shapes
- [Backbones](reference/architectures/ARCHITECTURES.md) - MLP, LSTM, Mamba, etc.

### Project Status
- [Goals](planning/GOALS.md) - Big picture priorities
- [Fixes](planning/FIXES.md) - Technical debt tracking

## Generated API Docs

ExDoc-generated documentation (from code comments) is in `doc/` (gitignored).
Regenerate with:

```bash
mix docs
open doc/index.html
```

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project instructions for Claude Code
- [README.md](../README.md) - Project overview
