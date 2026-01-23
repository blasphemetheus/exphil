# Melee AI Case Studies

This directory contains comprehensive documentation of existing Super Smash Bros. Melee AI projects, research, and technical challenges. The purpose is to provide context for Claude Code instances working on ExPhil without requiring deep knowledge of our specific implementation.

## Document Index

### Core Project Case Studies

| Document | Description |
|----------|-------------|
| [slippi-ai.md](slippi-ai.md) | vladfi1's TensorFlow BC+RL framework (primary reference) |
| [phillip.md](phillip.md) | Original pure deep RL bot (deprecated, historical) |
| [libmelee.md](libmelee.md) | Python API for Melee game state access |
| [smashbot.md](smashbot.md) | Rule-based behavior tree approach |
| [eric-gu-transformer.md](eric-gu-transformer.md) | GPT-style Transformer approach (2024) |
| [project-nabla.md](project-nabla.md) | Self-play stability research (2022) |

### Additional RL & ML Projects

| Document | Description |
|----------|-------------|
| [athena.md](athena.md) | DQN-based approach with convolutional state encoding |
| [a3c-melee.md](a3c-melee.md) | A3C architecture with TensorFlow 1.x |
| [ssbm-gym.md](ssbm-gym.md) | OpenAI Gym-compatible environment for RL training |
| [q-learning-melee.md](q-learning-melee.md) | Tabular Q-learning educational implementation |

### Analytics & Statistics Tools

| Document | Description |
|----------|-------------|
| [slippi-stats.md](slippi-stats.md) | Browser-based instant stats computation (vinceau) |
| [slippipedia.md](slippipedia.md) | QML/Felgo desktop replay manager (cbartsch) |
| [slip-py.md](slip-py.md) | Flask-based replay browser and analyzer (pcrain) |

### Prediction & Classification

| Document | Description |
|----------|-------------|
| [melee-prediction.md](melee-prediction.md) | XGBoost match/tournament prediction (Erdos Institute) |
| [svm-melee.md](svm-melee.md) | SVM character classification from replays |

### Infrastructure & Ecosystem

| Document | Description |
|----------|-------------|
| [project-slippi.md](project-slippi.md) | The foundational Slippi ecosystem (Fizzi) |
| [peppi-ecosystem.md](peppi-ecosystem.md) | Rust parser + Python/Julia bindings (hohav) |
| [smashscan.md](smashscan.md) | YOLO-based video analysis for tournaments |

### Technical Deep Dives

| Document | Description |
|----------|-------------|
| [melee-ai-landscape.md](melee-ai-landscape.md) | Overview of the entire Melee AI research space |
| [action-space.md](action-space.md) | Melee's 30 billion input action space problem |
| [state-representation.md](state-representation.md) | How to embed Melee game state for ML |
| [training-approaches.md](training-approaches.md) | BC vs RL vs hybrid methodology comparison |
| [reward-shaping.md](reward-shaping.md) | **Reward design: PBRS theory, character-specific, hacking** |
| [research-papers.md](research-papers.md) | Key academic papers and technical blogs |
| [other-fighting-game-ai.md](other-fighting-game-ai.md) | AlphaStar, OpenAI Five, FightingICE, Street Fighter |
| [deployment-inference.md](deployment-inference.md) | ONNX, quantization, real-time inference |

### Reference Materials

| Document | Description |
|----------|-------------|
| [characters.md](characters.md) | Character-specific considerations for AI |
| [community-projects.md](community-projects.md) | Additional notable projects summary |
| [all-projects.md](all-projects.md) | **Complete index of all known projects** |
| [DOCUMENTATION_GAPS.md](DOCUMENTATION_GAPS.md) | **Roadmap of areas needing more research** |

## How to Use This Directory

**For understanding the landscape:**
Start with [melee-ai-landscape.md](melee-ai-landscape.md) for a high-level overview.

**For implementation reference:**
Read [slippi-ai.md](slippi-ai.md) as it's the most complete and active project.

**For self-play stability:**
Study [project-nabla.md](project-nabla.md) for critical lessons on population-based training.

**For alternative architectures:**
See [eric-gu-transformer.md](eric-gu-transformer.md) for GPT-style approach.

**For understanding why certain approaches work:**
See [training-approaches.md](training-approaches.md) and [research-papers.md](research-papers.md).

**For technical architecture decisions:**
Check [action-space.md](action-space.md) and [state-representation.md](state-representation.md).

**For reward function design:**
See [reward-shaping.md](reward-shaping.md) for PBRS theory, character-specific rewards, and avoiding reward hacking.

**For a complete list of all projects:**
See [all-projects.md](all-projects.md) for the comprehensive index.

**For tracking documentation progress:**
See [DOCUMENTATION_GAPS.md](DOCUMENTATION_GAPS.md) for the research roadmap.

## Context for ExPhil

ExPhil is an Elixir-based successor to slippi-ai targeting lower-tier characters (Mewtwo, G&W, Link, Ganondorf, Zelda, Ice Climbers). Key differentiators:

- **Language**: Elixir/Nx/Axon instead of Python/TensorFlow
- **Architecture**: Mamba backbone (O(n) vs O(n²) Transformers)
- **Target Characters**: Low-tier specialists vs Fox/Falco mains
- **Inference**: Sub-10ms for 60 FPS real-time play

The case studies here document what worked (and didn't) in prior projects to inform ExPhil's design decisions.

## Quick Reference: Project Comparison

| Aspect | Phillip | slippi-ai | SmashBot | Eric Gu | Nabla | ExPhil |
|--------|---------|-----------|----------|---------|-------|--------|
| **Year** | 2017 | 2020+ | 2019+ | 2024 | 2022 | 2024+ |
| **Language** | Python/TF | Python/TF | Python | Python | Python | Elixir/Nx |
| **Approach** | Pure RL | BC → RL | Rules | IL (Transformer) | BC → RL | BC → RL |
| **Status** | Deprecated | Active | Active | Research | Research | Active |
| **Characters** | Fox, Falcon | Fox, Falco | Fox only | All | Various | Low-tier |
| **Data Source** | Self-play | Replays | None | Replays | Replays | Replays |
| **Self-Play** | Single | Historical | N/A | None | Population | TBD |
| **Key Insight** | RL works | BC+RL best | Rules limit | Scale matters | Pop. required | Mamba fast |

## Key Lessons Summary

### From Phillip
- Pure RL is possible but impractical (months of compute)
- Superhuman reactions (33ms) are unrealistic

### From slippi-ai
- Two-stage BC → RL is the winning approach
- Historical sampling prevents self-play collapse
- 18+ frame delay training enables online play

### From Eric Gu
- Transformers work for fighting games
- All-character training beats single-character
- Simple objectives (next-token prediction) are effective

### From Project Nabla
- Single-opponent self-play causes rock-paper-scissors cycling
- Population-based training is necessary
- BC agents pass informal "Melee Turing Test"

### From SmashBot
- Rules achieve frame-perfect execution
- But cannot adapt or generalize
- Good baseline for comparison

## Contributing

When adding new case studies:
1. Focus on technical details useful for implementation
2. Include architecture diagrams where possible
3. Document lessons learned and pitfalls
4. Reference source repositories and papers
5. Note relevance to ExPhil's goals
6. Add entry to [all-projects.md](all-projects.md)
