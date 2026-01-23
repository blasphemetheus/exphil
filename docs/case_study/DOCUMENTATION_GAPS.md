# Documentation Gaps & Research Roadmap

This document tracks areas that need additional research and documentation for the Melee AI case studies.

## Status Key
- [ ] Not started
- [~] In progress
- [x] Completed

---

## High Priority (Directly Relevant to ExPhil)

### [x] Other Fighting Game AI
**Why**: Transferable lessons from related domains
**Topics covered**:
- FightingICE (academic competition)
- AlphaStar (real-time strategy, action space)
- OpenAI Five (reaction time, team coordination)
- Street Fighter AI research (DDQN, PPO approaches)
- FightLadder benchmark

**Output**: `other-fighting-game-ai.md` ✓

---

### [x] Deployment & Inference
**Why**: Critical for real-time 60 FPS play
**Topics covered**:
- ONNX export and quantization (INT8, FP16)
- TensorRT integration
- Netplay latency compensation strategies
- GPU vs CPU trade-offs
- Zero-copy strategies
- Hardware requirements and benchmarks

**Output**: `deployment-inference.md` ✓

---

### [ ] Reward Shaping Deep Dive
**Why**: Character-specific rewards are key to low-tier success
**Topics to cover**:
- Standard reward functions (KO, damage, stocks)
- Character-specific shaping (Mewtwo recovery, Ganon spacing, G&W bucket)
- Intrinsic motivation / curiosity-driven exploration
- Reward hacking failure modes
- Potential-based reward shaping theory
- What slippi-ai and Phillip used

**Output**: `reward-shaping.md`

---

### [ ] Dataset Curation
**Why**: Training data quality directly impacts model quality
**Topics to cover**:
- Skill-level filtering strategies
- Character/matchup balance
- Data augmentation techniques
- Anonymized vs named player data
- Where to obtain replays (Fizzi's collections, tournament archives)
- Legal/ethical considerations
- Dataset statistics (size, distribution)

**Output**: `dataset-curation.md`

---

## Medium Priority

### [ ] Matchup-Specific Training
**Why**: Different matchups require different strategies
**Topics to cover**:
- Single model vs per-matchup models
- Transfer learning between matchups
- Matchup-aware embeddings
- How pros handle matchup differences

**Output**: Add section to `training-approaches.md` or create `matchup-training.md`

---

### [ ] Stage Considerations
**Why**: Stages significantly affect gameplay
**Topics to cover**:
- Platform stages vs Final Destination
- Dynamic stage elements (Randall, FoD platforms)
- Stage-specific embeddings
- Tournament-legal stage list
- Stage counterpicking implications

**Output**: Add section to `state-representation.md` or create `stages.md`

---

### [ ] Slippi Replay Ecosystem
**Why**: Understanding data sources
**Topics to cover**:
- Replay file format details
- Fizzi's anonymized ranked collections
- Tournament replay archives
- Community replay sharing
- Data pipeline best practices

**Output**: `replay-ecosystem.md`

---

## Lower Priority (But Interesting)

### [ ] Conference Talks & Presentations
**Why**: Academic and industry perspectives
**Topics to cover**:
- GDC game AI talks
- NeurIPS/ICML game-playing papers
- Melee community presentations
- YouTube technical deep-dives

**Output**: Add to `research-papers.md`

---

### [ ] Community Research Channels
**Why**: Stay connected to ongoing work
**Topics to cover**:
- Slippi Discord AI channels
- Reddit r/SSBM technical threads
- Twitter/X researchers to follow
- Private research groups

**Output**: Add to `community-projects.md`

---

### [ ] Historical Context
**Why**: Understanding the evolution of Melee AI
**Topics to cover**:
- Timeline of major milestones
- How the meta evolved with AI
- Community reception over time
- Future directions

**Output**: `history-timeline.md`

---

## Completed Documentation

| Document | Date | Notes |
|----------|------|-------|
| `slippi-ai.md` | 2025-01-23 | Core reference |
| `phillip.md` | 2025-01-23 | Historical |
| `libmelee.md` | 2025-01-23 | API reference |
| `smashbot.md` | 2025-01-23 | Rule-based baseline |
| `eric-gu-transformer.md` | 2025-01-23 | Transformer approach |
| `project-nabla.md` | 2025-01-23 | Self-play lessons |
| `melee-ai-landscape.md` | 2025-01-23 | Overview |
| `action-space.md` | 2025-01-23 | Technical |
| `state-representation.md` | 2025-01-23 | Technical |
| `training-approaches.md` | 2025-01-23 | Methodology |
| `research-papers.md` | 2025-01-23 | Academic |
| `characters.md` | 2025-01-23 | Character-specific |
| `community-projects.md` | 2025-01-23 | Additional projects |
| `all-projects.md` | 2025-01-23 | Complete index |
| `other-fighting-game-ai.md` | 2025-01-23 | AlphaStar, OpenAI Five, FightingICE |
| `deployment-inference.md` | 2025-01-23 | ONNX, quantization, real-time |
| `athena.md` | 2026-01-23 | DQN approach |
| `a3c-melee.md` | 2026-01-23 | A3C architecture |
| `ssbm-gym.md` | 2026-01-23 | OpenAI Gym environment |
| `peppi-ecosystem.md` | 2026-01-23 | Rust parser + bindings |
| `smashscan.md` | 2026-01-23 | YOLO video analysis |
| `project-slippi.md` | 2026-01-23 | Slippi infrastructure |
| `q-learning-melee.md` | 2026-01-23 | Tabular Q-learning |
| `slippi-stats.md` | 2026-01-23 | Browser stats tool |
| `slippipedia.md` | 2026-01-23 | QML replay manager |
| `slip-py.md` | 2026-01-23 | Flask replay browser |
| `melee-prediction.md` | 2026-01-23 | XGBoost prediction |
| `svm-melee.md` | 2026-01-23 | SVM classification |

---

## How to Contribute

When completing a gap:
1. Mark as `[~]` when starting research
2. Create the output document
3. Mark as `[x]` when complete
4. Add to "Completed Documentation" table
5. Update `README.md` index if new file created
