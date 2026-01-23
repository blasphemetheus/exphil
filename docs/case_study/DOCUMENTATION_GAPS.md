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

### [x] Reward Shaping Deep Dive
**Why**: Character-specific rewards are key to low-tier success
**Topics covered**:
- Standard reward functions (KO, damage, stocks)
- Character-specific shaping (Mewtwo recovery, Ganon spacing, G&W bucket, Link projectiles, IC Nana)
- Intrinsic motivation / curiosity-driven exploration (ICM, skill-based curiosity)
- Reward hacking failure modes (approach oscillation, damage farming, stalling, ledge camping)
- Potential-based reward shaping theory (PBRS guarantees, 2024 research)
- What slippi-ai and Phillip used
- Practical recommendations for ExPhil training phases

**Output**: `reward-shaping.md` ✓

---

### [x] Dataset Curation
**Why**: Training data quality directly impacts model quality
**Topics covered**:
- Skill-level filtering strategies (rank tiers, heuristics, curriculum)
- Character/matchup balance (filtering, upsampling, loss weighting)
- Data augmentation techniques (mirror, noise, frame delay)
- Anonymized vs named player data
- Where to obtain replays (Fizzi's collections, ThePlayerDatabase, tournament archives)
- Legal/ethical considerations
- Dataset statistics (size, distribution, quality indicators)
- Cloud storage pipeline (B2, R2, RunPod)

**Output**: `dataset-curation.md` ✓

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

### [x] Stage Considerations
**Why**: Stages significantly affect gameplay
**Topics covered**:
- Platform stages vs Final Destination
- Dynamic stage elements (Randall cycle timing, FoD platform heights, Whispy wind, Stadium transformations)
- Stage-specific embeddings with code examples
- Tournament-legal stage list with blast zone data
- Stage asymmetries (Battlefield ledges, coordinate systems)
- Frozen vs unfrozen Pokemon Stadium
- Shy Guys on Yoshi's Story
- Mirror augmentation considerations per stage
- Character-specific stage preferences for low-tier

**Output**: `stages.md` ✓

---

### [x] Slippi Replay Ecosystem
**Why**: Understanding data sources
**Topics covered**:
- .slp UBJSON format specification with event types (0x35-0x3F)
- Parser comparison table (peppi, peppi-py, slippi-js, py-slippi, slippc)
- .slpp Arrow format vs .zlp LZMA compressed format
- Data sources: Fizzi anonymized, tournament archives, SmashLadder
- Quality filtering: skill tiers, character balance, heuristics
- Community tools: Slippipedia, slip.py, SlippiLab
- Pipeline architecture from acquisition to training

**Output**: `slippi-replay-ecosystem.md` ✓

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
| `reward-shaping.md` | 2026-01-23 | PBRS theory, character-specific rewards, hacking |
| `dataset-curation.md` | 2026-01-23 | Data sources, filtering, augmentation, storage |
| `stages.md` | 2026-01-23 | Dynamic elements, blast zones, asymmetries, embedding |
| `slippi-replay-ecosystem.md` | 2026-01-23 | .slp format, parsers, data sources, pipelines |

---

## How to Contribute

When completing a gap:
1. Mark as `[~]` when starting research
2. Create the output document
3. Mark as `[x]` when complete
4. Add to "Completed Documentation" table
5. Update `README.md` index if new file created
