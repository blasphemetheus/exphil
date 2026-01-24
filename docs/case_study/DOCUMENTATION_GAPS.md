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

### [x] Matchup-Specific Training
**Why**: Different matchups require different strategies
**Topics covered**:
- Single model vs per-matchup models (5 strategies compared)
- Transfer learning between matchups (base + fine-tuning)
- Matchup-aware embeddings (opponent embedding, MoE routing)
- How pros handle matchup differences (Zain, Cody, aMSa examples)
- Opponent modeling techniques (DQN encoding, TCN, online adaptation)
- Multi-task learning with matchup heads
- Data availability by matchup (Fox-Fox 30K vs Link-G&W <100)
- FightingICE competition insights (RHEA + opponent model)

**Output**: `matchup-training.md` ✓

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

### [x] Conference Talks & Presentations
**Why**: Academic and industry perspectives
**Topics covered**:
- DEF CON 24 SmashBot talk (altf4, 2016)
- Phillip arXiv paper presentation (2017)
- AlphaStar DeepMind demos + MIT/BU talks
- OpenAI Five Dota 2 paper and demos
- FightingICE IEEE CoG competition (2013-present)
- GDC rollback netcode talks (NRS, GGPO/Tony Cannon)
- Academic lectures (UC Berkeley, MIT, CMU, Waterloo)
- Community content (Samox documentaries, vladfi1 streams)
- Technical blog posts (Eric Gu, Project Nabla, Fizzi)

**Output**: `conference-talks.md` ✓

---

### [x] Community Research Channels
**Why**: Stay connected to ongoing work
**Topics covered**:
- Slippi Discord (#ai-ml channel, 40K+ members, replay data access)
- slippi-ai Discord (development support)
- Reddit r/SSBM (discussion threads)
- Twitter/X researchers (Fizzi, vladfi1, altf4, Eric Gu)
- Twitch/YouTube (x_pilot streams, exhibition matches)
- GitHub organizations (project-slippi, key repos)
- Smashboards legacy threads
- Frame data sites (ikneedata, meleeframedata)
- Melee Library (tech guides)
- Key people to follow with handles
- Getting involved guide

**Output**: `community-channels.md` ✓

---

### [x] Historical Context
**Why**: Understanding the evolution of Melee AI
**Topics covered**:
- Timeline of major milestones (2015-2025)
- SmashBot/libmelee foundation (2015-2016)
- Phillip pure RL breakthrough + Genesis 4 demo (2017)
- Slippi launch and infrastructure revolution (2018-2019)
- COVID pandemic catalyst + rollback netcode (2020)
- BC+RL hybrid approaches + Project Nabla (2021-2022)
- Eric Gu Transformer + Ranked mode (2023-2024)
- Top player exhibitions (Moky 10-3, Zain 5-3)
- Community reception evolution (skepticism → acceptance)
- Future directions and open questions

**Output**: `history-timeline.md` ✓

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
| `fall-2024-prediction.md` | 2026-01-23 | XGBoost/Glicko-2 prediction (79.89% accuracy) |
| `svm-melee.md` | 2026-01-23 | SVM classification |
| `reward-shaping.md` | 2026-01-23 | PBRS theory, character-specific rewards, hacking |
| `dataset-curation.md` | 2026-01-23 | Data sources, filtering, augmentation, storage |
| `stages.md` | 2026-01-23 | Dynamic elements, blast zones, asymmetries, embedding |
| `slippi-replay-ecosystem.md` | 2026-01-23 | .slp format, parsers, data sources, pipelines |
| `matchup-training.md` | 2026-01-23 | Single vs multi-model, opponent modeling, transfer |
| `history-timeline.md` | 2026-01-23 | 2015-2025 evolution, milestones, community reception |
| `conference-talks.md` | 2026-01-23 | DEF CON, GDC, NeurIPS, IEEE CoG, community content |
| `community-channels.md` | 2026-01-23 | Discord, GitHub, Twitter, streams, frame data sites |

---

## How to Contribute

When completing a gap:
1. Mark as `[~]` when starting research
2. Create the output document
3. Mark as `[x]` when complete
4. Add to "Completed Documentation" table
5. Update `README.md` index if new file created
