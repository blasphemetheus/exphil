# Complete Melee AI Project Index

This document catalogs all known Melee AI projects, tools, and research efforts. Projects are categorized by type and status.

## Primary AI Projects (Full Case Studies Available)

| Project | Author | Status | Case Study |
|---------|--------|--------|------------|
| [slippi-ai](https://github.com/vladfi1/slippi-ai) | vladfi1 | Active | [slippi-ai.md](slippi-ai.md) |
| [Phillip](https://github.com/vladfi1/phillip) | vladfi1 | Deprecated | [phillip.md](phillip.md) |
| [SmashBot](https://github.com/altf4/SmashBot) | altf4 | Active | [smashbot.md](smashbot.md) |
| [libmelee](https://github.com/altf4/libmelee) | altf4 | Active | [libmelee.md](libmelee.md) |
| [Eric Gu Transformer](https://ericyuegu.com/melee-pt1) | Eric Gu | Research | [eric-gu-transformer.md](eric-gu-transformer.md) |
| [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) | Bryan Chen | Research | [project-nabla.md](project-nabla.md) |

---

## Reinforcement Learning Projects

### Athena
**Repository**: [github.com/Sciguymjm/athena](https://github.com/Sciguymjm/athena)
**Author**: Sciguymjm
**Status**: Unknown
**Approach**: DQN (Deep Q-Network)
**Case Study**: [athena.md](athena.md)

DQN-based approach with convolutional state encoding and experience replay. Less documented than major projects but demonstrates core DQN concepts applied to Melee.

---

### Super-Smash-Bros-Melee-AI (A3C)
**Repository**: [github.com/KeganMc/Super-Smash-Bros-Melee-AI](https://github.com/KeganMc/Super-Smash-Bros-Melee-AI)
**Author**: KeganMc
**Status**: Inactive
**Approach**: Asynchronous Advantage Actor-Critic (A3C)
**Case Study**: [a3c-melee.md](a3c-melee.md)

Deep learning AI implementation using the A3C algorithm. Runs on Dolphin emulator. Includes code for a model-sharing website for different character matchups.

**Key features**:
- A3C architecture with TensorFlow 1.x
- Dolphin emulator integration via libmelee
- Model sharing infrastructure for matchup-specific weights

---

### Machine-Learning-Melee (Q-Learning)
**Repository**: [github.com/kape142/Machine-learning-melee](https://github.com/kape142/Machine-learning-melee)
**Author**: kape142
**Status**: Inactive
**Approach**: Tabular Q-learning
**Case Study**: [q-learning-melee.md](q-learning-melee.md)

An attempt at making an AI for SSBM using tabular Q-learning. Educational project demonstrating basic RL concepts with good documentation of challenges (state space explosion, reward shaping).

---

### ssbm-bot (Imitation Learning)
**Repository**: [github.com/effdotsh/ssbm-bot](https://github.com/effdotsh/ssbm-bot)
**Author**: effdotsh
**Status**: Unknown
**Approach**: Imitation learning

AI that learns via imitation learning from replay data.

---

## Gym Environments

### ssbm_gym
**Repository**: [github.com/Gurvan/ssbm_gym](https://github.com/Gurvan/ssbm_gym)
**Author**: Gurvan Priem
**Status**: MIT Licensed
**Type**: OpenAI Gym-compatible environment
**Case Study**: [ssbm-gym.md](ssbm-gym.md)

Gymnasium-compatible environment for SSBM RL training. Interfaces with Dolphin emulator for state observation and action execution. 117-dimensional continuous observation space with flexible action spaces.

```python
import ssbm_gym
env = ssbm_gym.make('SSBM-v0')
```

---

## Academic Projects

### Slippify (Stanford CS231N 2025)
**Paper**: [cs231n.stanford.edu/2025/papers/...](https://cs231n.stanford.edu/2025/papers/text_file_840589945-Parsing_Super_Smash_Bros__Melee_Frames.pdf)
**Author**: William Hu (Stanford)
**Type**: Computer vision + RL

Stanford CS231N project on parsing Melee frames. Explores visual-based policies using CNNs and direct RL (PPO). Highlights challenges in sample efficiency for visual learning.

---

### Perceptron Q-Learning (Stanford AA228)
**Paper**: [web.stanford.edu/class/aa228/reports/2018/final112.pdf](https://web.stanford.edu/class/aa228/reports/2018/final112.pdf)
**Class**: AA228 Decision Making under Uncertainty
**Year**: 2018

Stanford project applying Q-learning to SSBM. Educational focus on RL fundamentals.

---

### Learning with Delayed Actions
**Paper**: [yash-sharma.com/files/learning-play-super (3).pdf](https://www.yash-sharma.com/files/learning-play-super%20(3).pdf)
**Author**: Yash Sharma
**Focus**: Frame delay handling

Research on training agents with realistic human reaction times. Proposes DRQN (Deep Recurrent Q-Network) to handle delayed observations. Key finding: recurrent architectures help with delay.

---

## Video Analysis & Computer Vision

### SmashScan
**Repository**: [github.com/jpnaterer/smashscan](https://github.com/jpnaterer/smashscan)
**Author**: jpnaterer
**Blog**: [Medium series](https://blog.goodaudience.com/smashscan-using-neural-networks-to-analyze-super-smash-bros-melee-a7d0ab5c0755)
**Status**: Inactive
**Case Study**: [smashscan.md](smashscan.md)

Neural network video analysis for SSBM footage. Uses object detection to:
- Detect when Melee is on screen
- Identify characters and stages
- Extract match metadata

**Technical details**:
- DarkFlow (YOLO variant)
- Trained on 300+ annotated tournament matches
- 10GB+ of annotation data
- GTX 970 training

**Use case**: Automated video indexing for SmashVods database.

---

### ssbmachine-learning
**Repository**: [github.com/ZackMagnotti/ssbmachine-learning](https://github.com/ZackMagnotti/ssbmachine-learning)
**Author**: ZackMagnotti
**Status**: Unknown

Machine learning experiments on SSBM data.

---

## Replay Parsing & Data Tools

### Peppi (Rust Parser)
**Repository**: [github.com/hohav/peppi](https://github.com/hohav/peppi)
**Crates.io**: [crates.io/crates/peppi](https://crates.io/crates/peppi)
**Author**: hohav
**Status**: Active (v2.1.0)
**Case Study**: [peppi-ecosystem.md](peppi-ecosystem.md)

High-performance Rust parser for .slp (Slippi) replay files.

**Key features**:
- Fastest .slp parser available
- Round-trip capability (parse → write identical file)
- Alternative .slpp format (Arrow-based, compressible)
- Cross-language bindings (Python, Julia)

```rust
use peppi::game;
let game = game::read(&mut file)?;
```

---

### Peppi-py (Python Bindings)
**Repository**: [github.com/hohav/peppi-py](https://github.com/hohav/peppi-py)
**Author**: hohav
**Status**: Active

Python bindings for Peppi using PyO3 and Apache Arrow.

```python
from peppi_py import read_slippi
game = read_slippi("match.slp")
# Frame data as Arrow arrays
print(game.frames.port[0].leader.post.position.x[0])
```

---

### Peppi-slp (CLI Tool)
**Repository**: [github.com/hohav/peppi-slp](https://github.com/hohav/peppi-slp)
**Author**: hohav
**Status**: Active

Command-line inspector and converter for Slippi replays.

```bash
slp info match.slp
slp convert match.slp match.slpp
```

---

### Peppi.jl (Julia Bindings)
**Repository**: [github.com/jph6366/peppi-jl](https://github.com/jph6366/peppi-jl)
**Author**: jph6366
**Status**: Early prototype
**License**: MIT

Julia language bindings for the Peppi Slippi replay parser. Enables parsing `.slp` files within Julia applications for ML/data science workflows.

**Key features**:
- Apache Arrow columnar storage for frame data (efficient for analysis)
- Full access to game metadata, start/stop info, and frame-by-frame state
- Uses jlrs (Rust-Julia FFI) under the hood

```julia
using Pkg
Pkg.add(url="https://github.com/jph6366/peppi-jl")

using Peppi
game = Peppi.read("match.slp")
# Columnar access pattern
x_pos = game.frames.ports[1].leader.pre.position.x[1]
```

**Note**: Early prototype - API may change significantly. Interesting for Julia ML ecosystem integration (Flux.jl, MLJ.jl).

---

## Statistics & Analytics

### Lucky Stats
**Website**: [luckystats.gg](https://luckystats.gg/)
**Type**: Web platform
**Status**: Active

Advanced Melee analytics platform with:
- Glicko-2 ratings for 109,029 players
- 1,865,427 tracked sets
- Win probability calculator
- Matchup analysis
- Tournament statistics

---

### slippi-stats
**Repository**: [github.com/vinceau/slippi-stats](https://github.com/vinceau/slippi-stats)
**Website**: [vince.id.au/slippi-stats](https://vince.id.au/slippi-stats/)
**Author**: vinceau
**Status**: Active
**Case Study**: [slippi-stats.md](slippi-stats.md)

Browser-based Slippi stats computation using React + slippi-js. Runs locally with no server. Drag-and-drop .slp files for instant analysis with OBS integration.

---

### Slippipedia
**Repository**: [github.com/cbartsch/Slippipedia](https://github.com/cbartsch/Slippipedia)
**Author**: cbartsch
**Status**: Active
**Framework**: QML/Felgo (Qt-based)
**Case Study**: [slippipedia.md](slippipedia.md)

Desktop replay manager with persistent database caching. Batch processing of thousands of replays with rich filtering, video export via FFmpeg, and Slippi Desktop integration.

---

### slip.py
**Repository**: [github.com/pcrain/slip.py](https://github.com/pcrain/slip.py)
**Author**: pcrain
**Status**: Active
**License**: GPL-3.0
**Case Study**: [slip-py.md](slip-py.md)

Flask-based Slippi replay browser, search engine, and analyzer. Features indexed database, visual thumbnails, ledgedash precision calculations, and punish breakdowns. Uses slippc (C++) for high-performance parsing.

---

### slippi-set-stats
**Repository**: [github.com/project-slippi/slippi-set-stats](https://github.com/project-slippi/slippi-set-stats)
**Author**: Project Slippi
**Status**: Official tool

Official post-set stats generator for tournament broadcasts (used at Mainstage 2019).

---

### slippi-stats-database
**Repository**: [github.com/kyle-swygert/slippi-stats-database](https://github.com/kyle-swygert/slippi-stats-database)
**Author**: kyle-swygert
**Status**: Unknown

Database for storing replay information and computing stats.

---

### slippi-stats-display
**Repository**: [github.com/Jshauk/slippi-stats-display](https://github.com/Jshauk/slippi-stats-display)
**Author**: Jshauk
**Status**: Unknown

BTS-style stats screen generator using Melee Stream Tool.

---

### Slippi_Stats (Python)
**Repository**: [github.com/xaerincl/Slippi_Stats](https://github.com/xaerincl/Slippi_Stats)
**Author**: xaerincl
**Status**: Unknown

Python script for basic Slippi replay statistics. Requires Python >= 3.7.

---

## Prediction & Match Analysis

### Fall 2024 Melee Prediction
**Repository**: [github.com/jjaw89/fall-2024-smash-melee-prediction](https://github.com/jjaw89/fall-2024-smash-melee-prediction)
**Authors**: Dan Ursu, Jaspar Wiart
**Status**: Active (2024)
**Context**: Erdos Institute Data Science Boot Camp
**Case Study**: [melee-prediction.md](melee-prediction.md)

XGBoost-based prediction of match and tournament outcomes. Key innovation: character-adjusted ELO variants. Achieves 79.89% single-set accuracy and 70.1% top-8 winner prediction.

**Data**: ThePlayerDatabase (96K players, 1.8M sets from 2015-2024).

---

### Making Sense of Melee
**Blog**: [planetbanatt.net/articles/ambistats.html](https://planetbanatt.net/articles/ambistats.html)
**Author**: Ambi
**Focus**: ELO/prediction analysis

Analysis of 22,000+ tournament sets from 2017. Discusses:
- Limitations of ELO for double elimination
- Closed pool rating inflation
- ML prediction approaches

---

### SVM on SSBM
**Repository**: [github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee](https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee)
**Author**: trevorfirl
**Status**: Complete (Senior Project, July 2021)
**Institution**: Winona State University
**Case Study**: [svm-melee.md](svm-melee.md)

SVM character classification from 13 features per game (IPM, neutral win ratio, tech counts, etc.). Demonstrates that different characters have detectable statistical signatures. Uses slippi-js for replay processing.

---

## Infrastructure & Ecosystem

### Project Slippi
**Repository**: [github.com/project-slippi/project-slippi](https://github.com/project-slippi/project-slippi)
**Website**: [slippi.gg](https://slippi.gg/)
**Author**: Fizzi
**Status**: Active
**Case Study**: [project-slippi.md](project-slippi.md)

The foundational infrastructure for modern Melee AI development:
- Rollback netcode for online play
- .slp replay format (UBJSON-based frame data)
- Desktop launcher with spectator mode
- Ranked matchmaking
- Gecko code system for Dolphin mods

**AI Contributions**:
- Fast-forward gecko codes (enables faster-than-realtime RL)
- Anonymized replay collections (training data)
- slippi-js parsing library

---

### Dolphin Emulator (Slippi Fork)
**Repository**: [github.com/project-slippi/Ishiiruka](https://github.com/project-slippi/Ishiiruka)
**Status**: Active

Modified Dolphin with Slippi integration. Required for all Melee AI projects.

---

### slippi-ssbm-asm (Gecko Codes)
**Repository**: [github.com/project-slippi/slippi-ssbm-asm](https://github.com/project-slippi/slippi-ssbm-asm)
**Status**: Active

Assembly code for Slippi features. SmashBot requires specific gecko codes from here.

---

## Frame Data Resources

### FightCore
**Website**: [fightcore.gg](https://www.fightcore.gg/)
**Status**: Active

Interactive frame data with hitbox visualizations, crouch cancel calculator, and detailed move analysis.

---

### MeleeFrameData
**Website**: [meleeframedata.com](https://meleeframedata.com/)
**Status**: Active

Frame-by-frame seeker tool for analyzing move animations.

---

### IKneeData
**Website**: [ikneedata.com](https://ikneedata.com/)
**Status**: Active

Comprehensive frame data database for all characters.

---

## Historical / Deprecated

### Phillip (Original)
**Repository**: [github.com/vladfi1/phillip](https://github.com/vladfi1/phillip)
**Status**: Deprecated

Original deep RL Melee bot. Superseded by slippi-ai.

---

### st-tse Fork
**Repository**: [github.com/st-tse/Super-Smash-Bros-Melee-AI](https://github.com/st-tse/Super-Smash-Bros-Melee-AI)
**Status**: Fork

Fork of KeganMc's A3C implementation.

---

## Quick Reference by Category

### For Training AI
- **Primary**: [slippi-ai](https://github.com/vladfi1/slippi-ai), [libmelee](https://github.com/altf4/libmelee)
- **Environment**: [ssbm_gym](https://github.com/Gurvan/ssbm_gym)
- **Parsing**: [peppi](https://github.com/hohav/peppi), [peppi-py](https://github.com/hohav/peppi-py), [peppi-jl](https://github.com/jph6366/peppi-jl) (Julia)

### For Research/Learning
- **Papers**: [Phillip paper](https://arxiv.org/abs/1702.06230), [Delayed Actions](https://www.yash-sharma.com/files/learning-play-super%20(3).pdf)
- **Blogs**: [Eric Gu](https://ericyuegu.com/melee-pt1), [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
- **Academic**: Stanford CS231N, AA228 projects

### For Statistics
- **Platform**: [Lucky Stats](https://luckystats.gg/)
- **Tools**: [slippi-stats](https://github.com/vinceau/slippi-stats), [Slippipedia](https://github.com/cbartsch/Slippipedia)
- **Frame data**: [FightCore](https://www.fightcore.gg/), [IKneeData](https://ikneedata.com/)

### For Video Analysis
- **Project**: [SmashScan](https://github.com/jpnaterer/smashscan)

---

## Contributing

Know of a project not listed here? The Melee AI community is active and new projects emerge regularly. Key places to find new work:

- [GitHub ssbm topic](https://github.com/topics/ssbm)
- [GitHub melee topic](https://github.com/topics/melee)
- [Slippi Discord](https://discord.gg/slippi)
- [r/SSBM](https://reddit.com/r/ssbm)
