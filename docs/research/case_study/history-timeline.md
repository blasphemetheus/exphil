# Historical Timeline of Melee AI

This document chronicles the evolution of Super Smash Bros. Melee AI from early experiments to modern competitive agents, including key milestones, community reactions, and technical breakthroughs.

## Overview

Melee AI development spans nearly a decade, progressing from rule-based systems to deep RL agents capable of competing with top professionals. The journey parallels broader advances in game AI while being uniquely shaped by Melee's dedicated competitive community.

```
2015-2016: Foundation Era (SmashBot, libmelee)
    ↓
2017: Pure RL Breakthrough (Phillip, Genesis 4)
    ↓
2018-2019: Infrastructure Revolution (Slippi launch)
    ↓
2020: Pandemic Catalyst (Rollback netcode, mass adoption)
    ↓
2021-2022: Hybrid Approaches (BC+RL, Project Nabla)
    ↓
2023-2024: Scaling Era (Transformers, ranked mode)
    ↓
2025: Competitive Parity (Top player exhibitions)
```

---

## Pre-2015: The Dark Ages

### Game AI Landscape

Before dedicated Melee AI efforts:
- **StarCraft Brood War** (1998): Active AI competition scene
- **Chess engines**: Deep Blue (1997), Stockfish (2008)
- **Atari DQN** (2013): DeepMind's breakthrough

### Melee Modding Scene

- **20XX Hack Pack** development begins
- Community creates enhanced training tools
- Frame data documentation efforts
- No serious AI attempts

**Challenge**: No programmatic interface to Dolphin emulator.

---

## 2015-2016: Foundation Era

### December 2015 - SmashBot Begins

**Project**: [SmashBot](https://github.com/altf4/SmashBot)
**Creator**: Dan "altf4"

Dan begins work on the first serious Melee AI:
- Reverse engineers Dolphin memory structures
- Creates named pipe input mechanism
- Develops hierarchical decision system (Strategy → Tactics → Chains)

**Initial limitations**:
- Fox only
- Final Destination only
- vs Marth only
- Rule-based, not learning

### August 2016 - DEF CON 24

SmashBot presented at DEF CON 24 security conference.

**Talk**: "Game Over, Man! Reversing Video Games to Create an Unbeatable AI"

Key contributions:
- Documented Dolphin memory layout
- Open-sourced input injection method
- Inspired future AI developers

> "The road to creating SmashBot was challenging since he was charting new territory in the Smash world." - [Bishop Fox](https://bishopfox.com/blog/reversing-video-games-to-create-smashbot-ai)

### 2016 - libmelee Development

altf4 extracts SmashBot's interface code into standalone library:
- Python API for game state access
- Controller input via Dolphin pipes
- Foundation for all future Melee AI

---

## 2017: The Pure RL Breakthrough

### January 2017 - Phillip Training Begins

**Project**: [Phillip](https://github.com/vladfi1/phillip)
**Creators**: Vlad Firoiu (MIT), William F. Whitney (NYU), Joshua Tenenbaum (MIT)

Vlad Firoiu begins training pure RL agent:
- Actor-Critic architecture (A3C)
- Self-play on MIT supercomputer (MGHPCC)
- No human demonstrations

> "I just sort of forgot about it for a week." - Vlad Firoiu

Training infrastructure:
- CUDA / Tesla K20 / TITAN X GPUs
- TensorFlow framework
- Distributed across compute cluster

### February 2017 - Genesis 4 Demonstration

**Event**: Genesis 4 (Oakland, CA)
**Date**: January 20-22, 2017

Phillip demonstrated against professional players:

| Player | Result | Quote |
|--------|--------|-------|
| Gravy | 5-8 (loss) | "The AI is definitely godlike. I am not sure if anyone could beat it." |
| 10 pros total | All lost more stocks than AI | - |

**Key observations**:
- 33ms reaction time (vs human ~200ms)
- Dash dancing behavior emerged from self-play
- Alien-looking but effective strategies

### February 21, 2017 - Academic Paper

**Paper**: ["Beating the World's Best at Super Smash Bros. with Deep Reinforcement Learning"](https://arxiv.org/abs/1702.06230)

arXiv:1702.06230

Authors: Vlad Firoiu, William F. Whitney, Joshua B. Tenenbaum

**Contributions**:
- First peer-reviewed Melee AI paper
- Demonstrated superhuman performance
- Addressed multi-agent RL challenges

**Media coverage**:
- [Quartz](https://qz.com/917221/a-super-smash-bros-playing-ai-has-taught-itself-how-to-stomp-professional-players)
- [Slashdot](https://games.slashdot.org/story/17/02/25/1932231/machine-learning-ai-now-beats-humans-at-super-smash-bros-melee)
- [NVIDIA Blog](https://developer.nvidia.com/blog/self-taught-ai-bot-beat-professional-players-at-super-smash-bros/)

### Community Reaction (2017)

Mixed reception:

**Positive**:
- Excitement about AI capabilities
- Interest in training tool potential
- Academic credibility for Melee

**Skeptical**:
- "33ms reaction time is cheating"
- "Not playing like a human"
- "Can't use for practice - patterns too exploitable"

> "Training against bots (especially lvl 9s) offers little reward. Real players move a hell of a lot faster." - Smashboards user

---

## 2018: Infrastructure Revolution

### February 2018 - Slippi Private Beta

**Project**: [Project Slippi](https://github.com/project-slippi)
**Creator**: Jas "Fizzi" Laferriere

Slippi enters private beta testing:
- Automatic replay recording (.slp files)
- Post-game statistics
- Modified Dolphin emulator

### June 18, 2018 - Slippi Public Launch

Slippi publicly released with:
- Replay recording to .slp format
- Match statistics display
- Integration with tournament setups

**Impact on AI research**:
- Massive replay data now accessible
- Standardized game state format
- Enables imitation learning approaches

### Fall 2018 - SmashLadder Integration

SmashLadder adopts Slippi as primary netplay build:
- Community standardization
- Increased replay generation
- Growing dataset for AI training

---

## 2019: Building Momentum

### April 2019 - Slippi Web Visualizer

In-browser replay visualization added:
- Share matches easily
- Analyze gameplay
- Designed by Will Blackett

### Pound 2019

Major tournament uses Slippi with positive reception:
- Real-time stats display
- Professional broadcast integration
- Community enthusiasm grows

### November 2019 - Fizzi Goes Full-Time

Fizzi leaves smashgg to work on Slippi full-time:
- Patreon-funded development
- Focus on rollback netcode
- 7 months of dedicated work begins

**Community support**: Patreon and donations fund development.

---

## 2020: The Pandemic Catalyst

### March 2020 - COVID-19 Shutdown

In-person tournaments cease globally:
- Delay-based netplay inadequate
- Community in crisis mode
- Urgent need for better online play

### June 22, 2020 - Rollback Netcode Release

**Slippi 2.0** released with:
- **Rollback netcode** (fighting game standard)
- **Integrated matchmaking**
- **Auto-updates**
- **Automatic replays**

> "A team of 4 random dudes did a better job with netcode than literally any actual professionally released fighting game."

**Community reaction**:

| Player | Comment |
|--------|---------|
| Leffen | 20-minute video explaining benefits, footage from Sweden-to-US match |
| Axe | "Using Slippi felt like playing offline" |
| General | Widespread adoption within weeks |

### Netplay Boom

Slippi user statistics explode:
- Thousands of concurrent players
- New players enter scene
- International competition viable

### November 2020 - The Big House Online Controversy

**Event**: The Big House Online scheduled
**Issue**: Nintendo C&D letter over Slippi usage

> "While Nintendo has yet to actually prove Project Slippi infringes their copyright in a court of law, their action was enough to cancel the entire event."

**#FreeMelee movement** emerges:
- Community backlash against Nintendo
- Increased awareness of Slippi
- Legal uncertainty persists

### AI Implications (2020)

The rollback release transforms AI research:
- **Data explosion**: Orders of magnitude more replays
- **Online play viable**: AI can compete in real conditions
- **18-frame delay training**: Standard for netplay AI

Fizzi's fast-forward gecko codes enable:
- 100x+ training speed
- Practical RL iteration
- Essential for self-play

---

## 2021-2022: Hybrid Approaches

### January 2021 - Hidden MMR

Slippi v2.2.4 introduces:
- Hidden matchmaking rating (MMR)
- Better skill-based matching
- Data quality improvement (skill-stratified)

### 2021 - slippi-ai Development

Vlad Firoiu pivots from pure RL to hybrid approach:

**slippi-ai** (Phillip II):
- Stage 1: Behavioral cloning on Slippi replays
- Stage 2: PPO refinement with self-play
- KL regularization to teacher policy

> "While the original Phillip used pure deep RL, this one starts with behavioral cloning on slippi replays, which makes it play a lot more like a human."

### August 2022 - Project Nabla

**Project**: [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
**Creator**: Bryan Chen (bycn)

Key findings:
- BC agents pass informal "Melee Turing Test"
- Single-opponent self-play causes RPS cycling
- Population-based training necessary

Technical approach:
- ~100K tournament games for BC
- PPO fine-tuning
- Consumer GPU training (days, not months)

**Capabilities demonstrated**:
- Wavedashing, waveshining, L-canceling
- Shield drops, short combos
- Human-like movement patterns

### December 2022 - Ranked Early Access

Fizzi announces Slippi Online Ranked:
- Subscriber-only early access
- Public rating system
- Regional leaderboards (NA, EU, Other)

---

## 2023: The Scaling Era

### March 2023 - Ranked Goes Public

Slippi Ranked becomes free for all:
- Mass adoption
- Competitive ladder culture
- High-quality ranked data

**Leaderboard structure**:
- North America
- Europe
- Other (SA, Oceania, Asia)
- Updates every 2 minutes

### Ongoing slippi-ai Development

Vlad continues improving agents:
- Matchup-specific training
- Multiple character support
- Regular checkpoint releases

### Data Abundance

By 2023:
- 100K+ anonymized ranked replays available
- Skill-stratified (Bronze → Grandmaster)
- Regular data drops from Fizzi

---

## 2024: Transformers and Scale

### Early 2024 - Eric Gu's Transformer

**Project**: [Melee Transformer](https://ericyuegu.com/melee-pt1)
**Creator**: Eric Gu

Revolutionary approach:
- **20M parameter** decoder-only Transformer
- **GPT-style** next-token prediction
- **3 billion frames** of training data
- **$5 training cost** (5 hours, 2× RTX 3090)
- **95% win rate** vs Level 9 CPU

Key insight:
> "I strongly believe a single Transformer trained on all character replays would out-perform and be more efficient to train."

Finding: All-character training > single-character training

### November 2024 - Strongest Agents

slippi-ai releases new agent versions:
- `fox_d18_ditto_v3` (2024-11-18) - Strongest known
- `marth_d18_ditto_v3`
- `fox_d18_vs_falco_v3`
- `marth_d18_vs_falco_v3`
- `falco_d18_ditto_v3` (2025-04-07)

### Late 2024 - Top Player Exhibitions

Exhibition matches gain attention:

| Match | Format | Result | Notes |
|-------|--------|--------|-------|
| Moky vs Phillip | FT10 Fox ditto | **10-3 Phillip** | AI dominance |
| Zain vs Phillip | FT5 | **5-3 Zain** | Human wins |
| aMSa streams | Ongoing | Various | Community engagement |

**YouTube**: [Moky vs Phillip](https://www.youtube.com/watch?v=1kviVflqXc4)

### Community Reception (2024)

Shift from skepticism to acceptance:
- Recognition of human-like play
- Interest in AI as training partner
- Manifold prediction markets on AI vs humans

---

## 2025: Competitive Parity

### Current State

**slippi-ai**:
- Active development by vladfi1
- Regular Twitch streams (x_pilot)
- Competitive with top 50 players
- Matchup-specific specialists

**Community tools**:
- Slippi Ranked mature
- Peppi (fast Rust parser)
- Multiple analysis tools

**Research directions**:
- Population-based training
- Real-time inference optimization
- Multi-character generalization

### Top Player Engagement

Pro players regularly interact with AI:
- aMSa streaming matches
- Training tool discussions
- Tournament appearance speculation

### Open Questions

| Question | Status |
|----------|--------|
| Can AI win a major? | Unresolved |
| Will Nintendo intervene? | Ongoing tension |
| Can AI be legitimate training? | Growing acceptance |
| What's the skill ceiling? | Still climbing |

---

## Key Lessons Across Eras

### Technical Evolution

| Era | Approach | Limitation Overcome |
|-----|----------|---------------------|
| 2016 | Rule-based | No adaptation |
| 2017 | Pure RL | Sample inefficiency |
| 2020 | BC only | Bounded by data |
| 2021+ | BC+RL | Best of both worlds |
| 2024+ | Transformers | Scale and generalization |

### Infrastructure Dependency

AI progress directly tied to infrastructure:

```
SmashBot (2016) → libmelee → Enabled all future projects
Slippi (2018) → .slp format → Enabled imitation learning
Rollback (2020) → Mass data → Enabled scaling
Ranked (2023) → Skill labels → Enabled quality filtering
```

### Community-Research Symbiosis

Melee AI uniquely benefits from:
- Passionate player community
- Open-source infrastructure
- Academic interest
- Patreon funding model

---

## Timeline Summary

| Year | Milestone | Significance |
|------|-----------|--------------|
| **2015** | SmashBot begins | First serious attempt |
| **2016** | DEF CON talk | Public awareness |
| **2017** | Phillip Genesis 4 | Pure RL beats pros |
| **2017** | arXiv paper | Academic recognition |
| **2018** | Slippi launch | Replay infrastructure |
| **2019** | Fizzi full-time | Dedicated development |
| **2020** | Rollback release | Pandemic adoption boom |
| **2020** | Big House C&D | Nintendo tension |
| **2021** | slippi-ai BC+RL | Hybrid approach |
| **2022** | Project Nabla | Population training insights |
| **2022** | Ranked announced | Competitive ladder |
| **2023** | Ranked public | Mass participation |
| **2024** | Eric Gu Transformer | Scaling laws apply |
| **2024** | Top player exhibitions | Near-competitive parity |
| **2025** | Ongoing development | Active research |

---

## Future Directions

### Near-Term (2025-2026)

- Real-time inference on consumer hardware
- Tournament participation (if allowed)
- Training tool integration
- More character coverage

### Medium-Term (2026-2028)

- Population-based competitive training
- Multi-game transfer learning
- Automated coaching tools
- Style-specific agents

### Long-Term Speculation

- Superhuman play indistinguishable from human
- AI-human collaboration (augmented play)
- Preservation of Melee through AI

---

## References

### Primary Sources
- [Phillip Paper](https://arxiv.org/abs/1702.06230) - Firoiu et al., 2017
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html) - Bryan Chen, 2022
- [Eric Gu Blog](https://ericyuegu.com/melee-pt1) - Eric Gu, 2024
- [Slippi About](https://slippi.gg/about) - Official

### Repositories
- [SmashBot](https://github.com/altf4/SmashBot) - altf4
- [libmelee](https://github.com/altf4/libmelee) - altf4
- [Phillip](https://github.com/vladfi1/phillip) - vladfi1
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - vladfi1
- [Project Slippi](https://github.com/project-slippi) - Fizzi

### Media Coverage
- [Quartz (2017)](https://qz.com/917221/a-super-smash-bros-playing-ai-has-taught-itself-how-to-stomp-professional-players)
- [NVIDIA Blog (2017)](https://developer.nvidia.com/blog/self-taught-ai-bot-beat-professional-players-at-super-smash-bros/)
- [SmashWiki - Project Slippi](https://www.ssbwiki.com/Project_Slippi)

### Community
- [Slippi Discord](https://discord.gg/slippi)
- [Smashboards](https://smashboards.com/)
- [r/SSBM](https://reddit.com/r/SSBM)
