# Conference Talks & Presentations

This document catalogs relevant conference talks, academic presentations, technical lectures, and community content related to Melee AI and game-playing AI more broadly.

## Overview

Knowledge about game AI is spread across multiple venues:
- **Academic conferences**: NeurIPS, ICML, IEEE CoG
- **Industry conferences**: GDC (Game Developers Conference)
- **Security conferences**: DEF CON
- **Community content**: YouTube, Twitch, blogs

---

## Melee-Specific Presentations

### DEF CON 24: SmashBot (2016)

**Title**: "Game Over, Man! Reversing Video Games to Create an Unbeatable AI Player"
**Speaker**: Dan "altf4" Petro
**Date**: August 7, 2016
**Venue**: DEF CON 24, Las Vegas

**Links**:
- [Video (TIB AV-Portal)](https://av.tib.eu/media/36229)
- [Blog Post (Bishop Fox)](https://bishopfox.com/blog/reversing-video-games-to-create-smashbot-ai)
- [Hackaday Coverage](https://hackaday.com/2016/08/16/creating-unbeatable-videogame-ai/)

**Abstract**: First public presentation of a serious Melee AI. Covers reverse engineering Dolphin emulator memory, creating programmatic controller input, and building rule-based decision systems.

**Key Technical Content**:
- Dolphin memory structure reverse engineering
- Named pipe input mechanism
- Strategy → Tactics → Chains hierarchy
- Frame-perfect execution challenges

**Significance**: Foundation talk that enabled all subsequent Melee AI work by documenting how to interface with the game.

**Limitations Discussed**:
- Fox only, Final Destination only, vs Marth only
- Rule-based (no learning)
- Superhuman reaction time

---

### Phillip Paper Presentation (2017)

**Title**: "Beating the World's Best at Super Smash Bros. with Deep Reinforcement Learning"
**Authors**: Vlad Firoiu, William F. Whitney, Joshua B. Tenenbaum
**Venue**: arXiv preprint (not formally presented at conference)
**Date**: February 21, 2017

**Links**:
- [arXiv Paper](https://arxiv.org/abs/1702.06230)
- [MIT CSAIL Page](https://www.csail.mit.edu/research/beating-worlds-best-super-smash-bros-deep-reinforcement-learning)
- [NVIDIA Blog](https://developer.nvidia.com/blog/self-taught-ai-bot-beat-professional-players-at-super-smash-bros/)

**Key Contributions**:
- First peer-reviewed Melee AI paper
- A3C with self-play
- Demonstrated superhuman performance at Genesis 4
- Multi-agent RL challenges addressed

**Technical Details**:
- Actor-Critic architecture
- MIT supercomputer training (MGHPCC)
- Tesla K20/TITAN X GPUs, TensorFlow
- 33ms reaction time achieved

---

### Community Documentaries

#### The Smash Brothers (2013)

**Creator**: Travis "Samox" Beauchamp
**Format**: 9-part documentary series
**Budget**: $12,000 (~$7,000 community-funded)

**Links**:
- [Internet Archive](https://archive.org/details/the-smash-brothers-documentary)
- [SmashWiki](https://www.ssbwiki.com/The_Smash_Brothers)

**Significance**: Credited with reviving competitive Melee interest. While not AI-focused, provides essential context for understanding the community that produces training data.

#### Metagame (2020)

**Creator**: Travis "Samox" Beauchamp
**Format**: 8-part documentary series
**Premiere**: December 11-13, 2020 (Twitch)

**Links**:
- [Official Site](https://www.metagamedoc.com/)
- [SmashWiki](https://www.ssbwiki.com/Metagame_(documentary))

**Content**: Covers the "Five Gods" era (2008-2015). Provides context for understanding high-level play that AI attempts to replicate.

---

## Fighting Game AI (General)

### FightingICE Competition (IEEE CoG)

**Venue**: IEEE Conference on Games (CoG)
**Organizer**: Intelligent Computer Entertainment Lab, Ritsumeikan University
**Running Since**: 2013

**Links**:
- [Official Site](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/)
- [CoG 2024 Competitions](https://2024.ieee-cog.org/competitions/)
- [DareFightingICE Paper](https://arxiv.org/abs/2203.01556)

**Format**: Annual competition with multiple tracks:
1. **AI Track**: Build agents to play FightingICE
2. **Sound Design Track**: Design audio for visually impaired players (DareFightingICE)

**Technical Constraints**:
- Java-based platform (Python supported)
- Resources from The Rumble Fish 2 (Dimps Corporation)
- 64GB RAM + 32GB VRAM limits

**Notable Papers from Competition**:
- "Mastering Fighting Game Using Deep Reinforcement Learning With Self-play" (IEEE CoG 2020)
- "Enhanced Rolling Horizon Evolution Algorithm with Opponent Model Learning" (2019 runner-up)
- "Genetic state-grouping algorithm for deep reinforcement learning"

**Relevance to Melee AI**: Academic benchmark for fighting game AI techniques. Many approaches (MCTS + self-play, opponent modeling) transfer to Melee.

---

### GDC: Rollback Netcode (NetherRealm Studios)

**Title**: (Rollback Netcode Implementation)
**Speaker**: Michael Stallone, Lead Software Engineer
**Venue**: GDC
**Duration**: ~1 hour

**Links**:
- [GDC Vault](https://www.gdcvault.com/) (requires subscription)
- [Test Your Might Discussion](https://testyourmight.com/threads/technical-insight-of-ggpo-netcode-by-nrs.68431/)

**Key Content**:
- Adapting rollback for modern games (Mortal Kombat, Injustice 2)
- Performance optimization challenges
- 7-8 man-years of development effort
- Determinism requirements

**Relevance to Melee AI**: Understanding rollback is essential for:
- Training AI for online play (18+ frame delay)
- Building real-time inference systems
- Understanding Slippi's implementation

---

### EVO 2017: GGPO Talk

**Title**: GGPO and Rollback Netcode
**Speaker**: Tony Cannon (GGPO creator, EVO co-founder)
**Venue**: Evolution Championship Series 2017

**Content**:
- How GGPO works
- History and motivation
- Integration with FinalBurn Alpha emulator
- Technical content starts ~10:38

**Key Concepts**:
- Input prediction with "sticky inputs"
- Speculative execution
- Rollback on misprediction
- Why delay-based netcode fails for fighting games

**Relevance**: Slippi's rollback implementation follows GGPO principles. Understanding this is crucial for building AI that plays online.

---

### GDC: Reinforcement Learning in Production

**Title**: "ML Tutorial Day: Smart Bots for Better Games: Reinforcement Learning in Production"
**Venue**: GDC

**Links**:
- [GDC Vault](https://www.gdcvault.com/play/1026281/ML-Tutorial-Day-Smart-Bots)

**Content**:
- Overview of RL algorithms for game studios
- Training bots for automated testing
- Design assistance applications
- Challenges in Ubisoft games

**Relevance**: Practical considerations for deploying RL agents in production game environments.

---

### GDC: NetEase Games RL Applications

**Title**: "Applying Reinforcement Learning to Develop Game AI in Netease Games"
**Venue**: GDC 2022

**Links**:
- [GDC Vault](https://gdcvault.com/play/1026636/Applying-Reinforcement-Learning-to-Develop)

**Content**:
- RL applications in NetEase games
- Development problems and solutions
- Tools and process specifications
- Creating human-like AI behavior

---

## Landmark Game AI Presentations

### AlphaStar (DeepMind)

**Title**: "AlphaStar: Mastering the Real-Time Strategy Game StarCraft II"
**Speaker**: Oriol Vinyals (Senior Staff Research Scientist, DeepMind)
**Venues**: MIT, Boston University, BlizzCon 2018

**Links**:
- [DeepMind Blog (Initial)](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
- [DeepMind Blog (Grandmaster)](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [Blizzard VoD](https://news.blizzard.com/en-gb/article/22871520/watch-the-deepmind-starcraft-ii-demonstration-vod)
- [Wikipedia](https://en.wikipedia.org/wiki/AlphaStar_(software))

**Key Results**:
- First AI to reach Grandmaster in major esport
- Beat pros TLO (5-0) and MaNa (5-1)
- Achieved top 0.2% of Battle.net players
- All three races (Protoss, Terran, Zerg)

**Technical Details**:
- 16 TPUs per agent, 14-day training
- 200 years of real-time play experience
- League training with exploiters
- 217ms average reaction time (human-like)

**Relevance to Melee AI**:
- League/population-based training approach
- Reaction time constraints
- Multi-agent self-play stability
- Imitation learning + RL pipeline

---

### OpenAI Five (Dota 2)

**Title**: "Dota 2 with Large Scale Deep Reinforcement Learning"
**Organization**: OpenAI
**Key Events**: The International 2018, April 2019 OG match

**Links**:
- [OpenAI Blog](https://openai.com/index/openai-five/)
- [Technical Paper (PDF)](https://cdn.openai.com/dota-2.pdf)
- [Wikipedia](https://en.wikipedia.org/wiki/OpenAI_Five)
- [NVIDIA Blog](https://developer.nvidia.com/blog/ai-learns-to-play-dota-2-with-human-precision/)

**Key Results**:
- Beat world champions OG (April 2019)
- First AI to beat reigning esports champions
- 40-2 vs top professional teams

**Technical Details**:
- 256 NVIDIA Tesla P100 GPUs
- 128,000 CPU cores
- 180 years of gameplay per day
- 150 million parameters
- PPO (Proximal Policy Optimization)
- 217ms reaction time
- Acts every 4th frame (30 FPS engine)

**Limitations**:
- Only 17 heroes supported (vs 100+)
- Hand-scripted item/ability logic

**Relevance to Melee AI**:
- Scale of compute required for complex games
- Reaction time design choices
- Multi-agent coordination challenges

---

### NeurIPS Papers

#### Honor of Kings (2020)

**Title**: "Towards Playing Full MOBA Games with Deep Reinforcement Learning"
**Venue**: NeurIPS 2020

**Links**:
- [Paper (PDF)](https://papers.nips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf)
- [NeurIPS Review](https://proceedings.neurips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Review.html)

**Key Results**:
- 40 hero pool (vs OpenAI Five's 17)
- Beat top professionals 40-2
- 97.7% win rate in 642K matches vs top amateurs

**Relevance**: Demonstrates scaling beyond OpenAI Five's limitations.

#### ROA-Star (2023)

**Title**: "A Robust and Opponent-Aware League Training Method for StarCraft II"
**Venue**: NeurIPS 2023

**Links**:
- [Paper (PDF)](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)

**Content**: Improvements to AlphaStar's league training, better exploiter effectiveness.

---

## Academic Lectures & Courses

### UC Berkeley: Deep RL Course

**Course**: CS 285 (formerly CS 294)
**Topic**: "Deep Learning, Dynamical Systems, and Behavior Cloning"

**Links**:
- [Lecture Slides (PDF)](https://rll.berkeley.edu/deeprlcoursesp17/docs/week_2_lecture_1_behavior_cloning.pdf)

**Content**:
- Behavioral cloning fundamentals
- Distribution shift problem
- DAgger algorithm
- Connection to dynamical systems

**Relevance**: Foundational understanding of imitation learning used in slippi-ai.

---

### University of Waterloo: CS885

**Course**: CS885 - Reinforcement Learning
**Instructor**: Pascal Poupart
**Topic**: Imitation Learning

**Links**:
- [Class Central](https://www.classcentral.com/course/youtube-cs885-imitation-learning-142565)

**Content** (~30 minutes):
- Behavioral cloning
- Autonomous driving applications
- GAIL (Generative Adversarial Imitation Learning)
- Inverse dynamics
- Robotics experiments

---

### MIT: Underactuated Robotics

**Course**: 6.832
**Chapter**: 21 - Imitation Learning

**Links**:
- [Course Notes](https://underactuated.mit.edu/imitation.html)

**Content**:
- Theoretical foundations
- Behavior cloning analysis
- Distribution mismatch
- Connections to control theory

---

### CMU: Deep RL and Control

**Course**: 10-403
**Topic**: Imitation Learning & Behavior Cloning

**Links**:
- [Lecture Slides (PDF)](https://www.andrew.cmu.edu/course/10-403/slides/S19_lecture2_behaviorcloning.pdf)

**Content**:
- BC algorithm details
- Compounding error analysis
- DAgger and variants
- Practical considerations

---

## Technical Blog Posts

### Eric Gu: Melee Transformer (2024)

**Title**: "Training AI to play Super Smash Bros. Melee"
**Author**: Eric Gu

**Link**: [ericyuegu.com/melee-pt1](https://ericyuegu.com/melee-pt1)

**Content**:
- 20M parameter Transformer
- GPT-style next-token prediction
- $5, 5-hour training cost
- All-character > single-character finding

**Key Insight**: "I strongly believe a single Transformer trained on all character replays would out-perform."

---

### Project Nabla Technical Writeup (2022)

**Title**: "Project Nabla Writeup"
**Author**: Bryan Chen (bycn)

**Link**: [bycn.github.io](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)

**Content**:
- BC + PPO pipeline
- Population-based training necessity
- "Melee Turing Test" concept
- Self-play cycling problem

---

### Fizzi: Slippi Medium Posts

**Author**: Jas "Fizzi" Laferriere

**Links**:
- [Project Slippi Public Release](https://medium.com/project-slippi/project-public-release-4080c81d7205)
- [Slippi Backend Migration](https://medium.com/project-slippi/slippi-backend-migration-f557d00d2474)

**Content**:
- Slippi architecture decisions
- Rollback implementation challenges
- Community building

---

### Rollback Netcode Deep Dives

**Title**: "Delta Rollback: New optimizations for Rollback Netcode"
**Author**: David Dehaene

**Link**: [Medium](https://medium.com/@david.dehaene/delta-rollback-new-optimizations-for-rollback-netcode-7d283d56e54b)

**Content**: Advanced rollback optimizations applicable to game AI inference.

---

## YouTube Content

### Moky vs Phillip (2024)

**Title**: Games vs Moky
**Channel**: vladfi1

**Link**: [YouTube](https://www.youtube.com/watch?v=1kviVflqXc4)

**Content**: FT10 exhibition match, Phillip wins 10-3.

**Significance**: Demonstrates current state-of-the-art Melee AI vs top human player.

---

### x_pilot Twitch Streams

**Channel**: [twitch.tv/x_pilot](https://twitch.tv/x_pilot)
**Host**: vladfi1 (slippi-ai creator)

**Content**:
- Live matches vs community
- Development updates
- Agent demonstrations

---

### Siraj Raval: OpenAI Five Explained

**Title**: "OpenAI Five vs DOTA 2 Explained"

**Link**: [GitHub with Code](https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained)

**Content**: Accessible explanation of OpenAI Five architecture and training.

---

## Recommended Viewing Order

### For Understanding Melee AI History

1. **DEF CON 24 SmashBot** - Foundation
2. **The Smash Brothers documentary** - Community context
3. **Phillip arXiv paper** - First ML approach
4. **Project Nabla writeup** - BC+RL insights
5. **Eric Gu blog** - Modern Transformer approach

### For Understanding Game AI Generally

1. **AlphaStar Blizzard VoD** - Landmark achievement
2. **OpenAI Five technical paper** - Scale challenges
3. **GDC Rollback talk** - Real-time constraints
4. **FightingICE papers** - Fighting game specific

### For Technical Implementation

1. **UC Berkeley Deep RL lectures** - BC fundamentals
2. **CMU 10-403 slides** - Practical BC
3. **MIT Underactuated notes** - Theory
4. **Eric Gu blog** - Melee-specific implementation

---

## Summary Table

| Talk/Paper | Year | Venue | Focus | Link Type |
|------------|------|-------|-------|-----------|
| SmashBot DEF CON | 2016 | DEF CON 24 | Reverse engineering | Video |
| Phillip Paper | 2017 | arXiv | Pure RL | Paper |
| GGPO EVO Talk | 2017 | EVO | Rollback netcode | Video |
| AlphaStar Demo | 2019 | BlizzCon | StarCraft AI | Video |
| OpenAI Five Paper | 2019 | OpenAI | Dota 2 AI | Paper |
| FightingICE | 2013-present | IEEE CoG | Competition | Papers |
| Project Nabla | 2022 | Blog | BC+RL | Blog |
| Eric Gu Transformer | 2024 | Blog | Transformer IL | Blog |
| NRS Rollback GDC | - | GDC | Implementation | Video |

---

## References

### Academic
- [arXiv: Phillip Paper](https://arxiv.org/abs/1702.06230)
- [arXiv: DareFightingICE](https://arxiv.org/abs/2203.01556)
- [NeurIPS 2020: Honor of Kings](https://papers.nips.cc/paper/2020/file/06d5ae105ea1bea4d800bc96491876e9-Paper.pdf)
- [OpenAI: Dota 2 Paper](https://cdn.openai.com/dota-2.pdf)

### Industry
- [GDC Vault](https://www.gdcvault.com/)
- [DeepMind Blog](https://deepmind.google/discover/blog/)
- [OpenAI Blog](https://openai.com/blog/)

### Community
- [FightingICE Official](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/)
- [GGPO Official](https://www.ggpo.net/)
- [Slippi Official](https://slippi.gg/)
