# Community Projects and Alternative Approaches

This document covers additional Melee AI projects, community efforts, and alternative approaches beyond the major projects (slippi-ai, Phillip, SmashBot).

## Notable Projects

### Eric Gu's Transformer Agent

**Author**: Eric Gu
**Link**: [ericyuegu.com/melee-pt1](https://ericyuegu.com/melee-pt1)
**Status**: Research project (2024)

**Approach**:
- Decoder-only Transformer (GPT-style)
- Next-token prediction on action sequences
- 20M parameters, 3B frames training data

**Architecture**:
```
Input: [s_0, a_0, s_1, a_1, ..., s_t]
Transformer: Causal attention over sequence
Output: a_t distribution
```

**Key Results**:
- 95% win rate vs Level 9 CPU
- $5 training cost
- 5 hours on 2× RTX 3090

**Key Insight**: All-character model outperformed single-character models

**Implications for ExPhil**:
- Transformers viable for fighting games
- Multi-character training beneficial
- Simple objective (next-token prediction) works

### Project Nabla

**Author**: bycn
**Link**: [bycn.github.io](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
**Status**: Research writeup (2022)

**Key Findings**:

1. **BC learns modular skills**:
   - Agents trained on BC exhibit recognizable human tech
   - Wavedashing, L-canceling emerge naturally
   - Skills transfer across situations

2. **Single-opponent self-play fails**:
   ```
   Policy A > Policy B
   Policy B' > Policy A
   Policy A' > Policy B'
   → Rock-paper-scissors cycling
   ```

3. **Solutions explored**:
   - Historical sampling
   - Population-based training
   - Diversity rewards

**Lessons for ExPhil**:
- Don't use single-opponent self-play
- Historical checkpoints or population required
- BC provides strong initialization

### Fizzi's Fast-Forward Codes

**Author**: Fizzi
**Status**: Integrated into Slippi

**What it enables**:
- Frame-advance capability in Dolphin
- Essential for RL training (run faster than real-time)
- Gecko codes for memory manipulation

**Impact**: Made RL training practical (previously real-time only)

### CPU Mod Projects

**20XX Training Pack**: Enhanced CPU behaviors for human practice
**Uncle Punch**: Frame data visualization and practice tools
**Akaneia**: Custom build with AI enhancements

**Relevance**: Show what hand-tuned CPU can achieve (still loses to ML)

### Cross-Language Replay Parsing

The Peppi ecosystem provides replay parsing across multiple languages:

| Language | Project | Status |
|----------|---------|--------|
| Rust | [peppi](https://github.com/hohav/peppi) | Active, canonical |
| Python | [peppi-py](https://github.com/hohav/peppi-py) | Active |
| Julia | [peppi-jl](https://github.com/jph6366/peppi-jl) | Early prototype |

**Peppi.jl** ([github.com/jph6366/peppi-jl](https://github.com/jph6366/peppi-jl)) is notable for bringing Slippi parsing to Julia's ML ecosystem (Flux.jl, MLJ.jl). Uses Apache Arrow columnar storage for efficient frame data access.

**Relevance for ExPhil**: Demonstrates community interest in non-Python ML stacks. Similar to how ExPhil uses Elixir/Nx, peppi-jl enables Julia-based Melee AI research.

## Alternative Approaches

### Hierarchical RL

**Concept**: Decompose decision-making into levels

```
High-Level (5s): Strategic goals (approach, defend, edgeguard)
Mid-Level (1s): Tactical decisions (which combo, which recovery)
Low-Level (frame): Execution (button timing, angles)
```

**Benefits**:
- Easier credit assignment
- Reusable skills
- More interpretable

**Challenges**:
- Defining hierarchy boundaries
- Communication between levels
- Learning all levels simultaneously

**Status**: Explored in research, not yet dominant in Melee

### Model-Based RL

**Concept**: Learn world model, plan using it

```
World Model: s_t, a_t → s_t+1
Planner: Simulate futures, select best action sequence
```

**Benefits**:
- Sample efficient
- Can plan ahead
- Uncertainty quantification

**Challenges**:
- Melee physics are complex
- Model errors compound
- Real-time planning is slow

**Phillip used**: Partial model for delay compensation

### Inverse RL

**Concept**: Infer reward function from demonstrations

```
Demonstrations → Inferred Reward → RL Training
```

**Benefits**:
- Learns "what" to optimize, not just "how"
- Can generalize better than BC
- Avoids reward hacking

**Challenges**:
- Computationally expensive
- Reward ambiguity
- Requires good demonstrations

**Status**: Not yet applied to Melee at scale

### Offline RL

**Concept**: RL from fixed dataset (no environment interaction)

```
Static Dataset → Conservative Policy → Deployment
```

**Benefits**:
- No simulator needed
- Uses existing replay data
- Safer (no live mistakes)

**Challenges**:
- Distribution shift
- Conservative estimates
- Limited by data coverage

**Potential for Melee**: Use massive replay archives without live training

## Community Tools

### Slippi Ecosystem

**Slippi Launcher**: Desktop app for online play and replays
**Slippi.gg**: Web platform for rankings and stats
**Slippi DB**: Replay database and statistics

**AI-Relevant Features**:
- `.slp` format for replay parsing
- Ranked queue provides skill-labeled games
- Spectator mode for data collection

### Dolphin Modifications

**Memory Access**: Reading game state
**Named Pipes**: Controller input
**Gecko Codes**: Game behavior modification
**Save States**: For deterministic training

### Analysis Tools

**IKneeData**: Frame data database
**Fizzi's Frame Display**: In-game frame counter
**Melee Library**: Tech skill documentation

## Research Directions

### Explored but Not Solved

1. **Perfect neutral game**: AI still makes suboptimal neutral decisions
2. **Adaptation**: No agent learns opponent habits during match
3. **Character generalization**: Single model for all 26 characters
4. **Style transfer**: Imitating specific player styles

### Open Problems

1. **Superhuman low-tier**: Can AI make Mewtwo top-tier?
2. **Novel technique discovery**: Beyond human metagame
3. **Explainable decisions**: Why did agent make that choice?
4. **Transfer to other games**: Melee → Ultimate → ?

### What Would Help

1. **More compute**: Scale existing approaches
2. **Better simulators**: Faster than Dolphin
3. **Diverse replay data**: More characters, more skill levels
4. **Standardized benchmarks**: Compare approaches fairly

## Community Engagement

### Twitch/YouTube

**x_pilot (vladfi1)**: Live streams of slippi-ai vs professionals
**Melee Stats**: Community commentary on AI matches
**Various**: AI tournaments and exhibitions

### Tournaments

**AI brackets**: Occasional AI vs AI competitions
**Human vs AI**: Exhibitions at majors
**Community interest**: High engagement with AI content

### Open Source

Most Melee AI is open source:
- slippi-ai: MIT license
- Phillip: MIT license
- SmashBot: MIT license
- libmelee: LGPL

**Implications**: Can build on existing work directly

## Comparison Table

| Project | Approach | Characters | Results | Compute |
|---------|----------|------------|---------|---------|
| slippi-ai | BC+RL | Fox, Falco | Top 20 competitive | Weeks |
| Phillip | Pure RL | Fox, Falcon | Beat pros (2017) | Months |
| Eric Gu | Transformer IL | All | 95% vs CPU | Hours |
| SmashBot | Rules | Fox | Predictable | None |
| Project Nabla | BC+RL research | Various | Research insights | Variable |
| ExPhil | BC+RL (Mamba) | Low-tier | In progress | TBD |

## Lessons from Community

### What Works

1. **Two-stage training**: BC → RL proven effective
2. **Open data**: Slippi replays enable research
3. **Open source**: Building on prior work
4. **Community testing**: Human players find weaknesses

### What Doesn't

1. **Pure RL from scratch**: Too expensive
2. **Single self-play opponent**: Causes cycling
3. **Ignoring delay**: Must train with realistic delay
4. **Superhuman reactions**: Unrealistic, unfair

### Emerging Consensus

1. **BC is essential**: Bootstrap from human data
2. **Mamba/Transformer viable**: Modern architectures work
3. **Multi-character helps**: Don't over-specialize
4. **Population training needed**: Avoid collapse

## ExPhil's Position

**Unique contributions**:
1. Elixir/BEAM: Native concurrency for self-play
2. Mamba backbone: O(n) temporal modeling
3. Low-tier focus: Unexplored character space
4. ONNX deployment: Framework-agnostic inference

**Building on community**:
- slippi-ai architecture patterns
- libmelee for Dolphin integration
- Slippi replays for training data
- Community knowledge of characters

## References

- [Eric Gu's Blog](https://ericyuegu.com/melee-pt1)
- [Project Nabla](https://bycn.github.io/2022/08/19/project-nabla-writeup.html)
- [Slippi](https://slippi.gg/)
- [20XX Training Pack](https://20xx.me/)
- [Uncle Punch](https://github.com/UnclePunch/Training-Mode)
- [Melee Library](https://www.meleelibrary.com/)
