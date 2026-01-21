# Melee AI Research & Prior Art

This document summarizes research relevant to ExPhil, including lessons learned from existing projects and papers.

## Table of Contents

- [Key Projects](#key-projects)
- [Training Approaches](#training-approaches)
- [Model Architectures](#model-architectures)
- [State Representation](#state-representation)
- [Action Space Handling](#action-space-handling)
- [Frame Delay & Reaction Time](#frame-delay--reaction-time)
- [Self-Play & RL](#self-play--rl)
- [Lessons for ExPhil](#lessons-for-exphil)
- [Research Roadmap](#research-roadmap)

---

## Key Projects

### slippi-ai (Active, Primary Reference)
- **Repository:** https://github.com/vladfi1/slippi-ai
- **Approach:** Behavioral cloning from Slippi replays + RL self-play refinement
- **Architecture:** GRU core + autoregressive action heads
- **Results:** Competitive with top professionals (Moky 10-3, Zain 5-3)
- **Delay:** 18+ frames (designed for online play feel)

### Phillip (Original, Deprecated)
- **Repository:** https://github.com/vladfi1/phillip
- **Paper:** [Beating the World's Best at SSBM with Deep RL](https://arxiv.org/abs/1702.06230)
- **Approach:** Pure deep RL (no imitation learning)
- **Architecture:** Actor-Critic with frame stacking
- **Results:** Beat professionals at tournaments (2017)

### Project Nabla
- **Blog Post:** https://bycn.github.io/2022/08/19/project-nabla-writeup.html
- **Approach:** BC + RL self-play (similar to slippi-ai)
- **Key Finding:** BC agents learn human-like skills (wavedashing, L-cancelling), RL "stitches" them together
- **Warning:** Single-opponent self-play causes policy collapse

### Eric Gu's Transformer Approach
- **Blog:** https://ericyuegu.com/melee-pt1
- **Architecture:** 20M parameter decoder-only Transformer
- **Training:** Next-token prediction (GPT-style) on controller inputs
- **Results:** 95% win rate vs Level 9 CPU, $5 training cost, 5 hours on 2x 3090s
- **Insight:** Transformer trained on all characters could outperform character-specific models

### SmashBot (Rule-Based)
- **Repository:** https://github.com/altf4/SmashBot
- **Approach:** Hand-coded behavior trees and frame data
- **Use Case:** Good baseline opponent for testing, no ML required

---

## Training Approaches

### Two-Stage Training (Proven Effective)

The consensus approach from slippi-ai and Project Nabla:

```
Stage 1: Behavioral Cloning
├── Dataset: ~100k Slippi replays
├── Loss: Cross-entropy on action prediction
├── Result: Human-like but error-prone play
└── Duration: Days to week on good GPU

Stage 2: RL Self-Play
├── Algorithm: PPO with clipped objective
├── Opponent: Self (latest weights) or population
├── Result: Stronger play, fixes BC mistakes
└── Duration: Weeks of continuous training
```

### Why Two Stages?

1. **Pure RL is slow** - The action space is huge, reward is sparse (winning)
2. **BC provides strong prior** - Agents learn tech skill from humans
3. **RL refines** - Fixes suboptimal patterns, discovers new strategies

### Alternatives to Explore

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **DAgger** | Interactive imitation, queries expert on agent's states | Handles distributional shift | Needs online expert |
| **GAIL** | Adversarial imitation learning | More robust than BC | Harder to train |
| **Offline RL** | RL from fixed dataset (no environment) | No simulator needed | Conservative policies |

**Reference:** [Interactive and Hybrid Imitation Learning](https://arxiv.org/abs/2412.07057)

---

## Model Architectures

### Current Approaches in Use

| Project | Core Architecture | Temporal Handling | Action Head |
|---------|------------------|-------------------|-------------|
| slippi-ai | Dense + GRU (256) | 1-layer recurrent | Autoregressive |
| Phillip | MLP + frame stack | Concatenated frames | Independent heads |
| Eric Gu | Decoder Transformer | Causal attention | Next-token prediction |
| **ExPhil** | MLP/Mamba/LSTM | Configurable backbone | Autoregressive (6 heads) |

### Why Mamba for ExPhil?

Our choice of Mamba (State Space Model) is novel for game AI:

| Property | Transformer | Mamba | Impact for Melee |
|----------|------------|-------|------------------|
| Complexity | O(n²) | O(n) | Longer context windows possible |
| Throughput | Baseline | 5× faster | More headroom for inference |
| Memory | Grows with seq² | Linear | Can use 90+ frame windows |
| Selectivity | Fixed attention | Input-dependent | May adapt to game state better |

**Reference:** [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

**Our benchmarks:**
- Mamba inference: **8.93ms** (60 FPS ready)
- ONNX INT8: **0.55ms** (plenty of headroom)

### Hybrid Approaches to Consider

[Mamba-2](https://goombalab.github.io/blog/2024/mamba2-part1-model/) (2024) introduces State Space Duality (SSD), allowing hybrid SSM+attention models. NVIDIA research validates that hybrids can outperform pure architectures.

---

## State Representation

### libmelee Feature Set

**Reference:** https://libmelee.readthedocs.io/en/latest/gamestate.html

Per-player features (~26 values):

```python
# Position
x, y                    # Character position

# Velocity (5 values, not 2!)
speed_air_x_self       # Self-induced air speed
speed_ground_x_self    # Self-induced ground speed
speed_x_attack         # Knockback X from being hit
speed_y_attack         # Knockback Y from being hit
speed_y_self           # Self-induced Y speed

# State
action                 # Current action state enum
action_frame           # Frame within current action
facing                 # Boolean: facing right?
on_ground              # Boolean: grounded?
jumps_left             # Remaining jumps
invulnerability        # Invuln frames remaining

# Resources
percent                # Damage percent
stocks                 # Lives remaining
shield_strength        # Shield health (0-60)

# Collision (ECB)
ecb_bottom, ecb_left, ecb_right, ecb_top  # 4 x (x,y) offsets
```

**Key insight:** Melee has **5 velocity values**, not 2. This is critical for tech chasing and combo prediction. SmashBot explicitly needs all 5 for proper tech chase behavior.

### ExPhil Embedding Comparison

Our current embedding: **446 dims per player, ~1991 total**

This is comparable to reference implementations. Key differences:
- We include character one-hot encoding
- We include stage platform positions
- We track projectiles separately

---

## Action Space Handling

### The Challenge

Melee controller has ~30 billion possible input combinations per frame:
- 8 buttons (independent binary)
- Main stick: 256 × 256 positions
- C-stick: 256 × 256 positions
- L/R shoulders: 256 levels each

### Autoregressive Action Heads (Best Practice)

slippi-ai uses sequential prediction:

```
button1 → button2 → main_x → main_y → c_x → c_y → shoulder
```

**Why autoregressive?**
- Coordinates multi-button inputs (down + B for shine)
- Teacher forcing during training
- Each head can condition on previous outputs

**ExPhil implementation:** 6 autoregressive heads matching this pattern.

### Stick Discretization

**Key finding from Eric Gu's research:**
- K-means clustering on 100k samples → **21 cluster centers** worked well
- Finer discretization performed **worse** than coarser
- Melee engine discretizes internally, so matching it helps

**Current ExPhil:** 9 classes per stick axis (81 positions)

**Experiment to try:** K-means-based discretization from replay data

### Precise Inputs

Some tech requires pixel-perfect stick positions:
- Shield drops: specific notch angles
- Ledge dashes: precise angles
- Firefox angles: 16+ distinct useful angles

Consider adding hand-picked clusters for these via [Altimor's StickMap](https://github.com/altf4/libmelee).

---

## Frame Delay & Reaction Time

### Online vs Local Play

| Context | Typical Delay | Notes |
|---------|--------------|-------|
| Phillip online | 18+ frames | Feels like ~300ms ping |
| Local play | 0-3 frames | Controller + display lag |
| Human reaction | 10-15 frames | ~200ms visual reaction |

### Training with Delay

**From [Learning to Play SSBM with Delayed Actions](https://www.yash-sharma.com/files/learning-play-super%20(3).pdf):**

Training with explicit delay teaches agents to **predict** rather than **react**. Key hyperparameters:
- Frame delay during training
- Recurrent architecture (DRQN) handles partial observability
- Discount factor tuning for delayed rewards

### ExPhil Delay Strategy

For local play (our goal):
1. Train with **variable delay (0-18 frames)** - learns both reactive and predictive play
2. Deploy with **minimal delay** - use our fast inference (<16ms)
3. Add **optional delay flag** for online play simulation

---

## Self-Play & RL

### Policy Collapse Problem

**From Project Nabla:**
> "With certain characters, policies collapse into undesirable behaviors — future work should utilize historical sampling or population-based training."

Single-opponent self-play can lead to:
- Rock-paper-scissors cycles
- Exploitation of specific weaknesses
- Loss of general skill

### Solutions

1. **Historical Sampling** - Play against older versions of self
2. **Population-Based Training** - Multiple agents, league system
3. **Diverse Opponents** - Mix of rule-based, BC, RL agents

### League Training (AlphaStar-style)

```
Main Agents (trained)
    ↓ plays against
Exploiter Pool (finds weaknesses)
    ↓ added to
League (historical checkpoints)
    ↓ sampled from
Training Opponents
```

**Reference:** [AlphaStar Paper](https://www.nature.com/articles/s41586-019-1724-z)

### PPO for Fighting Games

**From [Mastering Fighting Game Using Deep RL](https://ieee-cog.org/2020/papers/paper_207.pdf):**

PPO with self-play achieved 62% win rate vs professionals in Blade & Soul:
- Clipped objective prevents catastrophic updates
- Self-play curriculum (easy → hard opponents)
- Data skipping for sample efficiency

---

## Lessons for ExPhil

### What We're Doing Right

1. **Two-stage training** - BC then RL matches best practices
2. **Autoregressive action heads** - Proper multi-button coordination
3. **Mamba backbone** - Novel, efficient, enables long context
4. **Comprehensive state embedding** - All 5 velocities, ECB, etc.

### What We Should Add

1. **Population-based self-play** - Avoid policy collapse
2. **Frame delay augmentation** - Train with variable delays
3. **K-means stick discretization** - May improve over uniform grid
4. **DAgger or similar** - Better than pure BC for distributional shift

### Low-Tier Specific Considerations

No existing research targets low-tiers. Our unique angle:

| Character | Context Window | Special Features |
|-----------|---------------|------------------|
| Mewtwo | 90+ frames | Teleport recovery timing, tail hitboxes |
| Ganondorf | 60 frames | Spacing reads, punish optimization |
| Link | 75 frames | Projectile tracking, item positions |
| G&W | 45 frames | No L-cancel, RNG moves (hammer/bucket) |
| Zelda | 60 frames | Transform state tracking |

### Reward Shaping Ideas

For RL stage, character-specific rewards:

```elixir
# Ganondorf: Reward spacing and punishes
def ganon_reward(state) do
  base_reward(state)
  + spacing_bonus(state)      # Reward optimal range
  + punish_damage(state)      # Big reward for hard punishes
  - whiff_penalty(state)      # Punish missed slow moves
end

# Mewtwo: Reward stage control and edgeguards
def mewtwo_reward(state) do
  base_reward(state)
  + center_stage_bonus(state) # Reward stage control
  + edgeguard_bonus(state)    # Reward keeping opponent offstage
  + recovery_bonus(state)     # Reward successful teleport recovery
end
```

---

## Research Roadmap

### Phase 1: Training Infrastructure (Current)
- [x] Behavioral cloning from replays
- [x] Multiple backbone support (MLP, LSTM, GRU, Mamba)
- [x] Training features (EMA, LR scheduling, augmentation)
- [x] **Frame delay augmentation** (`--online-robust` or `--frame-delay-augment`)
- [ ] **K-means stick discretization experiment**

### Phase 2: Self-Play (Next)
- [ ] **Population-based training** - Multiple agents
- [ ] **Historical sampling** - Play against old checkpoints
- [ ] **League system** - Exploiters + main agents
- [ ] **BEAM concurrency** - Elixir processes for parallel games

### Phase 3: RL Refinement
- [ ] **PPO integration** with self-play
- [ ] **Character-specific reward shaping**
- [ ] **Curriculum learning** - Easy → hard opponents
- [ ] **Opponent diversity** - Rule-based + learned

### Phase 4: Specialization
- [ ] **Mewtwo specialist** - Long context, recovery focus
- [ ] **Ganondorf specialist** - Spacing, reads
- [ ] **Multi-character model** - Single model, character conditioning

### Experiments to Run

| Experiment | Hypothesis | Metric | Status |
|------------|-----------|--------|--------|
| K-means stick discretization | Better than uniform grid | BC loss, action accuracy | Pending |
| Frame delay augmentation | More robust play | Win rate vs delayed self | ✅ Implemented |
| Mamba vs Transformer | Comparable quality, faster | Loss, inference time | Pending |
| Population vs single self-play | Avoids collapse | Strategy diversity | Pending |

---

## References

### Papers

1. **Firoiu et al. (2017)** - [Beating the World's Best at SSBM with Deep RL](https://arxiv.org/abs/1702.06230)
2. **Gu & Dao (2023)** - [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
3. **Sharma** - [Learning to Play SSBM with Delayed Actions](https://www.yash-sharma.com/files/learning-play-super%20(3).pdf)
4. **Parr et al.** - [SSBM: An "Untouchable" Agent](https://arxiv.org/abs/1712.03280)
5. **Oh et al. (2020)** - [Mastering Fighting Game Using Deep RL with Self-Play](https://ieee-cog.org/2020/papers/paper_207.pdf)
6. **Tang et al. (2020)** - [Discretizing Continuous Action Space for On-Policy Optimization](https://arxiv.org/abs/1901.10500)

### Code Repositories

1. **slippi-ai** - https://github.com/vladfi1/slippi-ai (Primary reference)
2. **Phillip** - https://github.com/vladfi1/phillip (Original, deprecated)
3. **libmelee** - https://github.com/altf4/libmelee (Game state API)
4. **SmashBot** - https://github.com/altf4/SmashBot (Rule-based baseline)
5. **Mamba** - https://github.com/state-spaces/mamba (SSM architecture)

### Documentation

1. **libmelee docs** - https://libmelee.readthedocs.io/en/latest/
2. **Mamba-2 explainer** - https://goombalab.github.io/blog/2024/mamba2-part1-model/
3. **Fighting Game AI Competition** - https://www.ice.ci.ritsumei.ac.jp/~ftgaic/

### Community Resources

1. **Slippi Discord** - Replay datasets, libmelee support
2. **Melee Library** - https://www.meleelibrary.com/ (Frame data, tech)
