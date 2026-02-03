# Matchup-Specific Training for Melee AI

This document covers strategies for training Melee AI agents that can handle different character matchups effectively. The core question: should you train one model for all matchups, specialized models per matchup, or something in between?

## Why Matchups Matter

Melee is fundamentally a matchup-based game. Playing Fox vs Marth requires completely different strategies than Fox vs Jigglypuff:

| Aspect | Fox vs Marth | Fox vs Jigglypuff |
|--------|--------------|-------------------|
| **Neutral** | Space outside sword | Aggressive, don't let Puff camp |
| **Combo** | Short combos, tech chases | Long vertical combos |
| **Edgeguard** | Gimp tether recovery | Bair wall offstage |
| **Kill %** | ~80-100% | ~60-80% (lighter) |
| **Pacing** | Explosive exchanges | Patient, bait-and-punish |

**Key insight**: A single "optimal" strategy doesn't exist. The best action depends heavily on who you're fighting.

---

## Current Approaches in Melee AI

### slippi-ai: Per-Matchup Models

[slippi-ai](https://github.com/vladfi1/slippi-ai) trains separate models for each matchup:

**Available agents** (as of 2025):
- `fox_d18_ditto_v3` - Fox mirror
- `fox_d18_vs_falco_v3` - Fox vs Falco
- `marth_d18_ditto_v3` - Marth mirror
- `marth_d18_vs_falco_v3` - Marth vs Falco
- `falco_d18_ditto_v3` - Falco mirror

**Observations**:
- Some matchup agents are much stronger than others
- Requires separate training runs per matchup
- Storage: N characters × N opponents = N² models

### Eric Gu's Transformer: All-Character Model

[Eric Gu's approach](https://ericyuegu.com/melee-pt1) trains a single Transformer on all character replays:

> "I strongly believe a single Transformer trained on all character replays would out-perform and be more efficient to train. General concepts like spacing and positioning are fundamentally the same across all matchups."

**Results**: Single model achieved 95% win rate vs CPU across all characters.

**Key finding**: All-character training > single-character training (at least for imitation learning).

---

## Training Strategies

### Strategy 1: Single Universal Model

Train one model on all matchups simultaneously.

```
All Replays (all characters, all matchups)
    ↓
Single Policy Network
    ↓
Output: Actions given (my_character, opponent_character, game_state)
```

**Architecture considerations**:

```elixir
defmodule UniversalPolicy do
  def embed_game_state(state) do
    # Include both characters' identities
    my_char = one_hot(state.player.character, 26)
    opp_char = one_hot(state.opponent.character, 26)

    # Character-conditioned embedding
    Nx.concatenate([
      my_char,
      opp_char,
      embed_player(state.player),
      embed_player(state.opponent),
      embed_stage(state.stage)
    ])
  end
end
```

**Pros**:
- Single model to maintain
- Learns shared concepts (spacing, timing, fundamentals)
- More data per model (full dataset)
- Better generalization to rare matchups

**Cons**:
- May underfit specific matchup knowledge
- Larger model needed for same performance
- Conflicting gradients from different matchups

**When to use**:
- Limited compute budget
- Targeting many characters/matchups
- Early exploration phase

### Strategy 2: Per-Matchup Specialists

Train separate models for each matchup.

```
Fox vs Marth Replays → Fox vs Marth Model
Fox vs Falco Replays → Fox vs Falco Model
Fox vs Puff Replays → Fox vs Puff Model
...
```

**Pros**:
- Maximum matchup-specific optimization
- Smaller models per matchup
- Clear performance attribution

**Cons**:
- N² models for full coverage
- Less data per model
- No knowledge transfer
- Maintenance burden

**When to use**:
- Targeting specific tournament matchups
- Abundant data for target matchups
- Maximum performance in known scenarios

### Strategy 3: Base + Fine-Tuning

Train a universal base model, then fine-tune per matchup.

```
All Replays → Universal Base Model
    ↓ (fine-tune)
Matchup-Specific Data → Specialized Model
```

**Implementation**:

```elixir
# Stage 1: Train base model on all data
base_params = train_universal(all_replays, epochs: 50)

# Stage 2: Fine-tune on specific matchup
fox_vs_marth_params = fine_tune(
  base_params,
  filter_matchup(replays, :fox, :marth),
  epochs: 10,
  learning_rate: base_lr / 10  # Lower LR for fine-tuning
)
```

**Pros**:
- Best of both worlds
- Knowledge transfer from base
- Smaller fine-tuning datasets work
- Shared infrastructure

**Cons**:
- Still N² final models
- Risk of catastrophic forgetting
- More complex training pipeline

**When to use**: Recommended default for most scenarios.

### Strategy 4: Multi-Task Learning

Train a single model with matchup-specific heads.

```
Shared Backbone (learned representations)
    ↓
Matchup-Specific Heads
├── Fox vs Marth Head
├── Fox vs Falco Head
├── Fox vs Puff Head
└── ...
```

**Architecture**:

```elixir
defmodule MultiTaskPolicy do
  def forward(state, params) do
    # Shared feature extraction
    shared = backbone(state, params.backbone)

    # Route to matchup-specific head
    matchup_key = {state.player.character, state.opponent.character}
    head_params = params.heads[matchup_key]

    action_head(shared, head_params)
  end

  def compute_loss(batch, params) do
    # Group by matchup for balanced updates
    grouped = Enum.group_by(batch, fn {state, _} ->
      {state.player.character, state.opponent.character}
    end)

    losses = for {matchup, samples} <- grouped do
      compute_matchup_loss(samples, params, matchup)
    end

    Nx.mean(Nx.stack(losses))
  end
end
```

**Pros**:
- Single model, multiple specializations
- Shared feature learning
- Efficient storage
- Balanced gradient updates possible

**Cons**:
- Complex architecture
- Head routing overhead
- Potential negative transfer between very different matchups

### Strategy 5: Opponent Embedding

Instead of discrete matchup routing, learn continuous opponent representations.

```
Game State + Opponent Embedding → Policy
```

**Approach** ([He et al., 2016](https://arxiv.org/abs/1609.05559)):

```elixir
defmodule OpponentAwarePolicy do
  @doc """
  Learn opponent behavior embedding from observation history.
  """
  def embed_opponent(opponent_history, params) do
    # Encode opponent's recent behavior
    behavior_features = Enum.map(opponent_history, fn frame ->
      [
        frame.action_state,
        frame.position,
        frame.velocity
      ]
    end)

    # RNN/Transformer to summarize opponent style
    opponent_embedding = encode_behavior(behavior_features, params.opponent_encoder)

    opponent_embedding
  end

  def forward(state, opponent_history, params) do
    game_embed = embed_game_state(state)
    opp_embed = embed_opponent(opponent_history, params)

    combined = Nx.concatenate([game_embed, opp_embed])
    policy_head(combined, params.policy)
  end
end
```

**Benefits**:
- Adapts to opponent behavior, not just character
- Can handle unknown opponents
- Single model for all scenarios
- Potentially learns style-specific counters

**Research foundation**:
- [Opponent Modeling in Deep RL](https://arxiv.org/abs/1609.05559) - DQN with opponent encoding
- [LOLA](https://arxiv.org/abs/1709.04326) - Learning with Opponent-Learning Awareness
- [Model-Based Opponent Modeling](https://proceedings.neurips.cc/paper/2022/file/b528459c99e929718a7d7e1697253d7f-Paper-Conference.pdf) - NeurIPS 2022

---

## Opponent Modeling Techniques

### Mixture of Experts (MoE)

Automatically discover opponent strategy patterns:

```elixir
defmodule MixtureOfExperts do
  @num_experts 8

  def forward(state, opponent_history, params) do
    # Gating network chooses expert weights
    opponent_embed = encode_opponent(opponent_history)
    expert_weights = softmax(gate(opponent_embed, params.gate))

    # Each expert produces action distribution
    expert_outputs = for i <- 0..(@num_experts - 1) do
      expert_forward(state, params.experts[i])
    end

    # Weighted combination
    Enum.zip(expert_weights, expert_outputs)
    |> Enum.reduce(fn {w, out}, acc -> Nx.add(acc, Nx.multiply(w, out)) end)
  end
end
```

**Use case**: Opponent types emerge without supervision (e.g., "aggressive", "defensive", "technical").

### Online Adaptation

Adapt during the match based on opponent behavior:

```elixir
defmodule OnlineAdaptation do
  @doc """
  Rolling Horizon Evolution Algorithm with opponent model.
  """
  def adapt_policy(base_policy, opponent_model, horizon \\ 30) do
    # Predict opponent's next N actions
    predicted_opponent_actions = predict_sequence(opponent_model, horizon)

    # Optimize policy against predicted opponent
    optimized_policy = evolve(base_policy, predicted_opponent_actions)

    optimized_policy
  end
end
```

**Research**: [Enhanced RHEA with Opponent Model Learning](https://arxiv.org/abs/2003.13949) achieved 2nd place in FightingICE 2019.

### Temporal Convolutional Network (TCN) for Opponent Modeling

Model opponent behavior changes over time:

```python
# OM-TCN: Opponent Modeling with Temporal Convolutions
class OpponentModelTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.tcn = TCN(input_dim, hidden_dim, kernel_size=3, dropout=0.1)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, opponent_history):
        # opponent_history: (batch, time, features)
        encoded = self.tcn(opponent_history)
        prediction = self.predictor(encoded[:, -1, :])  # Predict next action
        return prediction
```

**Advantage**: Handles non-stationary opponent policies (they adapt too!).

---

## Matchup Data Considerations

### Data Availability by Matchup

Not all matchups have equal data:

| Matchup | Estimated Games | Quality |
|---------|-----------------|---------|
| Fox vs Fox | 30K+ | High |
| Fox vs Falco | 25K+ | High |
| Fox vs Marth | 20K+ | High |
| Marth vs Sheik | 15K+ | Medium |
| Fox vs Puff | 10K+ | Medium |
| Mewtwo vs Fox | 2K | Low |
| Ganondorf vs Marth | 1K | Low |
| Link vs G&W | <100 | Very Low |

**Implication**: Low-tier matchups have severe data scarcity.

### Handling Rare Matchups

#### Option 1: Transfer from Similar Matchups

```elixir
def get_transfer_matchups(target) do
  case target do
    {:ganondorf, :fox} ->
      # Similar: heavy vs spacie
      [{:falcon, :fox}, {:bowser, :fox}]

    {:mewtwo, :marth} ->
      # Similar: floaty vs swordie
      [{:peach, :marth}, {:puff, :marth}]

    _ -> []
  end
end
```

#### Option 2: Synthetic Data via Self-Play

Generate rare matchup data through self-play:

```elixir
def generate_matchup_data(char_a, char_b, base_model, num_games) do
  # Load base models for each character
  policy_a = load_model(char_a, base_model)
  policy_b = load_model(char_b, base_model)

  # Run self-play
  replays = for _ <- 1..num_games do
    run_game(policy_a, policy_b)
  end

  # Filter for quality (close games)
  Enum.filter(replays, &high_quality?/1)
end
```

#### Option 3: Cross-Character Generalization

Eric Gu's finding: training on all characters helps rare matchups:

```
When trained on Fox + Marth + Falco + ... + Mewtwo data together,
the model learns general concepts that transfer to rare Mewtwo matchups
better than training on Mewtwo data alone.
```

---

## How Pros Handle Matchups

Understanding human expertise informs AI design.

### Matchup-Specific Preparation

Top players prepare differently for each opponent:

| Player | Main | Known For |
|--------|------|-----------|
| **Zain** | Marth | Matchup-specific tech (Puff, Spacies) |
| **Cody** | Fox | Consistent across matchups |
| **aMSa** | Yoshi | Deeply optimized vs top tiers |
| **Axe** | Pikachu | Character-specific combos per MU |

**Insight**: Even "universal" players have matchup-specific optimizations.

### The Matchup Chart

Community-maintained matchup charts ([SmashWiki](https://www.ssbwiki.com/Character_matchup_(SSBM))):

```
           Fox  Falco  Marth  Sheik  Puff
Fox        50   55     55     60     55
Falco      45   50     50     55     50
Marth      45   50     50     50     55
Sheik      40   45     50     50     55
Puff       45   50     45     45     50
```

**Numbers**: % chance of winning (>50 = favorable)

**AI use**: Weight training based on matchup difficulty:
- Focus more data on losing matchups
- Less data needed for favorable matchups

### Adaptation During Sets

Pro players adapt between games:

1. **Game 1**: Default strategy
2. **Game 2**: Adjust based on opponent tendencies
3. **Game 3+**: Full adaptation

**AI parallel**: Could train adaptation networks that take opponent history.

---

## Practical Recommendations

### For ExPhil Low-Tier Characters

Given limited data for low-tier matchups, recommended approach:

```
Phase 1: Universal Base Model
├── Train on ALL character data (Fox, Marth, etc. included)
├── Learn general Melee concepts
└── ~50 epochs, full dataset

Phase 2: Character Fine-Tuning
├── Fine-tune on target character replays
├── Lower learning rate (1/10 base)
└── ~10 epochs, filtered data

Phase 3: Matchup Specialization (Optional)
├── Further fine-tune on specific matchups
├── Or use multi-head architecture
└── Only for matchups with sufficient data
```

### Architecture Recommendation

```elixir
defmodule ExPhilMatchupPolicy do
  @doc """
  Hybrid approach: character-conditioned with optional matchup heads.
  """

  def build(config) do
    %{
      # Shared backbone
      backbone: build_mamba_backbone(config),

      # Character conditioning (always used)
      char_embed: Axon.embedding(26, config.char_dim),

      # Optional matchup heads (for common matchups)
      matchup_heads: build_matchup_heads(config.common_matchups),

      # Default head (for rare matchups)
      default_head: build_controller_head(config)
    }
  end

  def forward(state, params) do
    # Embed with character conditioning
    my_char = params.char_embed[state.player.character]
    opp_char = params.char_embed[state.opponent.character]

    features = backbone(state, my_char, opp_char, params.backbone)

    # Route to matchup head if available
    matchup = {state.player.character, state.opponent.character}

    if Map.has_key?(params.matchup_heads, matchup) do
      params.matchup_heads[matchup].(features)
    else
      params.default_head.(features)
    end
  end
end
```

### Training Schedule

```
Week 1-2: Universal pre-training
├── All character replays
├── Large batch size
├── Character + opponent embedding

Week 3: Character specialization
├── Filter to target characters
├── Fine-tune with lower LR
├── Evaluate per-matchup

Week 4+: Matchup refinement
├── Identify weak matchups (from evaluation)
├── Collect/generate more data
├── Fine-tune specific heads
```

### Evaluation Protocol

Test each matchup separately:

```elixir
def evaluate_matchups(policy, opponents) do
  results = for opp_char <- opponents do
    win_rate = run_evaluation(policy, opp_char, num_games: 100)
    {opp_char, win_rate}
  end

  # Identify weak matchups
  weak = Enum.filter(results, fn {_, wr} -> wr < 0.4 end)

  %{
    overall: mean_win_rate(results),
    per_matchup: Map.new(results),
    needs_work: weak
  }
end
```

---

## Research Directions

### Open Questions

1. **Optimal knowledge sharing**: How much should matchups share?
2. **Adaptation speed**: How quickly should AI adapt to new opponent?
3. **Style detection**: Can we identify opponent playstyle from first stock?
4. **Meta-learning**: Learn to learn new matchups quickly?

### Promising Approaches

| Approach | Complexity | Potential |
|----------|------------|-----------|
| Multi-task + MoE | Medium | High |
| Online adaptation (TCN) | High | Very High |
| Meta-learning (MAML) | Very High | Unknown |
| Simple fine-tuning | Low | Good baseline |

### FightingICE Competition Insights

The [FightingICE AI competition](https://arxiv.org/abs/2003.13949) findings:
- RHEA + opponent model beat all 2018 bots
- Online learning crucial for adaptation
- Mixed MCTS + self-play achieved 94.4% win rate

---

## Summary

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Universal** | Simple, data efficient | May underfit | Early stage, limited data |
| **Per-matchup** | Maximum performance | N² models | Tournament prep |
| **Base + fine-tune** | Best of both | More complex | Recommended default |
| **Multi-task** | Single model, multiple heads | Complex arch | Production systems |
| **Opponent embed** | Adapts to behavior | Research-y | Advanced applications |

**Bottom line**: Start with universal training, fine-tune for important matchups, and consider opponent modeling for advanced adaptation.

---

## References

### Melee AI
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Per-matchup agents
- [Eric Gu's Transformer](https://ericyuegu.com/melee-pt1) - All-character approach

### Opponent Modeling
- [He et al., 2016](https://arxiv.org/abs/1609.05559) - Opponent Modeling in Deep RL
- [LOLA](https://arxiv.org/abs/1709.04326) - Learning with Opponent-Learning Awareness
- [Model-Based Opponent Modeling](https://proceedings.neurips.cc/paper/2022) - NeurIPS 2022

### Fighting Game AI
- [Enhanced RHEA](https://arxiv.org/abs/2003.13949) - FightingICE competition
- [FightLadder](https://arxiv.org/abs/2406.02081) - Competitive MARL benchmark
- [Pro-Level Fighting AI](https://arxiv.org/abs/1904.03821) - Blade & Soul research

### Multi-Task Learning
- [Multi-Task Learning Overview](https://arxiv.org/abs/1706.05098) - Survey paper
- [Mixture of Experts](https://arxiv.org/abs/1701.06538) - Sparsely-gated MoE

### Matchup Data
- [SmashWiki Matchup Chart](https://www.ssbwiki.com/Character_matchup_(SSBM))
- [Melee Library](https://www.meleelibrary.com/) - Matchup guides
