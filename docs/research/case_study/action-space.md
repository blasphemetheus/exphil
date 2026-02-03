# Melee Action Space Deep Dive

This document explores the challenges and solutions for handling Melee's massive action space in machine learning systems.

## The Problem

### Raw Input Space

A GameCube controller has:
- **8 buttons**: A, B, X, Y, Z, L, R, Start (plus D-pad)
- **Main stick**: 256×256 positions (analog)
- **C-stick**: 256×256 positions (analog)
- **L/R triggers**: 256 positions each (analog)

**Theoretical combinations per frame**:
```
2^8 × 256^2 × 256^2 × 256 × 256 ≈ 30 billion
```

This is intractable for direct policy learning.

### Why Discretization is Necessary

**Continuous action spaces** fail because:
1. Most positions do nothing different (dead zones)
2. Melee engine quantizes internally anyway
3. Exploration is inefficient in high-dimensional continuous space

**Fine-grained discretization** fails because:
1. Too many actions → slow convergence
2. Action frequencies highly imbalanced
3. Similar actions treated as completely different

## Melee Engine Internals

### Stick Processing

The game processes analog stick inputs through several stages:

```
Raw Input (0-255) → Deadzone → Scaling → Engine Value (-1 to +1)
```

**Deadzone**: Inner ~23 units ignored (neutral)
**Scaling**: Remaining range mapped to -1 to +1

**Practical result**: Many raw positions map to same engine value

### Button vs Analog

**Buttons**: Truly binary (pressed/not pressed)
**Analog**: Continuous but with thresholds:
- Light shield: Analog L/R < 0.3
- Full shield: Analog L/R ≥ 0.3
- Angles: Specific ranges for techs

## Discretization Approaches

### Approach 1: Grid Discretization

**Method**: Divide stick space into uniform grid

```python
# 5×5 grid
positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
# 25 stick positions × 6 buttons = 150 actions
```

**Problems**:
- Misses critical angles (shield drop at exactly 0.6875 down)
- Wastes capacity on unused regions
- Uniform spacing doesn't match usage patterns

### Approach 2: Cardinal + Diagonals

**Method**: Include only meaningful directions

```python
positions = {
    'neutral': (0, 0),
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0),
    'up_left': (-0.7, 0.7),
    'up_right': (0.7, 0.7),
    'down_left': (-0.7, -0.7),
    'down_right': (0.7, -0.7),
}
# 9 positions × 6 buttons = 54 actions
```

**Problems**:
- Still misses wavedash angles
- No tilt attack positions
- Ignores analog precision for some techs

### Approach 3: K-Means Clustering (Recommended)

**Method**: Cluster replay data to find natural positions

```python
# Extract all stick positions from replays
stick_positions = extract_from_replays(replays)  # Millions of points

# Cluster
kmeans = KMeans(n_clusters=21)
kmeans.fit(stick_positions)
cluster_centers = kmeans.cluster_centers_

# Result: ~21 meaningful positions per axis
```

**Advantages**:
- Captures actual human usage patterns
- Includes tech-specific angles naturally
- Matches engine precision

**slippi-ai finding**: 21 clusters ≈ optimal for Fox

### Approach 4: Hierarchical Actions

**Method**: Decompose into components, sample sequentially

```
Sample buttons → Sample main_x | buttons → Sample main_y | buttons, x → ...
```

**Advantages**:
- Captures correlations (wavedash = L + angle)
- Smaller per-component action spaces
- More efficient exploration

**slippi-ai's autoregressive head**:
1. Buttons (8 Bernoulli)
2. Main stick X (given buttons)
3. Main stick Y (given buttons + X)
4. C-stick X
5. C-stick Y
6. Shoulder

## ExPhil's Approach

### Current Configuration

```elixir
# 9 classes per stick axis
main_stick_x: 9 classes  # [-1, -0.7, -0.35, -0.15, 0, 0.15, 0.35, 0.7, 1]
main_stick_y: 9 classes
c_stick_x: 9 classes
c_stick_y: 9 classes
shoulder: 4 classes      # [0, 0.3, 0.6, 1.0]

# Total: 9 × 9 × 9 × 9 × 4 = 26,244 combinations
# With 8 buttons: Much larger but handled autoregressively
```

### Autoregressive Sampling

```elixir
# Policy head outputs
def sample_action(hidden_state) do
  # Stage 1: Buttons (8 independent)
  buttons = sample_buttons(hidden_state)

  # Stage 2: Main stick X (conditioned on buttons)
  main_x = sample_main_x(hidden_state, buttons)

  # Stage 3: Main stick Y (conditioned on buttons + X)
  main_y = sample_main_y(hidden_state, buttons, main_x)

  # Stages 4-6: C-stick and shoulder
  c_x = sample_c_x(hidden_state, buttons, main_x, main_y)
  c_y = sample_c_y(hidden_state, buttons, main_x, main_y, c_x)
  shoulder = sample_shoulder(hidden_state, buttons, main_x, main_y, c_x, c_y)

  %Controller{
    buttons: buttons,
    main_stick: {main_x, main_y},
    c_stick: {c_x, c_y},
    shoulder: shoulder
  }
end
```

## Character-Specific Considerations

### Fox/Falco

**Critical inputs**:
- Shine: B + down (frame 1 active)
- Multishine: B + down, then specific up-angle
- Short hop: Y for exactly 3 frames
- Wavedash: Y → 45° angles

**Discretization needs**:
- Fast B press detection
- Precise diagonal angles for wavedash
- Rapid button transitions

### Mewtwo (ExPhil Target)

**Critical inputs**:
- Teleport angles: 8 directions
- Confusion timing: B + forward
- Shadow Ball charge: Hold B
- Tail attacks: Specific spacing inputs

**Discretization needs**:
- Recovery angle precision
- Hold vs tap B distinction
- Directional influence (DI) angles

### Ice Climbers (ExPhil Target)

**Critical inputs**:
- Desync techniques: Specific button/stick timing
- Nana control: Through Popo's inputs
- Wobble: A + forward, rhythmic

**Discretization needs**:
- Frame-precise button timing
- Desync initiation angles
- Synchronized vs desynced modes

### Game & Watch (ExPhil Target)

**Critical inputs**:
- Bucket (B + down): Hold vs tap
- Judgment (B + side): Random outcome
- Chef (B + neutral): Projectile spam
- No L-cancel: Different timing model

**Discretization needs**:
- Standard angles sufficient
- Focus on aerial timing instead

## Advanced Topics

### Altimor's StickMap

Alternative to K-means: Hand-picked optimal positions based on frame data analysis.

**Concept**: Include exactly the angles needed for:
- Shield drop (6.875° down from horizontal)
- Wavedash (optimal 17° angle)
- SDI (survival DI angles)
- Tech skill-specific positions

**Trade-off**: More precise but requires expert knowledge

### Action Repeat/Duration

**Problem**: Single action per frame ignores sequences

```python
# Wavedash is actually:
frame 0: Y (jump)
frame 1: nothing
frame 2: nothing
frame 3: L + angle (air dodge)
frame 4-14: nothing (slide)
```

**Solutions**:
1. **Frame-by-frame**: Model every frame (current approach)
2. **Action chains**: Treat sequences as atomic actions
3. **Macro actions**: Higher-level action space

### Temporal Abstraction

**Options framework**: Select high-level "options" that execute over multiple frames

```python
options = {
    'wavedash_left': [Y, _, _, L+left_angle, _, _, ...],
    'short_hop_nair': [Y, _, _, A, _, _, L, ...],
    'grab': [Z, _, _, _, _, ...],
}
```

**Trade-off**: Simpler decision-making but less flexibility

## Evaluation Metrics

### Action Accuracy

```python
# During BC training
predicted = model(state)
actual = replay_action

# Per-component accuracy
button_acc = (predicted.buttons == actual.buttons).mean()
stick_acc = (predicted.stick == actual.stick).mean()
```

### Action Diversity

```python
# Entropy of action distribution
entropy = -sum(p * log(p) for p in action_probs)

# Low entropy = deterministic (good for execution)
# High entropy = exploratory (good for learning)
```

### Tech Skill Success Rate

```python
# Wavedash success
wavedash_attempts = detect_wavedash_attempts(trajectory)
wavedash_successes = detect_wavedash_successes(trajectory)
wavedash_rate = successes / attempts

# L-cancel success
l_cancel_attempts = detect_aerial_landings(trajectory)
l_cancel_successes = detect_l_cancels(trajectory)
l_cancel_rate = successes / attempts
```

## Practical Recommendations

### For Imitation Learning

1. **Use autoregressive sampling**: Captures button-stick correlations
2. **K-means discretization**: 15-25 clusters per axis
3. **Per-character tuning**: Fox needs more angles than Ganon
4. **Include rare actions**: Z, D-pad for taunt/item play

### For Reinforcement Learning

1. **Start with BC discretization**: Known to work
2. **Entropy regularization**: Prevent action collapse
3. **Action masking**: Remove illegal actions per state
4. **Curriculum**: Start with fewer actions, expand

### For Low-Tier Characters (ExPhil)

1. **Mewtwo**: Focus on teleport angles, 8+ recovery directions
2. **Ganondorf**: Standard angles sufficient, focus on timing
3. **Link**: Include item throw angles
4. **G&W**: Standard is fine, timing matters more
5. **Ice Climbers**: Desync-specific angles critical

## References

- [Discretizing Continuous Actions](https://arxiv.org/abs/1901.10500) - Tang et al.
- [Altimor's StickMap](https://github.com/altimor/SmashBot) - Precision positions
- [IKneeDta](https://ikneedata.com/) - Frame data for all characters
- [slippi-ai controller_heads.py](https://github.com/vladfi1/slippi-ai) - Autoregressive implementation
