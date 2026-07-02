# Training Best Practices for Melee Behavioral Cloning

## Problem: Mode Collapse on Large Datasets

The model predicts "neutral/do nothing" for every input — all buttons 0%, all sticks center.
Works on 100 files (val_loss 2.06), collapses on 200 files.

### Root Cause

**Focal loss + class imbalance at scale.** Players idle 60-80% of frames. With focal_gamma=3.0,
focal loss says "neutral prediction is easy → down-weight it." But at epoch 1, ALL predictions
are "easy" (model always predicts neutral). The gradient signal from rare button presses gets
suppressed before the model can learn them. Chicken-and-egg problem.

Confirmed by:
- 100 files works (fewer steps per epoch → LR hasn't decayed before model escapes neutral)
- f32 and bf16 both collapse on 200 files (not a precision issue)
- Loss drops but action diversity stays at 1 (model learns to predict neutral better, not actions)

### Solutions (Priority Order)

#### 1. Lower focal_gamma (0-1.0 instead of 3.0)
- **Why:** gamma=3.0 is too aggressive at initialization. With gamma=1.0, focal loss still helps with rare actions but doesn't suppress gradient signal at epoch 1.
- **Config:** `--focal-gamma 1.0`

#### 2. Entropy Regularization
- **Why:** Adds a term that penalizes collapsed output distributions. Gives gradient signal to maintain prediction diversity.
- **Formula:** `total_loss = imitation_loss - entropy_weight * mean(sum(p * log(p)))`
- **Config:** `--entropy-weight 0.01` (to be implemented)

#### 3. Learning Rate Warmup
- **Why:** Without warmup, Adam's initial moment estimates are near-zero, causing effective LR to be much larger than nominal. Large early updates overshoot into degenerate basin.
- **Config:** `--warmup-steps 1000` (5-10% of epoch steps)

#### 4. Higher Button Weight
- **Why:** 4 stick heads + 1 shoulder dominate total loss over 1 button head. At button_weight=2.0, buttons are still underrepresented.
- **Config:** `--button-weight 5.0`

#### 5. Lower LR for Larger Datasets
- **Why:** More diverse data needs smaller steps to avoid overshooting.
- **Config:** `--learning-rate 5e-5` or `--learning-rate 3e-4 --lr-schedule constant`

#### 6. Gradient Clipping Reduction
- **Why:** SSMs can have gradient spikes with diverse input sequences.
- **Config:** `--max-grad-norm 0.5`

### Advanced Techniques (Future)

#### Curriculum Learning
Start training on 100-file subset for 2-3 epochs, then add remaining files. Model learns basic features before being overwhelmed by full data distribution.

#### DAgger (Dataset Aggregation)
Iteratively: train BC → deploy in game → collect expert corrections on model's actual states → retrain. Gold standard for BC but requires game loop.

#### Action-Conditioned Sampling
Oversample frames with button presses at the data level. Already partially implemented via `--action-oversample 3.0`.

#### Stratified Batching
Ensure each mini-batch contains examples from multiple action classes. Prevents all-neutral batches that give uniform "predict neutral" gradients.

## Training Hyperparameter Reference

### What Worked (100 files, val_loss 2.06)
```
--backbone mamba --max-files 100 --batch-size 64
--precision f32 --lr-schedule constant --learning-rate 1e-4
--focal-loss --focal-gamma 3.0 --button-weight 2.0
--stick-edge-weight nil --action-oversample 3.0
```

### Recommended for 200+ Files
```
--backbone mamba --max-files 200 --batch-size 16
--precision f32 --lr-schedule constant --learning-rate 3e-4
--focal-gamma 1.0 --button-weight 5.0
--warmup-steps 1000 --max-grad-norm 0.5
--stick-edge-weight 2.0 --action-oversample 3.0
```

### Scale-Dependent Considerations
| Setting | Small (50 files) | Medium (200 files) | Large (1000+ files) |
|---------|-----------------|--------------------|--------------------|
| focal_gamma | 2.0-3.0 | 1.0 | 0.5-1.0 |
| learning_rate | 1e-4 | 3e-4 | 1e-4 |
| lr_schedule | cosine_restarts | constant | cosine with warmup |
| warmup_steps | 0 | 1000 | 5000 |
| button_weight | 2.0 | 5.0 | 5.0 |
| batch_size | 64 | 16-32 | 16 |

## References
- [Towards Balanced BC from Imbalanced Datasets](https://arxiv.org/html/2508.06319v1)
- [Curriculum Offline Imitation Learning](https://arxiv.org/abs/2111.02056)
- [Why Warmup the Learning Rate](https://arxiv.org/html/2406.09405v1)
- [Mamba gradient explosion issue](https://github.com/state-spaces/mamba/issues/529)
- [Entropy Regularization in RL](https://towardsdatascience.com/entropy-regularized-reinforcement-learning-explained-2ba959c92aad/)
