# SVM on Super Smash Bros. Melee

**Repository:** https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee
**Author:** Trevor Firl
**Institution:** Winona State University
**Language:** Python, JavaScript
**Status:** Complete (Senior Project, July 2021)
**Purpose:** Character classification from gameplay statistics

## Overview

A senior capstone project demonstrating Support Vector Machine (SVM) classification applied to competitive Melee gameplay. The model predicts which character a player was using based on 13 gameplay features extracted from Slippi replay files.

## Classification Task

**Target:** Identify player character from gameplay statistics

**Classes (4 characters):**
1. Fox (ID 0)
2. Marth (ID 1)
3. Falco (ID 2)
4. Jigglypuff (ID 3)

**Approach:** Multi-class SVM with One-vs-One strategy (6 binary classifiers)

## Features (13 Total)

### High-Level Statistics (from Slippi stats)
| Feature | Description |
|---------|-------------|
| Opponent Character | Character ID of opponent |
| IPM | Inputs Per Minute (mechanical intensity) |
| Win | Binary outcome (1 if 4 kills) |
| Length | Game duration in frames |
| Opening Per Kill | Openings created per kill achieved |
| Damage Per Opening | Damage dealt per opening |
| Neutral Win Ratio | Proportion of neutral states won |
| Opening Conversion Rate | Successful conversions from openings |

### Action Counts (frame-by-frame analysis)
| Feature | Description |
|---------|-------------|
| DD Count | Dash dance count (directional oscillations) |
| WD Count | Wave dash count (technical movement) |
| LG Count | Ledge grab count (recovery option uses) |
| Num Grabs | Total grab attempts (action state ID 212) |

### Target Variable
| Feature | Description |
|---------|-------------|
| Character Played | Which character the player used |

## Dataset

- **Total samples:** 9,268 games
- **Source:** Competitive Slippi replays
- **Stage:** Battlefield only (for consistency)
- **Format:** CSV with randomized row order

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Kernel | Linear |
| Regularization (C) | 0.5 |
| Preprocessing | StandardScaler (zero-mean, unit-variance) |
| Train-Test Split | 75-25 |
| Cross-Validation | 10-fold (RBF kernel: C=1, gamma=0.1) |

## Data Extraction Pipeline

```javascript
// toCsv.js using Slippi-js
for each .slp file:
    1. Parse replay with Slippi-js
    2. Extract high-level stats
    3. Iterate frames for action counts
    4. Detect grabs via action state ID 212
    5. Output row to CSV
```

**Grab Detection:** Tracks action state continuity to avoid double-counting consecutive frames in same grab attempt.

## Key Findings

### 1. Characters Have Distinctive Playstyle Signatures
Fox, Marth, Falco, and Jigglypuff exhibit statistically distinguishable patterns in:
- Dash dance frequency
- Grab usage
- Neutral game success rates
- Technical action (wavedash) frequency

### 2. Technical Actions Discriminate Characters
| Action | Discriminative Power |
|--------|---------------------|
| Wave dash count | Favors technical characters (Fox, Marth) |
| Dash dance count | Defensive/footsie playstyle |
| Grab count | Character throw game diversity |

### 3. Linear Separability
Linear kernel worked well, suggesting reasonable linear separability in 13D feature space for these four characters.

### 4. Mixed Feature Types Help
Combining aggregated ratios (conversion rate) with raw action counts provides both strategic and mechanical signal.

## Limitations

- **Single stage:** Battlefield only limits generalization
- **Top-tier only:** Competitive player data may not represent casual play
- **Binary win:** 4-kill threshold may miss performance nuance
- **4 characters:** Limited to Fox, Marth, Falco, Jigglypuff

## Technical Stack

| Component | Technology |
|-----------|------------|
| Replay Parsing | Slippi-js (JavaScript) |
| Runtime | Node.js |
| ML Framework | scikit-learn (Python) |
| SVM | sklearn.svm.SVC |
| Data Format | CSV, pandas |

## Project Files

| File | Purpose |
|------|---------|
| `toCsv.js` | Feature extraction from replays |
| `training_testing_slippi.py` | SVM training and evaluation |
| `CSV RANDOMIZED.csv` | Dataset (9,268 samples) |

## Relevance to ExPhil

### Comparison Point
- ExPhil uses deep learning for full game state; this shows simpler features work for classification
- Demonstrates statistical feature extraction from Slippi replays
- Validates that character-specific patterns exist

### Feature Engineering Lessons
- Action counts (DD, WD, grabs) capture character-specific tech
- Ratio metrics (conversion rate) capture strategic patterns
- Mix of mechanical and strategic features provides signal

### Baseline Value
- Could serve as classification baseline for ExPhil benchmarks
- Shows what's achievable without temporal/sequence modeling

### Why Deep Learning is Better for Gameplay
| SVM Approach | Deep Learning (ExPhil) |
|--------------|------------------------|
| Static feature aggregates | Dynamic frame sequences |
| Post-hoc classification | Real-time action prediction |
| 13 hand-crafted features | 408-1204 dim embeddings |
| Character identification | Controller output |

## Key Insight

This project demonstrates that Melee characters have statistically distinguishable playstyles even with simple aggregate features. The success of linear SVM suggests these patterns are relatively clean in feature space - but gameplay AI needs temporal modeling that static classification cannot provide.

## Links

- [GitHub Repository](https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee)
- [Slippi-js](https://github.com/project-slippi/slippi-js)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
