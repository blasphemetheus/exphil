# SVM on SSBM Case Study

**Repository**: https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee
**Author**: trevorfirl
**Status**: Complete (Senior Project)
**Institution**: Winona State University
**Date**: July 2021

## Overview

A senior capstone project applying **Support Vector Machine (SVM)** classification to Melee replay data. Demonstrates how classical ML can extract meaningful patterns from competitive gameplay.

## Goals

Classify characters based on gameplay statistics extracted from Slippi replays.

## Architecture

### Pipeline

```
.slp Replay Files
    ↓
slippi-js (toCsv.js)
    ↓
CSV Dataset
    ↓
Python SVM Training
    ↓
Character Classification
```

### Language Split

- **JavaScript (53.3%)**: Replay processing
- **Python (46.7%)**: SVM training/evaluation

## Feature Extraction

**13 features per game**:

| Feature | Description |
|---------|-------------|
| Character ID | Player's character |
| Opponent ID | Opponent's character |
| IPM | Inputs Per Minute |
| Win/Loss | Binary outcome |
| Game Length | Duration in frames |
| Openings Per Kill | Offensive efficiency |
| Damage Per Opening | Punish quality |
| Neutral Win Ratio | Neutral game success |
| Opening Conversion | Advantage conversion |
| Dash Dance Count | Movement technique usage |
| Wavedash Count | Advanced tech usage |
| Ledge Grab Count | Recovery interactions |
| Grab Count | Derived from action state 212 |

## SVM Configuration

### Linear Kernel

```python
svm = SVC(kernel='linear', C=0.5)
```

### RBF Kernel (Alternative)

```python
svm = SVC(kernel='rbf', C=1, gamma=0.1)
```

### Validation

- 10-fold cross-validation
- Confusion matrix analysis
- Per-class accuracy

## Classification Target

**4 character classes**:
- Fox (0)
- Marth (1)
- Falco (2)
- Jigglypuff (3)

## Code Structure

```
SVM-on-Super-Smash-Bros.-Melee/
├── toCsv.js                    # Replay → CSV
├── training_testing_slippi.py  # SVM training
├── CSV RANDOMIZED.csv          # Dataset
└── README.md
```

### toCsv.js

```javascript
const { SlippiGame } = require('@slippi/slippi-js');

// Iterate replays
for (const file of replayFolder) {
    const game = new SlippiGame(file);
    const stats = game.getStats();
    // Extract features, write CSV row
}
```

### training_testing_slippi.py

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load CSV
X, y = load_data('CSV RANDOMIZED.csv')

# Train SVM
model = SVC(kernel='linear', C=0.5)
scores = cross_val_score(model, X, y, cv=10)
```

## Key Design Decisions

1. **Classical ML**: SVM for interpretability
2. **Domain features**: IPM, neutral ratio, tech counts
3. **Character focus**: "Low-tier" characters (including Fox/Falco)
4. **slippi-js**: Official SDK for replay parsing

## Relevance to ExPhil

**Not directly applicable** - Classification, not gameplay AI.

**However**:
- **Feature engineering**: Domain-informed features (neutral ratio, opening conversion) are useful
- **Character signatures**: Different characters have detectable statistical profiles
- **slippi-js patterns**: Shows how to extract stats from replays

**Potential applications**:
- Character-specific training data filtering
- Skill level estimation from stats
- Matchup analysis

## References

- [Repository](https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee)
- [slippi-js](https://github.com/project-slippi/slippi-js)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
