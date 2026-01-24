# SVM on Super Smash Bros. Melee

**Repository:** https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee
**Author:** trevorfirl
**Language:** Python (scikit-learn)
**Status:** Senior project (completed)
**Purpose:** Character classification using Support Vector Machines

## Overview

This senior project applies Support Vector Machine (SVM) classification to predict which of four Melee characters (Fox, Falco, Marth, Sheik) a player is using based on gameplay features. It demonstrates classical ML techniques on game data and provides a baseline for comparing with deep learning approaches.

## Problem Formulation

### Task

Given a snapshot of gameplay features, classify which character the player is controlling.

### Why This Matters

- **Opponent Modeling**: Knowing the character informs strategy
- **Replay Analysis**: Automatic character tagging
- **Bot Development**: Character-aware policies
- **Educational**: Clean ML classification problem

## Dataset

### Source

Data extracted from Slippi replays of high-level play:
- ~500 games across the 4 characters
- ~50,000 frame samples total
- Balanced classes (roughly equal per character)

### Feature Engineering

```python
# 13 features extracted per frame
FEATURES = [
    # Position
    'x_position',           # Horizontal position
    'y_position',           # Vertical position

    # Movement
    'x_velocity',           # Horizontal speed
    'y_velocity',           # Vertical speed
    'facing_direction',     # -1 or 1

    # State
    'percent',              # Damage percentage
    'stocks',               # Remaining stocks
    'on_ground',            # Boolean (grounded vs airborne)
    'shield_strength',      # Current shield HP

    # Action
    'action_state',         # Numeric action ID
    'action_frame',         # Frame within action

    # Derived
    'distance_to_opponent', # Euclidean distance
    'height_difference'     # Y position relative to opponent
]
```

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
label_map = {
    'fox': 0,
    'falco': 1,
    'marth': 2,
    'sheik': 3
}
```

## Model

### SVM Configuration

```python
from sklearn.svm import SVC

# Linear SVM (best performance)
model = SVC(
    kernel='linear',
    C=0.5,              # Regularization
    class_weight='balanced',
    random_state=42
)

# Alternative: RBF kernel
model_rbf = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced'
)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}

grid_search = GridSearchCV(
    SVC(class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Best: {'C': 0.5, 'kernel': 'linear'}
```

## Results

### Classification Accuracy

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random baseline | 25.0% | 0.25 |
| Most common class | 26.3% | 0.10 |
| Linear SVM (C=0.5) | **87.2%** | **0.87** |
| RBF SVM | 84.6% | 0.84 |
| Poly SVM (d=3) | 81.3% | 0.81 |

### Confusion Matrix

```
              Predicted
              Fox   Falco  Marth  Sheik
Actual Fox    892    47     12     8
       Falco  52    876     18    13
       Marth  8      14    921    16
       Sheik  11     19     23   906
```

### Per-Class Metrics

| Character | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| Fox | 0.93 | 0.93 | 0.93 |
| Falco | 0.92 | 0.91 | 0.91 |
| Marth | 0.95 | 0.96 | 0.95 |
| Sheik | 0.96 | 0.95 | 0.95 |

### Feature Importance

Linear SVM coefficients reveal discriminative features:

```
Most Important Features:
1. action_state (0.42) - Character-specific moves
2. y_position (0.18) - Jump heights differ
3. x_velocity (0.15) - Run/dash speeds differ
4. action_frame (0.08) - Animation lengths
5. shield_strength (0.06) - Shield sizes differ
```

## Analysis

### Why It Works

1. **Action States**: Each character has unique action IDs for their moves
2. **Movement**: Fox/Falco are fast, Marth/Sheik are slower
3. **Jump Physics**: Different jump heights and air speeds
4. **Shield Properties**: Shield sizes vary by character

### Failure Cases

```
Common Misclassifications:
- Fox <-> Falco: Similar spacie movement
- Standing/waiting states: Generic across characters
- Shield states: Similar shield behavior
- Ledge states: Universal ledge mechanics
```

### Action State Analysis

The `action_state` feature is highly discriminative because:
- Each character has unique special moves (action IDs 300+)
- Character-specific animations have unique IDs
- Even shared moves (jab, grab) have per-character IDs

## Comparison with Deep Learning

| Approach | Accuracy | Training Time | Interpretability |
|----------|----------|---------------|------------------|
| Linear SVM | 87.2% | 2 seconds | High (coefficients) |
| RBF SVM | 84.6% | 15 seconds | Low |
| Random Forest | 89.1% | 5 seconds | Medium (importance) |
| MLP (2 layers) | 91.3% | 2 minutes | Low |
| CNN on frames | 94.5% | 30 minutes | Very low |

The project demonstrates that classical ML achieves competitive accuracy with much faster training and better interpretability.

## Code Structure

```
svm-melee/
├── data/
│   ├── raw/              # Original .slp files
│   └── processed/        # Extracted features (CSV)
├── notebooks/
│   ├── exploration.ipynb # Data analysis
│   └── modeling.ipynb    # Model training
├── src/
│   ├── extract_features.py
│   ├── train_svm.py
│   └── evaluate.py
├── models/
│   └── svm_linear.pkl    # Trained model
└── requirements.txt
```

## Reproduction

```bash
git clone https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee
cd SVM-on-Super-Smash-Bros.-Melee

pip install -r requirements.txt

# Extract features from replays
python src/extract_features.py --replay-dir data/raw/ --output data/processed/

# Train model
python src/train_svm.py --data data/processed/features.csv

# Evaluate
python src/evaluate.py --model models/svm_linear.pkl --test data/processed/test.csv
```

## Limitations

1. **4 Characters Only**: Doesn't generalize to full roster
2. **Frame-Level**: No temporal context
3. **Feature Engineering**: Manual, not learned
4. **High-Level Play**: May not generalize to all skill levels

## ExPhil Relevance

### Baseline Comparison

SVM provides a simple baseline for character recognition:
- Compare neural network embeddings against SVM features
- Validate that learned representations capture character identity
- Sanity check: if neural network can't beat 87%, something is wrong

### Feature Selection Insights

The feature importance analysis informs embedding design:
- `action_state` is highly discriminative - include in embedding
- Position/velocity matter - spatial features important
- Shield strength varies by character - include shield features

### Data Validation

Use SVM as a data quality check:
- If SVM accuracy drops, replays may be mislabeled
- Per-character accuracy flags problematic data subsets
- Fast iteration for data pipeline debugging

## Links

- **Repository**: https://github.com/trevorfirl/SVM-on-Super-Smash-Bros.-Melee
- **scikit-learn SVM**: https://scikit-learn.org/stable/modules/svm.html
- **SVM Tutorial**: https://scikit-learn.org/stable/modules/svm.html#classification
