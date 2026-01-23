# Fall 2024 Melee Prediction Case Study

**Repository**: https://github.com/jjaw89/fall-2024-smash-melee-prediction
**Authors**: Dan Ursu, Jaspar Wiart
**Status**: Active (2024)
**Context**: Erdos Institute Data Science Boot Camp

## Overview

A capstone project predicting Melee tournament outcomes using XGBoost. Demonstrates that intelligent feature engineering (character-adjusted ELO) provides measurable improvements over baseline Glicko-2 approaches.

## Prediction Goals

1. **Single-Set Prediction**: Who wins a match between two players?
2. **Tournament Winner**: Given top 8, who takes the tournament?

## Data Source

**ThePlayerDatabase** SQLite:
- 96,689 players
- 39,675 tournaments
- 1,795,681 sets (2015-2024)

### Data Processing

- Tournament classification via Liquipedia
- JSON parsing for nested structures
- Unix timestamp conversion
- Player randomization (eliminated 70% "player 1" bias)
- Pickle serialization

## Feature Engineering

### Glicko-2 Implementation

```python
# Weekly updates 2015-01-01 to 2024-12-31
initial_rating = 1500
initial_rd = 350
min_rating = 500
max_rd = 350
```

Includes "did not compete" decay for inactive weeks.

### Character-Adjusted ELO Variants

| Variant | Description |
|---------|-------------|
| Default | Baseline ELO |
| Alt ELO | Character-inclusive |
| Alt2 ELO | Alternative character handling |
| Alt3 ELO | Third variant |

**Key insight**: Treats player-character combinations as separate entities (e.g., "Mango-Fox" vs "Mango-Falco").

### Single-Set Features (30 total)

- Default ELO (both players)
- Alt ELO variants (m1_alt2, m1_alt3)
- Rating deviation (RD)
- Update counts
- Primary character usage %
- Head-to-head history (10-match records)

### Top-8 Features

- Tournament momentum
- Pairwise win probabilities
- Relative performance vs. odds

## XGBoost Model

### Single-Set Configuration

**Training**: 2016-2022
**Test**: 2024
**No shuffling** (preserve temporal ELO relationships)

**Optimized hyperparameters** (Optuna):
```python
max_depth: 8
learning_rate: 0.011288
n_estimators: 850
subsample: 0.60077
colsample_bytree: 0.98525
gamma: 5
min_child_weight: 10
```

### Top-8 Configuration

- Training: 2023 only (prevent leakage)
- Uses single-set predictor as foundation
- Computes pairwise matchup probabilities

## Results

### Single-Set Prediction (2024, n=247,608)

| Model | All Sets | Top 8 Sets |
|-------|----------|-----------|
| Higher ELO baseline | 77.56% | 73.89% |
| XGBoost (ELO only) | 79.05% | 74.04% |
| XGBoost (all features) | **79.89%** | **75.03%** |

### Top-8 Winner Prediction

| Model | Accuracy |
|-------|----------|
| Highest ELO in top 8 | 67.6% |
| Highest ELO in winners' semis | 70.2% |
| XGBoost (engineered) | 70.0% |
| XGBoost (combined) | **70.1%** |

**Key finding**: Character features provide modest gains (~0.84% single-set, ~2.5% top-8).

## Code Structure

```
├── preprocessing/
│   ├── preprocessing_0_extract_data.ipynb
│   ├── preprocessing_1_majors.ipynb
│   ├── preprocessing_2_top_8.ipynb
│   └── preprocessing_3_top_8_previous_sets.ipynb
├── feature_engineering/
│   ├── compute_elos_for_splits.py
│   ├── default_elo.ipynb
│   ├── engineered_elo_alt.ipynb
│   ├── engineered_elo_alt2.ipynb
│   └── engineered_elo_alt3.ipynb
├── dataset_generation/
│   └── dataset_generator.ipynb
├── models/
│   ├── single_set_model.ipynb
│   └── top_8_model.ipynb
└── experimental/
```

## Key Insights

1. **ELO dominates**: Default ratings are top predictors
2. **Character matters less than expected**: Modest improvement
3. **Tournament outcomes involve luck**: Top-8 harder than single-set
4. **Temporal constraints**: Don't shuffle ELO data

## Relevance to ExPhil

**Not directly applicable** - Match prediction, not gameplay AI.

**However**:
- **Player skill quantification**: Could weight training data by ELO
- **Matchup analysis**: Character-adjusted ratings inform matchup difficulty
- **Data source**: ThePlayerDatabase useful for player filtering

## References

- [Repository](https://github.com/jjaw89/fall-2024-smash-melee-prediction)
- [Erdos Institute](https://www.erdosinstitute.org/)
- [Glicko-2](http://www.glicko.net/glicko/glicko2.pdf)
- [XGBoost](https://xgboost.readthedocs.io/)
