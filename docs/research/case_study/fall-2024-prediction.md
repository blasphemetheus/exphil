# Fall 2024 Smash Melee Prediction

**Repository:** https://github.com/jjaw89/fall-2024-smash-melee-prediction
**Authors:** Erdos Institute Data Science Boot Camp team
**Language:** Python
**Status:** Complete (2024)
**Purpose:** Tournament match outcome prediction using machine learning

## Overview

A capstone project from the Erdos Institute's Data Science Boot Camp (Fall 2024) that predicts Super Smash Bros. Melee tournament outcomes using XGBoost with sophisticated feature engineering. Achieved 79.89% accuracy on single-set predictions.

## Algorithm

**Primary Model:** XGBoost (Gradient Boosting)
- Python implementation using xgboost >= 2.1.1
- Emphasis on feature engineering over model complexity
- Parallel processing for scalability

## Features Used

### Core Rating System
- **Glicko-2 ratings** - Extended ELO with rating deviation (RD) for confidence
- **Character-specific matchup analysis** (three variants):
  - Alt: Basic character adjustments
  - Alt2: Character-adjusted ELO for specific pairings
  - Alt3: Unique (player, character) combinations as separate entities

### Additional Features
| Feature | Description |
|---------|-------------|
| Rating Deviation (RD) | Measurement uncertainty from Glicko-2 |
| Head-to-head history | Results from last 10 matches between players |
| Update frequency | Proxy for player activity level |
| Tournament momentum | Probability of exceeding expected performance |

## Dataset

**Source:** The Player Database
- 1.8M sets total
- 96K+ players
- 39K+ tournaments

**Preprocessing:**
- SQLite database extraction
- Data cleaning for missing entries
- Player 1/2 order randomization (bias correction)

## Accuracy Results

### Single-Set Predictions (2024 test data)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Baseline (higher ELO wins) | 77.56% ± 0.16% | Simple heuristic |
| XGBoost + ELO only | 79.05% ± 0.16% | No feature engineering |
| **XGBoost + all features** | **79.89% ± 0.16%** | Best result |
| Top 8 sets only | 75.03% ± 0.35% | Harder to predict |

**Improvement:** +2.33% over baseline (statistically significant on 247,608 test samples)

### Top 8 Winner Predictions

| Model | Accuracy |
|-------|----------|
| Baseline (highest ELO in top 8) | 67.6% ± 1.3% |
| Baseline (highest ELO in winners' semis) | 70.2% ± 1.3% |
| XGBoost + features | 70.1% ± 1.3% |

Strong baseline limits ML improvement potential.

## Methodology

### Pipeline Stages
1. **Preprocessing** - Extract tournament data, clean entries, randomize order
2. **Feature Engineering** - Compute Glicko-2 across data splits
3. **Dataset Generation** - Combine preprocessed data with features
4. **Model Training** - XGBoost with cross-validation

### Training Setup
- Python 3.10.12
- Jupyter notebooks for development
- glicko2==2.1.0 for rating calculations
- Docker for reproducibility

## Key Insights

### 1. Feature Engineering > Model Complexity
Success came from domain-specific features (character matchups, rating deviation), not algorithmic sophistication. Domain expertise drives accuracy.

### 2. Character Matchups Matter
Character-specific ELO variants (alt2/alt3) provided consistent improvements. Player skill is character-dependent.

### 3. Baseline Saturation
Top 8 predictions hit diminishing returns (~70%) where simple ELO was already strong (67.6%). Basic heuristics capture significant variance.

### 4. History Has Limited Impact
Head-to-head from last 10 matches helped modestly. For gameplay AI, frame-by-frame decisions likely matter more than historical aggregates.

### 5. Uncertainty Estimation Helps
Including Glicko-2's rating deviation improved predictions. Could inform curriculum learning or sample weighting in AI training.

## Comparison: Match Prediction vs Gameplay AI

| Aspect | Match Prediction | Gameplay AI (ExPhil) |
|--------|------------------|----------------------|
| Input | Player ratings, history | Frame-by-frame state |
| Features | ~10-20 engineered | 408-1204 dims |
| Prediction | Win/loss outcome | Controller actions |
| Temporal | Static snapshot | 60 FPS sequence |
| Data scale | 1.8M match sets | 100K+ replay files |

## Relevance to ExPhil

### Shared Insights
- Large datasets essential for meaningful gains
- Character matchups significantly impact outcomes
- Uncertainty quantification valuable for learning

### Key Differences
- Match prediction uses aggregated statistics; gameplay AI needs frame-level decisions
- Static features vs dynamic temporal sequences
- Outcome prediction vs action generation

### Potential Applications
- Glicko-2 ratings could weight training samples by player skill
- Character matchup analysis could inform character-specific models
- Tournament momentum metrics could identify high-variance situations

## Technical Stack

| Component | Technology |
|-----------|------------|
| ML Framework | XGBoost, scikit-learn |
| Rating System | glicko2 package |
| Data Processing | pandas, numpy |
| Environment | Python 3.10.12, Docker |
| Development | Jupyter notebooks |

## Links

- [GitHub Repository](https://github.com/jjaw89/fall-2024-smash-melee-prediction)
- [The Player Database](https://theplayerdatabase.com/)
- [Glicko-2 Paper](http://www.glicko.net/glicko/glicko2.pdf)
