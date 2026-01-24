# Fall 2024 Melee Prediction

**Repository:** https://github.com/jjaw89/fall-2024-smash-melee-prediction
**Author:** jjaw89
**Language:** Python (scikit-learn, XGBoost)
**Status:** 2024 project
**Purpose:** Tournament match outcome prediction using ELO/Glicko-2 and machine learning

## Overview

This project predicts Super Smash Bros. Melee tournament match outcomes using a combination of rating systems (Glicko-2) and gradient boosted trees (XGBoost). It achieved 79.89% accuracy on single-set predictions and 70.1% on top-8 winner predictions.

## Methodology

### Data Collection

```
Data Sources:
├── start.gg API
│   ├── Tournament results
│   ├── Set scores
│   ├── Player identifiers
│   └── Character selections (when available)
├── Liquipedia
│   └── Historical rankings
└── slippi.gg
    └── Ranked ladder data
```

### Rating Systems

#### Glicko-2 Implementation

```python
# Glicko-2 parameters
INITIAL_RATING = 1500
INITIAL_RD = 350  # Rating deviation
INITIAL_VOLATILITY = 0.06
TAU = 0.5  # System constant

def update_ratings(player1, player2, result):
    """
    Update Glicko-2 ratings after a match.

    result: 1.0 = player1 wins, 0.0 = player2 wins, 0.5 = draw
    """
    # Convert to Glicko-2 scale
    mu1 = (player1.rating - 1500) / 173.7178
    mu2 = (player2.rating - 1500) / 173.7178
    phi1 = player1.rd / 173.7178
    phi2 = player2.rd / 173.7178

    # Calculate expected score
    g_phi2 = 1 / sqrt(1 + 3 * phi2**2 / pi**2)
    E = 1 / (1 + exp(-g_phi2 * (mu1 - mu2)))

    # Update rating, RD, and volatility
    # (Full Glicko-2 algorithm)
    ...
```

#### Weekly Rating Updates

Ratings are updated weekly to:
- Account for activity decay (RD increases when inactive)
- Smooth out variance from single tournaments
- Allow new information to propagate

### Feature Engineering

```python
FEATURES = [
    # Rating features
    'player1_rating',
    'player2_rating',
    'rating_diff',
    'player1_rd',
    'player2_rd',
    'rd_diff',

    # Historical features
    'player1_win_rate_30d',
    'player2_win_rate_30d',
    'head_to_head_wins_p1',
    'head_to_head_total',

    # Character features (when available)
    'player1_main_char',
    'player2_main_char',
    'matchup_historical_wr',

    # Tournament context
    'tournament_tier',  # Major, regional, local
    'bracket_round',    # Winners, losers, grands
    'is_bo5',

    # Momentum features
    'player1_streak',
    'player2_streak',
    'player1_recent_upsets',
    'player2_recent_upsets'
]
```

### Model Architecture

#### XGBoost Configuration

```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20
)
```

#### Training Process

```
Training Pipeline:
1. Collect tournament data (2019-2024)
2. Build Glicko-2 rating history
3. Generate features for each set
4. Train/test split (temporal, not random)
5. Hyperparameter tuning with CV
6. Evaluate on held-out tournaments
```

## Results

### Single Set Prediction

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| Random | 50.0% | 0.693 |
| Higher seed | 62.3% | - |
| ELO only | 71.2% | 0.521 |
| Glicko-2 only | 73.8% | 0.498 |
| XGBoost | **79.89%** | **0.412** |

### Tournament Bracket Prediction

| Metric | Accuracy |
|--------|----------|
| Top 8 placements | 70.1% |
| Grand finals entrants | 68.4% |
| Tournament winner | 52.3% |
| Full bracket (8 players) | 12.1% |

### Feature Importance

```
Top 10 Features:
1. rating_diff (0.312)
2. player1_rating (0.156)
3. player2_rating (0.142)
4. head_to_head_wins_p1 (0.089)
5. player1_rd (0.067)
6. tournament_tier (0.054)
7. player1_win_rate_30d (0.048)
8. bracket_round (0.041)
9. is_bo5 (0.035)
10. matchup_historical_wr (0.028)
```

## Character-Adjusted Ratings

The project experimented with character-specific ELO:

```python
# Track separate rating per character
class CharacterRatings:
    def __init__(self, player_id):
        self.overall = GlickoRating()
        self.by_character = {}  # char_id -> GlickoRating

    def get_matchup_rating(self, own_char, opp_char):
        """Blend overall and character-specific ratings"""
        char_rating = self.by_character.get(own_char, self.overall)
        # Weight by games played on character
        games_on_char = char_rating.games_played
        blend = min(games_on_char / 50, 1.0)
        return blend * char_rating + (1 - blend) * self.overall
```

### Character ELO Results

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Overall only | 79.89% | Baseline |
| Character-specific | 78.2% | Insufficient data |
| Blended | 80.1% | Slight improvement |
| Matchup-adjusted | 79.5% | No improvement |

Character-specific ratings didn't significantly improve predictions, likely due to:
- Insufficient games per character for most players
- Top players are consistent across characters
- Character selection often responds to matchup (confounding)

## Upset Analysis

The model identifies upsets (lower-rated player wins):

```
Upset Factors (when lower-rated wins):
- Smaller rating difference: avg 142 vs 287 for expected
- Higher RD for favorite: less certain ratings
- Losers bracket: 8% more upsets than winners
- Bo3 vs Bo5: 12% more upsets in Bo3
- Character counterpick: visible in top 50 only
```

## Limitations

1. **Data Quality**: Character data sparse, especially for non-majors
2. **Meta Shifts**: Model trained on 2019-2024 may not generalize
3. **Player Variance**: Some players are inherently more volatile
4. **Context Missing**: Travel fatigue, motivation, practice partners

## ExPhil Relevance

### Rating System for Self-Play

Glicko-2 could rank AI agents in population-based training:
- Track rating per policy checkpoint
- Use RD to identify uncertain matchups
- Select training opponents based on rating proximity

### Match Prediction for Dataset Curation

- Filter replays to high-skill matches (both players rated)
- Weight training samples by match competitiveness
- Identify "clean" games (rating-expected outcomes)

### Feature Engineering Patterns

- Historical win rate features for opponent modeling
- Momentum/streak features for temporal patterns
- Tournament context as auxiliary prediction targets

## Reproduction

```bash
git clone https://github.com/jjaw89/fall-2024-smash-melee-prediction
cd fall-2024-smash-melee-prediction

# Install dependencies
pip install -r requirements.txt

# Fetch tournament data
python scripts/fetch_startgg.py --start 2019-01-01 --end 2024-12-01

# Build rating history
python scripts/build_ratings.py

# Train model
python scripts/train_model.py

# Evaluate
python scripts/evaluate.py --tournament "Genesis 10"
```

## Links

- **Repository**: https://github.com/jjaw89/fall-2024-smash-melee-prediction
- **Glicko-2 Paper**: http://www.glicko.net/glicko/glicko2.pdf
- **start.gg API**: https://developer.start.gg/docs/intro
- **XGBoost**: https://xgboost.readthedocs.io/
