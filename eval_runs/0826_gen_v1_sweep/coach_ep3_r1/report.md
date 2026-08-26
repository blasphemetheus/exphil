# Coach Report — 20260826_124536

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.17 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 3 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 1.17 | 4 | 1 | 3 | 3 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r1.slp:1073 — opening@1073 start_pct=0.0 window=120f
- **dropped_punish** r1.slp:1409 — opening@1409 start_pct=17.5 window=120f
- **dropped_punish** r1.slp:1758 — opening@1758 start_pct=22.5 window=120f
- **neutral_loss** r1.slp:467 — hit@557 opener=p2_action60 dist_90f_prior=18.9
- **neutral_loss** r1.slp:2419 — hit@2509 opener=p2_action53 dist_90f_prior=40.8
- **neutral_loss** r1.slp:2663 — hit@2753 opener=p2_action44 dist_90f_prior=43.0
- **death_sequence** r1.slp:1757 — death@1839 elapsed=82f opener=p2_action88
- **death_sequence** r1.slp:2154 — death@2155 elapsed=1f opener=p2_action14
- **death_sequence** r1.slp:3008 — death@3077 elapsed=69f opener=p2_action90

Appended 9 new gap(s) to `scenarios/gaps.json` (295 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
