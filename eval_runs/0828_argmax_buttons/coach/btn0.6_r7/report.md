# Coach Report — 20260828_162821

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 9 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.0 | 2 | 1 | 7 | 9 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:2336 — passive_run=353f mid_dist=8.3
- **dropped_punish** r7.slp:325 — opening@325 start_pct=18.0 window=120f
- **dropped_punish** r7.slp:817 — opening@817 start_pct=54.3 window=120f
- **dropped_punish** r7.slp:1069 — opening@1069 start_pct=58.8 window=120f
- **dropped_punish** r7.slp:3452 — opening@3452 start_pct=161.4 window=120f
- **dropped_punish** r7.slp:4699 — opening@4699 start_pct=190.9 window=120f
- **dropped_punish** r7.slp:5357 — opening@5357 start_pct=206.8 window=120f
- **dropped_punish** r7.slp:6673 — opening@6673 start_pct=24.2 window=120f
- **dropped_punish** r7.slp:6997 — opening@6997 start_pct=28.4 window=120f
- **dropped_punish** r7.slp:7279 — opening@7279 start_pct=32.2 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (1061 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
