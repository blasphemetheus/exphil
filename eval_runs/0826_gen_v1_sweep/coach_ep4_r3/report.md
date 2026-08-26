# Coach Report — 20260826_125355

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 2 | 1 | 9 | 5 | 1 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:4786 — opening@4786 start_pct=73.3 window=120f
- **dropped_punish** r3.slp:5026 — opening@5026 start_pct=79.8 window=120f
- **dropped_punish** r3.slp:5335 — opening@5335 start_pct=84.8 window=120f
- **dropped_punish** r3.slp:6186 — opening@6186 start_pct=125.4 window=120f
- **dropped_punish** r3.slp:6794 — opening@6794 start_pct=143.8 window=120f
- **neutral_loss** r3.slp:384 — hit@474 opener=p2_action44 dist_90f_prior=11.4
- **neutral_loss** r3.slp:2300 — hit@2390 opener=p2_action44 dist_90f_prior=85.4
- **neutral_loss** r3.slp:2690 — hit@2780 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:4194 — hit@4284 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r3.slp:5071 — hit@5161 opener=p2_action57 dist_90f_prior=9.1

Appended 10 new gap(s) to `scenarios/gaps.json` (345 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
