# Coach Report — 20260827_182847

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 8 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.49 | 3 | 0 | 11 | 8 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r1.slp:693 — opening@693 start_pct=91.2 window=120f
- **dropped_punish** r1.slp:1805 — opening@1805 start_pct=120.2 window=120f
- **dropped_punish** r1.slp:2142 — opening@2142 start_pct=128.9 window=120f
- **dropped_punish** r1.slp:3412 — opening@3412 start_pct=150.6 window=120f
- **dropped_punish** r1.slp:4178 — opening@4178 start_pct=194.6 window=120f
- **dropped_punish** r1.slp:5031 — opening@5031 start_pct=28.2 window=120f
- **dropped_punish** r1.slp:5271 — opening@5271 start_pct=32.2 window=120f
- **dropped_punish** r1.slp:6488 — opening@6488 start_pct=85.9 window=120f
- **neutral_loss** r1.slp:444 — hit@534 opener=p2_action60 dist_90f_prior=4.2
- **neutral_loss** r1.slp:806 — hit@896 opener=p2_action44 dist_90f_prior=60.5

Appended 10 new gap(s) to `scenarios/gaps.json` (634 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
