# Coach Report — 20260827_181714

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.51 |
| Approaches (total) | 5 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 7 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.51 | 5 | 0 | 11 | 7 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r5.slp:1830 — opening@1830 start_pct=78.2 window=120f
- **dropped_punish** r5.slp:2188 — opening@2188 start_pct=102.3 window=120f
- **dropped_punish** r5.slp:3107 — opening@3107 start_pct=111.5 window=120f
- **dropped_punish** r5.slp:3641 — opening@3641 start_pct=116.0 window=120f
- **dropped_punish** r5.slp:3910 — opening@3910 start_pct=131.0 window=120f
- **dropped_punish** r5.slp:5455 — opening@5455 start_pct=15.1 window=120f
- **dropped_punish** r5.slp:6191 — opening@6191 start_pct=24.6 window=120f
- **neutral_loss** r5.slp:719 — hit@809 opener=p2_action44 dist_90f_prior=17.4
- **neutral_loss** r5.slp:1272 — hit@1362 opener=p2_action60 dist_90f_prior=20.1
- **neutral_loss** r5.slp:1865 — hit@1955 opener=p2_action53 dist_90f_prior=54.9

Appended 10 new gap(s) to `scenarios/gaps.json` (624 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
