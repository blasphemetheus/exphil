# Coach Report — 20260829_231235

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 6 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 7 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.49 | 6 | 3 | 9 | 7 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:4339 — passive_run=388f mid_dist=6.9
- **dropped_punish** r1.slp:507 — opening@507 start_pct=14.0 window=120f
- **dropped_punish** r1.slp:1200 — opening@1200 start_pct=51.4 window=120f
- **dropped_punish** r1.slp:1570 — opening@1570 start_pct=68.1 window=120f
- **dropped_punish** r1.slp:1861 — opening@1861 start_pct=73.1 window=120f
- **dropped_punish** r1.slp:3616 — opening@3616 start_pct=131.4 window=120f
- **dropped_punish** r1.slp:3972 — opening@3972 start_pct=4.3 window=120f
- **dropped_punish** r1.slp:7042 — opening@7042 start_pct=136.0 window=120f
- **neutral_loss** r1.slp:542 — hit@632 opener=p2_action44 dist_90f_prior=54.9
- **neutral_loss** r1.slp:1247 — hit@1337 opener=p2_action60 dist_90f_prior=53.4

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
