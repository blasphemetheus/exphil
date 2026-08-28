# Coach Report — 20260828_171135

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.35 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 15 |
| Dropped punishes | 4 |
| Death sequences | 4 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T170250.slp | 0.35 | 7 | 1 | 15 | 4 | 4 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T170250.slp:6912 — passive_run=702f mid_dist=11.5
- **dropped_punish** Game_20260828T170250.slp:3468 — opening@3468 start_pct=67.5 window=120f
- **dropped_punish** Game_20260828T170250.slp:3725 — opening@3725 start_pct=76.5 window=120f
- **dropped_punish** Game_20260828T170250.slp:7227 — opening@7227 start_pct=146.8 window=120f
- **dropped_punish** Game_20260828T170250.slp:8722 — opening@8722 start_pct=159.7 window=120f
- **neutral_loss** Game_20260828T170250.slp:327 — hit@417 opener=p2_action66 dist_90f_prior=43.4
- **neutral_loss** Game_20260828T170250.slp:702 — hit@792 opener=p2_action66 dist_90f_prior=12.1
- **neutral_loss** Game_20260828T170250.slp:2507 — hit@2597 opener=p2_action69 dist_90f_prior=57.8
- **neutral_loss** Game_20260828T170250.slp:3818 — hit@3908 opener=p2_action347 dist_90f_prior=13.7
- **neutral_loss** Game_20260828T170250.slp:4399 — hit@4489 opener=p2_action66 dist_90f_prior=9.1

Appended 10 new gap(s) to `scenarios/gaps.json` (1141 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
