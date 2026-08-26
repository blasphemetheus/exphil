# Coach Report — 20260826_131029

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 4 | 1 | 9 | 5 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:6706 — passive_run=406f mid_dist=8.3
- **dropped_punish** r3.slp:1988 — opening@1988 start_pct=115.2 window=120f
- **dropped_punish** r3.slp:2280 — opening@2280 start_pct=154.6 window=120f
- **dropped_punish** r3.slp:3387 — opening@3387 start_pct=5.0 window=120f
- **dropped_punish** r3.slp:3980 — opening@3980 start_pct=13.7 window=120f
- **dropped_punish** r3.slp:4854 — opening@4854 start_pct=62.4 window=120f
- **neutral_loss** r3.slp:571 — hit@661 opener=p2_action53 dist_90f_prior=34.8
- **neutral_loss** r3.slp:903 — hit@993 opener=p2_action60 dist_90f_prior=40.0
- **neutral_loss** r3.slp:1223 — hit@1313 opener=p2_action44 dist_90f_prior=6.5
- **neutral_loss** r3.slp:2797 — hit@2887 opener=p2_action44 dist_90f_prior=26.9

Appended 10 new gap(s) to `scenarios/gaps.json` (405 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
