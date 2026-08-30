# Coach Report — 20260829_231256

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 0 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.0 | 0 | 0 | 7 | 3 | 0 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:3182 — passive_run=396f mid_dist=6.9
- **passivity_window** r7.slp:3839 — passive_run=553f mid_dist=6.7
- **passivity_window** r7.slp:5156 — passive_run=569f mid_dist=8.3
- **passivity_window** r7.slp:6059 — passive_run=363f mid_dist=15.7
- **passivity_window** r7.slp:6514 — passive_run=763f mid_dist=8.3
- **dropped_punish** r7.slp:1331 — opening@1331 start_pct=90.5 window=120f
- **dropped_punish** r7.slp:3021 — opening@3021 start_pct=177.4 window=120f
- **dropped_punish** r7.slp:5025 — opening@5025 start_pct=260.6 window=120f
- **neutral_loss** r7.slp:399 — hit@489 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r7.slp:1496 — hit@1586 opener=p2_action53 dist_90f_prior=19.1

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
