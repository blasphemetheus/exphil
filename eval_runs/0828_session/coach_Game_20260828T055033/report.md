# Coach Report — 20260828_171203

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.52 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T055033.slp | 0.52 | 4 | 0 | 7 | 1 | 4 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T055033.slp:510 — passive_run=474f mid_dist=25.5
- **dropped_punish** Game_20260828T055033.slp:1132 — opening@1132 start_pct=82.8 window=120f
- **neutral_loss** Game_20260828T055033.slp:834 — hit@924 opener=p2_action63 dist_90f_prior=19.1
- **neutral_loss** Game_20260828T055033.slp:1958 — hit@2048 opener=p2_action67 dist_90f_prior=67.5
- **neutral_loss** Game_20260828T055033.slp:2591 — hit@2681 opener=p2_action63 dist_90f_prior=101.8
- **neutral_loss** Game_20260828T055033.slp:3995 — hit@4085 opener=p2_action69 dist_90f_prior=18.5
- **neutral_loss** Game_20260828T055033.slp:5137 — hit@5227 opener=p2_action69 dist_90f_prior=50.4
- **neutral_loss** Game_20260828T055033.slp:5557 — hit@5647 opener=p2_action69 dist_90f_prior=40.7
- **neutral_loss** Game_20260828T055033.slp:6506 — hit@6596 opener=p2_action213 dist_90f_prior=14.7
- **death_sequence** Game_20260828T055033.slp:2855 — death@2856 elapsed=1f opener=p2_action20

Appended 10 new gap(s) to `scenarios/gaps.json` (1259 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
