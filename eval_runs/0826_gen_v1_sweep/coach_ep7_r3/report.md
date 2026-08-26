# Coach Report — 20260826_131842

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 9 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 13 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.98 | 9 | 1 | 13 | 2 | 1 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:1714 — opening@1714 start_pct=138.7 window=120f
- **dropped_punish** r3.slp:3474 — opening@3474 start_pct=186.7 window=120f
- **neutral_loss** r3.slp:321 — hit@411 opener=p2_action44 dist_90f_prior=6.3
- **neutral_loss** r3.slp:987 — hit@1077 opener=p2_action44 dist_90f_prior=40.8
- **neutral_loss** r3.slp:1773 — hit@1863 opener=p2_action57 dist_90f_prior=57.5
- **neutral_loss** r3.slp:2535 — hit@2625 opener=p2_action63 dist_90f_prior=33.6
- **neutral_loss** r3.slp:2820 — hit@2910 opener=p2_action57 dist_90f_prior=22.8
- **neutral_loss** r3.slp:3141 — hit@3231 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:3652 — hit@3742 opener=p2_action60 dist_90f_prior=21.6
- **neutral_loss** r3.slp:4091 — hit@4181 opener=p2_action44 dist_90f_prior=12.3

Appended 10 new gap(s) to `scenarios/gaps.json` (435 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
