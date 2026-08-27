# Coach Report — 20260827_182850

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 1 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 2 | 0 | 1 | 2 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r5.slp:312 — opening@312 start_pct=22.5 window=120f
- **dropped_punish** r5.slp:758 — opening@758 start_pct=39.9 window=120f
- **neutral_loss** r5.slp:793 — hit@883 opener=p2_action60 dist_90f_prior=54.7
- **death_sequence** r5.slp:1023 — death@1024 elapsed=1f opener=p2_action16
- **death_sequence** r5.slp:1590 — death@1591 elapsed=1f opener=p2_action48
- **death_sequence** r5.slp:1854 — death@1855 elapsed=1f opener=p2_action14
- **death_sequence** r5.slp:2162 — death@2163 elapsed=1f opener=p2_action246

Appended 7 new gap(s) to `scenarios/gaps.json` (671 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
