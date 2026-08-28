# Coach Report — 20260828_162810

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 7 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 9 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.98 | 7 | 2 | 9 | 9 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:386 — opening@386 start_pct=5.0 window=120f
- **dropped_punish** r3.slp:1006 — opening@1006 start_pct=20.2 window=120f
- **dropped_punish** r3.slp:1350 — opening@1350 start_pct=31.6 window=120f
- **dropped_punish** r3.slp:2074 — opening@2074 start_pct=80.4 window=120f
- **dropped_punish** r3.slp:2874 — opening@2874 start_pct=116.8 window=120f
- **dropped_punish** r3.slp:3546 — opening@3546 start_pct=5.0 window=120f
- **dropped_punish** r3.slp:4305 — opening@4305 start_pct=24.7 window=120f
- **dropped_punish** r3.slp:6395 — opening@6395 start_pct=114.5 window=120f
- **dropped_punish** r3.slp:7032 — opening@7032 start_pct=138.3 window=120f
- **neutral_loss** r3.slp:744 — hit@834 opener=p2_action44 dist_90f_prior=38.9

Appended 10 new gap(s) to `scenarios/gaps.json` (981 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
