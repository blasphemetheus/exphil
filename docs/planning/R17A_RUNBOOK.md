# r17a Runbook — copy-paste launch (prepared 2026-07-23)

Everything r17a needs is staged. **This is a launch instruction, not an
auto-launch** — Bradley runs it when ready (deliberately not started
before Navarre).

Read alongside `R17_ACCEPTANCE_2026-07-23.md` (what "worked" means).

## What r17a is

r16's verdict: BC did NOT move gate-10 (armed approaches/min 0.17, target
≥1.0) — initiation isn't coming from more imitation *as previously fed*.
r17a is the first rung of the ESCALATE branch: keep imitation, but aim it
squarely at the going-in decision using the four levers built 2026-07-23.
If r17a still doesn't move armed/min ≥0.5, that's the decisive signal to
take the OTHER r17 branch (PPO approach-shaping).

## Deltas from r16 (this is the whole experiment)

| Lever | r16 | r17a |
|-------|-----|------|
| Opener weighting | none | `--opener-weight 3.0` (NEW — the gate-10 lever) |
| BC selection | `--bc-sample 20` RANDOM (mean armed/min ≈3.9) | curated top-20 (mean ≈5.54) |
| Probe-reg | set but a **silent no-op** (4 shield rows, `refit:false`) | actually applies (fixed) |
| Style conditioning | none (all Mewtwos averaged) | `--learn-player-styles` (9 distinct netplay Mewtwos) |

Everything else matches r16 so the comparison is clean.

## Pre-staged artifacts

- **Curated BC list:** `corpus/curated/r17a_bc.txt` (20 games, comma-joined,
  ready for `--bc-replays`). Regenerate any time with:
  `mix run scripts/curate_bc.exs 'corpus/archive/mewtwo/*.slp' --char mewtwo --top 20 --list-out corpus/curated/r17a_bc.txt`
- **Ranked report:** `logs/bc_curation_r17a.json` (per-game armed/min,
  opener entropy, conversion rate — the data-composition stamp).

## Launch

```bash
cd ~/git/exphil
BC=$(cat corpus/curated/r17a_bc.txt)
EXTRA="/home/blewf/Slippi/Game_20260719T213810.slp,/home/blewf/Slippi/Game_20260719T213957.slp,probes/newera8/r14/plain/p1/Game_20260718T073713.slp,probes/newera8/r14/plain/p2/Game_20260718T073726.slp,probes/newera8/r14/plain/p3/Game_20260718T073738.slp,probes/newera8/r14/debounce/p1/Game_20260718T074551.slp,probes/newera8/r14/debounce/p2/Game_20260718T074603.slp,probes/newera8/r14/debounce/p3/Game_20260718T074615.slp,probes/newera8/r16/r13/plain/p1/Game_20260723T022523.slp,probes/newera8/r16/r13/plain/p2/Game_20260723T022535.slp,probes/newera8/r16/r13/plain/p3/Game_20260723T022547.slp"

NEWERA8_TAG=r17a NEWERA8_BACKBONE=mamba_2 NEWERA8_MAX_EPOCHS=100 \
NEWERA8_ROUNDS=1 NEWERA8_DROPOUT=0.6 PROBE_STATEFUL=0 \
NEWERA8_CONVERSION_WEIGHT=3.0 NEWERA8_OPENER_WEIGHT=3.0 \
NEWERA8_PROBE_REG=0.1 NEWERA8_PROBE_REG_EVERY=5 NEWERA8_PROBE_EVAL=10 \
NEWERA8_BC_REPLAYS="$BC" NEWERA8_LEARN_STYLES=1 \
NEWERA8_PREFLIGHT=1 \
NEWERA8_EXTRA_ROLLOUTS="$EXTRA" \
scripts/overnight_newera8.sh > logs/r17a_launcher.log 2>&1 &
```

Launcher passthrough for the new levers (`NEWERA8_OPENER_WEIGHT`,
`NEWERA8_OPENER_LOOKBACK`, `NEWERA8_LEARN_STYLES`, `NEWERA8_STREAM_CHUNK`,
`NEWERA8_OPEN_SHARDS`) was ADDED 2026-07-23 — the command above is
turnkey. Still confirm the lever lines in the log (see Confirm below).

### Direct-drill equivalent (no launcher, no fan-out)

```bash
BC=$(cat corpus/curated/r17a_bc.txt)
mix run scripts/dagger_drill.exs \
  --expert mewtwo_combo --backbone mamba_2 --max-epochs 100 \
  --prev-action-dropout 0.6 --transition-weight 2.0 \
  --conversion-weight 3.0 --opener-weight 3.0 \
  --probe-reg 0.1 --probe-reg-every 5 --probe-eval-every 10 \
  --bc-replays "$BC" --learn-player-styles \
  --rollouts "<r13/r14/r16 probes + human demos>" \
  --out checkpoints/mewtwo_combo_newera_r17a_policy.bin
```

Add `--stream-chunk-size 500000 --open-shards 16` only if the pool
outgrows RAM — see the DATA_SCALING caveat (shuffle granularity moves the
loss; keep `--open-shards` high).

## Confirm the levers actually engaged (first 2 min of the log)

All four MUST appear, or the run isn't testing what you think:

1. `Opener weighting: N openers; ... upweighted x3.0 (dist %{...})`
2. `Conversion weighting: ... upweighted x3.0`
3. `Style conditioning: N distinct player(s) in the registry`
4. At the first refit: `probe-reg refit @ epoch 5: %{... refit: true, shield_rows: <~5-6% of rows>}`
   — **if this says `refit: false` / `shield_rows: 4`, the lever is dead
   again and the run is wasted.** (r16's exact failure.)

## On completion

Score against `R17_ACCEPTANCE_2026-07-23.md`:
- Primary: armed approaches/min — PASS ≥0.5, STRONG ≥1.0, FAIL <0.5.
- Secondary: report-card avg ≥6.5/10, conversions ≥ r16 rate, opener
  entropy ↑ toward 1.0, top-opener share ↓ toward 0.6.
- Then `coach_report.exs` on the fan-out probes for the gap ledger.
- Record the data-composition stamp in a dated handoff.

FAIL is a real result: it retires imitation-shaping and promotes PPO
approach-shaping (run the PPO smoke first — it's still UNRUN).
