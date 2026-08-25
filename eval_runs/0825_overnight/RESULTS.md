# Overnight 0825 — results

## P0: corpus extraction — DONE
All **7,911 master-master Fox replays** extracted
(replays/erickfm_ranked/FOX/extracted; 51s, tarball was fast).

## P1: "stage-internals arm" — the flag NEVER LANDED (silent drop)

The saved checkpoint reads `stage_internals: false`: dagger_drill
parses its own OptionParser strict list, `--stage-internals` isn't in
it, and OptionParser silently collects-and-ignores unknown flags. So
P1 trained the PLAIN champion recipe — i.e., it is **replicate #4**:

- argmax **ep59 at 434.4/min c434**, confirm x3 flat; mewtwo 77.9 c1
  (another low transfer draw). R1 spread update: argmaxes now
  {437.4, 438.4, 439.4, 439.4, 434.4} — still ~1% band; argmax epoch
  set now {4, 16, 45, 4, 59} — the epoch lottery spans the whole run.
- **The canary DID its first live tour**: saved (336-dim layout,
  correct for the queue-depth-4 config) and validated on all ~60
  snapshot loads during the sweep. train_delays [2,3,4] saved
  correctly (the 635e08e fix works end-to-end).

**New guard-class bug (GUARDS_BACKLOG candidate #9): scripts with
their own OptionParser strict lists silently ignore unknown flags.**
The 4-step CLI checklist in CLAUDE.md covers training/config.ex but
dagger_drill is its own parser. Fix: (a) add stage_internals to the
drill's strict list AND its embed-config path; (b) make the drill
ERROR on unrecognized --flags (OptionParser returns them; we drop
them). The honest stage-internals shakeout still hasn't run.

## P2: generalist pilot — died at GPU placement, cache BANKED

1,000 master replays parsed clean → **10.87M frames** embedded and
**cached to disk** (the expensive step is banked) → died placing
precomputed validation batches on device (EXLA binary_to_device_mem,
~1.1M val frames at fraction 0.75 — the trainer itself warned
"consider --stream-chunk-size"). No trainer bug — a scale wall the
non-streaming path is documented to hit.

Morning rerun options: `--stream-chunk-size` (streaming path; note it
excludes multi-delay/AWBC — fine for the pilot), or --max-files ~300
on the cached embeddings for a quick first checkpoint.

## Scoreboard

- Corpus: READY (7,911).
- Canary + train_delays metadata: production-validated.
- Champion recipe: n=5 now; height ceiling-stable, epoch lottery
  confirmed again.
- Stage-internals training shakeout: STILL OWED (drill flag fix
  first).
- Generalist pilot: one flag away (streaming), embeddings pre-baked.
