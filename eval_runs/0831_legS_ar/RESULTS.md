# Leg S on the AR head — RESULTS (0831_legS_ar)

**2026-08-31 02:01.** Three arms, one chain, knobs per PREREG.md (read it
first — the readings below follow its pre-registered rules). Fox-detected
ports (E1), n=16, T=0.5/0.5, 2000 decision frames, seed 20260831.
Per-arm tables: `{ep10,INDhead,ARhead}_table.md` (extracted from
`logs/legS_ar.log`; the `--out` files were eaten by a launcher quoting
bug, see Operational note).

## The table (joint head)

| arm | sampling | pass@1 | pass@16 | headroom |
|---|---|---:|---:|---:|
| ep10 | independent | 16.5 | 44.5 | +28.0 |
| v1.1_INDhead | independent | 8.1 | 36.6 | +28.5 |
| v1.1_ARhead | **sequential** | 8.5 | 38.6 | +30.1 |

(08-28 reference on ep10, mixed-character ports: 14.5 / 43.7 / +29.2.)

## Readings, against the prereg

1. **Joint pass@1 AR−IND = +0.4 pts → UNRESOLVED** (threshold ±3). The
   conditioning does NOT show up in open-loop master-match. This does
   not contradict 8a — 8a's rule deliberately gated on A2 recovery
   routes + live P(up|B), not on match — it *explains the instrument*:
   pass@k over decision frames is dominated by dense, near-independent
   moments; the AR head's payoff is concentrated in rare coincidence
   events (up-B routes) that 2000 sampled frames barely weight. A
   match-rate metric cannot see a rare-event fix (L5's cousin).
2. **Headroom AR +30.1 vs IND +28.5 → SURVIVES** (shrink threshold
   ≥5 pts; it did not shrink at all). The selection gap is NOT an
   artifact of the independent factorization: even with coherent joint
   samples, the master's action is in 16 samples ~4.7× as often as in
   one. **Consequence per the prereg follow-on: the critic program
   keeps its priority on the AR base, and the G3b learned-dynamics
   spike is justified as its rollout engine** (after the unfreeze).
3. **Instrument continuity ✓**: ep10 fox-only re-read 16.5/44.5/+28.0
   vs 08-28's 14.5/43.7/+29.2 — pass@1 +2 under correct expert ports,
   headroom stable. The 08-28 Leg S conclusions survive E1.

## Secondary observations (not pre-registered)

- **Both frozen-trunk refits sit at HALF of ep10's joint pass@1**
  (8.1/8.5 vs 16.5), driven by buttons (22 vs 43). Consistent with the
  TV polish story (0.63 vs 0.55) and Bradley's "sandbagging" read — and
  another proof pass@1 is not a skill score: ARhead *beats* ep10 live
  on recovery while matching the master half as often. The unfreeze
  (8/9) is exactly the run that should close this.
- ARhead's per-component curves are broader (main headroom +31.1 vs
  +23.8; shoulder +28.6 vs +17.7): sequential sampling spreads mass
  across coherent alternatives rather than concentrating per-component.

## Operational note

`systemd-run … devenv shell -- bash -c '<inline script>'` ate `${name}`
(brace-expansion form) somewhere in the systemd/devenv re-quoting while
unbraced `$name` survived — all three `--out` writes landed on one
hidden `.md`. Data recovered from the log (L3: logs survive what files
don't). **Law: no inline multi-line scripts under systemd-run — write a
`.sh` file and run the file.**
