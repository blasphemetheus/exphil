# g19 — first behavioral gate-sweep arm: results

Run 2026-08-20 10:53–13:04. Prereg in `run_g19.sh`. Full 60 epochs
(convergence exit disabled), snapshot every epoch, post-run gate-sweep
of all 60 snapshots (one deterministic sync stand-fox run each), argmax
confirmed x3 + mewtwo. Full table: `sweep/sweep_table.txt`.

## Headline

**`ms_g19_ep4.bin`: 437.4/min, max chain 438 (x3 identical) — NEW
ALL-TIME STAND RECORD**, beating old-grad g15 (430.4 c426) on the
fixed-grad stack, from the FOURTH epoch of a 60-epoch run whose final
epoch gates 122.8. Mewtwo on ep4: 93.9/min c1.

## The behavioral trajectory (first ever measured)

- ep1-2: skill forming (0 -> 78).
- **ep3-8: a WIDE peak — 424 / 437 / 435 / 437 / 428 / 380.** The
  entire skill is present by epoch THREE.
- ep9-42: violent oscillation between the peak basin and the ~87-150
  attractor — 1-epoch revisits at ep13-14 (~355), ep17 (419), ep19
  (350), ep22 (395), ep25/28/30 (~300-324), ep33 (424), ep42 (414),
  interleaved with ~80-130 epochs.
- ep43-60: terminal decay — never again above 180; settles ~110-150.

Loss over the same run descended monotonically-ish to 0.00187:
**loss and chain skill are anti-correlated after epoch ~8**, confirmed
across an entire run rather than inferred from endpoints.

## Prereg reads

- **G1: PASS, decisively.** Argmax 437.4 >= 304, vs final-epoch 122.8
  (3.6x). Gate-sweep IS the recipe lever; adopt --snapshot-all +
  scripts/gate_sweep.sh for every future arm (~35 min sweep cost).
- **G2 peak structure:** argmax ep4; primary peak width 6 epochs
  (ep3-8 all >= 380) — comfortably dense for per-epoch gating; later
  revisits are 1-epoch spikes (would be missed by sparser gating).
- G3: n/a.

## What this retro-explains

Every mystery of the 08-19 "variance" night: runs sample this
trajectory at whatever epoch their exit lottery lands on. g18a2's
ep13 exit (419.4) = a revisit spike; g16's ep58 (253.6) and g15r's
ep60 (116.8) and nxpin's ep83 (87.9) = the decay tail; the "mediocre
attractor" = ep43+ behavior. No nx bug required. The champion-era
430.4 (g15, old grads) was itself likely a peak sample.

## Caveats (g6 rule applies in full)

Stand-dummy FD numbers only. ep4 is a 4-epoch-old policy — off-
distribution robustness (YS, scramble/rung-0, moving opponents, human
play) is UNKNOWN and early-peak policies may be brittle there.
Before any promote-ladder move: YS x3 bucket, rung-0 probe, and the
blind human decider. NO CROWN from this file.

## Off-distribution battery (run 13:05-13:20, same day)

| gate | ms_g19_ep4 | reference |
|---|---|---|
| stand-fox d3 x3 | **437.4 c438** | g15 430.4 c426 (record beaten) |
| real d4, id-3 override | **388.4 c389** | sp1-era 332.4; deploy-rung strong |
| YS x3 | 44.9 c1 / 118.8 c99 / 67.9 c17 | no collapse; softer than g16 |
| stand-mewtwo d3 | 93.9 c1 | g16 109.8 c14 |
| real d2 (id2 or id3) | 2.0 c1 / 75.9 c2 | WEAK — d2 mode never formed |
| real d4, native id4 | 105.8 c1 | weak natively; id3 override is the knob |

Revisit peaks ep6/17/33/42 also have NO working d2/d4 modes (~80-112)
— this run never consolidated multi-rung modes anywhere; the peaks are
d3-specialist peaks, and the id-3 override covers d4. Practical
deploy story: **d3 local 437.4 / d4-netplay-with-id3 388.4 — the
practical band is covered by one knob**, d2 excluded.

## Next steps proposed

1. Off-distribution gate battery on ep4 (and ep6, the tied twin):
   YS x3, scramble probe, d2/d4 delay-id overrides.
2. Adopt sweep as standard; add fuller sweep metrics (mewtwo column)
   if the 30s/epoch budget allows.
3. Revisit the WHY (interp): what do updates after ep8 trade away?
   The pool is majority rollout/scramble — late training may fit
   opponent-reaction noise at the expense of the cycle. CycleMargins
   per epoch over the snapshot series is now a cheap offline probe.
4. The retune program's knob arms (lr/epochs) are OBSOLETE as framed —
   every recipe question must now be re-asked as "does it raise/widen
   the PEAK", measured by sweep.
