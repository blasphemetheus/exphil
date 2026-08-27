# g17 LR retune arms — results vs prereg

Run 2026-08-19 18:30–21:16 (detached pipeline; both arms trained clean,
~76 s/epoch, no divergence). Prereg in `run_g17.sh` header.
Checkpoints: `ms_g17_lr8.0e-4.bin`, `ms_g17_lr4.0e-4.bin`.

## Card (fixed-grad stack, champion recipe + --awbc; refs from 0814)

| arm | final loss | stand-fox d3 x3 | stand-mewtwo d3 |
|---|---|---|---|
| g16 = lr 2e-4 (ref) | 0.00127 | 253.6/min c203 | 109.8/min c14 |
| g17 lr 4e-4 | 0.009 | 151.8/min c88 (x3 identical) | **129.8/min c11** |
| g17 lr 8e-4 | 0.023 | 103.9/min c5 (x3 identical) | 94.9/min c7 |

## Verdict: prereg read #2 — **lr is NOT the recovery lever**

Both arms hard-fail the >=304 bar and land BELOW g16; quality is
monotone in (lower) lr. Next knob per prereg: **epochs (90) at 2e-4**
(the "best g17 lr" is the baseline itself).

Findings worth carrying:

1. **The fixture sweep's speed signal INVERTED at full scale.** 8e-4's
   epoch-1 convergence on the fixture predicted nothing good here; at
   full scale higher lr just raised the loss FLOOR (0.00127 / 0.009 /
   0.023), and stand-fox rate tracked the floor almost linearly.
   Lesson for future sweeps: fixture-scale lr reads transfer for
   STABILITY (no divergence — that held) but not for quality; screen
   lr arms by final full-scale loss before spending gate time.
2. **Hot lr flattens the specialist profile.** The fox↔mewtwo gap
   shrinks as lr rises (2e-4: 253.6/109.8; 4e-4: 151.8/129.8; 8e-4:
   103.9/94.9) — and 4e-4 actually BEAT g16 on mewtwo (+18%, 129.8).
   The rough optimum trades chain-critical fox precision for uniform
   competence. If a generalization-first arm is ever wanted, 4e-4 is
   a datapoint, not a recipe (n=1, stand numbers).
3. Chains scale with rate as usual (c5/c88/c203) — no
   rate-without-chains anomaly; deterministic FD triple confirmed on
   both arms.

## Next (per handoff GPU queue)

- GPU #2: g15r2 replicate (`eval_runs/0819_g15r2/run_g15r2.sh`) —
  launched right after this run.
- GPU #3 (now triggered by this negative read): 90-epoch arm at 2e-4
  (`eval_runs/0819_g18_ep90/run_g18.sh`), queued after #2.
- NO CROWN implications (stand numbers never crown).
