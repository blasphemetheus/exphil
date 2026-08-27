# nx-pin dynamics probe — results (run 08-19 23:37 → 08-20 01:33)

Prereg in `run_nxpin.sh`: exact g18 recipe (champion+awbc, 2e-4, 90ep
cap) at nx a7497612 (the g16/g15r-era pin) vs the f843aa1a HEAD all of
tonight's anomalous runs used.

## Verdict: **N1 — the nx bump is IMPLICATED in the dynamics change**

| stack | trajectory | outcome |
|---|---|---|
| a7497612 (this probe) | CALM: smooth 0.135→0.0011 over 83 epochs, max one-epoch swing ~4x, no collapse, genuine sub-target convergence | fox 87.9/min c3, mewtwo 83.9/min c6 |
| f843aa1a (g15r2, g18a1, g18a2) | WILD: order-of-magnitude epoch swings (ep7 spike to 1.01), two one-epoch collapses to ~0, one 13-epoch champion-class landing | dead / dead / **419.4 c415** |

The dynamics signature is categorical and matches era: every a7497612
run (g16, g15r, this probe) is calm; every f843aa1a run is wild. One
probe run, but the signature is built from 83 epoch observations, not
one number.

## The uncomfortable second finding

The wild stack's BEST draws (g15r2-ep50 362.5 c353, g18a2 419.4 c415)
are far above anything the calm stack has EVER produced on the fixed
grads (87.9 / 116.8 / 253.6, n=3, no collapse). Whatever the bump
changed acts like a higher-temperature optimizer: mostly worse, twice
spectacular. With the collapse guard (GOTCHA #99) making wild runs
survivable, "HEAD + guard + best-of-N gating" empirically beat every
calm-stack recipe tonight. That is a lottery, not a recipe — but it
produced the two best fixed-stack checkpoints in existence.

## What this does NOT settle

- WHICH of the 10 commits (a7497612..f843aa1a) changes training math,
  and whether it is a correctness BUG (wrong grads/values that could
  also affect anything else running on HEAD) or a benign numerical
  change. The candidates touching semantics: defn/expr donatable
  rewrite, exla defn output-donation fix, impl! dispatch changes
  (reduce/clip/gather/put_slice with mixed args), from_binary
  bitstring. exphil doesn't use donatable directly — but EXLA/defn
  internals might exercise those paths.
- Whether g17's negative lr read holds on the calm stack (it was run
  entirely on HEAD).

## Recommended next steps (Bradley's call)

1. **Bisect the 10 commits** with this script capped at ~20 epochs.
   Classifier (validated on all four 2e-4 runs): WILD = any one-epoch
   loss ratio > ~8x within 20 epochs (HEAD runs: 10.7x @ep11 /
   collapse @ep10 / 153x @ep8; pinned probe max 4.2x). ~25 min/probe,
   3-4 probes for the bisect. Caveat: dynamics are stochastic and the
   4.2x-vs-10.7x margin is workable but not huge — rerun any probe
   landing in the 4-10x gray zone before trusting its label.
2. Report the culprit upstream-of-fork (the fuzz campaign's own find:
   a semantic change that alters real training).
3. Meanwhile: TRAINING PINNED at a7497612 (current checkout state);
   deploy/play candidates from tonight (ms_g18_ep90.bin) are
   behaviorally validated in-game and unaffected by which stack
   TRAINED them.
4. The strategic fork (replicates on calm stack vs guarded lottery on
   HEAD) is a real decision — the lottery won tonight, but n=2.
