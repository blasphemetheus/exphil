# Conversion Arm — design spec (2026-08-11)

Target: the biggest open behavioral gap in the generalist — **231%
damage dealt / 0 stocks taken** — now that edgeB closed the SD loop
(1.9 stocks lost/run). The bot opens but never funnels openers into
payoff.

## Evidence (why this arm, why now)

From the full-corpus knowledge model
(`eval_runs/0811_fox_stats_full/stats.json`, 1.51M events / 4,387
games):

- In `conversion_open` (the 90 frames after an opener) vs Falco, human
  Fox option payoffs: **throw Δ+15.7 (n=2,068)**, **aerial Δ+5.3
  (n=5,621)**, dash Δ+3.0 (n=14,079), shield Δ+0.3. Same ordering holds
  across the top matchups. Humans convert through GRABS and AERIALS;
  dash is filler.
- The bot's own eval replays (run the same stats on
  eval_runs/0810_edgeB_pool) are the baseline measurement: its
  conversion_open option mix vs the human mix is the gap, in one table.

Mechanistically this is the r16 lesson ("converts fine, never
initiates" inverted: now it initiates but never converts) — the
training signal contains openers but the corpus-frequency-weighted
average drowns the payoff sequences.

## Design: two mechanisms, one arm each, measure both

Unlike the edge arm there is NO scripted expert (the combo game is
contextual — what converts depends on percent/DI/character). Both
mechanisms therefore use HUMAN data as the teacher, not rules:

### Arm C1 — successful-conversion snippet mix (the edgeB recipe)

1. **Harvest** (new script `conversion_snippet_mine.exs`, the
   edge-miner pattern): scan the human corpus (replays/fox_il_v1,
   per-file Fox port detection); anchor = opener event
   (`Situations` :conversion_open onset) whose outcome window shows
   REAL payoff (>=25 dmg dealt within 150 frames, or a stock taken);
   cut [anchor-45, anchor+150]; KEEP recorded labels (the human's
   conversion IS the label — no relabeling, rule 2 satisfied because
   the recorder is the teacher); emit MixFrames envelope.
   Expected volume: sub-sample to ~100-300k frames from the ~500k+
   qualifying windows (payoff openers are common in human play —
   unlike edge_danger this is oversampling REPRESENTED-but-diluted
   behavior).
2. **Pack** via build_snippet_corpus.exs (windows never cross
   snippets).
3. **Train** the v2 recipe x 14 epochs on fox_v2 +
   `--mix-corpus cache/corpus/conversion_snippets_v1
   --mix-oversample N` — pick N so the mix lands at 5-10% gradient
   share (higher than edge's 2.3%: this teaches a POSITIVE skill
   distribution, not a rare correction).

### Arm C2 — conversion-weighted corpus sampling (cheaper, less controlled)

`ExPhil.Training.ConversionSampling` already computes per-frame weights
for converting-approach spans and feeds `sampling_weights` — but only
in the DRILL pipeline. Port the weight computation to corpus mode
(MmapCorpus.batched_sequences would need a sampling_weights path — or
approximate by weighting at sequence-start selection). C2 is the
one-flag version of C1; build it ONLY if C1 shows signal and we want
the tunable-knob form.

**Start with C1** — it reuses this week's validated pipeline end to end
and needs zero lib changes.

## What C1 does NOT fix (scope honesty)

Imitation shifts the OPTION DISTRIBUTION toward throws/aerials in
conversion windows; it cannot optimize follow-ups it never sees or
prefer good-but-rare lines — that is F5 offline RL's job (this arm's
outcome plumbing and snippet corpora feed straight into it). If C1
moves the option mix but stocks-taken stays 0, the failure is deeper
than distribution (e.g. kill-percent awareness) and F5 jumps the queue.

## Measurement (pre-registered)

Pool evals n=8, temp 0.3, vs edgeB as baseline:
1. **stocks_taken** (the headline; currently 0.0).
2. **damage_dealt** per run (currently 0.0 in the sync regime — if it
   stays 0 the regime can't measure this arm and the async rung
   becomes the primary eval; note before launching).
3. **Option-mix shift**: situation_stats on the arm's replays —
   conversion_open option mix should move toward throw/aerial (human
   target: ~8% throw / ~20% aerial of conversion-window events).
4. Regressions: SDs must stay ~edgeB (<=1/run); stand multishine
   untested (generalist track, not the specialist crown).

REGIME CAVEAT: the sync pool showed 0 damage for every arm to date —
this arm's primary eval may need the async runner
(EVAL_PROTOCOL regime rule) or CPU level > 1 to see damage at all.
Decide the eval regime BEFORE training (the g6 lesson).

## Queue position

After tonight's viewer/stats work; before v3-edge (per-stage ledge is
mechanical and can run any idle GPU hour, but conversion has the
evidence-backed payoff). Bradley signs off on arm launches per the
standing heads-up rule.
