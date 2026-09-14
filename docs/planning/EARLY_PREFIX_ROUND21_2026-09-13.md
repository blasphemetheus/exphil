# Early-prefix 21-epoch bounded round

Follow-up: [three-window diagnosis](EARLY_WINDOW_DIAGNOSIS_2026-09-13.md).
No exact target conflicts; random GRU initial states cause batch-row dependence,
and training/export precision differs. Neither diagnostic toggle alone repairs
the three decisions. Fix those contracts before another fitting run.

Authorized 2026-09-13. Fresh optimizer, original saved initial parameter tree;
GRU64, two layers, autoregressive head, window16, queue3, cold history,
delay2, clean loss, unchanged canonical and six recorded teacher clips.

Fixed budget: 21 epochs of 14,589 draws / 228 batches = 4,788 updates.
First 18 targets of every recorded teacher clip repeated x64; all remaining
targets retained once per epoch. No architecture, label, or delay sweep.

Select the ordinary best aggregate-training-loss export, not a post-hoc epoch
snapshot. Frozen evaluation remains unweighted. Require all six early-case
gates (18/18 conditional matches and minimum joint target probability >=0.95).
Only a passing frozen gate proceeds to the fixed matched-teacher 12-response
live gate (cold delay2, T=1, chain>=10 each, valid timing, no drift/errors).
No budget extension or threshold adjustment if a gate fails.

- [x] Train the fixed 21 epochs and retain logs/checksums.
- [x] Measure the single selected checkpoint; apply frozen readiness gate.
- [ ] If qualified, run and audit the matched-teacher live gate — withheld: frozen gate failed.
- [x] Record outcome and next blocker without checkpoint fishing or promotion.

Launcher: `bash eval_runs/0913_early_prefix_fit/run_round.sh`.
Artifacts: `eval_runs/0913_early_prefix_fit/round21/`.

## Outcome

Completed all 21 epochs, 228 batches and mass 14,589 each: **4,788 updates**.
Peak RSS approximately 2.1 GB. Initial parameter trees compare exactly equal
to the original initialization (`initial_comparison.log`). No numerical rollback
or early stop. Best aggregate objective was **0.0035028206661710287 at epoch 20**;
the ordinary export selected that checkpoint, not the worse final epoch 21.

Frozen results: **705/708 teacher targets** have correct teacher-forced
conditional all-component argmax, up from 696/708. Canonical remains
7,077/7,077. Early targets improve from 96/108 to **105/108**. Four of six cases
pass the stricter per-target confidence gate; the full readiness gate fails.

| Case | Early matches | Minimum early teacher-action probability | Ready |
|---|---:|---:|---|
| Neutral 2228 | 18/18 | 0.999778 | yes |
| Neutral 2566 | 18/18 | 0.999784 | yes |
| Sustain 900 | 16/18 | 0.032104 | no |
| Recovery 4 | 17/18 | 0.476856 | no |
| Recovery 75 | 18/18 | 0.998748 | yes |
| Recovery 146 | 18/18 | 0.998023 | yes |

The three remaining conditional errors are all X-button decisions:
- Recovery 4, relative frame 0, action363/af11: target X, predicted no X;
  P(X)=0.4913. B has a small unwanted probability 0.0294.
- Sustain 900, relative frame 1 (frame901), action365/af3: target B only,
  but P(X)=0.9679 adds an unwanted jump.
- Sustain 900, relative frame 3 (frame903), action361/af1: target B+X,
  but P(X)=0.1714 misses the jump.

All other teacher targets match conditionally. These are frozen teacher-forced
results, **not evidence of a new closed-loop success rate**. The launcher exits
1 at the readiness gate, exactly as predeclared. No Dolphin evaluation was
started, no threshold relaxed, no extra epochs added, and no epoch snapshots
searched. The candidate is not promoted. The known inhibitor shutdown warning
appears after the successful training/export process; it is not the gate failure.

Checkpoint SHA-256:
`a5f703a2acd74a03afc990bd106a2f0a2487c98dcaa8040128d2b892a25837ba`.
Reports: `fit.json`, `fit_gate.json`, `train.log`, `sources.sha256`.

## Next blocker

Inspect the exact windows and delayed targets for these three X decisions:
check for identical/near-identical model inputs with conflicting targets,
and compare frozen predictions on their actual training windows. Distinguish
representation/history ambiguity from incomplete optimization before changing
the training budget or adding another weighting factor. Keep the matched-teacher
live gate unchanged and only claim behavioral progress after running it.
