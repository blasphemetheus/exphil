# R1/R2 implementation handoff — 2026-09-10

**Activated 2026-09-11 after the user confirmed Claude's work had stopped.**
The 23-file patch was integrated and its 367 tests plus 14 doctests passed again
against the active checkout's source, using independent CPU dependency binaries.
The activation instructions below are historical; do not apply the patch again.

- [Durable source/test patch](patches/bptt-eval-label-delay.patch)
- [Behavior and usage guide](../guides/BPTT_EVALUATION.md)
- Isolated source: `/tmp/exphil-improvements.MggyAe/exphil`
- Independent companion sources: `/tmp/exphil-improvements.MggyAe/edifice`
  and `/tmp/exphil-improvements.MggyAe/libmelee_ex`

## Activation

Completed 2026-09-11. Original procedure retained for the implementation record:

Only after confirming the sweep no longer uses this checkout, review and apply:

```bash
git apply --check docs/planning/patches/bptt-eval-label-delay.patch
git apply docs/planning/patches/bptt-eval-label-delay.patch
git diff --check
```

The patch contains 23 runtime, test, and generated-reference files. It excludes
the existing sweep edits and the earlier run-status work. Its application check
passes against the current working tree. If the tree changes, repeat that check;
do not force application or overwrite concurrent work. After activation, rerun
the contracts in an independently configured CPU build, then update R1/R2 from
staged to done. Nothing has been committed, pushed, or deployed.

## Validation performed

**367 tests and 14 doctests passed, zero failures**, including native replay tests:

- `test/exphil/evaluation/bptt_test.exs`
- `test/exphil/training/label_delay_test.exs`
- `test/exphil/training/config_test.exs`
- `test/exphil/training/config/flag_parity_test.exs`
- `test/exphil/training/config/flag_docs_test.exs`
- `test/exphil/training/mix_frames_test.exs`
- `test/exphil/training/trajectory_cursors_test.exs`
- `test/exphil/training/bptt_train_test.exs`
- `test/exphil/training/checkpoint_roundtrip_test.exs`
- `test/exphil/data/label_convention_test.exs`
- `test/exphil/data/label_alignment_test.exs`

This covers full versus chunked GRU logits, carry resets, short tails,
independent/autoregressive input contracts, a training step and standalone
export, fresh-process real-replay evaluation through `scripts/eval_model.exs`,
diagnostics, and matching successor targets across standard, streaming BPTT,
held-out validation, and mixed-frame loaders. Delay conflicts, explicit zero,
YAML/CLI precedence, legacy resume conversion, exported metadata, and live
numbering also have regressions. Generated flag-reference parity passes.

Validation used Elixir 1.18.4 with two schedulers and EXLA's host client. Compiled
dependency beams/native libraries were independently copied and dereferenced
into `/tmp/exphil-improvements.MggyAe/runtime`; changed modules were recompiled
in the isolated process. No hardlinks or native-library symlinks point back to
the active build. Peppi used an independent copy of its native library.

This is **not** a clean dependency bootstrap or a complete repository suite.
Required unpublished companion APIs still block R4's reproducible remote setup.
No GPU training, Dolphin gameplay, production checkpoint quality comparison,
or active-job restart was performed. The local runner and final output remain
at `/tmp/exphil-improvements.MggyAe/verify.exs` and
`/tmp/exphil-improvements.MggyAe/validated-regression.log`; `/tmp` is disposable,
so the durable patch is the source of record.

## Deliberate limits

The BPTT evaluator rejects unsupported modes rather than reconstructing a
different graph or reporting misleading scores. See the usage guide for its
GRU, discretization, queue/delay-ID, legacy-label, and export restrictions.
Teacher-forced scores do not replace live sampled-policy evaluation. The sampler
probe remains unchanged. BPTT gradient diagnostics remain skipped until a
carry-aware gradient diagnostic is implemented.

Delay augmentation is explicit additional jitter and remains non-temporal-only.
Precomputed corpora cannot be shifted at training time. Legacy mixed drill files
still warn and should be re-exported; their historical label construction is not
rewritten by the alias resolver. R3 CI integration and R5 evaluation identity
remain separate follow-ups.
