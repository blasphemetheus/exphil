# Current project status

Updated 2026-09-11. Start here for current direction; use dated handoffs for
experimental evidence and the exact checkpoint-specific commands.

## Direction

The immediate program is causal behavioral cloning for the general Fox policy,
alongside drill-based multishine skill acquisition and gameplay evaluation.
Architecture breadth, large-scale PPO, and additional kernel exploration are
available research directions, not the immediate next milestone.

- [Latest reviewed handoff: 2026-09-09](HANDOFF_2026-09-09.md): causal-label
  correction, checkpoint layout changes, v16 shakeouts, and the v3 recipe.
- [General-policy experiment record](V2_PREP.md): measurements and decisions.
- [Repository improvement checklist](REPO_IMPROVEMENTS.md): the seven engineering
  changes approved for incremental work, with validation requirements.
- [Structural invariants](INVARIANTS.md): the contracts implemented on 09-09.
- [Technical issues](FIXES.md): detailed findings; older entries can be superseded
  by the invariant ledger or later dated additions.

## Active work

On 09-11 the user confirmed Claude had no work in progress and authorized
integration of the staged runtime patch and a multishine benchmark.

Keep this work independent of the active run: no rebuilds, dependency updates,
launcher changes, artifact cleanup, or GPU evaluation in its environment.
Multi-stage launchers may read modified source on their next stage. Runtime
development needs independently isolated source, dependencies, and build output.

The first runtime improvement adds [explicit run completion records](../operations/RUN_STATUS.md)
for future launches. Launcher/watchdog contracts pass 17 isolated tests; training
callbacks still need automatic progress and diagnostic reporting. Remote
dependency defaults also await publication of required companion-repository
APIs; the improvement checklist records the exact revisions inspected.

R1 BPTT evaluation and R2 canonical training delay are integrated:
**367 tests and 14 doctests pass again** against this checkout's source using
independent CPU dependency binaries. The [implementation record](BPTT_DELAY_HANDOFF.md)
retains the original patch. The [multishine benchmark](../guides/MULTISHINE_BENCHMARK.md)
adds frozen replay scoring, censored recovery measurements, and teacher-label
audits. Fresh held-out gameplay and closed-loop teacher validation remain open.

## Current workflow and limitations

1. Follow [contributor setup](../../CONTRIBUTING.md#development-setup) for the
   current sibling dependency requirements.
2. Use `scripts/train.exs` for new training. See the
   [training guide](../guides/TRAINING.md#quick-start); historical
   `train_from_replays.exs` examples are not the primary entry point.
3. New parser output is causal: state[t] pairs with controller[t+1]. Training
   delays are additional reaction delay. Live Dolphin delay numbering is
   unchanged; use the [deployment cards](../guides/DEPLOY_KNOBS.md).
4. `scripts/eval_model.exs` now supports checkpoint-driven, carry-aware GRU BPTT
   evaluation and shared diagnostics. See the [usage and limits](../guides/BPTT_EVALUATION.md).
   Queued-action/delay-ID multishine policies are outside that evaluator's scope;
   gameplay measurements remain separate.
5. Evaluate gameplay at the checkpoint's deployment settings. Do not compare
   legacy leaked-label losses with causal-label losses or promote policies on
   stand-dummy results alone.

The README's architecture benchmark table is historical and lacks a recorded
benchmark date/revision. It does not establish current recurrent-step latency
or performance of the causal-policy training line.

## Maintaining this page

Update this page when the active recipe, supported workflow, or known limitations
change. Link evidence instead of copying experiment logs. Keep historical
handoffs intact; do not use their old process-state assertions as live status.
