# Current project status

Updated 2026-09-10. Start here for current direction; use dated handoffs for
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

The user clarified on 09-10 that Claude is running a sweep, not training.
Its identity and progress have not been independently verified for this work.
The 09-09 handoff's statement that nothing was running is historical, not current.

Keep this work independent of the active run: no rebuilds, dependency updates,
launcher changes, artifact cleanup, or GPU evaluation in its environment.
Multi-stage launchers may read modified source on their next stage. Runtime
development needs independently isolated source, dependencies, and build output.

The first runtime improvement adds [explicit run completion records](../operations/RUN_STATUS.md)
for future launches. Launcher/watchdog contracts pass 17 isolated tests; training
callbacks still need automatic progress and diagnostic reporting. Remote
dependency defaults also await publication of required companion-repository
APIs; the improvement checklist records the exact revisions inspected.

R1 BPTT evaluation and R2 canonical training delay are now implemented and tested
in an isolated copy: **367 tests and 14 doctests pass**. The
[source patch and activation handoff](BPTT_DELAY_HANDOFF.md) are saved here;
runtime changes are not yet activated while sweep completion is unconfirmed.

## Current workflow and limitations

1. Follow [contributor setup](../../CONTRIBUTING.md#development-setup) for the
   current sibling dependency requirements.
2. Use `scripts/train.exs` for new training. See the
   [training guide](../guides/TRAINING.md#quick-start); historical
   `train_from_replays.exs` examples are not the primary entry point.
3. New parser output is causal: state[t] pairs with controller[t+1]. Training
   delays are additional reaction delay. Live Dolphin delay numbering is
   unchanged; use the [deployment cards](../guides/DEPLOY_KNOBS.md).
4. The active `scripts/eval_model.exs` still lacks BPTT evaluation. The staged
   R1 patch adds checkpoint-driven, carry-aware teacher-forced evaluation and
   shared diagnostics. See the [usage and limits](../guides/BPTT_EVALUATION.md);
   apply only after the sweep is finished. Gameplay measurements remain separate.
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
