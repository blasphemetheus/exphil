# ExPhil Test-Coverage & Cleanliness Audit (2026-07-16, static analysis)

Scope: 233 modules under lib/, ~160 test files, ~120 scripts. No tests run.

## 1. Coverage map (by blast radius)

### Tier 0 — Teacher/expert labeling (mislabel = lost training round). Strong, one hole.
Every scripted expert has a dedicated test: mewtwo_fair_expert ✓,
mewtwo_combo_expert ✓, mewtwo_tech_chase_expert ✓, multishine_expert ✓,
fox_recovery_expert ✓, dummies/tech_random ✓.

**HOLE: `lib/exphil/interp/replay_stats.ex` — NO TESTS.** `shield_stats/1`,
`run_lengths/2`, `knockdown_episodes/1`, `load/1` (replay_stats.ex:26-146)
are the measurement backbone of BOTH `scripts/report_card.exs` (all 8 gates)
and `scripts/teacher_coverage.exs` (jump-label audit). A bug here silently
miscalibrates the gate that decides whether an overnight round is kept.
Highest-value missing test in the repo.

### Tier 1 — Training loop. Covered via facades; real gaps:
- Config submodules (parser.ex 726 LOC, presets, validator, yaml, diff,
  inference) exercised only through Config facade; parser + yaml deserve
  dedicated tests.
- Imitation submodules covered via facade (imitation.ex:491-503).
- Genuinely untested: `training/chunk_pipeline.ex`, `training/plots.ex`,
  `training/batch_tuner.ex`, `training/debug.ex`, `training/utils.ex` (on
  the Agent inference path — agent.ex:434,438,1004), callbacks
  validation/test_eval/policy_export/graceful_shutdown/loss_plot/
  progress_bar/diagnostics.

### Tier 2 — Agent runtime. The inference path is untested.
`agents_test.exs:66-308` only covers start/stop/config and no-policy errors —
never loads a policy, so `compute_action`, `apply_jump_debounce`,
temporal/incremental inference are entirely uncovered.

### Tier 3 — Embeddings. No unit tests on leaves; end-to-end covered
(game_test, player_test, snapshot_test, embed_path_parity_test,
representational_limits_test). Low residual risk.

### Tier 4 — Utilities/native/experimental (lowest). Untested:
native/{futhark,thunderkittens,triton,xla_selective,rust_linear,
flash_attention}_scan.ex, networks/{mamba_cumsum,mamba_hillis_steele,
mamba_nif,onnx_layers}.ex, mix/tasks/*, nx_safe.ex. Mostly benchmark-only.

## 2. Recently-changed items — is the NEW behavior tested?

| Item | New behavior | Covered? |
|---|---|---|
| agent.ex jump-debounce | `apply_jump_debounce/3` (agent.ex:559-589), airborne gating (:519-525), cooldown arm-on-release | **NO.** Zero references in test/. Highest-risk new gap. |
| fair-expert defaults + scrub | `scrub_early_air_jumps/1` (:104-112), `side_bucket/2` (:207-213), `:turn_toward`/`:approach` (:268-277) | **YES, thorough** (test :182-191, :152-170, :101-115, :70-89). Minor: `:turn_toward` opponent-behind branch only implicitly hit at test:158 — no label_traced assertion naming it. |
| dagger_drill.exs atomic _latest.bin publish | write-tmp-then-rename every 10 epochs (:524-532), epoch-5 logging (:517) | **NO.** No test; relies on untested Imitation.export_policy + rename atomicity. |
| report_card.exs 8 gates | gate vector + percentile math (:101-114, `pct` :23-28) | **NO.** Untested, atop untested ReplayStats. Off-by-one risk in `pct` unverified. |

## 3. Dead code

Module-level dead (referenced only in own file across lib/+scripts/+test/):
- `lib/exphil/training/flash_training.ex`
- `lib/exphil/inference/policy_serving.ex`
- `lib/exphil/native/xla_selective_scan.ex` — NOTE: the Nx audit
  (AUDIT_NX_USAGE_2026-07-16.md §6) recommends WIRING THIS IN, not deleting.
Near-dead (1 external ref): bridge/pytorch_port.ex, networks/mamba_nif.ex.

Superseded dated scripts: archived to scripts/archive/ 2026-07-16 (done).
One-off March bf16-conv investigation scripts (concluded, archive
candidates): bisect_bf16_conv_crash.exs, bisect_model_complexity.exs,
repro_bf16_conv_sigbus.exs, stress_test_conv_steps.exs,
stress_test_full_mamba.exs, profile_* septet.

## 4. Duplication worth consolidating

- `scripts/dagger_multishine.exs` (272 lines) superseded by
  `scripts/dagger_drill.exs` (header: "generalizes dagger_multishine";
  supports `--expert multishine`). Fold and delete.
- Script preamble duplication: 55 scripts alias Training.Output, 29 hand-roll
  OptionParser.parse, 9 repeat Logger.configure. `lib/exphil/script_template.ex`
  exists (tested) but under-used.
- Forensics helpers duplicated in-script: flatten_tensors, find_adam_state,
  numeric_stats, safe_max (dagger_drill.exs:345-420) — extract into
  ExPhil.Training.Debug.

## 5. Cleanliness

- TODO/FIXME/HACK: only 2 in lib/ (embeddings/player.ex:986,
  policy/backbone.ex:3115). Very clean.
- Oversized modules: policy/backbone.ex 3191 LOC, training/data.ex 2647,
  training/config.ex 1640, error.ex 1495, embeddings/player.ex 1147,
  agents/agent.ex 1125, training/output.ex 1042. backbone + data are top
  split candidates; data.ex sits on the labeling→training path.
- apply_jump_debounce calls Nx.to_number per-frame in hot inference path
  (agent.ex:564-565) — correctness-fine, bench note.

## Prioritized fix list

1. Unit-test ReplayStats against an existing .slp fixture. (High risk, med velocity)
2. Lock down report_card gate vector + pct percentile; extract gate math into
   a testable module. (High risk, med velocity)
3. Unit-test apply_jump_debounce (airborne suppressed during cooldown,
   grounded passes, held jump never cut, cooldown armed on release edge);
   extract to pure function. (Med-high risk, high velocity)
4. label_traced :turn_toward assertion for opponent-behind. (Low risk, high velocity)
5. Delete dead FlashTraining + PolicyServing (keep XLASelectiveScan — see Nx
   audit). (Neutral, high velocity)
6. Consolidate dagger_multishine.exs into dagger_drill.exs. (Low risk, high velocity)
7. Archive dated scripts — DONE 2026-07-16.
8. Cover training/utils.ex (Agent inference path). (Med risk, med velocity)
9. Extract + test forensics helpers into Training.Debug. (Med risk, med velocity)
10. Split backbone.ex and data.ex. (Low immediate risk, lower velocity)
