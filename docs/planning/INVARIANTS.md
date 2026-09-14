# INVARIANTS — make the wrong thing unrepresentable

Started 2026-09-09 after GOTCHA #113 (the delay-0 label leak: an
invariant that lived in a default and a comment, invisible to every
probe built to find it). Every item here is a fact the system must hold
that is currently enforced by convention, a guard, or a comment — and a
plan to make it *structural*: one source of truth the other copies are
derived from, so the violation cannot be written.

**The heuristic:** each bug on this list lived where two representations
of the same fact were maintained separately (two frame conventions, two
delay keys, four flag lists, three decode paths, training vs live
embedders). A guard is a third copy. Prefer collapsing copies.

**Work rule:** one item per PR-sized change; each lands with the test
that pins it; status here is the ledger. While a training beam is live
on slanka, items are edit-only — compile/test when the beam exits.

Status: `[ ]` open · `[~]` in progress · `[x]` structural · `[g]` guarded only

---

## Tier 1 — each retires a bug class we hit this week

### 1. [x] Causal labels: state[t] pairs with controller[t+1] (STRUCTURAL 2026-09-09 evening)
- **Fact:** Slippi records the controller on the frame whose post-update
  state it produced. Delay-0 targets leak through the state (GOTCHA #113).
- **Done:** `Peppi.to_training_frames` (and `ReplayParser`) always emit
  `(state[t], controller[t+1])` via `Peppi.causal_pairs/1`; `frame_delay`
  / `action_delay` are reaction delay on top (default 0 = causal). The
  guard and `--allow-leaky-labels` are deleted — the leak is
  unrepresentable. `ExPhil.Data.LabelConvention` is the ONE owner of the
  numbering: checkpoints stamp `label_convention: :causal`; unstamped =
  legacy (reaction = d-1, leak = -1). Comparability keys count in
  reaction terms (legacy delay-1 == causal delay-0). The live
  `--frame-delay N` flag is deliberately NOT rebased (Slippi's setting;
  all deploy cards keyed on it): live N = reaction N-1, and the Agent
  derives delay-id from the checkpoint's convention instead of copying
  the flag. Drill numbers renumbered (old `--action-delay 2` == new 1;
  the multishine expert table is now issued-input; composed labels
  identical).
- **Test:** `label_alignment_test` (format fact + causal-at-0 +
  additivity + no leak buildable) and `label_convention_test` (the deploy
  cards as cases: ms_g19 d3/id3, v16e d1, v1/v2 = leak, causal-0 ==
  v16e).

### 2. [x] One flag table generates parser, defaults, and docs
- **Fact:** a flag must be parsed, defaulted, validated, and documented
  identically.
- **Today:** four hand lists (`@valid_flags`, `Parser` pipeline,
  `defaults/0`, TRAINING.md table). Found: `--num-heads`
  accepted-never-parsed, `--log-file` documented-rejected,
  `--transition-weight` plumbed-no-flag, `save_every||checkpoint_every`
  key collision.
- **Structural:** adopt `ExPhil.CLI.flag_definitions/0`'s shape (name,
  flag, type, default, doc, group) for training; derive `@valid_flags`,
  the parser, `defaults/0`, and a `mix exphil.flags --md` docs table from
  it. A flag without a parser cannot exist.
- **Cost:** ~1 day mechanical (≈300 flags). Do it flag-group by
  flag-group behind the existing parser until parity, then swap.
- **Test:** `flag_table_parity_test`: every definition parses a sample
  value; every `defaults/0` key has a definition; TRAINING.md table is
  byte-identical to the generated one (fails on drift).

### 3. [x] One spec per backbone (spec map carries defaults + build recipe + output rule; 25 bespoke clauses stay code; edifice registry family-derived; cross-repo pinned)
- **Fact:** an architecture needs defaults, a dispatcher clause, an
  output-size rule, and a stateful/carry capability flag.
- **Today:** 4 hand tables (exphil `backbone_defaults/1` — 69 of 97
  fall through to `[]`; the 97-clause dispatcher;
  `temporal_backbone_output_size/2`; edifice `list_families/0`).
- **Structural:** each edifice arch module exports `spec/0`
  (`%{defaults:, output_size: fn, stateful?:, carry?:, family:}`);
  registration requires it; exphil's tables become lookups. The
  "silently untuned" class becomes a compile error.
- **Cost:** ~1 day; can start with the 28 archs that already have
  defaults and make the fallback a `raise` for the rest.
- **Test:** registry_integrity: every registered arch has a spec; every
  dispatcher atom is registered; families == registry.

### 4. [x] Train-time-absent input channels are absent live
- **Fact:** the live embedder may only populate channels the training
  data provided values for.
- **Today:** embedding canary checks LAYOUT only; projectiles were
  zeros in training (Peppi hardcodes `[]`) and populated live for every
  fox_gen checkpoint. `EXPHIL_ZERO_PROJECTILES=1` is an env A/B.
- **Structural:** the parser stamps `provides: [...]` channels; the
  embed config is the intersection with the requested config and is
  saved in the checkpoint; the live embedder is BUILT from the
  checkpoint's config, so an unprovided channel has no input dims.
- **Cost:** medium (embed config plumbing; one new stamp in Peppi).
- **Test:** train a 1-file probe with projectiles absent; assert the
  live embedder for that checkpoint has no projectile block; and the
  reverse when a parser provides them.

### 5. [x] Ports exist only at the parse boundary (live path; offline instruments ratcheted)
- **Fact:** downstream code speaks subject/opponent, never port numbers
  (SubjectResolver law).
- **Today:** enforced at 2 chokepoints (get_players_ego raise, Peppi
  remap warning); violated in `projectiles.ex` (owner as absolute
  port), `async_runner.ex` SD/stock logic, `melee_port.ex` blind-CSS
  read, `game.ex:730` opponent = flip(1↔2).
- **Structural:** the parse/bridge boundary returns
  `%{subject: Player, opponent: Player}`; no `players[n]` map leaves
  it. Credo/grep check forbids `players\[[12]\]` outside `data/` and
  `bridge/`.
- **Cost:** medium-large (touches embeddings, runner, navigator).
- **Test:** the grep check as a test; projectile-owner embeds as
  `owner == subject` boolean.

## Tier 2 — divergences we measured

### 6. [x] One decision per game frame, frame-driven
- **Today:** runner spin-polls (31 calls/frame headless, ~2 live); a
  guard in `compute_action` re-sends the cached action; confidence
  logging is diluted by cached returns (0.03 = 1/31).
- **Structural:** the frame loop invokes the agent exactly once per new
  frame; the guard, the cached branch, and the dilution disappear.
- **Cost:** small-medium (async_runner inference_loop).
- **Test:** inferences == frames in a headless run's stats.

### 7. [x] One decode struct, passed by value
- **Today:** three paths (windowed / stateful / incremental) each
  rebuild `sample_opts`; incremental drops critic selector, hysteresis,
  steering; `reconfigure(style_tag:)` is a silent no-op.
- **Structural:** `%Agent.Decode{}` built once at load/reconfigure and
  threaded to every path; paths take the struct, no `Keyword.get`
  defaults at use sites.
- **Test:** property test that all three paths receive identical decode
  fields for the same agent state.

### 8. [x] Loss types, not loss knobs
- **Today:** smoothing × pos_weight poison (found 07-02) is prevented
  by `smoothed = targets` + a comment; four copy-pasted loss-opts
  blocks already drift (`head` default differs by path).
- **Structural:** `ButtonLoss` (plain BCE | weighted BCE) and
  `CategoricalLoss` (with optional smoothing) as structs; smoothing is
  not a field of ButtonLoss. One `loss_opts(config)` builder.
- **Test:** the four builders produce the same loss struct from the
  same config.

### 9. [x] Instrument labels through one helper
- **Today:** each probe/scan defines its own "the input at state S"
  label; two same-frame instruments read "calibrated" through a 1000x
  miscalibration.
- **Structural:** `ExPhil.Interp.Labels.exit_input(frames, i)` /
  `issued_input(frames, i)` return the SUCCESSOR frame's controller;
  scans and probes must use it (grep-test forbids
  `:array.get(i, ...).controller` as a label in `scripts/probe_*`).
- **Cost:** small. **Status:** in progress 09-09.

### 10. [x] Budgets at construction
- **Today:** `cache_streaming: true` default inherited into the bptt
  path filled `/` (300GB/epoch); fixed by hard-coding `false` at the
  seam.
- **Structural:** `EmbeddingCache.new(budget_bytes:)` required; writes
  past budget raise; the bptt seam passes an explicit budget (0).
- **Test:** cache refuses the write that would exceed its budget.

## Tier 3 — comparability

### 11. [x] Checkpoints carry a comparability key
- **Today:** label_delay, embed canary, loss recipe, train_delays are
  stamped inconsistently; delay-0 val losses were compared to causal
  ones (leaked targets are easier — optimistic by construction).
- **Structural:** `Checkpoint.comparability_key/1` = {label_delay,
  canary hash, loss recipe hash, train_delays}; registry/leaderboard
  tools refuse to rank across differing keys.
- **Test:** ranking two checkpoints with different keys raises.

---


## Tier 4 — added 2026-09-12 (the closed-loop validation surfaced them)

### 12. [x] One table of harness latency; delay-ids derived from it (STRUCTURAL 2026-09-12)
- **Fact:** a decision is applied some frames after the observed state
  (latency); training names the same quantity reaction delay (k = latency
  - 1); a delay-conditioned checkpoint's id d was trained at reaction
  d + the drill's `--pipeline-offset`. Every harness adds its own
  pipeline on top of its knob.
- **Was:** four undeclared constants — the drill's offset (2, never
  stamped), the async runner's pipeline (+2, a comment), the sync
  runner's (a July note, WRONG on this rig), the scenario suite's (+1,
  nobody knew) — and a live law (`id = --frame-delay - 1`) that was right
  only because two of them cancelled. Symptom class: a chain that
  re-enters and floats (GOTCHA #115 = #81's twin).
- **Done:** `ExPhil.Eval.HarnessRung` is the table (`latency/2`,
  `knob/2`, `reaction_delay/2`, `delay_id/3`, `aligned_knob/3`,
  `deploy_knob/2`, `describe/3`), each row cited to its measurement
  (suite grid 09-12; LatencyProbe vs Slippi recording 09-12 evening:
  sync fd N = N+1, async policy path = N+2; the chain pin `eval_runs/
  0912_sync_rung` fd3/id2 427/436 is ep57 playing one frame faster than
  its labels on the sync runner, GOTCHA #115 addendum 3). The drill
  stamps `delay_id_reaction_offset` (Checkpoint stamps 0 for every
  other trainer; unstamped delay-conditioned checkpoints are read as
  the recipe's 2, tagged `:assumed_drill`). Each harness DECLARES
  itself to the Agent (`harness:` + `harness_knob:`; sync/async runners
  and the suite wired); the Agent derives the id from the table or
  warns when an explicit id disagrees; `LabelConvention.live_*` /
  `delay_id/2` delegate to the table. The suite resolves its
  `--response-delay` from the checkpoint by default (can no longer be
  run one frame fast by omission).
- **Structural form landed (09-12 evening, Bradley: "one --reaction-delay
  knob, drill ids physical, bridge measures latency"):** the units are
  unified instead of translated. (a) Drill delay-ids ARE physical reaction
  delays (`--pipeline-offset` retired with a fold-it-in error; stamp 0).
  (b) ONE knob `--reaction-delay k` on both runners and the suite,
  resolved by `HarnessRung.resolve/3` (flag > deprecated `--frame-delay`
  alias > checkpoint's smallest trained rung) into each harness's own
  knob; refuses below the floor with the fix in the message. (c) The
  bridge MEASURES: `ExPhil.Bridge.LatencyProbe` sends one marker input in
  the countdown and reads the frame the game reports it on; sync runner
  raises on mismatch, async logs at error level
  (`--allow-latency-mismatch`). The table is now each harness's FLOOR
  plus a cross-check, not a translator.
- **Consequence (Bradley to confirm live):** both runners' floor is
  latency 2 = reaction 1. A reaction-0 checkpoint (v16e, v3's default)
  has NO exact live rung; `--frame-delay 0` is the nearest (one slower)
  and the 09-09 card (`--frame-delay 1`) is TWO slower. The one-frame-
  fast direction is the one that breaks, so the cards were safe, not
  aligned. v3 could instead train at reaction 1 and deploy at
  `--frame-delay 0` exactly.
- **Test:** `harness_rung_test` (the calibrations as cases),
  `label_convention_test` + `label_delay_test` (deploy cards re-derived).

### 13. [g] A stateful policy handed a mid-game state is warmed with the true history
- **Fact:** a windowed/stateful policy's decision depends on its
  history; an evaluation that starts it mid-game from an empty history
  measures a cold start, not the policy (09-12: ep57 could not chain at
  handoffs inside its own 438-chain game).
- **Done:** `Agent.observe/4` (observe-only advance, pinned observe-then-
  decide == play-then-decide) and the scenario suite observes every
  prefix frame with the recorded input. Guard: the Agent warns on a
  first decision at frame > window + 120 with an empty history.
- **Structural (open):** a handoff API that REQUIRES the history
  (`Agent.handoff(agent, frames, inputs)`) so a mid-game start without
  one is unrepresentable; the suite is the only such tool today.
- **Test:** `agent_observe_test` (equivalence + window fill + probe).


### 14. [x] Delayed labels come from the label source's OWN future (STRUCTURAL 2026-09-12 evening)
- **Fact:** a reaction-delay-k label answers "what does the expert do k
  frames after this state". For a RECORDED expert (fixture, human
  replays) that is the recording k frames later. For an EXPERT-labeled
  list (relabeled rollouts, openers, mined snippets, their redresses) the
  recording is the STUDENT's; where the student broke the loop, the
  label k frames later is a recovery input — the same state with two
  futures.
- **Was:** both kinds were plain `[%{game_state, controller}]` and the
  drill shifted both with `Data.shift_actions/2`. The pool audited clean
  at shift 0 and conflicted on 10 of 11 loop states at the trained shift
  (365/3a: fixture B+X 1.00 vs 3,185 rollout frames 0.10/0.06). Every
  DAgger round since g19 trained against it; the technique floor fell
  296 -> 111 -> 93 as rollout volume grew (RESULTS §8-9).
- **Done:** frames carry `:label_source` (`:recorded` | `{:expert, mod}`);
  `ExPhil.Training.Labels.at_delay/3` is the ONE producer of delayed
  labels — recorded lists shift along the recording, expert lists call
  `mod.label_ahead/4`; `Data.shift_actions/2` RAISES on an expert-tagged
  list. `MultishineExpert.label_ahead/4` is phase-indexed on the canonical
  cycle learned from the fixture (loop state at phase p -> table label at
  p+k; off-loop -> the current recovery held, or dropped). The drill tags
  relabeled lists and snippets and routes every shift through
  `Labels.at_delay`; the pool auditor audits through the same function at
  `--shifts`, and gates every prereg.
- **Test:** `labels_test` (the fixture as oracle: label_ahead == the
  recorded future on every loop frame for k 1..5; recorded vs expert-
  tagged twins agree through at_delay; shift_actions on an expert list
  raises; off-loop hold/drop), `multishine_expert_test` (cycle learned).
  First run: g26 (`eval_runs/0912_g26_phase`).
- **Amended 2026-09-13 (off-loop = ABSTAIN):** g26a showed item 14 alone
  did not restore the floor (67/min), and the closed-loop audit
  (RECOVERY_LABEL_CONFIRMATION) measured the off-loop HOLD projection
  wrong on 18/21 recovery frames at shift 4 (0/327 on-loop): recoveries
  take 7+ frames through lockout/startup and the loop-entry frame is not
  a function of the state. `label_ahead/4` now returns `:skip` off the
  loop for k > 0; `Labels.at_delay` defaults `off_loop: :drop`; `:hold`
  is an explicit legacy A/B mode (`dagger_drill --off-loop-labels hold`,
  `audit_ms_pool_labels --off-loop hold`, fingerprinted). Consequence:
  at a delayed rung the expert supervises ONLY the loop; recovery
  supervision must be a RECORDING of the teacher recovering
  (`--recorded-frames`, item-14 recorded branch) — the two label kinds
  are complementary, not substitutes. An expert without `label_ahead/4`
  raises at k > 0 unless `:hold` is asked for (a silent whole-list drop
  would be worse than the bug). Not yet run at the g-line rungs (3/4/5);
  the equal-budget hold-vs-drop comparison is open.
## What's left (2026-09-12 15:00 — v3 is gated on this list being empty)

Open: item 13's structural form (handoff API); the sync/async floor
means v3's deploy card needs Bradley's call (train at reaction 1 for an
exact `--frame-delay 0`, or accept one-slower). Everything else [x].

### Earlier (2026-09-09 20:30)

No open work items. Item 1 was reopened and taken to its maximal form on
Bradley's call (09-09 evening). Item 3 is closed in an explicitly
ACCEPTED form (the middle road: uniform dispatcher clauses as spec data,
bespoke ones stay code).

| item | accepted form (not reopened unless Bradley says so) | why not the maximal form |
|---|---|---|
| 3 (middle road, Bradley's call 09-09) | Exphil: `@backbone_specs` rows carry training defaults AND, for 70 of 97 backbones, the construction recipe (`build: {module, embed_key, params}`) + output rule (`output: {opt_key, default}`); `build_temporal_backbone` keeps 25 bespoke clauses and falls through to `build_from_spec`. Edifice: `@registry_by_family` is the only registry/family source. Cross-repo: every module the dispatcher aliases OR a recipe names must be a registered architecture. Tests: `backbone_spec_build_test` (all 70 construct; output rules resolve or raise loudly; spec-only keys never leak into defaults), parity test forbids a clause shadowing a recipe. | The 25 that stayed code deviate from the template (hybrids composing two modules, literal fixed args like griffin's `use_local_attention: true`, env-gated Mamba scan variants, post-processing like TCN's last-frame slice, the xlstm variant injection). A recipe language for those is an interpreter; they remain pinned by the parity/link tests. 45 spec-built backbones have NO output rule (they had no clause before either) — still a loud raise, now with the fix named in the message. |

## Ledger

**2026-09-12 evening — 12 structural form (one knob):** `--reaction-delay`
on cli.ex (+ `--allow-latency-mismatch`; `--frame-delay` deprecated
alias, default nil), `HarnessRung.resolve/3` / `trained_reactions/1` /
`delay_id_for_reaction/2`, Agent `reaction_delay:` (id derived from the
physical number; harness/knob path kept for legacy callers),
`ExPhil.Bridge.LatencyProbe` (pure; 6 tests) wired into `play_dolphin.exs`
(stats.latency_probe, raise on mismatch) and `AsyncRunner` (ets probe per
game, error log, `latency_measured`), suite `--reaction-delay`, drill
`--pipeline-offset` retired + offset stamped 0, gate scripts on
`GATE_REACTION` / `--reaction-delay 4`. 61 tests green.

**2026-09-12 15:00 — 12 STRUCTURAL, 13 guarded:** `ExPhil.Eval.HarnessRung`
(one latency table, delay-ids derived; rows cited to measurements) +
drill/Checkpoint stamp `delay_id_reaction_offset` + every harness
declares `harness:`/`harness_knob:` to the Agent + suite auto-aligns
`--response-delay` + `LabelConvention.live_*` delegate. Sync pin
(`eval_runs/0912_sync_rung`, ep57 stand 60s): fd3/id2 427/436, fd4/id2
3/106, fd2/id2 2, fd1/id1 1/2 (id-1 cells chain c3-5 at every knob:
an id-1 cold-start weakness, not rung evidence) -> sync pipeline == async
== 2; the 07-28 "sync d3 == async d2" note is RETRACTED for this rig.
**CORRECTION (same evening):** the LatencyProbe checked against Slippi's
recording shows sync fd N = latency N+1 and async (policy path) N+2 —
the July note stands; the chain pin was ep57 playing one frame faster
than its labels on the sync runner (GOTCHA #115 addendum 3).
`Agent.observe/4` + cold-start-mid-game guard. Small fixes ridden along:
break miner `--tail-margin` (r2@4303 class), AR-head confidence probe
now advances history (was probing an empty window). 47 tests green
across harness_rung / label_convention / label_delay / agent_*.

**2026-09-09 23:00 — 3 middle road (Bradley's call):** a strict template
parser over `backbone.ex` (alias; N x `Keyword.get(opts, k, literal)`;
one `Mod.build(embed + those params)`) converted 70 of 97
`build_<arch>_backbone` functions into `build:` recipes on their
`Config.@backbone_specs` rows (+ `output:` from the 26 matching
output-size clauses); `backbone.ex` 3,259 -> ~1,340 lines. Anything off
the template stayed a clause (25: sliding_window/attention, lstm_hybrid,
griffin (literal arg), hawk, xlstm x3 (variant injection), hopfield, ntm,
snn, bayesian, decision_transformer, tcn, spla, infllm_v2, lstm, gru,
gated_ssm, the 7 mamba variants, reservoir (non-compact), mlp). The
`@type backbone_type` atom alternation (a 4th hand list) is now
`atom()`. `Config.backbone_spec/1`, `backbone_recipe/1`,
`backbone_output_rule/1`, `spec_built_backbones/0`; `backbone_defaults/1`
strips the spec-only keys (they must never reach opts or the checkpoint
JSON). Generator lesson: a comment-eating regex with `.` under `/s`
deleted 2,800 lines on the first pass — restored from git, regex fixed
to `[^\n]`. 80 tests green (spec build x70, parity, link).

**2026-09-09 22:00 — 1 taken to the MAXIMAL form (Bradley's call):**
`Peppi.causal_pairs/1` runs inside `to_training_frames` and
`to_training_frames_with_stats` (and `ReplayParser`): every frame's
controller is the input issued from it; `apply_frame_delay` /
`Data.shift_actions` / `Data.batched`'s in-batch offset are reaction
delay on top. Guard + `--allow-leaky-labels` deleted (parser row, flag
docs, TRAINING.md regenerated). New `ExPhil.Data.LabelConvention`:
`of/1` (stamp or legacy), `reaction_delay/1`, `train_reaction_delays/1`,
`leaky?/1`, `live_frame_delay/1`, `live_reaction_delay/1`,
`delay_id/2`. Stamped by `Config.build_config_json` and
`Checkpoint`; `Comparability.key` counts in reaction terms (legacy
delay-1 == causal delay-0, proven in test); Agent keeps
`:label_convention`, derives delay-id at load and on reconfigure when
frame_delay changes without an explicit id (play script no longer copies
the flag into the id); eval_model shifts by reaction delay; MixFrames
compares in reaction terms and warns on pre-rebase exports; drills'
`--action-delay` default 2 -> 1, export stamps the convention;
ParseStats cutoff = delay+1. Multishine expert table is now issued-input
(tests moved one key earlier; composed drill labels identical).
Fixture tests updated for the dropped last frame. Suites: data,
comparability, config, mix_frames, agents/multishine, recovery_synth —
green; full stale sweep 1,356 green.

**2026-09-09 20:30 — 2 phase C and 3 phase C CLOSED (list empty):**
- **2 phase C [x]** `ExPhil.Training.Config.FlagDocs` renders the flag
  reference from `Parser.flag_table/0` (+ a `@docs` description map,
  180 seeded from the old hand table) and `write!/0` writes it between
  `<!-- flag-reference:start/end -->` markers in TRAINING.md.
  `flag_docs_test`: the committed section must equal a fresh render
  (drift fails the suite, message says the one command to run), every
  parser flag appears exactly once, undocumented flags ratchet (31, may
  only fall). The hand table above the markers is now prose history.
- **3 phase C [x]** Edifice: the 267-entry `@architecture_registry` map
  and the 257-line hand-written `list_families/0` (a second copy, in
  sync today by luck — 28 comment groups vs 26 families, `hybrid_builder`
  filed under `# SSM` but `:meta`) collapsed into ONE family-grouped
  `@registry_by_family`; the map, `list_architectures/0` and
  `list_families/0` derive from it and a compile-time check rejects
  duplicate names. `family_derivation_test` pins the partition; the
  existing `registry_integrity_test` builds all 267 (282 edifice tests
  green). Exphil: `edifice_registry_link_test` pins every `alias
  Edifice.*` in the dispatcher to a registered, loadable architecture
  (the dispatcher bypasses `Edifice.build/2`, so this was the only way
  a rename would have surfaced before build time). Dispatcher-as-data
  NOT done — see the accepted-form table.

**2026-09-09 18:15 — 5 STRUCTURAL (live path):** `GameState.subject_port/
opponent_port/subject/opponent` resolve roles at the boundary (stamped
own_port; opponent = the other OCCUPIED port, never a 1<->2 flip).
Migrated: projectile owner is now a `mine?` ROLE bit (was absolute port
x0.5), `Game.get_players_ego` opponent, the runner's SD/stock/dummy
ports, the critic-selector opponent. `port_boundary_test`: the live
decision path (agent, embeddings, projectiles, async_runner) must be at
ZERO absolute-port reads; 14 offline instrument/training-internal files
that operate on REMAPPED frames (subject == port 1 by construction) are
a ratchet allowlist (30 reads) that may only fall. Boundary files
(data/, types.ex, melee_port's CSS local-port read) are exempt.
**2026-09-09 17:30 — 4 and 8 STRUCTURAL:**
- **8 [x]** `ExPhil.Training.Imitation.LossConfig`: one typed value built
  by `from_config/1` (absent keys -> `Config.defaults/0`, present
  nil/false honored so drills keep exact semantics); ALL seven builders
  in `imitation/loss.ex` derive from it and reach `Policy.imitation_loss`
  only through `to_loss_opts/1`. The button loss is `%{pos_weight,
  focal, weight}` — smoothing is not a field, so the July smoothing x
  pos_weight poison is unrepresentable. Drift closed: head defaulted
  :independent (windowed) vs :autoregressive (bptt), precision :bf16 vs
  :f32, button_weight 1.0 vs 2.0, focal_gamma 2.0 vs 3.0. Tripwire test
  greps loss.ex for any inline `config[:knob]` extraction.
- **4 [x]** `ExPhil.Data.Peppi.provides/0` = [:players, :stage,
  :stage_internals, :distance] (NOT projectiles/items — it hardcodes
  `projectiles: []`). `Embeddings.config_for_source(opts, provides)` turns
  off unprovided channels (warns once) at all three training config sites
  (Pipeline x2, Imitation) — new checkpoints have NO projectile block
  (smaller embedding). The checkpoint JSON records `provided_channels` +
  the resolved `with_projectiles`; the Agent builds its live embed config
  and canary from them, and `Agent.zero_projectiles?/2` (pure, tested)
  zeroes live projectiles ONLY for checkpoints that have the block but
  whose source never provided it (every old fox_gen checkpoint); env
  `EXPHIL_ZERO_PROJECTILES=0/1` remains the A/B override. Probes read
  `with_projectiles` from the checkpoint. NOTE for v3: its embedding will
  be `288 - 5*projectile_dims` wide — the canary handles it; old
  windowed/eval scripts that hardcode 288/296 will need the stamp
  (eval_model.exs listed in FIXES).

**2026-09-09 16:45 — 6 and 11 STRUCTURAL:**
- **6 [x]** `AsyncRunner` inference is frame-driven: the frame loop
  `send`s `:new_frame` after writing the ETS state; the inference process
  blocks in `receive`, coalesces any backlog, decides once. Headless
  verification: **3,782 frames, 3,780 inferences** (was 117,045 for 3,766
  — 31x). Confidence stats are no longer diluted by cached returns. The
  agent's same-frame cache branch stays as defense for other callers
  (sync runner, probes). Staleness 0.3% (a slow decision can miss a
  frame instead of re-sending — acceptable, measured).
- **11 [x]** `ExPhil.Training.Comparability.key/1` = {label_delay, embed
  canary hash, loss-recipe hash (12 knobs incl. head), train_delays};
  stamped into every `_config.json` by `Config.build_config_json/2`;
  `Registry.best/1` returns `{:error, %RegistryError{reason:
  :incomparable}}` for mixed keys unless `allow_incomparable: true`.
  Tests: key stability across JSON round-trip, leaky-vs-causal refusal
  (the exact trap: leaky 1.8 "beats" causal 2.4), within-key ranking.

**2026-09-09 16:30 — PHASE B for 3 (exphil side):** `@backbone_specs` is
the single map (98 rows: every dispatchable backbone + the `:hybrid`
alias); `valid_backbones/0` = its keys; `backbone_defaults/1` is a
lookup that RAISES for unknown atoms (typos are "not a backbone", not
"baseline"); `@untuned_backbones` (74) is the explicit ratchet list
with baseline defaults. The 97-clause hand list, the `_ ->` fallback,
and the dead `:mamba_2`-vs-`:mamba2` split are gone (`:mamba_2` is a
spec row, dispatchable and valid). 340 config/harness tests green.
Still open for 3: the DISPATCHER (`build_temporal_backbone`, 97 bespoke
clauses) and the output-size table are not generated from the specs —
a test pins dispatcher atoms == spec keys; and edifice-side "register
requires spec/0" is not done (cross-repo).

**2026-09-09 16:00 — PHASE B landed for 1, 2, 7, 9 (compiled + tested):**
- **1 [x]** defaults flipped `frame_delay 0->1`, `action_delay 0->1`: the
  leaky pairing now needs `--frame-delay 0 --allow-leaky-labels` (two
  explicit opt-ins). Guard + agent warning + format test retained.
  (Full loader-level rebasing of delay semantics NOT done — it would
  shift every delay-conditioned checkpoint's `train_delays` meaning;
  the default flip + refusal gets the structural property with no
  blast radius. Recorded as the accepted form.)
- **2 [x]** Parser is TABLE-DRIVEN: 170 rows in `@flag_table`, one
  `apply_flag_table/3`, 10 explicit multi-key steps in
  `@special_flags`; `Config.@valid_flags` is DERIVED from
  `Parser.flags/0` (the hand list of 182 deleted). Accepted <=> parsed
  is unrepresentable. Round-trip verified across every row type; 340
  config/harness tests green. Docs generation from the table = phase
  C (open, listed below).
- **7 [x]** `%ExPhil.Agents.Decode{}` typed struct (`@enforce_keys`,
  validated at construction: temperature > 0, release <= press,
  mode_of_n >= 1) held on `state.decode`; built ONLY by `from_state/1`
  at init + reconfigure; all five decision sites derive options via
  `Decode.opts/3`. 131 agent tests green.
- **9 [x]** `label_source_test` grep-ratchets same-frame controller reads
  in `scripts/probe_*` / `*_scan.exs` (4 legacy instruments allowlisted
  with counts that may only fall; one descriptive read marked).

**2026-09-09 15:20 — all six new tests GREEN after v16e's beam exited**
(labels, cache budget, flag parity, backbone table parity, decode,
label alignment on `mewtwo_ground_neutral.slp`). Findings during the
run: `:h3`/`:ttt`/`:ttt_e2e` clauses lacked `:precision` (fixed); the
ratchet measured 78 baseline-reliant backbones (not the audit's 75);
labels_test fixtures must be functions (module attrs can't call defp).

| date | item | change | test |
|---|---|---|---|
| 2026-09-09 | 1 | guard + agent warning + format test (not yet structural) | label_alignment_test |
| 2026-09-09 | 7 | `ExPhil.Agents.Decode.sample_opts/2` is the only decode-option builder: windowed, stateful-step, incremental-SSM, single-frame (MLP), and warmup all call it (there were FOUR paths, not three) (incremental previously dropped mode-of-N / critic / hysteresis). `resolve_style_id/2` extracted from init and now ALSO applied in `reconfigure` (was a silent no-op; agent stores `:player_registry`). Struct-typed phase B (a `%Decode{}` threaded by value, no Keyword access at use sites) still open. Compile/test queued. | decode_test |
| 2026-09-09 | 3 (phase A) | Source audit: 97 dispatchable / 22 with defaults clauses / 27 dispatchable-but-rejected-at-CLI / 52 without output-size clause (loud CaseClauseError, not silent) / `:hybrid` valid-but-undispatchable / `:mamba_2` vs `:mamba2` spelling split. FIXED: `_ -> []` is now a warned generic baseline (+ `backbone_defaults_baseline?/1`); the 27 added to `@valid_backbones`. `backbone_table_parity_test` pins valid↔dispatch both ways (2 documented exceptions) and RATCHETS baseline reliance (75, may only fall). Phase B = per-arch `spec/0` in edifice + registry requires it. Compile/test queued. | backbone_table_parity_test |
| 2026-09-09 | 2 (phase A) | Source-audit script found: `--num-heads` accepted-never-parsed, `--log-file` documented-rejected, 7 mode aliases parsed-but-rejected, 17 parsed keys without defaults (14 = `no_*` negations). FIXED: num_heads/head_dim/log_file parsed + defaulted (4/64/nil — Trainer's 2/32 no longer wins), aliases + log-file added to @valid_flags. `flag_parity_test` now pins all four lists against each other (parser read as source; meta flags + other scripts' doc tables allowlisted). Phase B (table-generated parser/docs) still open; draft table in scratchpad. Compile/test queued behind v16e. | flag_parity_test |
| 2026-09-09 | 10 | `EmbeddingCache.save` refuses writes past `:budget_bytes` / `EXPHIL_CACHE_BUDGET_GB` / 50GB default ({:error, :over_budget}, warns once; callers already handle the error tuple). Compile/test queued behind v16e. | embedding_cache_budget_test |
| 2026-09-09 | 9 | `ExPhil.Interp.Labels` (issued_input/producing_input + shape-agnostic predicates); failed_exit_scan, probe_wait_exit, probe_sampler_wait migrated — no instrument computes its own label now. Grep-test for `frames[i].controller`-as-label in scripts/ still TODO. Compile/test queued behind v16e. | labels_test (unit, synthetic) |
