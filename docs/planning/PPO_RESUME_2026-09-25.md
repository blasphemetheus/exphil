# PPO resume — September 25

Claude's results and the human playtests were read before resuming. The user
preferred Mewtwo PPO iteration150; iteration300's edge-roll/grab/back-throw
strategy is documented in DEGENERATE_ZOO. The frozen-prior win-rate result
does not establish broad strength. Neither checkpoint is promoted.

## Completed review and changes

- `scripts/degeneracy_check.py`: same-harness prior comparison with hashed
  sources, sample counts, explicit missing-data outcomes and retrospective
  thresholds. Iter300 matches the compound signature;150 receives a reduced-
  aerial warning;70 has no known signature. Five targeted Python tests pass.
  Report: `eval_runs/0924_mewtwo_ppo/eval/degeneracy_report.json`.
- Existing Mewtwo eval chain now runs that report automatically. It is not a
  validated universal detector, human-style gate or in-training tripwire.
- Both tested policies registered with parent `n8T_RBQtCMg`:
  `mewtwo-gru-ppo-v1-i150` (`xj-iSqZKs9Q`, user-preferred/drift-warning),
  `mewtwo-gru-ppo-v1-i300` (`usHd0wOuM3s`, human-reported-degenerate).
  Artifact hashes, human verdict sources and evaluations are attached.
- Live replay scans: five recordings per policy, no parser errors. Iter150:
  bot16 deaths,16 no-hit-in-90f flags; opponent3 deaths/3 flags. Iter300:
  bot16 deaths/11 flags; opponent4 deaths/4 flags. Reports in
  `eval_runs/0925_mewtwo_review/iter{150,300}_sd.md`. These heuristic labels
  include failed recovery after older hits; quit/reset and unequal durations
  also need review before treating counts as comparable rates. The claimed
  simulator survival improvement has NOT been established against humans.
- PPO now accepts `--opponent-heads` (comma-separated). Uses existing
  `OpponentPool` uniform historical sampling; the prior is always included.
  One opponent per rollout batch. Character, exact prior path and parameter
  shapes checked. SHA256 manifest in `opponents.json`; selected opponent in
  every metrics row. The KL anchor remains the original imitation prior.
- Evaluator accepts `--opponent-head`, checks character/prior and records it in
  summary. This enables a held-out-opponent panel with a matched prior control.
- Verification:3 targeted PPO tests pass;12-iteration GPU smoke exercised all
  four opponents (prior4, iter70×3, iter150×4, iter300×1), all numeric metrics
  finite, head/trainer checkpoints saved. Tiny4-game evaluation against iter100
  completed both ports (all draws at120-frame cap, a plumbing test only).

## Next experiment

`scripts/mewtwo_pool_chain.sh`, unit `exphil-mewtwo-pool-v2`.
See [live phase/status](MEWTWO_POOL_LIVE_STATUS.md) and
`eval_runs/0925_mewtwo_pool/status.json`; log `logs/exphil-mewtwo-pool-v2.log`.

Fresh training from the same imitation prior, up to200 iterations, KL0.01,
same optimizer/rollout recipe as v1, seed925. Uniform prior/70/150/300 pool
tests resistance to earlier strategies, including the zoo entry. Critic begins
from the existing Mewtwo fit and continues updating under the changed matchup
distribution. Do not interpret its old EV as validation on the new pool.

Final (or guard-stopped) head is evaluated on fresh seed9251 against the prior
and against v1 iter100, which is excluded from training. Each panel has100
candidate games and100 matched prior-control games, balanced ports; drift
reports compare within panel. This is exploratory, below the historical200-game
promotion gate, and all opponents still share one character/trunk/training
lineage. It is not an independent broad opponent league. No head selection uses
these test games. Export: `checkpoints/mewtwo_ppo_pool_v2_policy.bin`.

No claim that opponent diversity alone will fix the behavior. Results and
Dolphin play must decide. Remaining work: wider opponent-policy support,
matched in-training behavior checks, harness-confounded cstick measurements,
and a Mewtwo human-style corpus. Higher-KL comparison and Fox follow-ups remain
separate experiments; this arm keeps KL unchanged to avoid changing both at once.

While the chain is active, no Mix calls or edits to library files or scripts it
uses. STOP file saves training and then continues the queued evaluation; stop
the whole systemd unit to cancel all stages. All work remains uncommitted.

## Mewtwo resource-aware recovery panel (user follow-up)

Added standalone CPU scripts `resource_recovery_cases.exs` and
`resource_recovery_eval.exs`, with `scripts/lib/resource_recovery.exs`.
Guide: `docs/guides/RESOURCE_RECOVERY_EVAL.md`. Three standalone tests pass.
Fox's old probe/model cannot classify Mewtwo: its plans/physics are Fox-specific.
The new grid uses drift, available jump, teleport and air dodge; full savestates
preserve resource legality. Reports stratify by remaining jumps, action phase,
character/opponent/stage. No-witness means unknown, never proven checkmate;
neutral-opponent witnesses do not establish safety against an active edgeguard.

Smoke: live iter150 Battlefield f323 seeded exactly and found recoveries
(`eval_runs/0925_mewtwo_recovery/smoke`). A 32-case panel from recent iter150/300
human recordings is running separately, log `logs/mewtwo_recovery_live.log`,
outputs `eval_runs/0925_mewtwo_recovery/live`. This does not load a policy or
alter the active PPO chain. Next layer: score policies on the same fixed
snapshots and add adversarial edgeguard responses; neither is implemented yet.

Running at this update: PPO unit `exphil-mewtwo-pool-v2`, iteration12 starting,
and CPU recovery panel (terminal session50123). No Mix/library changes.

USER CLARIFICATION: checkmate means unable to return without future opponent
assistance, not adversarial/chess mate. A later rescue hit can take the fighter
out of checkmate. Wants a static deterministic state function, revised when
counterexamples expose missing information. The new search is offline
calibration only; Mewtwo static physics/resource model remains to implement.
Adversarial edgeguard responses are NOT a prerequisite. Exact contract is in
RESOURCE_RECOVERY_EVAL.md. Neutral-input witnesses need assistance filtering
because ongoing attacks/projectiles can still rescue the fighter.

CPU panel completed: 32/32 neutral-input recovery witnesses, zero divergent
seeds. These early-episode cases are too easy to validate a checkmate boundary.
They predate assistance filtering; do not use as unassisted-recovery ground
truth. Added conservative damage/hitstun-increase exclusion and a regression
test (4/4 standalone tests pass). Zero-damage interactions remain uncovered.
Revised one-case smoke: `eval_runs/0925_mewtwo_recovery/smoke_no_assist`.
The long PPO unit remains running independently; inspect its live-status/log
for the latest iteration. No static Mewtwo checkmate implementation claimed.
