# PPO live status — Codex, 2026-09-23

## Verified follow-up — 2026-09-23 20:46 CDT

Nothing is training or evaluating now. The only active `exphil-*` units are
`exphil-viewer-8012` and `exphil-msl-static` (page servers). The automatic
status block below is the final 15:15 snapshot, not a currently running watcher.

- Training completed all 200 iterations at 14:31 CDT; final KL 0.0616.
- Our held-out, port-balanced evaluation: **180 wins / 17 losses / 3 draws**
  over 200 games (90%); control prior vs prior: 16 wins / 16 losses.
- Claude's separate evaluation also completed: **182 wins / 18 losses**
  over 200 games (91%); reported Wilson 95% interval 86.2–94.2%, all games
  ended by stock-out. Results: `eval_runs/0923_ppo/eval_iter200/summary.json`.
  This corroborates the win-rate result; the full style gate remains open.
- Our descriptive style comparison: forward rolls 2.92 → 1.75/min,
  backward rolls 2.12 → 1.40/min; spotdodges 3.56 → 3.68/min (not improved).
  Aerials increased 19.02 → 33.82/min, above the human corpus's descriptive
  95th percentile of 24.76/min. First-30-second sim fingerprints and full-game
  human data are not directly matched, so these are concerns to investigate,
  not a new formal pass/fail threshold.
- Candidate export: `eval_runs/0923_ppo/eval_candidate/candidate_policy.bin`.
  No production policy was promoted. Dolphin/human transfer remains untested.
- The early CPU evaluation completed (23/32 wins at iteration 40), but its
  optional final style-report command failed because `python3` was absent
  from the service shell PATH. Game results survived. The final GPU style
  report succeeded and is available; this auxiliary failure does not affect it.

Next: validate style and real-game transfer before considering promotion.

<!-- PPO_COMPLETION_0923_V1 -->
## Automatic run status

Updated automatically at 2026-09-23T20:15:00+00:00.

Completed all 200 iterations.
Service state: `inactive`; result: `success`.
Completed metric rows: 200; final recorded iteration: 200.

| Latest metric | Value |
| --- | --- |
| reward | 0.15387210249900818 |
| kl | 0.06159571185708046 |
| entropy | 3.4759068727493285 |
| clip_frac | 0.0035026043566176667 |
| value_ev | 0.7115922670404027 |
| vloss | 0.0021931000519543886 |
| ms | 24399 |

Latest head checkpoint: `eval_runs/0923_ppo/v1_fixed/head_iter200.bin` (527159 bytes).
Latest trainer checkpoint: `eval_runs/0923_ppo/v1_fixed/trainer_iter200.bin` (3955055 bytes).

### Services present at this update

```text
exphil-msl-static.service          loaded active running [systemd-run] /nix/store/59avw9bz0ga2pqbaspc1isy5ridiwkg8-python3-3.12.13-env/bin/python3 -m http.server 8011 --bind 127.0.0.1
exphil-ppo-0923-completion.service loaded active running [systemd-run] /nix/store/59avw9bz0ga2pqbaspc1isy5ridiwkg8-python3-3.12.13-env/bin/python3 /home/blewf/git/exphil/scripts/ppo_completion_watch.py
exphil-viewer-8012.service         loaded active running [systemd-run] /nix/store/59avw9bz0ga2pqbaspc1isy5ridiwkg8-python3-3.12.13-env/bin/python3 -m http.server 8012 --bind 127.0.0.1
```

### Evaluation pipeline

```json
{
  "stage": "completed",
  "updated_at": "2026-09-23T20:14:43.357754+00:00",
  "checkpoint": "eval_runs/0923_ppo/v1_fixed/head_iter200.bin",
  "control": {
    "head": null,
    "seed": 920230,
    "policy": "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin",
    "games": 32,
    "outcomes": {
      "loss": 16,
      "win": 16
    },
    "win_rate_all_games": 0.5,
    "win_rate_decisive": 0.5,
    "mean_reward": 0.19906584814190856,
    "by_candidate_port": {
      "1": {
        "loss": 9,
        "win": 7
      },
      "2": {
        "loss": 7,
        "win": 9
      }
    },
    "max_frames": 28800,
    "r3_gate": "NOT ASSESSED: requires human-range identity and defense checks; timeouts are not wins"
  },
  "candidate": {
    "head": "eval_runs/0923_ppo/v1_fixed/head_iter200.bin",
    "seed": 920230,
    "policy": "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin",
    "games": 200,
    "outcomes": {
      "draw": 3,
      "loss": 17,
      "win": 180
    },
    "win_rate_all_games": 0.9,
    "win_rate_decisive": 0.9137055837563451,
    "mean_reward": 2.7119199706792823,
    "by_candidate_port": {
      "1": {
        "draw": 2,
        "loss": 9,
        "win": 89
      },
      "2": {
        "draw": 1,
        "loss": 8,
        "win": 91
      }
    },
    "max_frames": 28800,
    "r3_gate": "NOT ASSESSED: requires human-range identity and defense checks; timeouts are not wins"
  },
  "style_report": "eval_runs/0923_ppo/eval_candidate/style_comparison.json",
  "note": "R3 style/human-range gate remains unassessed; no production promotion"
}
```

eval_smoke: 4 game results recorded.
eval_prior_control: 32 game results recorded.
eval_candidate: 200 game results recorded.

### CPU checkpoint check

cpu_prior_control completed:
```json
{
  "head": null,
  "seed": 920240,
  "policy": "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin",
  "games": 32,
  "outcomes": {
    "draw": 2,
    "loss": 15,
    "win": 15
  },
  "win_rate_all_games": 0.46875,
  "win_rate_decisive": 0.5,
  "mean_reward": -0.1514790336787703,
  "by_candidate_port": {
    "1": {
      "draw": 1,
      "loss": 11,
      "win": 4
    },
    "2": {
      "draw": 1,
      "loss": 4,
      "win": 11
    }
  },
  "max_frames": 28800,
  "r3_gate": "NOT ASSESSED: requires human-range identity and defense checks; timeouts are not wins"
}
```

cpu_candidate40 completed:
```json
{
  "head": "eval_runs/0923_ppo/v1_fixed/head_iter40.bin",
  "seed": 920240,
  "policy": "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin",
  "games": 32,
  "outcomes": {
    "draw": 3,
    "loss": 6,
    "win": 23
  },
  "win_rate_all_games": 0.71875,
  "win_rate_decisive": 0.7931034482758621,
  "mean_reward": 1.4799532257020471,
  "by_candidate_port": {
    "1": {
      "loss": 3,
      "win": 13
    },
    "2": {
      "draw": 3,
      "loss": 3,
      "win": 10
    }
  },
  "max_frames": 28800,
  "r3_gate": "NOT ASSESSED: requires human-range identity and defense checks; timeouts are not wins"
}
```


Training completion is not an R3 gate pass. The preregistered human-range
style checks remain outstanding. No production policy was promoted.

Claude can take over from these artifacts. Check for other active training
services before invoking Mix or starting GPU evaluation.

<!-- END_PPO_COMPLETION_0923_V1 -->

## Ownership and coordination

The user is handing ongoing work to Claude. Codex changed `lib/exphil/sim/ppo.ex`, `scripts/ppo_r3.exs`,
`scripts/ppo_0923_run.sh`, and `test/exphil/sim/ppo_test.exs` for this run.
Do not compile or run another GPU training process while this run is active.
The user requested PPO progress and unattended training while away.
Unrelated untracked artifacts and `scripts/regret_traces.exs` are untouched.

Status watcher: `exphil-ppo-0923-completion` runs independently and updates
the generated block at the top of this file every 30 seconds, including
current services, checkpoints, metrics, early stops and failures. It stays
alive through the evaluation pipeline. Edit outside the generated markers.

## Queued work (13:30 CDT)

`exphil-ppo-0923-eval` is waiting for training to exit. It then runs:

1. Four-game, 1,800-frame smoke on the latest saved head, exercising both
   ports, policy export/reload and both players' style fingerprints.
2. A 32-game prior-vs-prior control, balanced equally between ports.
3. A 200-game candidate-vs-prior evaluation, 100 on each port, with seeds
   separate from training and a 28,800-frame cap. Timeouts are reported
   separately and never counted as wins. Fingerprints use the first 1,800
   in-game frames and do not yet constitute the human-range gate.

Runner: `scripts/ppo_after_train.py`; evaluator: `scripts/ppo_eval.exs`.
Stage record: `eval_runs/0923_ppo/eval_pipeline_state.json`.
Logs: `logs/ppo-eval_smoke.log`, `logs/ppo-eval_prior_control.log`,
`logs/ppo-eval_candidate.log`. Results live in matching `eval_runs/0923_ppo/`
directories (`summary.json`, `games.jsonl`, `fingerprints.jsonl`, and
`candidate_policy.bin` when evaluating a trained head).

The pipeline skips evaluation if the training STOP file exists, waits if
another BEAM GPU process is present, and stops with a recorded failure on
any failed stage. No compilation is requested for the evaluation commands.
To cancel queued evaluation alone: `systemctl --user stop exphil-ppo-0923-eval`.
Avoid starting conflicting GPU work while this queued service is enabled.

Syntax checks passed for both Python services and the Elixir evaluator.
A CPU-only two-game evaluator smoke PASSED at 13:28 CDT without Mix or
compilation, using the existing BEAM modules, four scheduler threads and
the iteration-40 head. Verified 2 game records, 4 fingerprints (both roles
on both ports), policy export and successful reload. Both short games hit
the 800-frame cap; they were correctly recorded as timeouts, not wins.
Log: `/tmp/ppo-eval-cpu-smoke.log`.

Additional running service: `exphil-ppo-0923-cpu-check`, capped at four CPUs,
uses CPU-only EXLA (`CUDA_VISIBLE_DEVICES` empty) with no Mix/compilation.
It runs 32 full-length prior-control games then 32 iteration-40 candidate
games, each balanced by port. This early diagnostic overlaps GPU training
without using GPU memory; CPU/GPU arithmetic can differ, so it is distinct
from the final GPU evaluation. Log: `logs/ppo-cpu-check.log`; results:
`eval_runs/0923_ppo/cpu_prior_control` and `cpu_candidate40`.
The generated status block records its results too.

`scripts/ppo_style_report.py` additionally writes `style_comparison.json`
for the candidate evaluations: candidate/prior means for 12 identity and
habit measures, alongside descriptive human-corpus percentiles from the
existing Fox fingerprint datasets. The report is explicitly not a new
acceptance threshold: human fingerprints cover full games, while the sim
sample is the first 1,800 in-game frames. The helper was exercised on the
CPU smoke output and both human datasets before queuing it.

At 13:30 CDT, the first 8 full control games completed with real terminal
outcomes (2 wins / 6 losses from candidate port 1). This is only the first
quarter of a port-balanced control, not evidence for a PPO improvement.
GPU training reached iteration 50 and saved its head and trainer state;
CPU evaluation temporarily raised some iteration times to ~33 seconds.

## Launch snapshot (live progress is in metrics.jsonl)

- First complete PPO iteration passed in `eval_runs/0923_ppo/repro`.
- Three targeted regression tests passed: optimizer update, GAE boundary
  semantics, real autoregressive head gradients and parameter changes.
- Two-iteration smoke PASSED: `/tmp/ppo-smoke-fixed.log`, output
  `eval_runs/0923_ppo/smoke_fixed`. Post-update KL 0.0027 / 0.0033, first
  iteration clip fraction 0.0067. Both head and trainer checkpoints saved.
- Full run active since 13:09 CDT: 64 envs × 600 frames × 200 iterations, two PPO epochs,
  minibatch 3840, LR 3e-5, frozen `fox_v3_1_step8_mix4` opponent/prior,
  R2 `refit_v2/critic_best.bin` initialization. No improvement claim yet.
- Verified 13:19 CDT: iteration 23 completed, ~24 seconds/iteration,
  post-update KL 0.0247 (stop threshold 0.5), checkpoints at 1/10/20.
  At this rate 200 iterations finish around 14:30 CDT, unless the KL
  guard or a requested STOP ends it earlier. Read metrics.jsonl for live
  progress; this timestamped note is only a snapshot.

## Root cause and changes

The handoff's attribution of the nil error to `grad_fn` was incorrect.
Current source line 323 was `Polaris.Updates.apply_updates`, whose default
third argument is nil. Nx 0.13.1 tries to traverse that default as a JIT
argument. The gradient had succeeded. Wrapping apply_updates inside a
two-argument JIT fixes it; no flattening of batches was needed.

Also fixed critic tracing (features/targets are explicit JIT arguments),
GAE bootstrapping at a truncated rollout, resetting completed games and
recurrent rows, partial minibatches, and critic shuffling once per pass.
The stop guard measures KL after the whole update, rather than an average
of stale minibatch measurements. Metrics append every iteration. Saves
include the first iteration, final iteration and guard/STOP exits, plus
head and trainer state (critic and both optimizers).

## Monitor / stop

Once launched, unit: `exphil-ppo-0923-v1`.
Log: `logs/exphil-ppo-0923-v1.log`.
Metrics: `eval_runs/0923_ppo/v1_fixed/metrics.jsonl`.
Checkpoints: `head_iter*.bin` and `trainer_iter*.bin` in that directory.

Create `eval_runs/0923_ppo/v1_fixed/STOP` to save and stop after the current
iteration. Do not delete or overwrite the prior. No checkpoint is promoted
to production automatically. R3 still requires ≥200 evaluation games with
win rate >60% and the preregistered style checks in `RL_ON_PRIOR.md`.

The desktop's Ollama process uses ~6.6 GB GPU memory. This run caps EXLA's
reservation at 35% of the GPU; Ollama is left running.
