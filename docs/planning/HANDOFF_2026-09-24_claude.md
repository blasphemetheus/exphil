# HANDOFF 2026-09-24 — Claude's thread (R3 evaluation)

Author-suffixed because two agents are handing off today: Astra writes
`HANDOFF_2026-09-24.md` for its own thread (Mewtwo imitation, registry
backfill). **This file covers only my work.** Where we overlap, read the
coordination log `docs/planning/COORDINATION_CLAUDE_ASTRA.md` — it is the
channel, append-only, entry format at the top.

Repo `/home/blewf/git/exphil`, branch `main`.

## 1. State of the world (verified 11:42)

- **NOTHING IS COMMITTED, deliberately.** Bradley's instruction on 09-23 was
  "continue its work without committing". `main` is still at `60d97d9b` (my
  coordination doc). Everything since — Astra's PPO fixes, its registry
  backfill and Mewtwo work, my evaluation script and results — lives in the
  working tree only. Do not commit without asking; do not `git checkout` or
  `git stash` anything, you would destroy two agents' days of work.
- **Running right now:** unit `exphil-r3-control`, started 11:40, ~13 min,
  log `logs/exphil-r3-control.log`, output `eval_runs/0923_ppo/eval_control/`.
  It holds the GPU. Result section §3 has what to do with it.
- Also up: `exphil-viewer-8012` (page server, mine, useful) and
  `exphil-msl-static` (port 8011, mine, unused — stop it any time).
- Astra's `exphil-mewtwo-il-v1` **finished at 06:03** (`phase:
  imitation_complete`, best val loss 3.1043,
  `checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin`). The GPU was
  free when I started the control. **Check before you start GPU work** —
  Astra said its next step is held-out test plus Dolphin validation.

## 2. Task list, verbatim, with status

1. **R3 win-rate gate: ≥200 complete games vs the frozen prior, win rate
   >60 %** — **DONE, PASSED.** 182W / 18L / 0D = **91.0 %**, Wilson 95 % CI
   86.2–94.2 %. §3.
2. **Harness control (prior vs prior must land near 50 %)** — Astra's 32-game
   control returned exactly 16W/16L. **My 200-game version is running now**
   to tighten the interval; it is confirmation, not a new claim.
3. **R3 style half: fingerprint stays inside the human range** — **NOT DONE.**
   This is the next real task. §5.
4. **Wire `ExPhil.Melee.Checkmate` into the GA edgeguard term** — not started.
5. **GA run with the corrected kill rule** — never exercised. The rule is
   ported (`GA.earned_stocks/3`, uncommitted in the working tree since 09-23).
6. **Handoff** — this file.

## 3. The R3 result, and why I believe it

`scripts/ppo_r3_eval.exs` (mine, new, untracked) plays the trained head
against the untouched prior. The challenger is the prior's frozen trunk with
Astra's PPO head loaded at runtime through `Agent.put_head_params/2`.

| measurement | result |
| --- | --- |
| my eval, `head_iter200`, 200 games | **182W 18L 0D — 91.0 %**, CI 86.2–94.2 % |
| Astra's independent eval, same head | 180W 17L 3D — corroborates |
| Astra's control, prior vs prior, 32 games | 16W 16L — exactly 50 % |
| my control, prior vs prior, 200 games | RUNNING, see `eval_control/summary.json` |
| games decided by stock-out | 200 / 200 (no timeouts) |
| by challenger port | p1 90 %, p2 92 % — not a port artifact |

Design choices, so you can defend or attack the number:

- Real games to stock-out; cap 28,800 frames is Melee's own 8-minute timeout,
  decided by stocks then percent exactly as the game does.
- The challenger plays port 1 for the first 100 games and port 2 for the
  second, because the prior is port-conditioned.
- The verdict requires the **Wilson lower bound** above 60 %, not the point
  estimate, so a marginal result cannot be talked over the line.

**Two independent implementations agreeing at ~91 %, with a 50 % control, is
strong evidence the win rate is real.** What it does NOT establish:

- **Transfer.** Bradley reported **four consecutive losses** to the exported
  candidate in real play, calling it his best version played. Astra logged
  this as a user report, not replay-verified. A 91 % sim win rate against one
  frozen opponent and a human beating it are not contradictory: the opponent
  pool is a single frozen policy, and exploiting one fixed opponent is the
  easiest thing RL does. Treat sim win rate as necessary, not sufficient.
- **Style.** Task 3 is exactly this and is unrun.
- The head at iteration 200 may not be the best head. Training reward peaked
  near iteration 81 and drifted after. Astra's position, which I agree with:
  select an earlier checkpoint only on **fresh** games, never on the test
  games already used.

## 4. Findings and decisions not recorded elsewhere

- **The coordination protocol works and is worth keeping.** Append-only log,
  fixed entry shape (STATUS / OWNS / RUNNING / DID / NEXT / ASKS), claims by
  timestamp, heartbeats because an idle agent and a busy one look identical
  from outside. Astra adopted it immediately and answered by number. It
  caught a real near-miss: I read its entry and learned a Mewtwo GPU job
  existed before assuming the machine was mine.
- **Astra corrected my root cause from 09-23.** The PPO nil error was never
  the gradient. Line 323 had shifted under my own edits onto
  `Polaris.Updates.apply_updates`, whose optional third argument defaults to
  nil, which Nx then tries to traverse as a JIT argument. That is why every
  isolated probe I ran passed. `HANDOFF_2026-09-23.md` §5 still carries my
  wrong diagnosis; the correction is in the coordination log and here.
- **Astra's parallel threads** (do not duplicate): a registry backfill of 765
  records over 839 total (`docs/reference/RECENT_BOTS.md`), and a Mewtwo
  imitation campaign — 418 deduped games, 5.24M frames, 334/42/42 split, GRU
  2×1024, 10.46M params, finished this morning.

## 5. Exact next action

1. **Read the control.** When `exphil-r3-control` exits:
   ```bash
   python3 -c "import json;d=json.load(open('eval_runs/0923_ppo/eval_control/summary.json'));print(d['wins'],d['losses'],d['draws'],round(d['win_rate']*100,1),d['wilson95'])"
   ```
   Expect ≈50 % with the interval spanning it. **If it comes back far from
   50 %, the 91 % is an artifact and the gate is not passed** — say so loudly
   rather than defending the earlier number.
2. **Run the style half** (task 3). The challenger's fingerprints are already
   written: `eval_runs/0923_ppo/eval_iter200/fingerprint.jsonl`, 16 rows, in
   the row shape `scripts/sim_prior_play.exs` emits, so
   `scripts/sim_r1_compare.exs` reads it unchanged. Compare against the
   prior's own spread in `eval_runs/0921_sim_r1/anon_self_n10/` and the human
   range used for the R1 verdict. The six tells are jump_x_ratio, short_hop,
   c-stick aerial rate, aerials/min, rolls and spotdodge. The preregistered
   bound is in `RL_ON_PRIOR.md`. Astra also wrote fingerprints for its own
   runs under `eval_runs/0923_ppo/eval_candidate/fingerprints.jsonl`.
3. Then either R3 is fully passed and the remaining question is transfer
   (Dolphin, against Bradley), or the style half fails and the KL coefficient
   needs raising for a rerun.

## 6. Gotchas

- **`:erlang.term_to_binary` on GPU-resident tensors produces a file that
  cannot be decoded back.** My first control run died on exactly this with
  `decode failed, none of the variant types could be decoded` from the agent.
  Always `ExPhil.Training.PPO.to_binary_backend/1` before serializing params.
  Astra's checkpoints do this; my hand-rolled one did not.
- **`scripts/launch_unit.sh UNIT 'mix run …'`** for anything long —
  `systemd-run --user` starts in `$HOME`, so a bare command dies instantly
  with no log. It prints the unit's first log lines, which is how I saw the
  failure above.
- One EXLA process on the GPU at a time. Check
  `systemctl --user list-units 'exphil-*'` and the coordination log first.
- `mix run` recompiles changed `.ex` files and replaces the EXLA NIF, which
  kills any live training beam. Safe when nothing is training; never while
  Astra has a GPU job up.
- Do not edit files Astra lists under OWNS: `lib/exphil/sim/ppo.ex`,
  `scripts/ppo_r3.exs`, `scripts/ppo_0923_run.sh`,
  `test/exphil/sim/ppo_test.exs`, plus its Mewtwo scripts and the registry.
  `scripts/ppo_r3_eval.exs` is mine.

## 7. Resume

```bash
cd ~/git/exphil
date '+%H:%M'; systemctl --user list-units 'exphil-*' --no-legend   # what is running
tail -25 docs/planning/COORDINATION_CLAUDE_ASTRA.md                  # what Astra said last
git log --oneline -3; git status --short | grep -v '^??' | head      # still uncommitted?
tail -4 logs/exphil-r3-control.log                                   # the control's verdict
```

Then §5. Append an entry to the coordination log before touching the GPU or
any file Astra owns.
