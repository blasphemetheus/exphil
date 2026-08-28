# 0828_livelook_btn05 — Bradley's live look at `--buttons-temperature 0.5`

Human-gated rung for the argmax-buttons bracket winner
(`eval_runs/0828_argmax_buttons/RESULTS.md`). fox_gen_v1 ep10, local
windowed, FD, `--frame-delay 0 --temperature 0.5 --buttons-temperature 0.5`,
bot port 1, Bradley port 2. Replays land in `2026-08-Mainline/` (Dolphin's
own subfolder under `--replay-dir`). 11 files written, 7 full games
(>1 MB; the 4 ~100 KB files are CSS restarts) — symlinked in `full/`.
No page-aligned sizes (GOTCHA #102 check passed).

Comparison arm: the 08-28 morning session, same policy, same human, same
day, buttons T=1.0 (`eval_runs/0828_session`, 11 full games).

## Loop / taunt axis (`loop_report`, bot port 1)

| arm | games | d_up press/min | taunts/min | held-action | loops/min | max reps | GRAB_WAIT>GRAB_PUMMEL episodes |
|---|---|---|---|---|---|---|---|
| T=1.0 (0828_session) | 12 | **455** [398-646] | 1.17 [0-3.13] | 0.34 | 0.71 | 2.9 | 16 (1.3/game) |
| T=0.5 (this) | 7 | **100.6** [87-125] | 1.79 [0.49-2.53] | 0.25 | 1.16 | 5.3 | 10 (1.4/game) |

- The d_up **impulse** dropped 4.5x, disjoint ranges — the vs-CPU
  dose-response reproduces against a human.
- Actual **taunt entries did NOT drop** (1.17 -> 1.79, ranges overlap:
  unresolved). Pummel-loop episodes per game unchanged. The knob cuts the
  press rate, not the idle-taunt or grab-loop *events* the human sees.

## Competence axis (`coach_report --char fox --bot-port 1`)

| arm | games | armed/min | conversions | deaths/game | dropped/game | neutral_loss/game | passive/game |
|---|---|---|---|---|---|---|---|
| T=1.0 | 11 | 0.98 | 8/87 (9%) | 3.64 | 1.3 | 7.1 | 0.7 |
| T=0.5 | 7 | 0.50 | 7/61 (11%) | 3.57 | 2.6 | 10.7 | 1.4 |

- Deaths identical (both arms lose nearly every stock — the human wins).
- Dropped conversions and neutral losses roughly double at T=0.5: colder
  buttons plausibly cost follow-up presses. n=7 vs 11, single day; all
  under-2x differences except dropped are unresolved by the standing law.

## Verdict

NOT crowned on numbers: T=0.5 cures the *mechanism* metric it was
selected on, not the human-visible pathology (taunt entries, pummel
loops), and shows a possible dropped-conversion cost. Bradley's
impression is the gate — recorded below.

Bradley's impression (08-28, after 7 games): "definitely less taunty, less loopy, and I got hints of it playing a very fast, scrappy game. It did feel like it was occasionally playing the game, but a lot of the time it would just throw out punishable options or taunt or shield grab repetitively. Sometimes it would do a short hop drill shine grab — that was pretty good."

**Read:** the human gate PASSES on the taunt/loop feel (contradicting the taunt-entry count, which is the coarser instrument at n=7); the residual complaint is SELECTION — punishable options thrown out, repeated shield-grab — i.e. the model contains the good sequences (SH drill-shine-grab) and picks them too rarely. That is the Leg S / critic question, not a temperature question.

**Decision:** `--buttons-temperature 0.5` becomes the DEFAULT decode for fox_gen_v1 live play (not a crown — v1 has no crown; a recipe note).
