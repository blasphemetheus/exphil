# 0828 loop/taunt rescore — the buttons sweep was NOT null

Retroactive rescore of **85 already-banked replays** with a new
instrument (`ExPhil.Interp.LoopStats`, `scripts/loop_report.exs`). No
new games were played; no GPU was used.

## Why

Two decode brackets (`ddbce62`, `1db23d6`) returned null, and the
recorded reason was that armed-approaches/min is rare and bursty — the
baseline arm moved **7x across days** (0.10 vs 0.69) at n=5. Separately,
the 08-28 human session flagged **taunts**, **laser/shield-grab/pummel
loops** and **dithering** — none of which `coach_report` measures at all.

So the sweeps were scored on a noisy metric that was also the wrong
metric. This rescore points the instrument at the human's actual
complaint, using events that are *dense per game* (button presses,
frame-runs) rather than rare (conversions).

## Headline: buttons temperature has a clean dose-response

`eval_runs/0828_buttons_temp`, n=5/arm, the exact same 20 games the null
sweep scored:

| arm (buttons T) | d_up press/min [range] | taunts/min | held-action frac | loops/min | max loop reps |
|---|---|---|---|---|---|
| 0.5 | **112.27** [95.4–119.5] | 2.25 | 0.26 | 1.57 | 10.8 |
| 0.6 | **170.54** [157.1–179.1] | 1.17 | 0.24 | 1.96 | 10.4 |
| 0.7 | **247.85** [236.5–259.4] | 2.94 | 0.30 | 1.96 | 12.0 |
| 1.0 | **430.16** [421.8–450.3] | 2.15 | 0.34 | 2.35 | 13.4 |

**d_up presses/min separates completely.** The four ranges are
**disjoint** — no arm's range touches another's — monotone in
temperature, and 3.83x end to end. This is the first arm-separating
result the buttons sweep has produced, and it comes from games that were
already on disk.

Everything else moves in the same direction but does not clear the 2x
law: held-action fraction 0.34 → 0.26 (1.3x), loops/min 2.35 → 1.57
(1.5x), max loop repeats 13.4 → 10.8. Consistent, not resolved.

**Executed taunts do NOT separate** (1.17–2.94, no order). That is the
power argument demonstrated inside a single dataset: the same pathology
is invisible when counted as a rare effect (~2–6 taunts/game) and
unmistakable when counted as its dense cause (~200–900 presses/game).

## The new metric is reproducible across days; the old one was not

Same configuration (buttons T = 1.0), three different days and contexts:

| source | d_up press/min |
|---|---|
| buttons sweep btn1.0 (08-28, vs CPU) | 430 [422–450] |
| epoch sweep, all 10 epochs (08-26, vs CPU) | 422–495 |
| human session (08-28, vs a person) | 455 [398–646] |

Within ~±10%, across days, opponents and checkpoints — against the
armed/min baseline's 7x swing between two of those same sessions. **This
metric can carry an A/B; armed/min could not.**

## Two controls that anchor the scale

**Argmax (known-bad).** The `async_ep10_clean` / `async_ep10_ctrl2` runs
— the deterministic-decode games that the 08-26 handoff describes as
"crouches, then runs off the stage four times" — score **frozen-input
fraction 0.98 and 0.99**, versus ~0.00 for every sampled arm. The
mode collapse everyone agreed was happening is now a number, and the
metric has a known-bad endpoint.

Note that frozen-input fraction is ~0.00 for *all* sampled arms: under
stochastic decode the sticks resample every frame, so nothing ever
freezes. Its value is as an **argmax/lock detector**, not as a
discriminator among sampled arms.

**A human (known-good).** In the 08-28 session, port 1 is the bot (455
d_up/min, pummel loops) and port 2 is Bradley (**0.00** d_up/min). A
human presses the taunt button zero times per minute; the bot presses it
seven times per second.

## The dominant pathology, named

The top repeated cycle in every corpus is the same one:

| corpus | `GRAB_WAIT>GRAB_PUMMEL` episodes |
|---|---|
| buttons sweep (20 games) | 78 |
| epoch sweep (53 games) | 94 |
| human session (12 games) | 16 |

It outnumbers every other detected cycle combined. From the human's
side of the 08-28 session, `GRABBED>GRAB_PUMMELED` appears 21 times —
the bot pummel-looping a real person. This is the "pummel loop" from the
human session's notes, now counted.

## Caveat: raw loop COUNT is not a pathology score

The human scores **9.64 loops/min to the bot's 0.71** — because
dash-dancing (`TURNING>DASHING`, 170 episodes) is a repeated cycle and
also correct Melee. The cycle detector finds repetition, and repetition
is not automatically bad. Rank on **specific patterns**
(`GRAB_WAIT>GRAB_PUMMEL`), never on the bare `loops/min` number.

## Caveat: this measures cost, not competence

These metrics score the pathology side only. The 0827 per-head bracket
found colder buttons trade passivity for deaths (2.0 → 3.6 deaths/run at
the full cold schedule). So the correct reading is: **we now have a
clean dose-response curve on the cost axis**, to be paired with a
competence axis before picking an operating point. Colder buttons
demonstrably buy fewer taunts and less repetition; whether that is worth
its price is still a human-gated call — but it is now a call made
against a measurement instead of an impression.

## Training-side note

`lib/exphil/networks/policy/loss.ex:302` documents that the taunt button
carries `pos_weight = 30`, and that composing it with label smoothing
moved the BCE optimum *above* the press threshold ("observed live as
constant taunts"). The smoothing half of that bug is fixed; the w=30
weight itself remains, and is the training-side candidate for the d_up
press rate that decode temperature is currently paying to suppress.

## Reproduce

```bash
mix run scripts/loop_report.exs --per-game --out DIR 'eval_runs/0828_buttons_temp/btn*/r*.slp'
mix run scripts/loop_report.exs --bot-port 2 'eval_runs/0828_session/*.slp'   # the human
```

Outputs: `buttons_sweep/`, `epoch_sweep/`, `human_p1/` (bot), `human_p2/`
(human control), each with `report.md` + `report.json`.
