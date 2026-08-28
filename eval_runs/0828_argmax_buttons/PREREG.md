# PRE-REGISTRATION — buttons-decode bracket (fox_gen_v1 ep10)

Written **before the bracket ran**. Read this before the numbers.

Run: `bash scripts/argmax_buttons_bracket.sh` (5 arms x 8 games x 120s,
CPU dummy, delay 0, sticks sampling at scalar 0.5).

## Question

The 0828 rescore (`eval_runs/0828_loop_rescore/RESULTS.md`) established a
clean dose-response: colder button temperature monotonically reduces the
d_up press rate (T=1.0 → 430/min, 0.5 → 112/min, disjoint ranges). It did
not test the **zero-temperature endpoint**. `--deterministic-buttons`
("argmax the buttons while sticks sample") has been in the CLI unused and
unbracketed the whole time.

**Does the cold end of the curve keep paying, or does it enter the argmax
failure regime?**

## Arms

| arm | decode |
|---|---|
| `base` | buttons raw at T=1.0 — the current decode, the anchor |
| `btn0.6` | `--buttons-temperature 0.6` |
| `btn0.5` | `--buttons-temperature 0.5` |
| `detbtn` | `--deterministic-buttons` (T→0 endpoint) |
| `detbtn_hyst` | `--deterministic-buttons --press-threshold 0.6 --release-threshold 0.4` |

The hysteresis arm exists because argmax's known failure is **holding**:
argmax over all heads produced frozen-input fraction 0.98 and the
crouch-SD collapse. Hysteresis turns presses into edges rather than
holds and is documented as applying to argmax button modes only
(`sampling.ex:130`).

## Metrics

**Axis 1 — cost** (`scripts/loop_report.exs`): d_up presses/min
(primary), taunts/min, `GRAB_WAIT>GRAB_PUMMEL` episodes, held-action
fraction, frozen-input fraction.

**Axis 2 — competence** (`scripts/coach_report.exs`): armed
approaches/min, conversion rate, deaths/run, passivity.

## Decision rule (pre-registered)

1. **Argmax-lock guard, applied first.** Disqualify any arm with
   **frozen-input fraction > 0.20**. The known-bad argmax runs score
   0.98–0.99 and every healthy sampled arm scores ~0.00, so this
   threshold sits in empty space between the two regimes. A
   disqualified arm is out regardless of how good its other numbers
   look — this is the guardrail against re-shipping the crouch-SD
   collapse.
2. **Competence gate, second.** An arm passes only if it is *not
   detectably worse* than `base`: deaths/run must not rise by ≥2x and
   conversion rate must not fall by ≥2x (the project's own <2x
   resolution law used as the noise band).
3. **Among arms passing 1 and 2, pick the COLDEST** (lowest d_up/min).
4. The winner goes to a **human look**. Nothing is crowned on CPU
   numbers — the g6 lesson stands.

## Declared in advance

- **We expect the competence axis to be UNRESOLVED at n=8.** Armed/min
  is bursty and its own baseline moved 7x across days. The rule above is
  built to tolerate that: it asks only "not detectably worse", never
  "better". An unresolved competence axis is a PASS, not a failure, and
  must not be reported as one arm beating another.
- **Expected `base` value: d_up ≈ 430/min, range ≈ [420, 450].** This is
  a prediction from the rescore, and the bracket re-measures it. If
  `base` lands far outside that band, something about the setup or the
  instrument changed and the whole bracket is suspect — treat this as a
  built-in replication check, not as a finding.
- **Falsification of the cold-end premise:** if `detbtn` scores
  frozen-input near 0.9, the zero endpoint is unusable and the answer is
  a finite temperature. That is a clean, informative result and should be
  recorded as such.
- **A null is possible and is not a failure.** If every arm passes the
  guard and the gate, and the cost axis separates as expected, the
  decision is mechanical (take the coldest). If the cost axis does *not*
  separate, that contradicts the rescore and the instrument needs
  re-examination before any decode change ships.

## Knob assertion

Each arm greps its own run log for the decode banner it asked for
(`Deterministic buttons: true`, `buttons: 0.6`, `press=0.6`) and marks
itself `KNOB ASSERTION FAILED` if absent. A silently dropped decode flag
would make two arms identical and produce a **fake null** — the
flag-drop bug class this repo already paid for once (guard #6). The
`play_dolphin_async` banner was extended in this commit to echo
`deterministic_buttons` and the hysteresis thresholds, which it did not
previously report at all.

## Mechanism note (from slippi-ai, read 2026-08-28)

slippi-ai samples every component at a **single global T=1.0** and has
**no argmax path at all** (`slippi_ai/eval_lib.py:402`; the only `argmax`
calls in the repo are one-hot extraction and a Q-learning training
target). So the arm we are testing is outside their design space, not
behind it.

More useful is a mechanism they expose without commenting on: their
buttons are **Bernoulli** and their sticks are **Categorical**
(`slippi_ai/jax/embed.py:126-133` vs `:256-260`) — and ExPhil's per-head
temperature has the same split (`sigmoid(logits/t)` for buttons). Those
two respond to temperature differently:

* Categorical: `logits/T` flattens toward **uniform** as T rises.
* Bernoulli: `sigmoid(logit/T)` pulls p toward **0.5** as T rises, and
  toward **0 or 1** as T falls.

For a rare button — a negative logit, p well under 0.5 — button
temperature is therefore not a "diversity" knob at all. It is close to a
direct **press-rate dial**: raising it pushes a rare press UP toward 0.5,
lowering it pushes it toward 0. That is exactly the monotone
dose-response the rescore measured (430/min at T=1.0 → 112/min at 0.5),
and it makes a sharper prediction for this bracket:

**`detbtn` should drive d_up presses to ~0**, because argmax presses a
button only when p > 0.5 and the taunt button is nowhere near that. The
real question for that arm is therefore not "does it stop taunting" —
it should — but **what legitimate sub-0.5 presses it also silences**
(Z-grab and shine timing are the ones to watch in the competence axis).
