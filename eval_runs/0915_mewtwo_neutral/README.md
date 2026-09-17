# Mewtwo versus Fox neutral experiments

Scope: Fox on Final Destination, ordinary local timing (reaction delay 0,
measured input latency 1), sampled temperature 1. No combo-route or offstage
recovery objective. **No student is promoted yet.**

The actual learned weights are each round's `vN/candidate.bin`. The teacher is
`lib/exphil/agents/mewtwo_neutral_teacher.ex`; it generates demonstrations and
is not a hidden action override in the learned-policy runner. Each round stores
its exact training arguments, source data, checkpoint and gameplay reports.

## Development qualification

Each entry below lists actual first neutral openings in three fresh 30-second
games. Counter-hits and combo continuations do not add neutral wins. These are
small development checks, not estimates of human-match win rate.

| Round | Standing Fox openings | CPU-6 openings | Standing stock losses | Result |
| --- | --- | --- | --- | --- |
| 1 | 0 / 0 / 0 | 1 / 1 / 2 | 0 / 0 / 0 | Failed openings and wavedashes; standing samples repeated due to seed bug |
| 2 | 12 / 9 / 5 | 0 / 3 / 2 | 1 / 0 / 0 | Failed consistency and standing survival |
| 3 | 4 / 2 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | Failed active-opponent openings |
| 4 | 0 / 1 / 1 | 0 / 2 / 2 | 0 / 0 / 0 | Failed opening consistency; repeated out-of-range down tilts |
| 5 | 5 / 2 / 0 | 0 / 1 / 1 | 1 / 0 / 1 | Failed opening consistency and standing survival |

All completed rounds delivered inputs at latency 1. Use `summary_rescored.json`
for rounds 1 and 2, and `summary.json` for later rounds. Original scoring reports
are preserved. The graphical Mewtwo runner is not yet qualified.

## Data and remaining gates

Teacher validation evidence is in `../0915_mewtwo_teacher_audit/README.md`.
The user's whole FD game is assigned to training; nineteen audited successful
opening clips enter rounds 3 onward. The other four whole human recordings
remain held out. Exact per-game CPU levels in those recordings are unknown.

Promotion requires all six sampled games to produce an opening, all requested
move categories to appear, no prolonged shield hold, no standing-opponent SD,
valid input delivery, and a separate ordinary-speed graphical check. Protocols
were declared before each fit in `docs/planning/MEWTWO_NEUTRAL_BASELINE_V1.md`.
Do not substitute a deterministic run or relax the gates to promote a candidate.

## Current conclusion

Round 5 fits its demonstrations closely (best weighted training loss 0.00727),
but fresh sampled games still fail. More fitting loss alone is not the next gate.
The next experiment should collect validated teacher corrections from states
the student actually visits: out-of-range down-tilt attempts and movement near
an edge. Keep the student's preceding inputs as history-only context, then
supervise only the teacher takeover; audit successful corrections before fitting.
No such corrective-data round has started. No Mewtwo graphical qualification has
been run, and no checkpoint should yet be advertised as a reliable neutral bot.
