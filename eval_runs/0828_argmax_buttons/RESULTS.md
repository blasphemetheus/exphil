# RESULTS — buttons-decode bracket (fox_gen_v1 ep10)

Read `PREREG.md` first; the decision rule below was fixed before any game
was played and is applied here mechanically.

5 arms x 8 games x 120s, CPU dummy, delay 0, sticks sampling at 0.5.
Ran 2026-08-28 14:56 → 16:28 on a quiet machine.

## The replication check passed

PREREG predicted, from the 0828 rescore: **base d_up ≈ 430/min, range
[420, 450]**.

Measured: **431.89 [415.84–454.10]**.

That is the fourth independent session in which buttons-T=1.0 lands at
~430 d_up/min, against an `armed/min` baseline that moved 7x between two
sessions. `btn0.5` also replicated: 110.43 [105.7–114.7] here vs 112.27
[95.4–119.5] in the sweep. **The instrument is sound.**

## Axis 1 — cost

| arm | scored/played | d_up press/min | frozen-input | held-action | loops/min |
|---|---|---|---|---|---|
| base (T=1.0) | 8/8 | 431.89 [416–454] | 0.00 | 0.36 | 2.26 |
| btn0.6 | 8/8 | 183.39 [164–197] | 0.00 | 0.30 | 1.53 |
| btn0.5 | 4/8 | **110.43 [106–115]** | 0.00 | 0.25 | 1.47 |
| detbtn | 1/8 | 0.00 | **0.86** | 0.60 | 3.43 |
| detbtn_hyst | 0/8 | — | — | — | — |

## Axis 2 — competence

| arm | n | armed/min | conversions | deaths/run |
|---|---|---|---|---|
| base | 8 | 0.37 | 3/15 (20%) | 1.12 |
| btn0.6 | 8 | 0.31 | 8/26 (31%) | 1.62 |
| btn0.5 | 4 | 0.73 | 6/15 (40%) | 2.00 |
| detbtn | 1 | 0.00 | 0/3 (0%) | **4.00** |

## Game duration — the decisive measurement

From the run LOGS, which replay truncation cannot touch:

| arm | mean game | games reaching the 120s cap |
|---|---|---|
| base | 124.6s | **8/8** |
| btn0.6 | 124.6s | **8/8** |
| btn0.5 | 124.1s | **8/8** |
| detbtn | **72.0s** | **0/8** |
| detbtn_hyst | **71.3s** | **2/8** |

24/24 sampled-button games ran to the cap; 2/16 argmax-button games did.
The sampled arms' frame counts span 74 frames total (they all hit the
timer); the argmax arms range 1206–7427. This is a categorical difference
in *whether the games finish*, not a ratio of two noisy means, so the <2x
law does not apply to it in the usual way.

Games end early because the bot loses its stocks: the one scoreable
`detbtn` game shows **deaths=4, armed/min=0.0**.

## Verdict, by the pre-registered rule

**Step 1 — argmax-lock guard (frozen-input > 0.20 disqualifies).**
`detbtn` scores **0.86**, four times the threshold, in the same regime as
the known-bad full-argmax runs (0.98–0.99). **DISQUALIFIED.**
`detbtn_hyst` produced no scoreable replay so its frozen-input could not
be computed; it is rejected on the duration evidence instead (71.3s mean,
2/8 reaching the cap), and that non-measurability is recorded rather than
papered over.

**PREREG's falsification branch has fired.** It said: *"if `detbtn`
scores frozen-input near 0.9, the zero endpoint is unusable and the answer
is a finite temperature."* That is exactly what happened. Hysteresis did
not rescue it. **`--deterministic-buttons` collapses the policy** — the
same failure as full argmax, reached through the buttons alone.

**Step 2 — competence gate (not detectably worse than base).**

| arm | deaths vs base | conv vs base | passes? |
|---|---|---|---|
| btn0.6 | 1.62 / 1.12 = 1.45x | 31% vs 20% | ✅ |
| btn0.5 | 2.00 / 1.12 = 1.79x | 40% vs 20% | ✅ (close) |

Both clear the 2x band. Per the prereg's own declaration the competence
differences are **within noise and must not be read as one arm converting
better** — the gate only asks "not detectably worse", and both pass.

**Step 3 — take the coldest survivor: `btn0.5`.**
110 d_up/min vs base's 432 — a **3.9x reduction in the taunt impulse** at
no measurable competence cost.

**Step 4 — a human look.** Required before this becomes the deploy decode.
Nothing is crowned on CPU numbers (the g6 lesson).

## The cost gradient is real, and it points somewhere

deaths/run rises monotonically as buttons get colder: **1.12 → 1.62 →
2.00 → 4.00**. Colder buttons buy fewer taunts and less repetition, and
they are paid for in deaths. `btn0.5` sits at 1.79x — inside the gate but
near it. So T=0.5 is close to the useful end of this knob, and `detbtn`
is past the cliff. That is consistent with the 0827 finding that colder
buttons trade passivity for aggression "not competence", and with the
Bernoulli mechanism: as T falls, every sub-0.5 press dies, not just the
taunt.

**Recommended next decode: `--buttons-temperature 0.5`, pending your live
look.** `btn0.6` is the conservative alternative (183/min, deaths 1.45x,
full 8/8 coverage).

## HARNESS BUG FOUND — 19 of 40 replays lost, and the loss is BIASED

`eval_live_protocol.sh:155-157` copies the newest `.slp` out of
`~/Slippi` **without waiting for Dolphin to finalize it**:

```sh
newest=$(ls -t "$HOME"/Slippi/*.slp ... | head -1)
cp "$newest" "$OUTDIR/r$i.slp"
```

A copy race. Every damaged file has a **page-aligned size** (a multiple of
4096) — the signature of an unflushed tail — while every intact file has
an arbitrary size. 19 of 40 replays here are unparseable.

**The loss is not random. It is biased against the arms that perform
worst**, because a game that ends early gives Dolphin less time to
finalize. Coverage was 8/8, 8/8, 4/8, 1/8, **0/8** — perfectly ordered by
how badly the arm did. `detbtn_hyst` produced zero scoreable replays and
is therefore invisible to every replay-based metric in this repo.

That is the worst possible shape for a data-loss bug: **the harness
silently deletes the evidence of failure.** Any replay-scored comparison
made while it was live is inflated in favour of bad arms, because their
bad games are simply missing. This bracket only survived it because game
duration is recorded in the LOGS.

Fixed in this commit (wait for the file size to stabilize before copying)
and recorded as a GOTCHA. `LoopStats.safe_load/2` now skips and counts
unreadable replays instead of crashing, and `loop_report` prints
scored/played coverage per arm.

## Note on the knob assertions

`base` and `detbtn` printed KNOB ASSERTION FAILED. Both are FALSE ALARMS:
the config banner writes an ANSI reset between the label and the value, so
a literal `"Deterministic buttons: false"` never matches. Both arms were
verified by hand to have run with the intended decode, and all replays
were collected. The assertion now strips ANSI before matching.
