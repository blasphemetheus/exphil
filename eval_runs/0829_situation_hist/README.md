# A1 + B2 first run — situation → next-option histograms, bot vs expert

Script `scripts/situation_hist.exs`; full tables in `RESULTS.md`. Expert =
800 erickfm ranked Fox games (299,100 option events, port 1). Bot sets:
ep10 vs CPU (8), buttons-0.6 vs CPU (8), mode-of-16 vs CPU (8), ep10 T=1.0
vs Bradley (12, 08-28), ep10 T=0.5 vs Bradley (7, 08-28), B1/B2/B3 vs
Bradley (7/10/5, 08-29). Ran 2026-08-29 23:06 — under one minute.

## 1. The bot has no ground movement game (the biggest A1 finding)

| option, all frames | expert | ep10 cpu | ep10 human | B1 human |
|---|---|---|---|---|
| **dash** | **31.6%** | 1.2 | 0.8 | 1.6 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| aerial | 14.1 | 5.8 | 5.6 | 6.5 |
| double_jump | 15.4 | 5.8 | 5.3 | 3.0 |
| special (shine/laser/illusion/firefox) | 9.9 | 21.6 | 19.1 | 17.0 |
| **grab** | **2.2** | **16.3** | 13.3 | 12.9 |
| **spotdodge** (in neutral) | **0.7** | **15.3** | 10.6 | 8.8 |
| shield_on | 6.2 | 6.7 | 8.9 | 8.9 |
| wavedash | 2.2 | 6.2 | 6.0 | 4.8 |
| missed_tech | 1.4 | 1.8 | 7.1 | 11.9 |
| options / min | 124.5 | 76.5 | 69.8 | 70.8 |

Verified on raw action states, not just the option detector: the expert
spends **9–12% of frames in DASHING and 1–2% in RUNNING** (40–57 dash
entries/min); the bot spends **0.1% in DASHING and 0.0% in RUNNING** (0.6–1.5
dash entries/min). It moves by wavedash, jump, and specials only. This is
one mechanism behind several complaints at once: no dash-dance (nothing
to bait with), grab from standing/shield instead of from a dash
(shield-grab spam), spotdodge as the substitute for dash-back (B2 the
"spot-dodger"), and a third fewer decisions per minute than the expert.
Open question for interp: is it the 17-bucket stick discretization at
T=0.5 never producing a neutral→full-tilt flick in one frame, or a
learned preference? (A dash needs the stick past ~0.8 within a frame.)

## 2. The pummel loop is "never throws", not "pummels too long"

`pummel_throw_decision` (grab held, throw available):

| option | expert | ep10 cpu | ep10 human | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| throw | **86.1%** | 28.1 | 8.5 | 14.3 | 20.0 | 35.7 |
| shield_on (grab broke → shield) | 9.8 | 21.3 | 40.4 | 33.3 | 32.5 | 28.6 |
| spotdodge | 0.5 | 22.5 | 29.8 | 14.3 | 15.0 | 17.9 |
| jab / special / smash | 0.6 | 23.6 | 14.9 | 33.3 | 25.0 | 10.7 |

The expert throws 86% of the time from a grab; the bot 9–36%. It holds
until the grab breaks, then shields or spotdodges. This answers A3
without running it: the fix is not "cap pummels", it is that **throw is
not being selected** — a decode mask that forces a throw after N pummels
would directly reproduce expert behaviour here.

## 3. Off-stage: airdodge instead of up-B

`recovery_low`: expert special (up-B / side-B) 59.8%, airdodge 4.7%; the bot
airdodges 20–60% (B1 35%, ep10 human 35%, btn0.6 60%). `being_edgeguarded`:
expert special 61%, bot 13–43%, airdodge 16–32% vs 12%. The unforced
off-stage deaths in `sd_scan` (55% of base's deaths) have their mechanism.

## 4. B2 — TV distance as a decode ranker: ordering matches the humans and the live rung

Mean total-variation distance to the expert over reported situations:

| set | mean TV | situations |
|---|---|---|
| B2 human | **0.46** | 17 |
| B1 human | 0.47 | 14 |
| B3 human | 0.47 | 15 |
| ep10 cpu (T=0.5/0.5) | 0.53 | 17 |
| ep10 T=1.0 human | 0.54 | 18 |
| ep10 T=0.5 human | 0.56 | 19 |
| buttons 0.6 cpu | 0.57 | 18 |
| **mode-of-16 cpu** | **0.72** | 4 (only 128 events in 8 games) |

- The three 3-epoch continuations sit closest to the expert (0.46–0.47) and
  ep10 further (0.53–0.56) — the same ordering as Bradley's "all three beat
  ep10". First offline metric that agrees with the human read.
- mode-of-16 is the farthest (0.72) with 19.8 options/min vs 124 — the
  narrowing shows up as distance, where pass@1 rewarded it. **B2 would have
  rejected mode-of-N offline.** Adopt as the decode ranker, paired with the
  live gate per L9.
- Caveat: bot sets are 5–12 games; per-situation n is 20–900. Differences
  under ~0.05 TV are inside what D1 will measure as noise.

## Status for EVAL_DIRECTIONS

A1 ran; B2 ran (adopt); A3's core question answered by A1 (§2); C4's
mechanism found (§3). Next: interp on the dash deficit (§1 open question),
throw-after-pummel mask as a decode experiment, D1 noise floors.
