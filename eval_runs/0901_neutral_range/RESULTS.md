# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(expert::auto, bot_v12ar:1).
toward = mean signed stick-x toward the opponent (deadzone-free mean);
%twd/%awy past a 0.3 deadzone; option columns = share of neutral frames in
those action states (dash/run/turn; grounded blaster 341-344; aerial
blaster 345-348; jumpsquat+rise).

### expert

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 89374 | 0.095 | 34.5 | 24.0 | 7.7 | 0.1 | 0.2 | 13.0 |
| 20-40 | 77674 | 0.129 | 36.2 | 22.2 | 18.1 | 0.4 | 0.9 | 16.3 |
| 40-70 | 50147 | 0.222 | 41.9 | 17.1 | 26.8 | 1.8 | 3.6 | 14.3 |
| 70-100 | 10955 | 0.435 | 56.8 | 7.7 | 13.6 | 2.6 | 6.2 | 9.0 |
| 100-140 | 3223 | 0.558 | 67.9 | 7.4 | 5.1 | 0.6 | 2.9 | 5.0 |
| 140+ | 833 | 0.609 | 70.5 | 8.0 | 7.3 | 0.8 | 3.8 | 6.7 |


### bot_v12ar

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 20846 | -0.062 | 25.1 | 32.3 | 0.1 | 4.1 | 2.8 | 1.0 |
| 20-40 | 17469 | -0.158 | 21.6 | 39.0 | 0.1 | 5.1 | 1.7 | 1.1 |
| 40-70 | 11659 | -0.202 | 20.8 | 42.8 | 0.3 | 3.8 | 2.4 | 1.0 |
| 70-100 | 4660 | -0.126 | 22.5 | 35.8 | 0.2 | 9.7 | 2.3 | 1.3 |
| 100-140 | 2044 | -0.020 | 27.6 | 28.8 | 0.2 | 7.3 | 2.7 | 0.6 |
| 140+ | 863 | 0.051 | 23.2 | 15.3 | 0.2 | 22.5 | 5.7 | 0.6 |


Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).

## F3c VERDICT — Bradley's pushback CONFIRMED; the 08-31 corpus-marginal story RETRACTED

The expert's neutral game at range is APPROACH, not retreat/laser zone:
toward rises monotonically with distance (0.095 → 0.609 at 140+; %twd
70.5 vs %awy 8.0 at range), with a dash-dance band at 40–70 (dash 26.8%,
twd/awy balanced) and SH/aerial lasers as a modest seasoning (laserA
peaks 6.2% at 70–100), not a game plan.

The bot inverts nearly every column: toward NEGATIVE at every bucket
below 140; dash 0.1–0.3% (vs 7.7–26.8); jump ~1% (vs 9–16); aerial laser
below expert exactly in the SH-laser band (1.7–2.4 vs 3.6–6.2 at
40–100) while grounded laser is ABOVE expert everywhere (4–22% vs
0.1–2.6) — standing lasers as another spam option, consistent with
Bradley's "no SH laser / FH double laser" observation.

Consequence for F3b's interpretation: the trunk's all-range retreat is
NOT a faithful copy of the corpus marginal — the corpus marginal at
those distances says approach. BC failed to reproduce the expert's
distance-conditional steering in the bot's own reached states (same
family as the closed-loop dash-dance drift finding, 08-29). This moves
weight from "curation must add approach data" toward "the data is
there; training/decode loses it" — raising the stakes on plan (c) and
on the closed-loop-drift line.

Caveat: opponent confound — expert rows are expert-vs-expert states,
bot rows are bot-vs-Bradley states; the F3b counterfactual probe is the
controlled half of this pair (same states, placed opponents), and both
agree on the bot side.
