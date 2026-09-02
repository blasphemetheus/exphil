# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(expert::auto, CRITIC::auto, BASE::auto).
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


### CRITIC

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 0 | — | — | — | — | — | — | — |
| 20-40 | 0 | — | — | — | — | — | — | — |
| 40-70 | 0 | — | — | — | — | — | — | — |
| 70-100 | 0 | — | — | — | — | — | — | — |
| 100-140 | 0 | — | — | — | — | — | — | — |
| 140+ | 0 | — | — | — | — | — | — | — |


### BASE

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 0 | — | — | — | — | — | — | — |
| 20-40 | 0 | — | — | — | — | — | — | — |
| 40-70 | 0 | — | — | — | — | — | — | — |
| 70-100 | 0 | — | — | — | — | — | — | — |
| 100-140 | 0 | — | — | — | — | — | — | — |
| 140+ | 0 | — | — | — | — | — | — | — |


Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).
