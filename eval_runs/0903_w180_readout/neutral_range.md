# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(w180:1, v14:1).
toward = mean signed stick-x toward the opponent (deadzone-free mean);
%twd/%awy past a 0.3 deadzone; option columns = share of neutral frames in
those action states (dash/run/turn; grounded blaster 341-344; aerial
blaster 345-348; jumpsquat+rise).

### w180

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 8565 | 0.052 | 22.7 | 16.3 | 0.1 | 29.8 | 2.2 | 0.8 |
| 20-40 | 5009 | -0.044 | 15.4 | 20.4 | 0.1 | 18.7 | 2.1 | 1.5 |
| 40-70 | 2142 | 0.042 | 22.0 | 16.9 | 0.0 | 16.4 | 0.7 | 1.1 |
| 70-100 | 512 | 0.166 | 26.8 | 8.2 | 0.0 | 20.9 | 0.2 | 0.8 |
| 100-140 | 67 | 0.357 | 52.2 | 9.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 140+ | 28 | 0.741 | 85.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |


### v14

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 7310 | -0.084 | 22.2 | 31.0 | 0.5 | 11.3 | 3.1 | 2.3 |
| 20-40 | 4296 | -0.139 | 18.3 | 33.0 | 0.9 | 7.9 | 3.6 | 2.0 |
| 40-70 | 3561 | -0.081 | 15.3 | 23.5 | 0.5 | 12.3 | 0.5 | 2.2 |
| 70-100 | 1094 | 0.220 | 36.9 | 12.6 | 0.2 | 11.2 | 2.4 | 0.9 |
| 100-140 | 464 | 0.188 | 23.3 | 2.6 | 0.0 | 32.8 | 0.0 | 1.3 |
| 140+ | 243 | 0.357 | 39.5 | 1.6 | 0.8 | 11.9 | 0.0 | 0.0 |


Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).
