# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(w180b:1, w180e2:1).
toward = mean signed stick-x toward the opponent (deadzone-free mean);
%twd/%awy past a 0.3 deadzone; option columns = share of neutral frames in
those action states (dash/run/turn; grounded blaster 341-344; aerial
blaster 345-348; jumpsquat+rise).

### w180b

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 7314 | -0.015 | 21.7 | 22.8 | 0.1 | 31.2 | 4.2 | 0.5 |
| 20-40 | 4148 | -0.046 | 17.0 | 21.7 | 0.1 | 13.0 | 2.0 | 1.0 |
| 40-70 | 2265 | 0.106 | 26.4 | 14.5 | 0.0 | 6.8 | 5.5 | 0.9 |
| 70-100 | 892 | 0.393 | 48.8 | 4.8 | 0.0 | 11.0 | 2.7 | 0.3 |
| 100-140 | 333 | 0.128 | 17.4 | 2.7 | 0.0 | 5.4 | 0.3 | 2.4 |
| 140+ | 85 | -0.251 | 3.5 | 35.3 | 0.0 | 0.0 | 0.0 | 4.7 |


### w180e2

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 8565 | 0.052 | 22.7 | 16.3 | 0.1 | 29.8 | 2.2 | 0.8 |
| 20-40 | 5009 | -0.044 | 15.4 | 20.4 | 0.1 | 18.7 | 2.1 | 1.5 |
| 40-70 | 2142 | 0.042 | 22.0 | 16.9 | 0.0 | 16.4 | 0.7 | 1.1 |
| 70-100 | 512 | 0.166 | 26.8 | 8.2 | 0.0 | 20.9 | 0.2 | 0.8 |
| 100-140 | 67 | 0.357 | 52.2 | 9.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 140+ | 28 | 0.741 | 85.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |


Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).
