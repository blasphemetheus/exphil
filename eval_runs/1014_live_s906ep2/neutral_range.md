# Neutral-range scorecard (F3c) — expert vs bot, neutral frames by distance

Same detector both sets: :neutral frames only, subject = Fox
(expert::auto, bot:1).
toward = mean signed stick-x toward the opponent (deadzone-free mean);
%twd/%awy past a 0.3 deadzone; option columns = share of neutral frames in
those action states (dash/run/turn; grounded blaster 341-344; aerial
blaster 345-348; jumpsquat+rise).

### expert

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 102619 | 0.107 | 35.6 | 23.8 | 8.6 | 0.1 | 0.2 | 14.2 |
| 20-40 | 88736 | 0.143 | 38.4 | 22.7 | 19.7 | 0.4 | 0.8 | 17.6 |
| 40-70 | 57975 | 0.264 | 46.3 | 16.8 | 29.5 | 1.9 | 4.0 | 15.3 |
| 70-100 | 13382 | 0.463 | 59.6 | 7.7 | 14.8 | 2.9 | 7.2 | 9.6 |
| 100-140 | 3735 | 0.503 | 62.9 | 8.4 | 4.4 | 1.2 | 5.9 | 6.3 |
| 140+ | 1079 | 0.515 | 63.2 | 9.5 | 8.6 | 0.6 | 3.0 | 7.3 |


### bot

| \|dx\| | frames | toward | %twd | %awy | dash% | laserG% | laserA% | jump% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20 | 49055 | 0.058 | 29.8 | 22.3 | 9.5 | 0.1 | 0.7 | 12.0 |
| 20-40 | 36977 | 0.117 | 35.2 | 21.1 | 19.6 | 0.3 | 1.1 | 17.7 |
| 40-70 | 23940 | 0.191 | 41.3 | 18.7 | 27.5 | 0.8 | 2.5 | 15.6 |
| 70-100 | 6731 | 0.302 | 48.9 | 13.4 | 21.5 | 2.2 | 5.7 | 10.0 |
| 100-140 | 1853 | 0.425 | 59.4 | 9.7 | 13.6 | 1.4 | 5.8 | 3.5 |
| 140+ | 400 | 0.677 | 80.0 | 4.5 | 0.8 | 0.0 | 0.3 | 3.8 |


Reading: if the expert holds toward > 0 (or high dash% with balanced
twd/awy = dash-dance) at ranges where the bot's F3b mechanistic
approach_delta is negative, the corpus DOES contain approach at that
range and BC lost it (selection lever, plan c). If the expert also
retreats at range, F3b matches the corpus and the lever is curation.
Laser columns test the "retreat without the laser half" hypothesis
(Bradley 09-01: bot does no SH laser / FH double laser).
