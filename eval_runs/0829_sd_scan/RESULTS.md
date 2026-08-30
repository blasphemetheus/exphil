# Self-destruct scan — is the training corpus poisoned with "run off left"?

Hypothesis (Bradley, 08-29 evening): the higher mode-of-N decodes run off
the LEFT edge over and over, never the right, because the training set
contains scripted/SD games where Fox walks off left several times per
game. Scanner: `scripts/sd_scan.exs` (stock loss with no hitstun and no
percent gain in the prior 90 frames = self-destruct; side = blastzone
crossed; walk-off = last grounded action was walk/dash/run past the edge).

## 1. The bot's own games confirm the OBSERVATION

| arm (8 games each, bot port 1, CPU dummy, FD) | deaths | SDs | SD left : right | walk-off SDs |
|---|---|---|---|---|
| mode-of-16 | 32 | 28 (88%) | **24 : 3** | **25** (23 left, 2 right) |
| base (T=0.5 sampling) | 18 | 10 (56%) | 5 : 0 | 0 (all falls, no walk-offs) |

Every mode-16 game had ≥2 left SDs; 7/8 had ≥3. The pattern is real.

## 2. The training corpus does NOT contain it — hypothesis FALSIFIED

7,911 erickfm ranked Fox games, subject = port 1 (the Fox v1 imitates),
opponent = port 2, same games:

| | port 1 (imitated) | port 2 |
|---|---|---|
| deaths | 24,968 | 25,536 |
| SDs (no hit in 90 f) | 12,048 (48%) | 12,303 (48%) |
| **SD left : right** | **3,833 : 4,088** | 4,026 : 3,999 |
| walk-off SDs | **97** (54 left, 43 right) | 101 (51 left, 50 right) |
| games with ≥3 / ≥4 left SDs | 142 / 16 | 156 / 17 |
| games with ≥3 / ≥4 right SDs | 186 / 19 | 146 / 20 |

- Left:right is balanced (port 1 leans slightly RIGHT, 0.94:1).
- Walk-offs are 0.8% of self-destructs — 97 across 7,911 games, and
  split evenly by side. No batch of "Fox walks off left four times."
- The 16 games with ≥4 left SDs are matched by 19 with ≥4 right SDs;
  the top file (6 left SDs) is one game, not a batch.
- The 48% "SD" rate is the detector's 90-frame window counting recovery
  deaths whose hit was earlier; it is identical on both ports, so it does
  not bias the left:right read.

**There is no directional poison in what v1 trained on.** The mode-16
walk-offs come from the DECODE, not the data.

## 3. What it is instead (open, one cheap test)

v1 trained with NO mirroring (`augment: false` gates `mirror_prob`,
pipeline.ex:854), so a learned asymmetry is possible — but the corpus
gives it nothing directional to learn. The remaining candidates:

- **Spawn/side effect, not "left" per se.** In every eval the bot is port
  1 and spawns on the LEFT of FD; a frozen "hold toward my side / away
  from the opponent" input walks off left. Base also leans left (5:0)
  but by falling during recovery, not walking.
- **Mode lock on a stick direction.** frozen-input 0.74 means the modal
  joint input is held for most of the game; whichever main-stick bucket
  is modal in neutral gets held until the edge.

Cheap discriminating test (not run): play `--mode-of-n 16` with the bot
on **port 2** (`--port 2 --opponent-port 1`, spawn on the right). If the
walk-offs flip to the RIGHT, it is "toward my spawn side"; if they stay
left, it is an absolute-direction bias in the model and worth an interp
probe (facing × x-position). Either way it does not rescue mode-of-N —
the freeze is the problem, the direction is a symptom.
