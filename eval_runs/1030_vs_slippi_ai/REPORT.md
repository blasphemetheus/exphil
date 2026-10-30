# ExPhil vs slippi-ai — first head-to-head (2026-10-30, 02:50–03:08, headless)

Rig: `scripts/play_vs_slippi_ai.exs` (our async Dolphin harness hosts; slippi-ai's agent drives port 2 via
`priv/python/slippi_ai_worker.py` in its own venv, 0.58 ms/step). Dolphin = `~/.local/share/slippi/exi-ai/
dolphin-emu-headless` (the Launcher AppImage cannot run headless — connect timeout). FD, Fox dittos, T=1.0 both.
Opponent = `fox_d18_imitation_v3` = vladfi1's IMITATION-ONLY Fox (no RL), playing at ITS trained delay 18.
Ours at `--reaction-delay 1` (checkpoints trained at 0; async floor is 1 — one rung slower than trained).

| game | ours | mode | result (our stocks / their stocks) |
|---|---|---|---|
| 1 | master_ep7_plain | stateful | 0 / 4 |
| 2–4 | master_ep7_plain | stateful | 0/3, 0/3, 0/3 |
| 5 | master_ep7_plain | WINDOWED (deploy-equivalent) | 0 / 4 |
| 6 | s906_ep2 (candidate) | stateful | 0 / 4 |
**Set: 0–6. Every game lost all four stocks in ~130–150 s; they lost 3 stocks across six games.**

Play stats over the first 4 games (`corpus_skill_score.exs` on the replays):
| | ExPhil (p1) | slippi-ai (p2) |
|---|---|---|
| deaths / min | 1.78 | 0.33 |
| sd / min (PlayStats) | 1.44 | 0.33 |
| offstage return rate | 0.82 | 0.97 |
| kills / min | 0.33 | 1.78 |
| damage dealt / min | 72 | 137 |
| L-cancel rate | 0.85 | 0.87 |
| APM | 204 | 245 |
| dashes / min | 26 | 58 |
Review export (moments): 7–21 EDGEGUARD moments per game against us, 1–2 "sd" moments. So the deaths are
offstage deaths — failed recoveries under edgeguard pressure plus outright SDs — not being outplayed on stage.
Execution (L-cancel) is level; movement density (dashes) is half theirs.

Read: their plain per-frame imitation bot at delay 18, trained on ~100k+ replays, does not die offstage. Ours,
the strongest we have, dies offstage ~1.4×/min against it. This is the "obvious deficiency" Bradley named, now
measured against an external bot rather than our own instruments. It is also the one our recovery curve has
tracked since 10-11 (decided-trip return 0.38–0.60 vs expert 0.916) — scale/epochs/width/data-selection did not
move it. Their recipe differences that could matter: data scale, delay-18 training (forces commitment), polar
stick buckets, tech-animation masking, and a frame-skip-with-live-controller observation.

Caveats: 6 games; headless; our bot one rung slower than trained (reaction 1 vs 0 — a real handicap but not a
4-stock one: the same checkpoints hold their own at this rung in our league); their bot at its own delay.
Watch it: http://127.0.0.1:8012/review/?run=/runs/1030_vs_slippi_ai/review (six games, moments + ghosts + controllers).
Next rung: `fox_d18_ditto_v4` (their RL Fox) for the RL-gain number; our ms_g19 line at d18-ish for a delay-matched read.

## RL rung (03:10): vs `fox_d18_ditto_v4` (their RL-refined Fox ditto, d18) — 0–3
Games lasted 67–79 s (vs 130–150 s against their imitation bot): 4 stocks each, they lost 0 stocks, took 7–58 %.
RL gain on their side ≈ halving the game length against us. Both their bots beat us 4-0 per game; the imitation
one is the right comparison (same training stage as ours), and it already shows the gap.

## 10-30 13:40 — Bradley's read, measured (6 games vs their imitation Fox)
- **Neutral:** openings 10.0/game (ours) vs 19.8 (theirs) — they win neutral 2:1.
- **Combos:** mean follow-up damage per opening 13.3 vs 17.5; openings converted to >=30 %: 3 % vs 13 %. Their
  top opening moves: bair 23, UP-THROW 15, nair 14. Ours: shine 13, dair 10, bair 10, grab-pummel 4 (we grab and
  pummel; we do not up-throw into the bread-and-butter).
- **Offstage:** sd 1.44/min vs 0.33; return 0.82 vs 0.97 (above).
- **Delay (isolated):** master_ep7 vs a STANDING dummy in Dolphin at reaction 1 (7 games): sd/min 0.60 — identical
  to the NIF sim at reaction 0 (0.625). The one-rung mismatch is NOT the cause; opponent pressure doubles the
  offstage death rate (0.60 -> 1.44). Delay cleared as the main issue.
- **Data (the instinct):** we hold ONE of SIX master-master Fox parts (FOX_master-master_a1 = 7,911 games, 6.3 GB
  of 40.8 GB). HF FOX slice: master-master 40.8 GB, master-diamond 51.6, master-platinum 41.7, diamond-diamond
  20.0, diamond-platinum 61.9, platinum-platinum 85.3 (301 GB, 36 files). Extraction ratio ~4.6x. Disk: 16 GB
  free on / and on /data — the blocker.
