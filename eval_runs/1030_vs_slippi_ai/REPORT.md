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
