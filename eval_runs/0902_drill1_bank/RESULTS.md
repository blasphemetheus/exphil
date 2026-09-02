# Drill bank score — eval_runs/0902_drill1_bank (uthrow_low_mid)

388 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 239/388.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot 0-19% | 388 | 2.9 | 30 | 16.4 | 0 |
| expert 0-19% | 39 | 3.9 | 87 | 27.0 | 0 |


0-19% hits histogram: 1:88  2:183  3:52  4:8  5:4  6:3  7:3  8:24  9:14  10:6  14:2  17:1


## Bank notes (2026-09-02)

- 388/500 episodes: the run tripped a CUMULATIVE menu_steps limit after 97
  auto_menu game transitions (driver bug, fixed same day — the guard now
  resets per menu visit). No gameplay wedge; all 388 episodes + 97 replays
  are clean. Top-up bank (laddered, new driver) runs separately.
- Baseline verdict: 271/388 (70%) episodes stop at 1-2 hits — the F4 live
  cap at scale. The 8-17 "hit" tail is laser strings (counted identically
  in the expert reference; histogram is the honest read).
- This bank is the PRE-RETRAIN reference the AWBC arm re-measures against.
- Training use: slice [handoff, handoff+240] windows ONLY (episodes.jsonl
  has anchors; both ports are Fox — pass port 1 explicitly). Raw-file
  training would clone scripted idling and credit dummy SDs as kills.
