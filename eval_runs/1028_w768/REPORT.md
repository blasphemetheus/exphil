
# width arm 768 on master data (2026-10-28 15:08)

# width arm 768 on master data (2026-10-28 21:33)

## w768 epoch 1 (lr 0.0002) (21:33)

# width arm 768 on master data (2026-10-28 21:44)

## w768 epoch 1 (lr 0.0002) (21:44)
STALE_BUILD (lib newer than _build/dev; run EDIFICE_LOCAL_NX=1 devenv shell -- mix compile): lib/exphil/agents/agent.ex

## w768 epoch 2 (lr 0.0002) (21:44)
STALE_BUILD (lib newer than _build/dev; run EDIFICE_LOCAL_NX=1 devenv shell -- mix compile): lib/exphil/agents/agent.ex

## w768 epoch 3 (lr 0.0001) (21:44)
STALE_BUILD (lib newer than _build/dev; run EDIFICE_LOCAL_NX=1 devenv shell -- mix compile): lib/exphil/agents/agent.ex



# width arm 768 on master data (2026-10-28 21:46)

## w768 epoch 1 (lr 0.0002) (21:46)
val_loss=3.141
RESULT fox_mamba_master_w768_ep1: frames=30583 change_recall=0.277 hold_agreement=0.586 repeat_prev=0.763 (expert 0.665) neutral=0.234 (expert 0.275)
RESULT fox_mamba_master_w768_ep1 press edges/min model:expert  A 21.31:19.19  B 26.84:18.72  X 22.95:30.96  Y 33.78:26.72  Z 5.53:4.36  L 28.96:33.31  R 15.54:7.89
RESULT fox_mamba_master_w768_ep1 jab1 episodes=6 A-edges/episode model=0.667 expert=0.333
RESULT fox_mamba_master_w768_ep1 change events recall/precision  ±0: press 0.033/0.03 release 0.043/0.039 stick 0.031/0.035  |  ±2: press 0.236/0.219 release 0.153/0.141 stick 0.149/0.168  |  ±5: press 0.324/0.296 release 0.254/0.242 stick 0.256/0.297
RESULT fox_mamba_master_w768_ep1 vs idle: 16.0 env-min  SD/min 1.0  deaths/min 1.063  kills/min 1.688  dmg dealt/taken per min 97.29/0.063
RESULT fox_mamba_master_w768_ep1 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 16 (1.0/min)
RESULT fox_mamba_master_w768_ep1 offstage eps/min 5.563  recovery rate 0.802  neutral 0.35  repeat_prev 0.757  max frozen run 410 f  (23.0s wall)
RESULT fox_mamba_master_w768_ep1: 36 cases x 8 trials  recovery 0.622  died 0.378  timeout 0.0  never 1/36  always 2/36
RESULT fox_mamba_master_w768_ep1 by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.534 (11 cases)  4+ routes: 0.651 (24 cases)
DONE fox_mamba_master_w768_ep1
RESULT ALL frames: n 39430  joint 75.7%  buttons 93.4%  families 85.8%  main dir 92.5% (bucket 85.7%, ±1 89.1%)  c dir 99.1%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.84  change recall 69.1%  false change 4.3%
RESULT CHANGE frames (real input changed): n 11028  joint 24.2%  buttons 76.9%  families 51.6%  main dir 73.4% (bucket 49.0%, ±1 61.4%)  c dir 96.9%  trigger 93.6%  jump 90.5%  attack 97.1%  special 96.4%  surprise 3.236  change recall 69.1%  false change %
RESULT frames with a real press edge: n 1314  joint 0.8%  buttons 19.5%  families 8.4%  main dir 88.5% (bucket 76.9%, ±1 85.0%)  c dir 99.5%  trigger 79.1%  jump 57.5%  attack 86.8%  special 84.4%  surprise 4.867  change recall 30.3%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.2%  buttons 80.6%  families 62.9%  main dir 77.6% (bucket 57.1%, ±1 67.5%)  c dir 99.2%  trigger 96.5%  jump 92.0%  attack 97.5%  special 97.0%  surprise 2.498  change recall 70.1%  false change 4.2%
RESULT press calibration jump: AUC 0.864  mean P on press frames 0.073 vs no-press 0.024  press frames above the no-press median 96.4%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.915  mean P on press frames 0.091 vs no-press 0.009  press frames above the no-press median 98.2%  (n 165 / 38183)
RESULT press calibration special: AUC 0.949  mean P on press frames 0.158 vs no-press 0.01  press frames above the no-press median 98.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.894  mean P on press frames 0.343 vs no-press 0.035  press frames above the no-press median 94.8%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 44.9% families 33.8%
EPOCH 1 DONE light (01:18)

## w768 epoch 2 (lr 0.0002) (01:18)

## epoch 1 read (Claude, 01:20): w768 ep1 val 3.141 / fam 51.6 / joint 24.2 / AUC 0.864 / STANDING 44.9  vs  512 ep1 val 3.140 / 50.8 / 23.8 / 0.854 / 47.0
Within cross-run drift (~2 fam) after one epoch; 65 ms/it vs 48. The read is epoch 3 vs the 512 arm's plateau (53.4 / 26.4).
val_loss=2.9792
RESULT fox_mamba_master_w768_ep2: frames=30583 change_recall=0.29 hold_agreement=0.592 repeat_prev=0.746 (expert 0.665) neutral=0.235 (expert 0.275)
RESULT fox_mamba_master_w768_ep2 press edges/min model:expert  A 21.89:19.19  B 22.6:18.72  X 25.43:30.96  Y 40.38:26.72  Z 5.65:4.36  L 28.49:33.31  R 15.54:7.89
RESULT fox_mamba_master_w768_ep2 jab1 episodes=6 A-edges/episode model=0.333 expert=0.333
RESULT fox_mamba_master_w768_ep2 change events recall/precision  ±0: press 0.043/0.038 release 0.046/0.04 stick 0.034/0.035  |  ±2: press 0.259/0.233 release 0.165/0.146 stick 0.158/0.166  |  ±5: press 0.347/0.309 release 0.277/0.246 stick 0.265/0.293
RESULT fox_mamba_master_w768_ep2 vs idle: 16.0 env-min  SD/min 0.813  deaths/min 0.813  kills/min 2.0  dmg dealt/taken per min 108.674/0.0
RESULT fox_mamba_master_w768_ep2 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 13 (0.813/min)
RESULT fox_mamba_master_w768_ep2 offstage eps/min 4.625  recovery rate 0.817  neutral 0.367  repeat_prev 0.752  max frozen run 892 f  (22.4s wall)
RESULT fox_mamba_master_w768_ep2: 36 cases x 8 trials  recovery 0.747  died 0.253  timeout 0.0  never 0/36  always 6/36
RESULT fox_mamba_master_w768_ep2 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.602 (11 cases)  4+ routes: 0.802 (24 cases)
DONE fox_mamba_master_w768_ep2
RESULT ALL frames: n 39430  joint 76.1%  buttons 93.5%  families 86.0%  main dir 92.8% (bucket 86.2%, ±1 89.6%)  c dir 99.2%  trigger 97.7%  jump 97.3%  attack 99.2%  special 98.9%  surprise 2.556  change recall 70.8%  false change 4.5%
RESULT CHANGE frames (real input changed): n 11028  joint 26.0%  buttons 77.1%  families 52.6%  main dir 74.4% (bucket 51.1%, ±1 63.0%)  c dir 97.0%  trigger 93.7%  jump 90.6%  attack 97.1%  special 96.4%  surprise 3.211  change recall 70.8%  false change %
RESULT frames with a real press edge: n 1314  joint 1.1%  buttons 19.6%  families 8.1%  main dir 89.3% (bucket 78.5%, ±1 85.4%)  c dir 99.5%  trigger 78.5%  jump 57.6%  attack 86.7%  special 84.6%  surprise 4.809  change recall 27.9%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 41.6%  buttons 80.8%  families 63.6%  main dir 78.4% (bucket 58.7%, ±1 68.7%)  c dir 99.2%  trigger 96.6%  jump 92.1%  attack 97.5%  special 97.0%  surprise 2.394  change recall 71.8%  false change 4.4%
RESULT press calibration jump: AUC 0.875  mean P on press frames 0.08 vs no-press 0.023  press frames above the no-press median 97.1%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.932  mean P on press frames 0.1 vs no-press 0.009  press frames above the no-press median 98.8%  (n 165 / 38183)
RESULT press calibration special: AUC 0.955  mean P on press frames 0.175 vs no-press 0.009  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.902  mean P on press frames 0.349 vs no-press 0.033  press frames above the no-press median 96.1%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 48.3% families 37.6%
EPOCH 2 DONE light (04:51)

## w768 epoch 3 (lr 0.0001) (04:51)

## epoch 2 read (Claude, 07:55): w768 ep2 val 2.979 / fam 52.6 / joint 26.0 / AUC 0.875 / STANDING 48.3  vs  512 ep2 2.982 / 52.3 / 25.8 / 0.865 / 47.4 — level. 72 ms/it at lr 1e-4 on ep3.
val_loss=3.0465
RESULT fox_mamba_master_w768_ep3: frames=30583 change_recall=0.272 hold_agreement=0.595 repeat_prev=0.738 (expert 0.665) neutral=0.208 (expert 0.275)
RESULT fox_mamba_master_w768_ep3 press edges/min model:expert  A 29.19:19.19  B 19.66:18.72  X 27.66:30.96  Y 41.08:26.72  Z 3.88:4.36  L 30.72:33.31  R 21.66:7.89
RESULT fox_mamba_master_w768_ep3 jab1 episodes=6 A-edges/episode model=0.5 expert=0.333
RESULT fox_mamba_master_w768_ep3 change events recall/precision  ±0: press 0.043/0.035 release 0.051/0.041 stick 0.04/0.043  |  ±2: press 0.309/0.255 release 0.195/0.16 stick 0.176/0.188  |  ±5: press 0.398/0.335 release 0.325/0.272 stick 0.289/0.322
RESULT fox_mamba_master_w768_ep3 vs idle: 16.0 env-min  SD/min 1.188  deaths/min 1.188  kills/min 2.125  dmg dealt/taken per min 110.365/0.0
RESULT fox_mamba_master_w768_ep3 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 19 (1.188/min)
RESULT fox_mamba_master_w768_ep3 offstage eps/min 5.75  recovery rate 0.791  neutral 0.325  repeat_prev 0.742  max frozen run 389 f  (23.5s wall)
RESULT fox_mamba_master_w768_ep3: 36 cases x 8 trials  recovery 0.712  died 0.288  timeout 0.0  never 0/36  always 4/36
RESULT fox_mamba_master_w768_ep3 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.602 (11 cases)  4+ routes: 0.76 (24 cases)
DONE fox_mamba_master_w768_ep3
RESULT ALL frames: n 39430  joint 73.7%  buttons 93.5%  families 86.0%  main dir 93.0% (bucket 86.3%, ±1 89.6%)  c dir 99.1%  trigger 97.4%  jump 97.4%  attack 99.2%  special 99.0%  surprise 3.136  change recall 73.1%  false change 7.6%
RESULT CHANGE frames (real input changed): n 11028  joint 25.7%  buttons 77.2%  families 52.9%  main dir 75.3% (bucket 51.5%, ±1 63.3%)  c dir 97.0%  trigger 92.7%  jump 90.7%  attack 97.1%  special 96.5%  surprise 3.404  change recall 73.1%  false change %
RESULT frames with a real press edge: n 1314  joint 1.6%  buttons 20.1%  families 7.9%  main dir 89.5% (bucket 78.5%, ±1 85.5%)  c dir 99.4%  trigger 76.9%  jump 57.4%  attack 86.8%  special 85.2%  surprise 5.151  change recall 29.5%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.7%  buttons 80.9%  families 63.8%  main dir 78.9% (bucket 59.0%, ±1 68.9%)  c dir 99.2%  trigger 95.8%  jump 92.2%  attack 97.5%  special 97.1%  surprise 2.511  change recall 74.2%  false change 7.1%
RESULT press calibration jump: AUC 0.868  mean P on press frames 0.072 vs no-press 0.02  press frames above the no-press median 96.7%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.928  mean P on press frames 0.118 vs no-press 0.01  press frames above the no-press median 98.8%  (n 165 / 38183)
RESULT press calibration special: AUC 0.956  mean P on press frames 0.181 vs no-press 0.009  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.907  mean P on press frames 0.312 vs no-press 0.025  press frames above the no-press median 96.1%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 57.7% families 43.6%
EPOCH 3 DONE light (08:24)

W768 DONE (08:24)

## walk-off origins w768 ep3 (08:24)
RESULT w768_ep3 walk-off origins: 89 trips in 16.0 env-min (5.56/min)
RESULT w768_ep3   walk_off      n  12  0.75/min  died 0.17  stick toward edge (10 f before) 0.33  full-out 0.17  jumps left 1.0  last ground action [["DASHING", 7], ["CROUCH_START", 2], ["CROUCH_END", 1]]
RESULT w768_ep3   jump_off      n  50  3.13/min  died 0.16  stick toward edge (10 f before) 0.26  full-out 0.07  jumps left 0.58  last ground action [["KNEE_BEND", 22], ["DASHING", 3], ["CROUCHING", 2]]
RESULT w768_ep3   special_off   n   7  0.44/min  died 0.86  stick toward edge (10 f before) 0.36  full-out 0.27  jumps left 0.29  last ground action [["FIREFOX_GROUND", 2], ["KNEE_BEND", 2], ["CROUCH_START", 1]]

## league (08:25)
RESULT pair 1/6 s906_ep2 vs master_ep6: 13-18-1 (0/32 finished)  75 s  1535 game-frames/s
RESULT pair 2/6 s906_ep2 vs master_ep7_plain: 12-19-1 (0/32 finished)  70 s  1627 game-frames/s
RESULT pair 3/6 s906_ep2 vs w768_ep3: 20-12-0 (0/32 finished)  81 s  1415 game-frames/s
RESULT pair 4/6 master_ep6 vs master_ep7_plain: 13-17-2 (0/32 finished)  70 s  1642 game-frames/s
RESULT pair 5/6 master_ep6 vs w768_ep3: 21-11-0 (0/32 finished)  77 s  1486 game-frames/s
RESULT pair 6/6 master_ep7_plain vs w768_ep3: 14-17-1 (0/32 finished)  78 s  1466 game-frames/s
RESULT standings (Elo, Bradley–Terry over 192 games):
RESULT league -> eval_runs/1028_w768/league/league.json
RESULT standing 1530 master_ep6 win 0.557
RESULT standing 1522 master_ep7_plain win 0.542
RESULT standing 1489 s906_ep2 win 0.479
RESULT standing 1459 w768_ep3 win 0.422
RESULT matrix master_ep6 - 0.438 0.578 0.656
RESULT matrix master_ep7_plain 0.563 - 0.609 0.453
RESULT matrix s906_ep2 0.422 0.391 - 0.625
RESULT matrix w768_ep3 0.344 0.547 0.375 -

W768 LEAGUE DONE (08:33)

## READ (Claude, 08:40) — width is not the lever
            val     fam%   joint  recall  STANDING  SD/min  league Elo (4 bots)
512 ep3     2.977   53.4   25.5   72.6    50.0      —       —
512 ep6     2.977   53.4   26.4   73.1    54.3      0.69    1530
512 ep7     2.869   53.9   27.3   71.9    50.9      0.63    1522
768 ep3     3.047   52.9   25.7   73.1    57.7      1.19    1459  (beat ep7 55/45, lost to ep6 34/66 and the candidate 38/62)
At equal data and epochs the 768-wide model is level or slightly behind on agreement, its held-out loss went UP from epoch 2
to 3 (2.979 -> 3.047: more capacity, same 7,853 games = earlier overfit), it self-destructs more in the sim and sits last in the
league. The master arm's plateau is a DATA ceiling, not a capacity one. Capacity arms are parked; the lever is more master-grade
data (the master player's games against other tiers, if the corpus carries the tier per port) or a better objective.
