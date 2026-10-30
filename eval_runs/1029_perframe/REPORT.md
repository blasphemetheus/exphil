
# Per-frame control arm — 3 epochs (2026-10-29 17:47); corpus 7853 files

## epoch 1 (lr 0.0002) (17:47)
val_loss=0.9668
RESULT fox_mamba_perframe_ep1: frames=30583 change_recall=0.281 hold_agreement=0.558 repeat_prev=0.717 (expert 0.665) neutral=0.234 (expert 0.275)
RESULT fox_mamba_perframe_ep1 press edges/min model:expert  A 21.66:19.19  B 26.96:18.72  X 21.19:30.96  Y 34.25:26.72  Z 4.59:4.36  L 25.43:33.31  R 14.71:7.89
RESULT fox_mamba_perframe_ep1 jab1 episodes=6 A-edges/episode model=0.167 expert=0.333
RESULT fox_mamba_perframe_ep1 change events recall/precision  ±0: press 0.016/0.015 release 0.019/0.018 stick 0.024/0.022  |  ±2: press 0.139/0.133 release 0.117/0.111 stick 0.15/0.138  |  ±5: press 0.216/0.203 release 0.193/0.184 stick 0.261/0.242
RESULT fox_mamba_perframe_ep1 vs idle: 16.0 env-min  SD/min 1.25  deaths/min 1.25  kills/min 1.063  dmg dealt/taken per min 63.529/0.0
RESULT fox_mamba_perframe_ep1 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 20 (1.25/min)
RESULT fox_mamba_perframe_ep1 offstage eps/min 5.375  recovery rate 0.765  neutral 0.317  repeat_prev 0.71  max frozen run 167 f  (20.7s wall)
RESULT fox_mamba_perframe_ep1: 36 cases x 8 trials  recovery 0.365  died 0.635  timeout 0.0  never 7/36  always 0/36
RESULT fox_mamba_perframe_ep1 by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.17 (11 cases)  4+ routes: 0.443 (24 cases)
DONE fox_mamba_perframe_ep1
RESULT ALL frames: n 39430  joint 76.1%  buttons 93.7%  families 86.2%  main dir 92.3% (bucket 83.2%, ±1 89.0%)  c dir 99.1%  trigger 98.5%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.081  change recall 51.4%  false change 2.6%
RESULT CHANGE frames (real input changed): n 11028  joint 21.1%  buttons 77.8%  families 52.4%  main dir 73.8% (bucket 45.3%, ±1 62.2%)  c dir 97.0%  trigger 95.0%  jump 90.6%  attack 97.1%  special 96.4%  surprise 3.229  change recall 51.4%  false change %
RESULT frames with a real press edge: n 1314  joint 0.8%  buttons 19.6%  families 5.6%  main dir 90.9% (bucket 80.0%, ±1 88.0%)  c dir 99.5%  trigger 75.2%  jump 57.6%  attack 86.7%  special 84.6%  surprise 4.802  change recall 22.9%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.1%  buttons 93.7%  families 86.2%  main dir 92.3% (bucket 83.2%, ±1 89.0%)  c dir 99.1%  trigger 98.5%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.081  change recall 51.4%  false change 2.6%
RESULT press calibration jump: AUC 0.846  mean P on press frames 0.067 vs no-press 0.022  press frames above the no-press median 93.7%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.941  mean P on press frames 0.095 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.956  mean P on press frames 0.158 vs no-press 0.009  press frames above the no-press median 98.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.932  mean P on press frames 0.249 vs no-press 0.013  press frames above the no-press median 97.4%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 6.4% families 13.2%
EPOCH 1 DONE light (20:41)
### agreement on the master run's 16 held-out games (8 used) (20:41)
RESULT ALL frames: n 39430  joint 76.1%  buttons 93.7%  families 86.2%  main dir 92.3% (bucket 83.2%, ±1 89.0%)  c dir 99.1%  trigger 98.5%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.081  change recall 51.4%  false change 2.6%
RESULT CHANGE frames (real input changed): n 11028  joint 21.1%  buttons 77.8%  families 52.4%  main dir 73.8% (bucket 45.3%, ±1 62.2%)  c dir 97.0%  trigger 95.0%  jump 90.6%  attack 97.1%  special 96.4%  surprise 3.229  change recall 51.4%  false change %
RESULT frames with a real press edge: n 1314  joint 0.8%  buttons 19.6%  families 5.6%  main dir 90.9% (bucket 80.0%, ±1 88.0%)  c dir 99.5%  trigger 75.2%  jump 57.6%  attack 86.7%  special 84.6%  surprise 4.802  change recall 22.9%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.1%  buttons 93.7%  families 86.2%  main dir 92.3% (bucket 83.2%, ±1 89.0%)  c dir 99.1%  trigger 98.5%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.081  change recall 51.4%  false change 2.6%
RESULT press calibration jump: AUC 0.846  mean P on press frames 0.067 vs no-press 0.022  press frames above the no-press median 93.7%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.941  mean P on press frames 0.095 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.956  mean P on press frames 0.158 vs no-press 0.009  press frames above the no-press median 98.6%  (n 207 / 37830)

## epoch 2 (lr 0.0002) (20:43)
val_loss=0.9089
RESULT fox_mamba_perframe_ep2: frames=30583 change_recall=0.283 hold_agreement=0.566 repeat_prev=0.725 (expert 0.665) neutral=0.254 (expert 0.275)
RESULT fox_mamba_perframe_ep2 press edges/min model:expert  A 21.66:19.19  B 25.9:18.72  X 23.54:30.96  Y 30.25:26.72  Z 4.83:4.36  L 25.07:33.31  R 15.42:7.89
RESULT fox_mamba_perframe_ep2 jab1 episodes=6 A-edges/episode model=0.5 expert=0.333
RESULT fox_mamba_perframe_ep2 change events recall/precision  ±0: press 0.018/0.017 release 0.03/0.029 stick 0.032/0.03  |  ±2: press 0.161/0.154 release 0.118/0.114 stick 0.179/0.168  |  ±5: press 0.244/0.234 release 0.198/0.193 stick 0.301/0.285
RESULT fox_mamba_perframe_ep2 vs idle: 16.0 env-min  SD/min 0.563  deaths/min 0.563  kills/min 2.0  dmg dealt/taken per min 112.847/0.0
RESULT fox_mamba_perframe_ep2 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 9 (0.563/min)
RESULT fox_mamba_perframe_ep2 offstage eps/min 3.938  recovery rate 0.855  neutral 0.308  repeat_prev 0.7  max frozen run 153 f  (20.5s wall)
RESULT fox_mamba_perframe_ep2: 36 cases x 8 trials  recovery 0.476  died 0.524  timeout 0.0  never 3/36  always 0/36
RESULT fox_mamba_perframe_ep2 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.33 (11 cases)  4+ routes: 0.531 (24 cases)
DONE fox_mamba_perframe_ep2
RESULT ALL frames: n 39430  joint 76.9%  buttons 93.7%  families 86.4%  main dir 92.5% (bucket 83.8%, ±1 89.2%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.1%  false change 1.8%
RESULT CHANGE frames (real input changed): n 11028  joint 22.1%  buttons 78.3%  families 53.3%  main dir 74.3% (bucket 45.7%, ±1 62.6%)  c dir 97.2%  trigger 95.1%  jump 90.8%  attack 97.1%  special 96.6%  surprise 3.068  change recall 51.1%  false change %
RESULT frames with a real press edge: n 1314  joint 1.7%  buttons 20.7%  families 5.3%  main dir 91.5% (bucket 79.8%, ±1 88.1%)  c dir 99.5%  trigger 74.7%  jump 58.5%  attack 86.5%  special 85.1%  surprise 4.407  change recall 22.6%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.9%  buttons 93.7%  families 86.4%  main dir 92.5% (bucket 83.8%, ±1 89.2%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.1%  false change 1.8%
RESULT press calibration jump: AUC 0.885  mean P on press frames 0.105 vs no-press 0.022  press frames above the no-press median 96.0%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.949  mean P on press frames 0.113 vs no-press 0.008  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.971  mean P on press frames 0.199 vs no-press 0.008  press frames above the no-press median 100.0%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.943  mean P on press frames 0.254 vs no-press 0.013  press frames above the no-press median 98.2%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 7.3% families 13.7%
EPOCH 2 DONE light (23:38)
### agreement on the master run's 16 held-out games (8 used) (23:38)
RESULT ALL frames: n 39430  joint 76.9%  buttons 93.7%  families 86.4%  main dir 92.5% (bucket 83.9%, ±1 89.2%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.1%  false change 1.8%
RESULT CHANGE frames (real input changed): n 11028  joint 22.1%  buttons 78.3%  families 53.3%  main dir 74.3% (bucket 45.7%, ±1 62.6%)  c dir 97.2%  trigger 95.1%  jump 90.8%  attack 97.1%  special 96.6%  surprise 3.068  change recall 51.1%  false change %
RESULT frames with a real press edge: n 1314  joint 1.7%  buttons 20.7%  families 5.3%  main dir 91.5% (bucket 79.9%, ±1 88.1%)  c dir 99.5%  trigger 74.7%  jump 58.5%  attack 86.5%  special 85.1%  surprise 4.407  change recall 22.6%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.9%  buttons 93.7%  families 86.4%  main dir 92.5% (bucket 83.9%, ±1 89.2%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.1%  false change 1.8%
RESULT press calibration jump: AUC 0.885  mean P on press frames 0.105 vs no-press 0.022  press frames above the no-press median 96.0%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.949  mean P on press frames 0.113 vs no-press 0.008  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.971  mean P on press frames 0.199 vs no-press 0.008  press frames above the no-press median 100.0%  (n 207 / 37830)

## epoch 3 (lr 0.0001) (23:40)
val_loss=0.9032
RESULT fox_mamba_perframe_ep3: frames=30583 change_recall=0.262 hold_agreement=0.549 repeat_prev=0.714 (expert 0.665) neutral=0.247 (expert 0.275)
RESULT fox_mamba_perframe_ep3 press edges/min model:expert  A 25.43:19.19  B 23.54:18.72  X 31.08:30.96  Y 24.25:26.72  Z 4.94:4.36  L 23.07:33.31  R 20.48:7.89
RESULT fox_mamba_perframe_ep3 jab1 episodes=6 A-edges/episode model=0.333 expert=0.333
RESULT fox_mamba_perframe_ep3 change events recall/precision  ±0: press 0.023/0.022 release 0.022/0.021 stick 0.037/0.034  |  ±2: press 0.188/0.175 release 0.111/0.103 stick 0.188/0.173  |  ±5: press 0.252/0.237 release 0.193/0.177 stick 0.306/0.284
RESULT fox_mamba_perframe_ep3 vs idle: 16.0 env-min  SD/min 0.875  deaths/min 0.875  kills/min 2.438  dmg dealt/taken per min 129.778/0.0
RESULT fox_mamba_perframe_ep3 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 14 (0.875/min)
RESULT fox_mamba_perframe_ep3 offstage eps/min 4.375  recovery rate 0.791  neutral 0.332  repeat_prev 0.711  max frozen run 132 f  (20.5s wall)
RESULT fox_mamba_perframe_ep3: 36 cases x 8 trials  recovery 0.5  died 0.5  timeout 0.0  never 2/36  always 0/36
RESULT fox_mamba_perframe_ep3 by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.295 (11 cases)  4+ routes: 0.589 (24 cases)
DONE fox_mamba_perframe_ep3
RESULT ALL frames: n 39430  joint 76.8%  buttons 93.7%  families 86.4%  main dir 92.4% (bucket 83.7%, ±1 89.0%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.7%  false change 2.0%
RESULT CHANGE frames (real input changed): n 11028  joint 22.0%  buttons 78.1%  families 53.5%  main dir 74.3% (bucket 45.9%, ±1 62.6%)  c dir 97.1%  trigger 95.1%  jump 91.0%  attack 97.1%  special 96.5%  surprise 3.049  change recall 51.7%  false change %
RESULT frames with a real press edge: n 1314  joint 2.1%  buttons 21.1%  families 6.0%  main dir 91.2% (bucket 80.2%, ±1 87.9%)  c dir 99.5%  trigger 74.9%  jump 58.3%  attack 86.7%  special 84.9%  surprise 4.32  change recall 23.6%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.8%  buttons 93.7%  families 86.4%  main dir 92.4% (bucket 83.7%, ±1 89.0%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.7%  false change 2.0%
RESULT press calibration jump: AUC 0.889  mean P on press frames 0.105 vs no-press 0.02  press frames above the no-press median 96.4%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.954  mean P on press frames 0.132 vs no-press 0.009  press frames above the no-press median 100.0%  (n 165 / 38183)
RESULT press calibration special: AUC 0.971  mean P on press frames 0.209 vs no-press 0.009  press frames above the no-press median 99.5%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.945  mean P on press frames 0.242 vs no-press 0.014  press frames above the no-press median 98.4%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 7.3% families 13.7%
EPOCH 3 DONE light (02:35)
### agreement on the master run's 16 held-out games (8 used) (02:35)
RESULT ALL frames: n 39430  joint 76.8%  buttons 93.7%  families 86.4%  main dir 92.4% (bucket 83.7%, ±1 89.0%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.6%  false change 2.0%
RESULT CHANGE frames (real input changed): n 11028  joint 22.0%  buttons 78.1%  families 53.5%  main dir 74.3% (bucket 45.9%, ±1 62.6%)  c dir 97.1%  trigger 95.1%  jump 91.0%  attack 97.1%  special 96.5%  surprise 3.049  change recall 51.6%  false change %
RESULT frames with a real press edge: n 1314  joint 2.1%  buttons 21.1%  families 6.0%  main dir 91.2% (bucket 80.2%, ±1 87.9%)  c dir 99.5%  trigger 74.9%  jump 58.3%  attack 86.7%  special 84.9%  surprise 4.319  change recall 23.6%  false change %
RESULT DECISION frames (stick head trained here): n 39430  joint 76.8%  buttons 93.7%  families 86.4%  main dir 92.4% (bucket 83.7%, ±1 89.0%)  c dir 99.2%  trigger 98.6%  jump 97.4%  attack 99.2%  special 99.0%  surprise 1.021  change recall 51.6%  false change 2.0%
RESULT press calibration jump: AUC 0.889  mean P on press frames 0.105 vs no-press 0.02  press frames above the no-press median 96.4%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.954  mean P on press frames 0.132 vs no-press 0.009  press frames above the no-press median 100.0%  (n 165 / 38183)
RESULT press calibration special: AUC 0.971  mean P on press frames 0.21 vs no-press 0.009  press frames above the no-press median 99.5%  (n 207 / 37830)

## walk-off origins perframe ep3 (02:36)
RESULT perframe_ep3 walk-off origins: 68 trips in 16.0 env-min (4.25/min)
RESULT perframe_ep3   walk_off      n   6  0.38/min  died 0.33  stick toward edge (10 f before) 0.5  full-out 0.25  jumps left 1.0  last ground action [["DASHING", 4], ["CROUCH_START", 2]]
RESULT perframe_ep3   jump_off      n  43  2.69/min  died 0.21  stick toward edge (10 f before) 0.31  full-out 0.04  jumps left 0.77  last ground action [["KNEE_BEND", 21], ["LANDING_SPECIAL", 3], ["DASHING", 1]]
RESULT perframe_ep3   special_off   n   6  0.38/min  died 0.67  stick toward edge (10 f before) 0.48  full-out 0.0  jumps left 0.17  last ground action [["FIREFOX_GROUND", 1], ["NEUTRAL_B_FULL_CHARGE_AIR", 1]]

## league (02:37)
RESULT pair 1/6 s906_ep2 vs master_ep3: 15-16-1 (0/32 finished)  73 s  1572 game-frames/s
RESULT pair 2/6 s906_ep2 vs master_ep7_plain: 12-19-1 (0/32 finished)  69 s  1657 game-frames/s
RESULT pair 3/6 s906_ep2 vs perframe_ep3: 23-8-1 (0/32 finished)  70 s  1633 game-frames/s
RESULT pair 4/6 master_ep3 vs master_ep7_plain: 11-21-0 (0/32 finished)  70 s  1637 game-frames/s
RESULT pair 5/6 master_ep3 vs perframe_ep3: 20-12-1 (1/33 finished)  69 s  1646 game-frames/s
RESULT pair 6/6 master_ep7_plain vs perframe_ep3: 26-6-0 (0/32 finished)  70 s  1644 game-frames/s
RESULT standings (Elo, Bradley–Terry over 193 games):
RESULT league -> eval_runs/1029_perframe/league/league.json
RESULT standing 1610 master_ep7_plain win 0.693
RESULT standing 1521 s906_ep2 win 0.536
RESULT standing 1497 master_ep3 win 0.495
RESULT standing 1373 perframe_ep3 win 0.278
RESULT matrix master_ep7_plain - 0.609 0.656 0.813
RESULT matrix s906_ep2 0.391 - 0.484 0.734
RESULT matrix master_ep3 0.344 0.516 - 0.621
RESULT matrix perframe_ep3 0.188 0.266 0.379 -

MLIKE DONE (02:44)

## VERDICT (10-30 02:45) — the coherence representation EARNS ITS KEEP IN PLAY

Agreement on the master run's 16 held-out games (same 11,028 change frames; "families" = option-level):
| epoch | per-frame fam / joint | coherence recipe fam / joint |
|---|---|---|
| 1 | 52.4 / 21.1 | 50.8 / 23.8 |
| 2 | 53.3 / 22.1 | 52.3 / 25.8 |
| 3 | 53.5 / 22.0 | 53.4 / 25.5 |
Teacher-forced: LEVEL on option families, 3.5 points lower on exact joint (stick bucket exactness).
Epochs 1.9 h vs 2.5–4.5 h. Val loss not comparable (different heads).

League (NIF sim, Fox dittos, 193 games, Bradley–Terry):
| | Elo | win rate |
|---|---|---|
| master_ep7_plain (coherence, 7 ep) | 1610 | 0.69 |
| s906_ep2 (candidate) | 1521 | 0.54 |
| master_ep3 (coherence, EQUAL epochs) | 1497 | 0.50 |
| **perframe_ep3** | **1373** | **0.28** |
Head-to-head: perframe_ep3 beats master_ep3 38 %, master_ep7 19 %. Closed loop vs idle: SD/min 0.875 (ep7 0.625), kills 2.44 (ep7 2.06) — noise band.

READ: the plain slippi-ai-style per-frame objective matches the coherence recipe on what-option-when (teacher-
forced) but is clearly weaker when it has to play its own trajectories. That is the signature of exposure drift:
events + hold-length + chunk futures make the bot's own rollouts coherent, and the teacher-forced metric cannot see
it (the same lesson as the mixed-in data checklist). So: the representation divergence is VALIDATED, not an
accident to undo. The external yardstick (slippi-ai d18 imitation vs our bots, both representations) is the next
check — their per-frame bot plays with 18 frames of delay, which is itself a different coherence mechanism.
Caveat: one seed each, 3 epochs; the league sim is our sim.
