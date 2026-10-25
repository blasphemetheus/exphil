
# Overnight 2026-10-24 — started 2026-10-24 04:11


# Overnight 2026-10-24 — started 2026-10-24 04:22

## Stage 0 — null-label sets (04:22)
RESULT null set: 9061 lists, 164744 labelled frames, 145206 labels differed from the actual press (88.1 %) -> data/silent_fall/sim_dagger_expert_r9o_2091_null_lean.frames (77 MB)
RESULT null set: 9172 lists, 166012 labelled frames, 145381 labels differed from the actual press (87.6 %) -> data/silent_fall/sim_dagger_expert_r9o_2094_null_lean.frames (78 MB)
RESULT null set: 8868 lists, 165052 labelled frames, 144710 labels differed from the actual press (87.7 %) -> data/silent_fall/sim_dagger_expert_r9o_2097_null_lean.frames (74 MB)
RESULT null set: 8981 lists, 164905 labelled frames, 144168 labels differed from the actual press (87.4 %) -> data/silent_fall/sim_dagger_expert_r9o_2100_null_lean.frames (76 MB)
null mix: data/silent_fall/sim_dagger_expert_r9o_2091_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2094_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2097_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2100_null_lean.frames,data/silent_fall/slp_dagger_expert_live1014_null_lean.frames

## Stage 1 — testbed queue: press commitment + passivity (04:24)

### arm pq_base — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 (04:24)
TRAIN_FAILED pq_base

### arm pq_base906 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 (04:26)
TRAIN_FAILED pq_base906

### arm pq_on10 — --onset-weight 10 --neutral-weight 1.0 --action-oversample 1.0 (04:27)
TRAIN_FAILED pq_on10

### arm pq_tw3 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --transition-weight 3 (04:27)
TRAIN_FAILED pq_tw3

### arm pq_tw3on10 — --onset-weight 10 --neutral-weight 1.0 --action-oversample 1.0 --transition-weight 3 (04:28)
TRAIN_FAILED pq_tw3on10

### arm pq_cw2 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --chunk-weight 2 (04:28)
TRAIN_FAILED pq_cw2

### arm pq_bw4 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --button-weight 4 (04:29)
TRAIN_FAILED pq_bw4

### arm pq_nw05 — --onset-weight 5 --neutral-weight 0.5 --action-oversample 1.0 (04:29)
TRAIN_FAILED pq_nw05

### arm pq_nw025 — --onset-weight 5 --neutral-weight 0.25 --action-oversample 1.0 (04:29)
TRAIN_FAILED pq_nw025

### arm pq_sf3 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --silent-fall-weight 3 (04:30)
TRAIN_FAILED pq_sf3

### arm pq_ao2 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 2.0 (04:30)
TRAIN_FAILED pq_ao2

### arm pq_combo — --onset-weight 10 --neutral-weight 0.5 --action-oversample 1.0 --transition-weight 3 (04:31)
TRAIN_FAILED pq_combo

STAGE 1 DONE (04:31)

## Stage 2 — master-only full arm, epochs 1-6 (04:31)

### master epoch 1 (lr 0.0002) (04:31)
disk: moved eval_runs/0921_evals + 0921_step8 (15.6 GB) to /data/exphil/eval_runs_archive (symlinked) after ENOSPC on / killed stage 1 at 04:34; relaunch

# Overnight 2026-10-24 — started 2026-10-24 04:34

## Stage 0 — null-label sets (04:34)
null mix: data/silent_fall/sim_dagger_expert_r9o_2091_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2094_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2097_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2100_null_lean.frames,data/silent_fall/slp_dagger_expert_live1014_null_lean.frames

## Stage 1 — testbed queue: press commitment + passivity (04:34)

### arm pq_base — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 (04:34)
val_loss=3.2265
RESULT pq_base: frames=32394 change_recall=0.291 hold_agreement=0.574 repeat_prev=0.832 (expert 0.764) neutral=0.261 (expert 0.303)
RESULT pq_base press edges/min model:expert  A 19.23:13.22  B 18.0:11.0  X 16.23:30.12  Y 17.56:19.89  Z 4.22:10.0  L 13.11:26.34  R 15.78:8.33
RESULT pq_base jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_base change events recall/precision  ±0: press 0.008/0.01 release 0.006/0.006 stick 0.014/0.02  |  ±2: press 0.043/0.049 release 0.033/0.038 stick 0.056/0.08  |  ±5: press 0.083/0.095 release 0.064/0.074 stick 0.117/0.169
RESULT pq_base vs idle: 16.0 env-min  SD/min 1.063  deaths/min 1.313  kills/min 0.125  dmg dealt/taken per min 11.728/0.25
RESULT pq_base deaths by verdict at the decision frame: kill 4 (0.25/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 17 (1.063/min)
RESULT pq_base offstage eps/min 3.625  recovery rate 0.625  neutral 0.548  repeat_prev 0.867  max frozen run 1424 f  (33.4s wall)
RESULT pq_base: 36 cases x 8 trials  recovery 0.403  died 0.597  timeout 0.0  never 7/36  always 1/36
RESULT pq_base by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.102 (11 cases)  4+ routes: 0.526 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.8%  buttons 69.6%  families 36.3%  main dir 66.0% (bucket 50.1%, ±1 57.3%)  c dir 96.2%  trigger 91.9%  jump 87.5%  attack 95.9%  special 96.3%  surprise 3.744  change recall 50.9%  false change %
RESULT frames with a real press edge: n 1029  joint 0.6%  buttons 17.3%  families 11.2%  main dir 89.1% (bucket 81.1%, ±1 87.4%)  c dir 99.9%  trigger 83.4%  jump 58.5%  attack 87.1%  special 88.7%  surprise 5.44  change recall 24.2%  false change %
RESULT press calibration jump: AUC 0.791  mean P on press frames 0.02 vs no-press 0.018  press frames above the no-press median 89.4%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.889  mean P on press frames 0.044 vs no-press 0.008  press frames above the no-press median 94.0%  (n 133 / 30709)
RESULT press calibration special: AUC 0.883  mean P on press frames 0.055 vs no-press 0.008  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 32.3%  families 15.5%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 45.6%

### arm pq_base906 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 (05:02)
val_loss=3.2412
RESULT pq_base906: frames=32394 change_recall=0.297 hold_agreement=0.575 repeat_prev=0.766 (expert 0.764) neutral=0.259 (expert 0.303)
RESULT pq_base906 press edges/min model:expert  A 21.78:13.22  B 19.78:11.0  X 25.34:30.12  Y 29.01:19.89  Z 2.67:10.0  L 20.0:26.34  R 18.45:8.33
RESULT pq_base906 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_base906 change events recall/precision  ±0: press 0.01/0.009 release 0.014/0.012 stick 0.021/0.021  |  ±2: press 0.064/0.055 release 0.052/0.045 stick 0.087/0.088  |  ±5: press 0.12/0.109 release 0.104/0.095 stick 0.186/0.193
RESULT pq_base906 vs idle: 16.0 env-min  SD/min 1.688  deaths/min 1.813  kills/min 0.125  dmg dealt/taken per min 34.571/0.25
RESULT pq_base906 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 27 (1.688/min)
RESULT pq_base906 offstage eps/min 6.875  recovery rate 0.721  neutral 0.347  repeat_prev 0.756  max frozen run 518 f  (35.8s wall)
RESULT pq_base906: 36 cases x 8 trials  recovery 0.365  died 0.635  timeout 0.0  never 6/36  always 2/36
RESULT pq_base906 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.17 (11 cases)  4+ routes: 0.438 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.2%  buttons 69.6%  families 35.9%  main dir 65.7% (bucket 49.6%, ±1 56.7%)  c dir 96.2%  trigger 91.8%  jump 87.5%  attack 95.9%  special 96.3%  surprise 3.646  change recall 51.8%  false change %
RESULT frames with a real press edge: n 1029  joint 0.2%  buttons 17.3%  families 10.8%  main dir 89.3% (bucket 80.9%, ±1 87.5%)  c dir 100.0%  trigger 82.7%  jump 58.5%  attack 87.1%  special 88.7%  surprise 5.239  change recall 23.7%  false change %
RESULT press calibration jump: AUC 0.797  mean P on press frames 0.03 vs no-press 0.021  press frames above the no-press median 90.4%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.886  mean P on press frames 0.05 vs no-press 0.009  press frames above the no-press median 94.7%  (n 133 / 30709)
RESULT press calibration special: AUC 0.878  mean P on press frames 0.074 vs no-press 0.009  press frames above the no-press median 90.4%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 32.3%  families 16.8%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 43.4%

### arm pq_on10 — --onset-weight 10 --neutral-weight 1.0 --action-oversample 1.0 (05:31)
val_loss=3.2554
RESULT pq_on10: frames=32394 change_recall=0.259 hold_agreement=0.529 repeat_prev=0.762 (expert 0.764) neutral=0.195 (expert 0.303)
RESULT pq_on10 press edges/min model:expert  A 20.67:13.22  B 23.23:11.0  X 25.67:30.12  Y 22.0:19.89  Z 6.67:10.0  L 20.78:26.34  R 29.78:8.33
RESULT pq_on10 jab1 episodes=4 A-edges/episode model=0.5 expert=0.25
RESULT pq_on10 change events recall/precision  ±0: press 0.012/0.01 release 0.007/0.006 stick 0.023/0.022  |  ±2: press 0.067/0.054 release 0.046/0.037 stick 0.084/0.082  |  ±5: press 0.116/0.095 release 0.089/0.073 stick 0.156/0.162
RESULT pq_on10 vs idle: 16.0 env-min  SD/min 1.313  deaths/min 1.438  kills/min 0.375  dmg dealt/taken per min 37.515/0.188
RESULT pq_on10 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 21 (1.313/min)
RESULT pq_on10 offstage eps/min 5.813  recovery rate 0.736  neutral 0.258  repeat_prev 0.747  max frozen run 279 f  (35.1s wall)
RESULT pq_on10: 36 cases x 8 trials  recovery 0.375  died 0.625  timeout 0.0  never 5/36  always 2/36
RESULT pq_on10 by routes: 1 routes: 0.375 (1 cases)  2-3 routes: 0.205 (11 cases)  4+ routes: 0.453 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.3%  buttons 69.5%  families 36.6%  main dir 66.3% (bucket 49.5%, ±1 56.8%)  c dir 96.2%  trigger 92.0%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.681  change recall 50.9%  false change %
RESULT frames with a real press edge: n 1029  joint 0.7%  buttons 17.1%  families 12.3%  main dir 89.7% (bucket 81.0%, ±1 87.5%)  c dir 100.0%  trigger 84.0%  jump 58.5%  attack 87.1%  special 88.8%  surprise 5.193  change recall 24.2%  false change %
RESULT press calibration jump: AUC 0.791  mean P on press frames 0.023 vs no-press 0.019  press frames above the no-press median 89.7%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.893  mean P on press frames 0.067 vs no-press 0.01  press frames above the no-press median 94.7%  (n 133 / 30709)
RESULT press calibration special: AUC 0.875  mean P on press frames 0.07 vs no-press 0.009  press frames above the no-press median 90.4%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.1% of 30362
RESULT STANDING change frames: n 161 change recall 31.7%  families 18.6%  | press edges same-family: jump 0% attack 0% special 0.9% grab 0% trigger 47.6%

### arm pq_tw3 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --transition-weight 3 (05:59)
val_loss=3.2343
RESULT pq_tw3: frames=32394 change_recall=0.273 hold_agreement=0.559 repeat_prev=0.756 (expert 0.764) neutral=0.215 (expert 0.303)
RESULT pq_tw3 press edges/min model:expert  A 29.23:13.22  B 24.89:11.0  X 26.23:30.12  Y 31.23:19.89  Z 5.0:10.0  L 26.56:26.34  R 24.67:8.33
RESULT pq_tw3 jab1 episodes=4 A-edges/episode model=0.5 expert=0.25
RESULT pq_tw3 change events recall/precision  ±0: press 0.013/0.009 release 0.012/0.009 stick 0.018/0.017  |  ±2: press 0.082/0.06 release 0.069/0.049 stick 0.086/0.084  |  ±5: press 0.147/0.107 release 0.12/0.09 stick 0.175/0.176
RESULT pq_tw3 vs idle: 16.0 env-min  SD/min 1.688  deaths/min 1.75  kills/min 0.25  dmg dealt/taken per min 44.386/0.063
RESULT pq_tw3 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 27 (1.688/min)
RESULT pq_tw3 offstage eps/min 7.25  recovery rate 0.757  neutral 0.386  repeat_prev 0.786  max frozen run 644 f  (35.3s wall)
RESULT pq_tw3: 36 cases x 8 trials  recovery 0.372  died 0.628  timeout 0.0  never 2/36  always 0/36
RESULT pq_tw3 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.193 (11 cases)  4+ routes: 0.438 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.6%  buttons 69.7%  families 36.4%  main dir 66.1% (bucket 50.0%, ±1 57.0%)  c dir 96.2%  trigger 91.9%  jump 87.6%  attack 95.9%  special 96.4%  surprise 3.637  change recall 52.3%  false change %
RESULT frames with a real press edge: n 1029  joint 0.3%  buttons 17.3%  families 11.9%  main dir 90.3% (bucket 81.5%, ±1 88.2%)  c dir 99.9%  trigger 83.5%  jump 58.5%  attack 87.1%  special 88.6%  surprise 5.101  change recall 24.0%  false change %
RESULT press calibration jump: AUC 0.791  mean P on press frames 0.033 vs no-press 0.021  press frames above the no-press median 90.1%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.882  mean P on press frames 0.061 vs no-press 0.01  press frames above the no-press median 93.2%  (n 133 / 30709)
RESULT press calibration special: AUC 0.872  mean P on press frames 0.064 vs no-press 0.009  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 36%  families 18.6%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 45.6%

### arm pq_tw3on10 — --onset-weight 10 --neutral-weight 1.0 --action-oversample 1.0 --transition-weight 3 (06:28)
val_loss=3.2488
RESULT pq_tw3on10: frames=32394 change_recall=0.292 hold_agreement=0.567 repeat_prev=0.796 (expert 0.764) neutral=0.218 (expert 0.303)
RESULT pq_tw3on10 press edges/min model:expert  A 22.34:13.22  B 22.89:11.0  X 20.0:30.12  Y 29.67:19.89  Z 7.45:10.0  L 16.34:26.34  R 23.12:8.33
RESULT pq_tw3on10 jab1 episodes=4 A-edges/episode model=0.25 expert=0.25
RESULT pq_tw3on10 change events recall/precision  ±0: press 0.009/0.008 release 0.013/0.011 stick 0.013/0.015  |  ±2: press 0.052/0.045 release 0.045/0.038 stick 0.07/0.079  |  ±5: press 0.11/0.096 release 0.096/0.085 stick 0.146/0.165
RESULT pq_tw3on10 vs idle: 16.0 env-min  SD/min 1.75  deaths/min 1.813  kills/min 0.438  dmg dealt/taken per min 29.747/0.063
RESULT pq_tw3on10 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 28 (1.75/min)
RESULT pq_tw3on10 offstage eps/min 6.313  recovery rate 0.698  neutral 0.351  repeat_prev 0.786  max frozen run 1090 f  (34.5s wall)
RESULT pq_tw3on10: 36 cases x 8 trials  recovery 0.389  died 0.611  timeout 0.0  never 5/36  always 1/36
RESULT pq_tw3on10 by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.17 (11 cases)  4+ routes: 0.479 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.8%  buttons 69.7%  families 36.5%  main dir 66.2% (bucket 49.9%, ±1 56.9%)  c dir 96.2%  trigger 91.8%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.655  change recall 52.2%  false change %
RESULT frames with a real press edge: n 1029  joint 0.4%  buttons 17.5%  families 11.2%  main dir 89.6% (bucket 81.4%, ±1 87.9%)  c dir 100.0%  trigger 82.8%  jump 58.5%  attack 87.1%  special 88.9%  surprise 5.172  change recall 23.5%  false change %
RESULT press calibration jump: AUC 0.794  mean P on press frames 0.032 vs no-press 0.021  press frames above the no-press median 89.9%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.889  mean P on press frames 0.051 vs no-press 0.009  press frames above the no-press median 95.5%  (n 133 / 30709)
RESULT press calibration special: AUC 0.878  mean P on press frames 0.107 vs no-press 0.01  press frames above the no-press median 90.4%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.1% of 30362
RESULT STANDING change frames: n 161 change recall 31.7%  families 19.3%  | press edges same-family: jump 0% attack 0% special 1.7% grab 0% trigger 43.4%

### arm pq_cw2 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --chunk-weight 2 (06:56)
val_loss=3.2361
RESULT pq_cw2: frames=32394 change_recall=0.292 hold_agreement=0.593 repeat_prev=0.796 (expert 0.764) neutral=0.297 (expert 0.303)
RESULT pq_cw2 press edges/min model:expert  A 23.23:13.22  B 11.56:11.0  X 24.56:30.12  Y 29.34:19.89  Z 4.33:10.0  L 12.45:26.34  R 15.11:8.33
RESULT pq_cw2 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_cw2 change events recall/precision  ±0: press 0.002/0.002 release 0.008/0.008 stick 0.018/0.022  |  ±2: press 0.037/0.038 release 0.039/0.039 stick 0.066/0.084  |  ±5: press 0.084/0.086 release 0.079/0.079 stick 0.14/0.178
RESULT pq_cw2 vs idle: 16.0 env-min  SD/min 1.063  deaths/min 1.063  kills/min 0.188  dmg dealt/taken per min 34.849/0.063
RESULT pq_cw2 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 17 (1.063/min)
RESULT pq_cw2 offstage eps/min 4.125  recovery rate 0.742  neutral 0.382  repeat_prev 0.796  max frozen run 650 f  (35.0s wall)
RESULT pq_cw2: 36 cases x 8 trials  recovery 0.344  died 0.656  timeout 0.0  never 7/36  always 2/36
RESULT pq_cw2 by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.102 (11 cases)  4+ routes: 0.443 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.6%  buttons 69.5%  families 37.0%  main dir 66.8% (bucket 49.8%, ±1 57.0%)  c dir 96.2%  trigger 91.8%  jump 87.5%  attack 95.9%  special 96.3%  surprise 3.704  change recall 51.6%  false change %
RESULT frames with a real press edge: n 1029  joint 0.4%  buttons 17.2%  families 11.2%  main dir 88.8% (bucket 80.1%, ±1 86.7%)  c dir 100.0%  trigger 83.0%  jump 58.5%  attack 87.1%  special 88.7%  surprise 5.35  change recall 24.4%  false change %
RESULT press calibration jump: AUC 0.779  mean P on press frames 0.026 vs no-press 0.02  press frames above the no-press median 86.5%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.899  mean P on press frames 0.066 vs no-press 0.009  press frames above the no-press median 95.5%  (n 133 / 30709)
RESULT press calibration special: AUC 0.886  mean P on press frames 0.051 vs no-press 0.007  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 31.7%  families 20.5%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 44.3%

### arm pq_bw4 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --button-weight 4 (07:24)
val_loss=3.3224
RESULT pq_bw4: frames=32394 change_recall=0.258 hold_agreement=0.541 repeat_prev=0.772 (expert 0.764) neutral=0.227 (expert 0.303)
RESULT pq_bw4 press edges/min model:expert  A 14.45:13.22  B 18.34:11.0  X 18.67:30.12  Y 18.45:19.89  Z 5.67:10.0  L 24.23:26.34  R 32.34:8.33
RESULT pq_bw4 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_bw4 change events recall/precision  ±0: press 0.01/0.009 release 0.008/0.008 stick 0.018/0.018  |  ±2: press 0.055/0.05 release 0.042/0.038 stick 0.095/0.095  |  ±5: press 0.099/0.091 release 0.094/0.087 stick 0.178/0.187
RESULT pq_bw4 vs idle: 16.0 env-min  SD/min 1.563  deaths/min 1.563  kills/min 0.438  dmg dealt/taken per min 43.175/0.063
RESULT pq_bw4 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 25 (1.563/min)
RESULT pq_bw4 offstage eps/min 5.188  recovery rate 0.688  neutral 0.312  repeat_prev 0.764  max frozen run 403 f  (35.6s wall)
RESULT pq_bw4: 36 cases x 8 trials  recovery 0.382  died 0.618  timeout 0.0  never 8/36  always 4/36
RESULT pq_bw4 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.261 (11 cases)  4+ routes: 0.422 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.8%  buttons 69.5%  families 36.9%  main dir 66.2% (bucket 50.0%, ±1 56.9%)  c dir 96.2%  trigger 92.2%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.661  change recall 52.5%  false change %
RESULT frames with a real press edge: n 1029  joint 0.9%  buttons 16.7%  families 13.4%  main dir 89.3% (bucket 81.2%, ±1 87.4%)  c dir 99.9%  trigger 85.0%  jump 58.5%  attack 87.1%  special 88.9%  surprise 5.183  change recall 25.2%  false change %
RESULT press calibration jump: AUC 0.797  mean P on press frames 0.025 vs no-press 0.018  press frames above the no-press median 89.9%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.91  mean P on press frames 0.057 vs no-press 0.009  press frames above the no-press median 97.0%  (n 133 / 30709)
RESULT press calibration special: AUC 0.879  mean P on press frames 0.092 vs no-press 0.01  press frames above the no-press median 92.2%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.1% of 30362
RESULT STANDING change frames: n 161 change recall 32.3%  families 18%  | press edges same-family: jump 0% attack 0% special 1.7% grab 0% trigger 51.1%

### arm pq_nw05 — --onset-weight 5 --neutral-weight 0.5 --action-oversample 1.0 (07:52)
val_loss=3.3335
RESULT pq_nw05: frames=32394 change_recall=0.267 hold_agreement=0.523 repeat_prev=0.721 (expert 0.764) neutral=0.094 (expert 0.303)
RESULT pq_nw05 press edges/min model:expert  A 16.23:13.22  B 29.34:11.0  X 31.34:30.12  Y 35.01:19.89  Z 2.22:10.0  L 23.45:26.34  R 22.78:8.33
RESULT pq_nw05 jab1 episodes=4 A-edges/episode model=0.25 expert=0.25
RESULT pq_nw05 change events recall/precision  ±0: press 0.014/0.01 release 0.013/0.01 stick 0.018/0.014  |  ±2: press 0.053/0.04 release 0.055/0.041 stick 0.088/0.07  |  ±5: press 0.103/0.08 release 0.108/0.082 stick 0.169/0.144
RESULT pq_nw05 vs idle: 16.0 env-min  SD/min 3.563  deaths/min 3.75  kills/min 0.625  dmg dealt/taken per min 47.077/0.313
RESULT pq_nw05 deaths by verdict at the decision frame: kill 3 (0.188/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 57 (3.563/min)
RESULT pq_nw05 offstage eps/min 10.813  recovery rate 0.634  neutral 0.186  repeat_prev 0.729  max frozen run 136 f  (37.5s wall)
RESULT pq_nw05: 36 cases x 8 trials  recovery 0.465  died 0.535  timeout 0.0  never 3/36  always 3/36
RESULT pq_nw05 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.261 (11 cases)  4+ routes: 0.536 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 15.1%  buttons 69.6%  families 36.2%  main dir 65.9% (bucket 48.2%, ±1 56.0%)  c dir 96.2%  trigger 91.8%  jump 87.5%  attack 95.9%  special 96.3%  surprise 3.735  change recall 53.1%  false change %
RESULT frames with a real press edge: n 1029  joint 0.3%  buttons 17.3%  families 11.2%  main dir 89.6% (bucket 80.7%, ±1 87.2%)  c dir 99.9%  trigger 82.9%  jump 58.5%  attack 87.1%  special 88.7%  surprise 5.211  change recall 24.4%  false change %
RESULT press calibration jump: AUC 0.773  mean P on press frames 0.035 vs no-press 0.024  press frames above the no-press median 84.4%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.88  mean P on press frames 0.05 vs no-press 0.01  press frames above the no-press median 93.2%  (n 133 / 30709)
RESULT press calibration special: AUC 0.878  mean P on press frames 0.066 vs no-press 0.01  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 58.4%  families 28%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 43.7%

### arm pq_nw025 — --onset-weight 5 --neutral-weight 0.25 --action-oversample 1.0 (08:20)
val_loss=3.338
RESULT pq_nw025: frames=32394 change_recall=0.258 hold_agreement=0.512 repeat_prev=0.733 (expert 0.764) neutral=0.036 (expert 0.303)
RESULT pq_nw025 press edges/min model:expert  A 24.0:13.22  B 27.89:11.0  X 21.67:30.12  Y 27.56:19.89  Z 5.33:10.0  L 16.89:26.34  R 31.01:8.33
RESULT pq_nw025 jab1 episodes=4 A-edges/episode model=0.25 expert=0.25
RESULT pq_nw025 change events recall/precision  ±0: press 0.006/0.004 release 0.008/0.006 stick 0.015/0.015  |  ±2: press 0.061/0.048 release 0.057/0.045 stick 0.069/0.071  |  ±5: press 0.121/0.094 release 0.096/0.077 stick 0.131/0.14
RESULT pq_nw025 vs idle: 16.0 env-min  SD/min 1.75  deaths/min 1.875  kills/min 0.438  dmg dealt/taken per min 35.639/0.125
RESULT pq_nw025 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 28 (1.75/min)
RESULT pq_nw025 offstage eps/min 7.125  recovery rate 0.727  neutral 0.072  repeat_prev 0.731  max frozen run 199 f  (34.9s wall)
RESULT pq_nw025: 36 cases x 8 trials  recovery 0.503  died 0.493  timeout 0.003  never 0/36  always 2/36
RESULT pq_nw025 by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.307 (11 cases)  4+ routes: 0.578 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 14.6%  buttons 69.6%  families 36.5%  main dir 66.3% (bucket 47.8%, ±1 55.5%)  c dir 96.3%  trigger 91.7%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.799  change recall 51.2%  false change %
RESULT frames with a real press edge: n 1029  joint 0.3%  buttons 17.5%  families 11.0%  main dir 89.4% (bucket 81.3%, ±1 87.5%)  c dir 100.0%  trigger 82.6%  jump 58.5%  attack 87.1%  special 88.8%  surprise 5.138  change recall 23.6%  false change %
RESULT press calibration jump: AUC 0.744  mean P on press frames 0.026 vs no-press 0.023  press frames above the no-press median 82.2%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.884  mean P on press frames 0.077 vs no-press 0.012  press frames above the no-press median 94.0%  (n 133 / 30709)
RESULT press calibration special: AUC 0.878  mean P on press frames 0.071 vs no-press 0.012  press frames above the no-press median 92.2%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 48.4%  families 28.6%  | press edges same-family: jump 0% attack 0% special 0.9% grab 0% trigger 42.7%

### arm pq_sf3 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 1.0 --silent-fall-weight 3 (08:48)
val_loss=3.2627
RESULT pq_sf3: frames=32394 change_recall=0.304 hold_agreement=0.589 repeat_prev=0.78 (expert 0.764) neutral=0.355 (expert 0.303)
RESULT pq_sf3 press edges/min model:expert  A 13.67:13.22  B 18.78:11.0  X 21.78:30.12  Y 24.89:19.89  Z 5.89:10.0  L 15.11:26.34  R 19.11:8.33
RESULT pq_sf3 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_sf3 change events recall/precision  ±0: press 0.007/0.007 release 0.012/0.012 stick 0.018/0.016  |  ±2: press 0.048/0.048 release 0.05/0.05 stick 0.095/0.084  |  ±5: press 0.099/0.104 release 0.091/0.09 stick 0.19/0.177
RESULT pq_sf3 vs idle: 16.0 env-min  SD/min 1.688  deaths/min 1.688  kills/min 0.0  dmg dealt/taken per min 33.021/0.0
RESULT pq_sf3 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 27 (1.688/min)
RESULT pq_sf3 offstage eps/min 6.0  recovery rate 0.71  neutral 0.364  repeat_prev 0.761  max frozen run 363 f  (34.8s wall)
RESULT pq_sf3: 36 cases x 8 trials  recovery 0.385  died 0.615  timeout 0.0  never 4/36  always 1/36
RESULT pq_sf3 by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.193 (11 cases)  4+ routes: 0.464 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.9%  buttons 69.7%  families 35.7%  main dir 64.3% (bucket 49.5%, ±1 56.4%)  c dir 96.3%  trigger 92.3%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.679  change recall 53.2%  false change %
RESULT frames with a real press edge: n 1029  joint 2.4%  buttons 17.2%  families 15.5%  main dir 89.1% (bucket 81.7%, ±1 87.4%)  c dir 99.9%  trigger 87.3%  jump 58.5%  attack 87.1%  special 88.8%  surprise 5.233  change recall 27.9%  false change %
RESULT press calibration jump: AUC 0.783  mean P on press frames 0.027 vs no-press 0.02  press frames above the no-press median 88.9%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.889  mean P on press frames 0.043 vs no-press 0.008  press frames above the no-press median 95.5%  (n 133 / 30709)
RESULT press calibration special: AUC 0.873  mean P on press frames 0.063 vs no-press 0.009  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.4% of 30362
RESULT STANDING change frames: n 161 change recall 31.1%  families 17.4%  | press edges same-family: jump 0% attack 0% special 0.9% grab 0% trigger 59.5%

### arm pq_ao2 — --onset-weight 5 --neutral-weight 1.0 --action-oversample 2.0 (09:17)
val_loss=3.2409
RESULT pq_ao2: frames=32394 change_recall=0.293 hold_agreement=0.572 repeat_prev=0.776 (expert 0.764) neutral=0.21 (expert 0.303)
RESULT pq_ao2 press edges/min model:expert  A 23.78:13.22  B 24.23:11.0  X 23.0:30.12  Y 18.67:19.89  Z 4.33:10.0  L 14.45:26.34  R 23.0:8.33
RESULT pq_ao2 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_ao2 change events recall/precision  ±0: press 0.012/0.011 release 0.007/0.006 stick 0.017/0.017  |  ±2: press 0.062/0.056 release 0.04/0.036 stick 0.081/0.084  |  ±5: press 0.115/0.108 release 0.096/0.09 stick 0.17/0.189
RESULT pq_ao2 vs idle: 16.0 env-min  SD/min 2.188  deaths/min 2.25  kills/min 0.375  dmg dealt/taken per min 30.342/0.125
RESULT pq_ao2 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 35 (2.188/min)
RESULT pq_ao2 offstage eps/min 7.125  recovery rate 0.676  neutral 0.298  repeat_prev 0.755  max frozen run 682 f  (36.7s wall)
RESULT pq_ao2: 36 cases x 8 trials  recovery 0.375  died 0.625  timeout 0.0  never 2/36  always 1/36
RESULT pq_ao2 by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.193 (11 cases)  4+ routes: 0.438 (24 cases)
RESULT CHANGE frames (real input changed): n 6454  joint 16.7%  buttons 69.5%  families 36.7%  main dir 66.5% (bucket 50.0%, ±1 57.2%)  c dir 96.2%  trigger 91.9%  jump 87.5%  attack 95.9%  special 96.3%  surprise 3.683  change recall 52.4%  false change %
RESULT frames with a real press edge: n 1029  joint 0.3%  buttons 17.3%  families 11.4%  main dir 90.1% (bucket 81.3%, ±1 87.8%)  c dir 100.0%  trigger 83.1%  jump 58.5%  attack 87.1%  special 88.7%  surprise 5.332  change recall 24.7%  false change %
RESULT press calibration jump: AUC 0.79  mean P on press frames 0.02 vs no-press 0.018  press frames above the no-press median 88.7%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.901  mean P on press frames 0.049 vs no-press 0.009  press frames above the no-press median 97.7%  (n 133 / 30709)
RESULT press calibration special: AUC 0.878  mean P on press frames 0.061 vs no-press 0.008  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 32.3%  families 17.4%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 44.7%

### arm pq_combo — --onset-weight 10 --neutral-weight 0.5 --action-oversample 1.0 --transition-weight 3 (09:45)
val_loss=3.3243
RESULT pq_combo: frames=32394 change_recall=0.243 hold_agreement=0.505 repeat_prev=0.725 (expert 0.764) neutral=0.104 (expert 0.303)
RESULT pq_combo press edges/min model:expert  A 20.67:13.22  B 16.78:11.0  X 33.45:30.12  Y 28.45:19.89  Z 5.11:10.0  L 25.23:26.34  R 23.89:8.33
RESULT pq_combo jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT pq_combo change events recall/precision  ±0: press 0.014/0.011 release 0.007/0.006 stick 0.018/0.016  |  ±2: press 0.078/0.06 release 0.054/0.042 stick 0.088/0.076  |  ±5: press 0.14/0.113 release 0.11/0.086 stick 0.178/0.166
RESULT pq_combo vs idle: 16.0 env-min  SD/min 1.875  deaths/min 1.875  kills/min 0.5  dmg dealt/taken per min 30.062/0.0
RESULT pq_combo deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 30 (1.875/min)
RESULT pq_combo offstage eps/min 8.625  recovery rate 0.781  neutral 0.173  repeat_prev 0.698  max frozen run 165 f  (34.3s wall)
RESULT pq_combo: 36 cases x 8 trials  recovery 0.396  died 0.604  timeout 0.0  never 7/36  always 1/36
RESULT pq_combo by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.114 (11 cases)  4+ routes: 0.505 (24 cases)

## Stage 1 READ (Claude, 10:10) — see `node scripts/pq_digest.js`
Seed floor (base vs base906): fam% ±0.4, joint ±0.6, STANDING recall ±0, SD/min ±0.6 (closed loop is NOISY), AUC ±0.01.
PRESS COMMITMENT: no loss weight moves it. Exact-frame same-family press agreement is 0-2 % on EVERY arm
(onset 10, transition 3, chunk 2, button 4, combos); calibration AUC flat (jump 0.77-0.80, attack 0.88-0.91).
=> the objective is the lever, not its weights: the target "press on exactly this frame" has to become
"press within the next k frames" (hazard / onset-window head) — code, not flags. Queue for tomorrow.
PASSIVITY: --neutral-weight 0.5 doubles STANDING change recall (32 -> 58 %; 0.25 -> 48 %) — far beyond the
floor — but SD/min 1.1 -> 3.6 and joint -1.7: the bot moves more everywhere, including offstage, and dies.
=> a blanket anti-passivity weight trades standing-still for self-destructs (the 09-05 history again).
The targeted version — weight neutral frames only when GROUNDED and ONSTAGE — needs a flag; code for tomorrow.
Nothing here is promotable to full scale as-is. Stage 2 (master-only) and stage 3 (null arm) continue.
RESULT CHANGE frames (real input changed): n 6454  joint 15.1%  buttons 69.6%  families 36.2%  main dir 65.9% (bucket 48.4%, ±1 55.9%)  c dir 96.2%  trigger 91.9%  jump 87.5%  attack 95.9%  special 96.4%  surprise 3.74  change recall 51.2%  false change %
RESULT frames with a real press edge: n 1029  joint 0.6%  buttons 17.4%  families 11.5%  main dir 89.9% (bucket 81.0%, ±1 87.7%)  c dir 99.9%  trigger 83.1%  jump 58.5%  attack 87.1%  special 88.8%  surprise 5.128  change recall 24.4%  false change %
RESULT press calibration jump: AUC 0.759  mean P on press frames 0.041 vs no-press 0.025  press frames above the no-press median 82.9%  (n 416 / 28954)
RESULT press calibration attack: AUC 0.887  mean P on press frames 0.068 vs no-press 0.011  press frames above the no-press median 94.0%  (n 133 / 30709)
RESULT press calibration special: AUC 0.88  mean P on press frames 0.051 vs no-press 0.008  press frames above the no-press median 91.3%  (n 115 / 30420)
RESULT false press (model press edge on a frame with no real press): 0.0% of 30362
RESULT STANDING change frames: n 161 change recall 35.4%  families 17.4%  | press edges same-family: jump 0% attack 0% special 0.9% grab 0% trigger 44.7%

STAGE 1 DONE (10:13)

## Stage 2 — master-only full arm, epochs 1-6 (10:13)

### master epoch 1 (lr 0.0002) (10:13)
val_loss=3.1396
DONE fox_mamba_master_ep1
RESULT ALL frames: n 39430  joint 75.7%  buttons 93.4%  families 85.6%  main dir 92.3% (bucket 85.6%, ±1 89.0%)  c dir 99.1%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 1.822  change recall 68.2%  false change 4.2%
RESULT CHANGE frames (real input changed): n 11028  joint 23.8%  buttons 76.7%  families 50.8%  main dir 72.7% (bucket 48.7%, ±1 60.7%)  c dir 97.0%  trigger 93.7%  jump 90.4%  attack 97.0%  special 96.3%  surprise 3.25  change recall 68.2%  false change %
RESULT frames with a real press edge: n 1314  joint 0.9%  buttons 19.5%  families 7.9%  main dir 88.7% (bucket 77.2%, ±1 85.3%)  c dir 99.4%  trigger 78.6%  jump 57.6%  attack 86.8%  special 84.4%  surprise 4.876  change recall 29.8%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.0%  buttons 80.5%  families 62.3%  main dir 77.0% (bucket 56.8%, ±1 66.9%)  c dir 99.2%  trigger 96.7%  jump 92.0%  attack 97.5%  special 96.9%  surprise 2.51  change recall 69.2%  false change 3.9%
RESULT press calibration jump: AUC 0.854  mean P on press frames 0.073 vs no-press 0.024  press frames above the no-press median 94.5%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.916  mean P on press frames 0.082 vs no-press 0.009  press frames above the no-press median 98.8%  (n 165 / 38183)
RESULT press calibration special: AUC 0.948  mean P on press frames 0.16 vs no-press 0.01  press frames above the no-press median 97.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.887  mean P on press frames 0.34 vs no-press 0.038  press frames above the no-press median 93.8%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 47% families 34.6%
EPOCH 1 DONE light (13:01)
### candidate s906_ep2 on the master held-out split (the comparison)
RESULT ALL frames: n 39430  joint 75.0%  buttons 93.5%  families 85.9%  main dir 92.6% (bucket 86.4%, ±1 89.6%)  c dir 99.2%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.978  change recall 72.1%  false change 6.0%
RESULT CHANGE frames (real input changed): n 11028  joint 26.2%  buttons 77.4%  families 52.1%  main dir 73.9% (bucket 51.6%, ±1 62.9%)  c dir 97.0%  trigger 93.6%  jump 90.7%  attack 97.0%  special 96.6%  surprise 3.263  change recall 72.1%  false change %
RESULT frames with a real press edge: n 1314  joint 1.9%  buttons 20.8%  families 10.2%  main dir 90.6% (bucket 78.9%, ±1 86.8%)  c dir 99.5%  trigger 79.5%  jump 57.6%  attack 86.7%  special 85.5%  surprise 4.861  change recall 29.6%  false change %
RESULT press calibration jump: AUC 0.874  mean P on press frames 0.092 vs no-press 0.023  press frames above the no-press median 97.5%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.939  mean P on press frames 0.11 vs no-press 0.009  press frames above the no-press median 97.6%  (n 165 / 38183)

### master epoch 2 (lr 0.0002) (13:11)
val_loss=2.9819
DONE fox_mamba_master_ep2
RESULT ALL frames: n 39430  joint 76.3%  buttons 93.5%  families 86.0%  main dir 92.7% (bucket 86.2%, ±1 89.5%)  c dir 99.1%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.29  change recall 70.5%  false change 4.0%
RESULT CHANGE frames (real input changed): n 11028  joint 25.8%  buttons 77.0%  families 52.3%  main dir 74.1% (bucket 50.9%, ±1 62.7%)  c dir 97.0%  trigger 93.8%  jump 90.6%  attack 97.1%  special 96.4%  surprise 3.167  change recall 70.5%  false change %
RESULT frames with a real press edge: n 1314  joint 1.1%  buttons 20.3%  families 8.6%  main dir 89.2% (bucket 78.9%, ±1 85.9%)  c dir 99.5%  trigger 78.8%  jump 57.7%  attack 86.8%  special 84.9%  surprise 4.805  change recall 28.8%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 41.5%  buttons 80.8%  families 63.4%  main dir 78.2% (bucket 58.6%, ±1 68.6%)  c dir 99.2%  trigger 96.6%  jump 92.1%  attack 97.5%  special 97.0%  surprise 2.389  change recall 71.5%  false change 4.3%
RESULT press calibration jump: AUC 0.865  mean P on press frames 0.081 vs no-press 0.023  press frames above the no-press median 96.2%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.93  mean P on press frames 0.095 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.954  mean P on press frames 0.18 vs no-press 0.009  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.895  mean P on press frames 0.348 vs no-press 0.034  press frames above the no-press median 93.5%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 47.4% families 35.9%
EPOCH 2 DONE light (15:59)

### master epoch 3 (lr 0.0001) (15:59)

## Stage 2 interim READ (Claude, 16:05) — apples to apples on the MASTER held-out split
                      fam%(change)  joint  change recall  AUC jump   STANDING recall
candidate s906_ep2        52.1      26.2      72.1         0.874       (49 on corpus)
master ep1 (lr 2e-4)      50.8      23.8      68.2         0.854        47.0
master ep2 (lr 2e-4)      52.3      25.8      70.5         0.865        47.4
ep1 -> ep2 = +1.5 fam / +2.0 joint: beyond the testbed seed floor (±0.4 / ±0.6), so the arm keeps
running (epochs 3-6 at lr 1e-4, ~2.8 h each, done ~03:00). Read so far: two passes over the 7,853
master games (~1/4 of the candidate's training frames) MATCH the candidate on option families and
trail it by 0.4 joint / 1.6 recall; data quality buys compute, it has not yet bought agreement
above the candidate. Timing: 109 min of 48 ms steps + ~50 min re-embedding per epoch (caches off).
val_loss=2.9773
DONE fox_mamba_master_ep3
RESULT ALL frames: n 39430  joint 74.5%  buttons 93.4%  families 86.2%  main dir 93.1% (bucket 86.3%, ±1 89.7%)  c dir 99.1%  trigger 97.6%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.642  change recall 72.6%  false change 6.5%
RESULT CHANGE frames (real input changed): n 11028  joint 25.5%  buttons 77.1%  families 53.4%  main dir 75.5% (bucket 51.5%, ±1 63.6%)  c dir 97.0%  trigger 93.2%  jump 90.7%  attack 97.1%  special 96.4%  surprise 3.242  change recall 72.6%  false change %
RESULT frames with a real press edge: n 1314  joint 1.2%  buttons 20.3%  families 8.1%  main dir 89.9% (bucket 78.8%, ±1 86.0%)  c dir 99.5%  trigger 77.9%  jump 57.8%  attack 86.8%  special 84.9%  surprise 4.917  change recall 29.5%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.9%  buttons 80.8%  families 64.3%  main dir 79.3% (bucket 59.1%, ±1 69.3%)  c dir 99.1%  trigger 96.2%  jump 92.2%  attack 97.6%  special 97.0%  surprise 2.428  change recall 73.7%  false change 5.9%
RESULT press calibration jump: AUC 0.866  mean P on press frames 0.082 vs no-press 0.023  press frames above the no-press median 96.4%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.927  mean P on press frames 0.096 vs no-press 0.009  press frames above the no-press median 98.8%  (n 165 / 38183)
RESULT press calibration special: AUC 0.959  mean P on press frames 0.179 vs no-press 0.009  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.898  mean P on press frames 0.337 vs no-press 0.031  press frames above the no-press median 95.6%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 50% families 38.9%
EPOCH 3 DONE light (18:48)

### master epoch 4 (lr 0.0001) (18:48)
val_loss=2.9042
DONE fox_mamba_master_ep4
RESULT ALL frames: n 39430  joint 76.3%  buttons 93.5%  families 86.2%  main dir 93.0% (bucket 86.5%, ±1 89.8%)  c dir 99.1%  trigger 97.7%  jump 97.4%  attack 99.2%  special 99.0%  surprise 2.597  change recall 71.4%  false change 4.4%
RESULT CHANGE frames (real input changed): n 11028  joint 26.6%  buttons 77.1%  families 53.2%  main dir 75.1% (bucket 52.1%, ±1 63.7%)  c dir 97.0%  trigger 93.7%  jump 90.6%  attack 97.1%  special 96.4%  surprise 3.149  change recall 71.4%  false change %
RESULT frames with a real press edge: n 1314  joint 1.3%  buttons 20.0%  families 9.0%  main dir 90.5% (bucket 80.1%, ±1 86.8%)  c dir 99.5%  trigger 78.9%  jump 57.8%  attack 86.6%  special 84.9%  surprise 4.78  change recall 28.5%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 42.1%  buttons 80.8%  families 64.1%  main dir 78.9% (bucket 59.6%, ±1 69.4%)  c dir 99.1%  trigger 96.6%  jump 92.2%  attack 97.5%  special 97.0%  surprise 2.339  change recall 72.4%  false change 4.4%
RESULT press calibration jump: AUC 0.875  mean P on press frames 0.087 vs no-press 0.022  press frames above the no-press median 97.1%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.937  mean P on press frames 0.099 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.959  mean P on press frames 0.186 vs no-press 0.009  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.903  mean P on press frames 0.359 vs no-press 0.033  press frames above the no-press median 95.3%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 49.6% families 39.3%
EPOCH 4 DONE light (21:36)

### master epoch 5 (lr 0.0001) (21:36)

## Stage 2 READ at epoch 4 (Claude, 21:40) — master held-out split
                      fam%(change)  joint  AUC jump  STANDING recall
candidate s906_ep2        52.1      26.2    0.874     (49 corpus)
master ep1                50.8      23.8    0.854      47.0
master ep2                52.3      25.8    0.865      47.4
master ep3 (lr 1e-4)      53.4      25.5    0.866      50.0
master ep4 (lr 1e-4)      53.2      26.6    0.875      49.6
Epoch 4 is at or above the candidate on every agreement number (fam +1.1, joint +0.4, AUC +0.001,
STANDING +0.6 vs its corpus value) with ~half the candidate's training frames; ep3->ep4 is flat on
families and +1.1 joint, i.e. the curve is bending. Epochs 5-6 run as queued; the league in stage 3
puts master_last against the candidate in play.
val_loss=3.0224
DONE fox_mamba_master_ep5
RESULT ALL frames: n 39430  joint 73.4%  buttons 93.4%  families 85.9%  main dir 92.9% (bucket 86.3%, ±1 89.6%)  c dir 99.1%  trigger 97.4%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.784  change recall 73.0%  false change 8.0%
RESULT CHANGE frames (real input changed): n 11028  joint 25.5%  buttons 77.2%  families 52.4%  main dir 74.8% (bucket 51.4%, ±1 63.2%)  c dir 97.0%  trigger 92.7%  jump 90.8%  attack 97.1%  special 96.6%  surprise 3.306  change recall 73.0%  false change %
RESULT frames with a real press edge: n 1314  joint 0.9%  buttons 19.7%  families 6.7%  main dir 89.9% (bucket 78.5%, ±1 85.9%)  c dir 99.5%  trigger 76.0%  jump 57.5%  attack 86.7%  special 84.8%  surprise 5.07  change recall 29.7%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.5%  buttons 80.9%  families 63.5%  main dir 78.7% (bucket 59.1%, ±1 69.0%)  c dir 99.1%  trigger 95.8%  jump 92.3%  attack 97.5%  special 97.1%  surprise 2.475  change recall 74.1%  false change 7.3%
RESULT press calibration jump: AUC 0.866  mean P on press frames 0.074 vs no-press 0.021  press frames above the no-press median 96.2%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.929  mean P on press frames 0.1 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.957  mean P on press frames 0.161 vs no-press 0.008  press frames above the no-press median 98.1%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.905  mean P on press frames 0.289 vs no-press 0.024  press frames above the no-press median 95.6%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 53% families 35.9%
EPOCH 5 DONE light (00:24)

### master epoch 6 (lr 0.0001) (00:24)
val_loss=2.977
DONE fox_mamba_master_ep6
RESULT ALL frames: n 39430  joint 73.5%  buttons 93.5%  families 86.2%  main dir 93.2% (bucket 86.6%, ±1 90.0%)  c dir 99.1%  trigger 97.4%  jump 97.3%  attack 99.2%  special 99.0%  surprise 3.129  change recall 73.1%  false change 8.2%
RESULT CHANGE frames (real input changed): n 11028  joint 26.4%  buttons 77.2%  families 53.4%  main dir 76.0% (bucket 52.5%, ±1 64.5%)  c dir 97.0%  trigger 92.8%  jump 90.7%  attack 97.1%  special 96.5%  surprise 3.33  change recall 73.1%  false change %
RESULT frames with a real press edge: n 1314  joint 1.4%  buttons 20.2%  families 7.7%  main dir 90.4% (bucket 79.1%, ±1 86.4%)  c dir 99.5%  trigger 76.7%  jump 57.7%  attack 86.8%  special 85.2%  surprise 5.121  change recall 30.3%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 41.2%  buttons 80.9%  families 64.3%  main dir 79.7% (bucket 60.0%, ±1 70.0%)  c dir 99.1%  trigger 95.8%  jump 92.2%  attack 97.6%  special 97.1%  surprise 2.472  change recall 74.3%  false change 7.6%
RESULT press calibration jump: AUC 0.873  mean P on press frames 0.079 vs no-press 0.021  press frames above the no-press median 96.7%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.936  mean P on press frames 0.109 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.96  mean P on press frames 0.186 vs no-press 0.008  press frames above the no-press median 98.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.909  mean P on press frames 0.318 vs no-press 0.025  press frames above the no-press median 97.2%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 54.3% families 40.2%
EPOCH 6 DONE light (03:14)

STAGE 2 DONE (03:14)

## Stage 3 — dose-matched null arm (epoch 3 from s906 ep2, null mix) (03:14)

## Stage 2 FINAL READ (Claude, 03:20) — master held-out split; candidate = fam 52.1 / joint 26.2 / recall 72.1 / AUC 0.874
      ep   fam%   joint  recall  AUCjump  STANDING
      1    50.8   23.8   68.2    0.854    47   (the parser had taken the candidate row for epoch 1; corrected)
      2    52.3   25.8   70.5    0.865    47.4
      3    53.4   25.5   72.6    0.866    50
      4    53.2   26.6   71.4    0.875    49.6
      5    52.4   25.5   73.0    0.866    53
      6    53.4   26.4   73.1    0.873    54.3
Plateau from epoch 3: families 53.2-53.4 (candidate 52.1), joint 25.5-26.6 (26.2), AUC 0.866-0.875 (0.874).
STANDING change recall climbs every epoch (47 -> 54 %; candidate 49 on its corpus split): the master-only model is
LESS passive from standing without any loss weighting. Read: training on master games only reaches the candidate’s
agreement with ~1/2 the frames and beats it by ~1 point on option families and ~5 points on standing-state activity;
it does not run away from it. The league (stage 3) tests whether that is worth anything in play.
val_loss=3.0826
DONE fox_mamba_v3s906_ep3_null9o
expert (FD reference)                        side-B first-means trips 88 / 1788 eps  (carried 5, later 83)
evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3      side-B first-means trips 17 / 155 eps  (carried 17, later 0)
evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 side-B first-means trips 18 / 298 eps  (carried 18, later 0)
fox_mamba_v3s906_ep1                         side-B first-means trips 17 / 332 eps  (carried 10, later 7)
fox_mamba_v3s906_ep2                         side-B first-means trips 6 / 319 eps  (carried 3, later 3)
expert (FD reference)                        carried side-B: n=5  jump spent inside the 90 f before: 4 (80%)  already gone 90 f out: 0  still in hand at the window start, no drop: 1
expert (FD reference)                        all near trips starting with no jump: n=120  jump spent inside the 90 f before: 89 (74%)  already gone 90 f out: 24  still in hand at the window start, no drop: 7
evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3      carried side-B: n=17  jump spent inside the 90 f before: 15 (88%)  already gone 90 f out: 0  still in hand at the window start, no drop: 2
evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3      all near trips starting with no jump: n=29  jump spent inside the 90 f before: 25 (86%)  already gone 90 f out: 0  still in hand at the window start, no drop: 4
evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 carried side-B: n=18  jump spent inside the 90 f before: 14 (78%)  already gone 90 f out: 2  still in hand at the window start, no drop: 2
evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 all near trips starting with no jump: n=36  jump spent inside the 90 f before: 31 (86%)  already gone 90 f out: 2  still in hand at the window start, no drop: 3
fox_mamba_v3s906_ep1                         carried side-B: n=10  jump spent inside the 90 f before: 8 (80%)  already gone 90 f out: 1  still in hand at the window start, no drop: 1
fox_mamba_v3s906_ep1                         all near trips starting with no jump: n=32  jump spent inside the 90 f before: 26 (81%)  already gone 90 f out: 5  still in hand at the window start, no drop: 1
fox_mamba_v3s906_ep2                         carried side-B: n=3  jump spent inside the 90 f before: 2 (67%)  already gone 90 f out: 0  still in hand at the window start, no drop: 1
fox_mamba_v3s906_ep2                         all near trips starting with no jump: n=19  jump spent inside the 90 f before: 15 (79%)  already gone 90 f out: 3  still in hand at the window start, no drop: 1
expert (FD reference)                        firefox 365 / 1788 eps  angle err q50/q75 19°/34°  >60° 7%
evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3      firefox 8 / 155 eps  angle err q50/q75 14°/48°  >60° 0%
evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 firefox 47 / 298 eps  angle err q50/q75 29°/49°  >60° 11%
fox_mamba_v3s906_ep1                         firefox 89 / 332 eps  angle err q50/q75 25°/43°  >60° 11%
fox_mamba_v3s906_ep2                         firefox 87 / 319 eps  angle err q50/q75 20°/30°  >60° 6%
expert    n=2194  stick-up share 42.5%  up-onset/3f 12.3%  P(B|up) 14.7%  P(B|not up) 2.6%
evt2ctx_ck8_off3_dur8e_on5u_dag7w_x1_e3 n=485  stick-up share 9.9%  up-onset/3f 2.1%  P(B|up) 8.3%  P(B|not up) 0.5%
evt2ctx_ck8_off3_dur8e_on5u_dag6w_x1_e3_s906 n=742  stick-up share 35.4%  up-onset/3f 8.0%  P(B|up) 11.4%  P(B|not up) 0.8%
fox_mamba_v3s906_ep1 n=701  stick-up share 24.0%  up-onset/3f 11.0%  P(B|up) 21.4%  P(B|not up) 3.2%
fox_mamba_v3s906_ep2 n=542  stick-up share 33.6%  up-onset/3f 15.8%  P(B|up) 25.3%  P(B|not up) 0.6%
fox_mamba_v3s906_ep3_null9o              (no recovery_means.json)
EPOCH 3 DONE (13:50)
curve rows (eval_runs/1011_full/curve.txt tail):
fox_mamba_v3s906_ep3_dag9o               3.186   263   193      0.689  31       0.742   10       0.327  0.0207 [532] ret✗ sideB✗ cdied✗  
== lr 0.0001 seed 906 mix 'data/silent_fall/sim_dagger_expert_r9o_2091_split_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2094_split_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2097_split_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2100_split_lean.frames,data/silent_fall/slp_dagg
fox_mamba_v3s906_ep3_dag9o_r             3.186   238   162      0.636  36       0.667   12       0.361  0.0199 [403] ret✗ sideB✗ cdied✗  
== re-readout of fox_mamba_v3s906_ep3_dag9o (same policy, second roll)
fox_mamba_v3s906_ep3_null9o              (no recovery_means.json)
== lr 0.0001 seed 906 mix 'data/silent_fall/sim_dagger_expert_r9o_2091_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2094_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2097_null_lean.frames,data/silent_fall/sim_dagger_expert_r9o_2100_null_lean.frames,data/silent_fall/slp_dagger_e

## Mini league (13:50)
RESULT pair 1/10 s906_ep2 vs s906_dag9o: 23-10-0 (1/33 finished)  201 s  570 game-frames/s
RESULT pair 2/10 s906_ep2 vs s906_null9o: 18-13-1 (0/32 finished)  208 s  553 game-frames/s
RESULT pair 3/10 s906_ep2 vs master_last: 14-15-3 (0/32 finished)  209 s  550 game-frames/s
RESULT pair 4/10 s906_ep2 vs V3.1_ep3: 23-8-1 (0/32 finished)  196 s  585 game-frames/s
RESULT pair 5/10 s906_dag9o vs s906_null9o: 11-21-0 (0/32 finished)  211 s  545 game-frames/s
RESULT pair 6/10 s906_dag9o vs master_last: 8-23-1 (0/32 finished)  213 s  541 game-frames/s
RESULT pair 7/10 s906_dag9o vs V3.1_ep3: 19-11-2 (0/32 finished)  207 s  554 game-frames/s
RESULT pair 8/10 s906_null9o vs master_last: 11-20-1 (0/32 finished)  212 s  542 game-frames/s
RESULT pair 9/10 s906_null9o vs V3.1_ep3: 15-16-1 (0/32 finished)  226 s  509 game-frames/s
RESULT pair 10/10 master_last vs V3.1_ep3: 16-15-1 (0/32 finished)  252 s  457 game-frames/s
RESULT standings (Elo, Bradley–Terry over 321 games):
RESULT league -> eval_runs/1024_overnight/league/league.json

ALL STAGES DONE (2026-10-25 14:26)

## Stage 3 READ (Claude, 14:45)
NULL ARM: val 3.0826 (parent ep2 3.0762; plain ep3 3.1671; dag9o 3.1860). The dose-matched null mix did NOT
regress the held-out loss while the expert-relabelled mix did — on the loss the harm was the LABELS, not the
mixing. Its closed-loop readout crashed (fused-handler crypto load, see coordination note); rerun pending.
MINI LEAGUE (321 games, 60 s, FD): s906_ep2 1571 (win 62 %) > master_ep6 1558 (60 %) > null9o 1489 (48 %) >
V3.1_ep3 1448 (41 %) > dag9o 1433 (38 %). Head to head: candidate vs master 48/52 — a coin flip; master beats
null 64/36 and dag9o 73/27; null beats dag9o 66/34. So in play: null arm > dag9o (the labels hurt there too),
and the master-only model is level with the candidate.
