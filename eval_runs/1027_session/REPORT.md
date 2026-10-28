
# 10-27 session — 2026-10-27 23:10 (Axon 0.9 / Polaris 0.2 build)
## 1. fused handler check
fused handlers still crash (see fusedcheck.log) — readouts run with EDIFICE_FUSED_CUSTOM_CALL=0
** (RuntimeError) Unsupported attribute type for attribute: operand_layouts
## 2. press-window arms

### arm pw_base —  (23:10)
val_loss=3.3378
RESULT pw_base: frames=30053 change_recall=0.307 hold_agreement=0.589 repeat_prev=0.818 (expert 0.818) neutral=0.371 (expert 0.336)
RESULT pw_base press edges/min model:expert  A 20.24:9.58  B 12.82:17.97  X 6.95:27.91  Y 36.66:18.69  Z 4.43:11.86  L 11.74:14.25  R 18.81:14.25
RESULT pw_base jab1 episodes=3 A-edges/episode model=0.0 expert=0.333
RESULT pw_base change events recall/precision  ±0: press 0.004/0.004 release 0.004/0.004 stick 0.016/0.019  |  ±2: press 0.038/0.039 release 0.038/0.04 stick 0.078/0.09  |  ±5: press 0.081/0.084 release 0.076/0.08 stick 0.163/0.193
RESULT pw_base vs idle: 16.0 env-min  SD/min 1.188  deaths/min 1.188  kills/min 0.0  dmg dealt/taken per min 2.5/0.063
RESULT pw_base deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 19 (1.188/min)
RESULT pw_base offstage eps/min 4.125  recovery rate 0.703  neutral 0.678  repeat_prev 0.893  max frozen run 1713 f  (16.5s wall)
RESULT pw_base: 36 cases x 8 trials  recovery 0.33  died 0.67  timeout 0.0  never 8/36  always 1/36
RESULT pw_base by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.159 (11 cases)  4+ routes: 0.391 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 8.7%  buttons 64.2%  families 21.7%  main dir 55.1% (bucket 43.9%, ±1 48.6%)  c dir 95.2%  trigger 92.6%  jump 85.5%  attack 96.3%  special 94.5%  surprise 4.183  change recall 35.9%  false change %
RESULT frames with a real press edge: n 481  joint 0.0%  buttons 7.5%  families 16.0%  main dir 86.1% (bucket 76.7%, ±1 83.6%)  c dir 100.0%  trigger 92.3%  jump 58.6%  attack 90.4%  special 86.1%  surprise 6.197  change recall 29.1%  false change %
RESULT press calibration jump: AUC 0.726  mean P on press frames 0.023 vs no-press 0.022  press frames above the no-press median 82.5%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.917  mean P on press frames 0.03 vs no-press 0.008  press frames above the no-press median 100.0%  (n 46 / 14647)
RESULT press calibration special: AUC 0.833  mean P on press frames 0.029 vs no-press 0.007  press frames above the no-press median 87.9%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.0% of 14428
RESULT STANDING change frames: n 60 change recall 36.7%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 73.5%
RESULT pw_base walk-off origins: 83 trips in 16.0 env-min (5.19/min)
RESULT pw_base   walk_off      n  11  0.69/min  died 1.0  stick toward edge (10 f before) 0.51  full-out 0.39  jumps left 1.0  last ground action [["DASHING", 8], ["RUNNING", 2], ["WALK_FAST", 1]]
RESULT pw_base   jump_off      n  47  2.94/min  died 0.38  stick toward edge (10 f before) 0.2  full-out 0.11  jumps left 0.74  last ground action [["KNEE_BEND", 26], ["DASHING", 1]]
RESULT pw_base   aerial_drift  n  18  1.13/min  died 0.0  stick toward edge (10 f before) 0.11  full-out 0.0  jumps left 0.33  last ground action []
RESULT pw_base   special_off   n   2  0.13/min  died 1.0  stick toward edge (10 f before) 1.0  full-out 0.95  jumps left 0.0  last ground action [["KNEE_BEND", 1]]

### arm pw_base906 —  (23:25)
val_loss=3.323
RESULT pw_base906: frames=30053 change_recall=0.28 hold_agreement=0.55 repeat_prev=0.771 (expert 0.818) neutral=0.251 (expert 0.336)
RESULT pw_base906 press edges/min model:expert  A 33.54:9.58  B 19.53:17.97  X 16.29:27.91  Y 56.42:18.69  Z 6.23:11.86  L 20.36:14.25  R 27.07:14.25
RESULT pw_base906 jab1 episodes=3 A-edges/episode model=0.333 expert=0.333
RESULT pw_base906 change events recall/precision  ±0: press 0.013/0.008 release 0.011/0.007 stick 0.017/0.02  |  ±2: press 0.065/0.043 release 0.041/0.027 stick 0.072/0.086  |  ±5: press 0.115/0.078 release 0.095/0.063 stick 0.135/0.167
RESULT pw_base906 vs idle: 16.0 env-min  SD/min 1.063  deaths/min 1.188  kills/min 0.438  dmg dealt/taken per min 12.048/0.125
RESULT pw_base906 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 17 (1.063/min)
RESULT pw_base906 offstage eps/min 6.938  recovery rate 0.826  neutral 0.292  repeat_prev 0.75  max frozen run 350 f  (15.6s wall)
RESULT pw_base906: 36 cases x 8 trials  recovery 0.326  died 0.674  timeout 0.0  never 8/36  always 1/36
RESULT pw_base906 by routes: 1 routes: 0.5 (1 cases)  2-3 routes: 0.125 (11 cases)  4+ routes: 0.411 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 8.5%  buttons 64.2%  families 21.8%  main dir 55.2% (bucket 44.0%, ±1 48.5%)  c dir 95.2%  trigger 92.6%  jump 85.5%  attack 96.3%  special 94.5%  surprise 4.099  change recall 34.6%  false change %
RESULT frames with a real press edge: n 481  joint 0.0%  buttons 7.5%  families 16.2%  main dir 86.5% (bucket 77.3%, ±1 83.8%)  c dir 99.8%  trigger 92.5%  jump 58.6%  attack 90.4%  special 86.1%  surprise 5.806  change recall 28.1%  false change %
RESULT press calibration jump: AUC 0.754  mean P on press frames 0.04 vs no-press 0.028  press frames above the no-press median 85.6%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.924  mean P on press frames 0.046 vs no-press 0.01  press frames above the no-press median 100.0%  (n 46 / 14647)
RESULT press calibration special: AUC 0.86  mean P on press frames 0.054 vs no-press 0.01  press frames above the no-press median 92.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.0% of 14428
RESULT STANDING change frames: n 60 change recall 36.7%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 73.5%
RESULT pw_base906 walk-off origins: 92 trips in 16.0 env-min (5.75/min)
RESULT pw_base906   walk_off      n   1  0.06/min  died 1.0  stick toward edge (10 f before) 1.0  full-out 0.0  jumps left 1.0  last ground action [["DASHING", 1]]
RESULT pw_base906   jump_off      n  55  3.44/min  died 0.13  stick toward edge (10 f before) 0.19  full-out 0.03  jumps left 0.6  last ground action [["KNEE_BEND", 31], ["DASHING", 1]]
RESULT pw_base906   aerial_drift  n  27  1.69/min  died 0.04  stick toward edge (10 f before) 0.2  full-out 0.03  jumps left 0.3  last ground action []
RESULT pw_base906   special_off   n   8  0.5/min  died 1.0  stick toward edge (10 f before) 0.55  full-out 0.0  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 3], ["KNEE_BEND", 2]]

### arm pw_pw2 — --press-window-weight 2 (23:40)
val_loss=3.35
RESULT pw_pw2: frames=30053 change_recall=0.312 hold_agreement=0.577 repeat_prev=0.818 (expert 0.818) neutral=0.283 (expert 0.336)
RESULT pw_pw2 press edges/min model:expert  A 19.05:9.58  B 34.98:17.97  X 10.3:27.91  Y 39.89:18.69  Z 2.52:11.86  L 11.26:14.25  R 15.69:14.25
RESULT pw_pw2 jab1 episodes=3 A-edges/episode model=0.0 expert=0.333
RESULT pw_pw2 change events recall/precision  ±0: press 0.004/0.004 release 0.003/0.003 stick 0.011/0.016  |  ±2: press 0.035/0.03 release 0.023/0.02 stick 0.062/0.09  |  ±5: press 0.071/0.064 release 0.06/0.052 stick 0.122/0.18
RESULT pw_pw2 vs idle: 16.0 env-min  SD/min 1.188  deaths/min 1.438  kills/min 0.438  dmg dealt/taken per min 16.275/0.313
RESULT pw_pw2 deaths by verdict at the decision frame: kill 4 (0.25/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 19 (1.188/min)
RESULT pw_pw2 offstage eps/min 6.313  recovery rate 0.765  neutral 0.423  repeat_prev 0.811  max frozen run 1504 f  (16.4s wall)
RESULT pw_pw2: 36 cases x 8 trials  recovery 0.385  died 0.615  timeout 0.0  never 7/36  always 2/36
RESULT pw_pw2 by routes: 1 routes: 0.5 (1 cases)  2-3 routes: 0.25 (11 cases)  4+ routes: 0.443 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 9.1%  buttons 64.2%  families 21.8%  main dir 55.4% (bucket 44.4%, ±1 49.1%)  c dir 95.2%  trigger 92.6%  jump 85.5%  attack 96.3%  special 94.5%  surprise 4.154  change recall 35.8%  false change %
RESULT frames with a real press edge: n 481  joint 0.0%  buttons 7.5%  families 15.6%  main dir 86.1% (bucket 77.3%, ±1 83.8%)  c dir 99.8%  trigger 91.9%  jump 58.6%  attack 90.4%  special 86.1%  surprise 6.101  change recall 27.0%  false change %
RESULT press calibration jump: AUC 0.727  mean P on press frames 0.027 vs no-press 0.023  press frames above the no-press median 80.9%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.922  mean P on press frames 0.044 vs no-press 0.007  press frames above the no-press median 97.8%  (n 46 / 14647)
RESULT press calibration special: AUC 0.83  mean P on press frames 0.134 vs no-press 0.015  press frames above the no-press median 86.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.0% of 14428
RESULT STANDING change frames: n 60 change recall 38.3%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 71.2%
RESULT pw_pw2 walk-off origins: 92 trips in 16.0 env-min (5.75/min)
RESULT pw_pw2   walk_off      n  10  0.63/min  died 0.8  stick toward edge (10 f before) 0.82  full-out 0.52  jumps left 1.0  last ground action [["RUNNING", 7], ["DASHING", 3]]
RESULT pw_pw2   jump_off      n  54  3.38/min  died 0.15  stick toward edge (10 f before) 0.31  full-out 0.06  jumps left 0.69  last ground action [["KNEE_BEND", 34], ["WALK_MIDDLE", 2], ["DASHING", 1]]
RESULT pw_pw2   aerial_drift  n  21  1.31/min  died 0.0  stick toward edge (10 f before) 0.06  full-out 0.0  jumps left 0.29  last ground action []
RESULT pw_pw2   special_off   n   6  0.38/min  died 0.83  stick toward edge (10 f before) 0.5  full-out 0.0  jumps left 0.33  last ground action [["KNEE_BEND", 2], ["NEUTRAL_B_FULL_CHARGE_AIR", 2]]

### arm pw_pw5 — --press-window-weight 5 (23:56)
val_loss=3.327
RESULT pw_pw5: frames=30053 change_recall=0.326 hold_agreement=0.626 repeat_prev=0.816 (expert 0.818) neutral=0.365 (expert 0.336)
RESULT pw_pw5 press edges/min model:expert  A 29.47:9.58  B 24.68:17.97  X 8.74:27.91  Y 23.72:18.69  Z 1.92:11.86  L 11.38:14.25  R 20.96:14.25
RESULT pw_pw5 jab1 episodes=3 A-edges/episode model=0.0 expert=0.333
RESULT pw_pw5 change events recall/precision  ±0: press 0.015/0.014 release 0.009/0.009 stick 0.011/0.013  |  ±2: press 0.038/0.036 release 0.033/0.033 stick 0.072/0.088  |  ±5: press 0.074/0.073 release 0.073/0.074 stick 0.151/0.187
RESULT pw_pw5 vs idle: 16.0 env-min  SD/min 1.438  deaths/min 1.563  kills/min 0.25  dmg dealt/taken per min 13.581/0.125
RESULT pw_pw5 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 23 (1.438/min)
RESULT pw_pw5 offstage eps/min 7.313  recovery rate 0.781  neutral 0.399  repeat_prev 0.758  max frozen run 983 f  (16.3s wall)
RESULT pw_pw5: 36 cases x 8 trials  recovery 0.285  died 0.715  timeout 0.0  never 14/36  always 1/36
RESULT pw_pw5 by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.102 (11 cases)  4+ routes: 0.344 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 8.3%  buttons 64.3%  families 21.4%  main dir 54.6% (bucket 43.4%, ±1 48.0%)  c dir 95.2%  trigger 92.6%  jump 85.5%  attack 96.3%  special 94.6%  surprise 4.162  change recall 35.9%  false change %
RESULT frames with a real press edge: n 481  joint 0.4%  buttons 7.7%  families 16.6%  main dir 85.9% (bucket 77.3%, ±1 83.2%)  c dir 99.8%  trigger 92.7%  jump 58.6%  attack 90.6%  special 86.1%  surprise 6.14  change recall 28.9%  false change %
RESULT press calibration jump: AUC 0.725  mean P on press frames 0.034 vs no-press 0.02  press frames above the no-press median 77.3%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.878  mean P on press frames 0.061 vs no-press 0.009  press frames above the no-press median 95.7%  (n 46 / 14647)
RESULT press calibration special: AUC 0.865  mean P on press frames 0.13 vs no-press 0.011  press frames above the no-press median 92.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.1% of 14428
RESULT STANDING change frames: n 60 change recall 35%  | press edges same-family: jump 0% attack 2.2% special 0% grab 0% trigger 75%
RESULT pw_pw5 walk-off origins: 100 trips in 16.0 env-min (6.25/min)
RESULT pw_pw5   walk_off      n   5  0.31/min  died 1.0  stick toward edge (10 f before) 0.64  full-out 0.24  jumps left 1.0  last ground action [["DASHING", 3], ["RUNNING", 1], ["WALK_MIDDLE", 1]]
RESULT pw_pw5   jump_off      n  61  3.81/min  died 0.13  stick toward edge (10 f before) 0.31  full-out 0.05  jumps left 0.7  last ground action [["KNEE_BEND", 44]]
RESULT pw_pw5   aerial_drift  n  26  1.63/min  died 0.0  stick toward edge (10 f before) 0.08  full-out 0.01  jumps left 0.42  last ground action []
RESULT pw_pw5   special_off   n   4  0.25/min  died 1.0  stick toward edge (10 f before) 0.9  full-out 0.1  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 2]]

### arm pw_pw10 — --press-window-weight 10 (00:11)
val_loss=3.3554
RESULT pw_pw10: frames=30053 change_recall=0.278 hold_agreement=0.522 repeat_prev=0.762 (expert 0.818) neutral=0.235 (expert 0.336)
RESULT pw_pw10 press edges/min model:expert  A 20.84:9.58  B 44.56:17.97  X 11.02:27.91  Y 44.56:18.69  Z 4.55:11.86  L 14.25:14.25  R 37.01:14.25
RESULT pw_pw10 jab1 episodes=3 A-edges/episode model=1.0 expert=0.333
RESULT pw_pw10 change events recall/precision  ±0: press 0.009/0.006 release 0.013/0.008 stick 0.016/0.015  |  ±2: press 0.052/0.034 release 0.047/0.031 stick 0.084/0.082  |  ±5: press 0.107/0.069 release 0.114/0.075 stick 0.173/0.174
RESULT pw_pw10 vs idle: 16.0 env-min  SD/min 1.125  deaths/min 1.125  kills/min 0.063  dmg dealt/taken per min 6.365/0.0
RESULT pw_pw10 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 18 (1.125/min)
RESULT pw_pw10 offstage eps/min 6.375  recovery rate 0.818  neutral 0.342  repeat_prev 0.766  max frozen run 887 f  (16.0s wall)
RESULT pw_pw10: 36 cases x 8 trials  recovery 0.316  died 0.684  timeout 0.0  never 6/36  always 1/36
RESULT pw_pw10 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.136 (11 cases)  4+ routes: 0.38 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 8.9%  buttons 64.0%  families 22.7%  main dir 55.7% (bucket 43.8%, ±1 48.5%)  c dir 95.2%  trigger 92.5%  jump 85.5%  attack 96.3%  special 94.6%  surprise 4.15  change recall 37.1%  false change %
RESULT frames with a real press edge: n 481  joint 1.9%  buttons 9.1%  families 18.5%  main dir 85.7% (bucket 76.3%, ±1 82.1%)  c dir 99.8%  trigger 93.1%  jump 58.6%  attack 90.6%  special 87.5%  surprise 6.112  change recall 32.4%  false change %
RESULT press calibration jump: AUC 0.707  mean P on press frames 0.048 vs no-press 0.027  press frames above the no-press median 79.4%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.882  mean P on press frames 0.054 vs no-press 0.008  press frames above the no-press median 95.7%  (n 46 / 14647)
RESULT press calibration special: AUC 0.835  mean P on press frames 0.189 vs no-press 0.016  press frames above the no-press median 86.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.4% of 14428
RESULT STANDING change frames: n 60 change recall 36.7%  | press edges same-family: jump 0% attack 2.2% special 10.6% grab 0% trigger 77.3%
RESULT pw_pw10 walk-off origins: 104 trips in 16.0 env-min (6.5/min)
RESULT pw_pw10   walk_off      n  10  0.63/min  died 0.7  stick toward edge (10 f before) 0.82  full-out 0.53  jumps left 1.0  last ground action [["DASHING", 6], ["RUNNING", 4]]
RESULT pw_pw10   jump_off      n  66  4.13/min  died 0.15  stick toward edge (10 f before) 0.2  full-out 0.03  jumps left 0.73  last ground action [["KNEE_BEND", 42], ["LANDING_SPECIAL", 1]]
RESULT pw_pw10   aerial_drift  n  23  1.44/min  died 0.0  stick toward edge (10 f before) 0.02  full-out 0.0  jumps left 0.09  last ground action []
RESULT pw_pw10   special_off   n   4  0.25/min  died 1.0  stick toward edge (10 f before) 0.47  full-out 0.33  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 3]]
## 3. master_ep6 / null9o readouts (00:27)
### fox_mamba_master_ep6
RESULT fox_mamba_master_ep6 vs idle: 16.0 env-min  SD/min 0.688  deaths/min 0.75  kills/min 2.25  dmg dealt/taken per min 89.305/0.125
RESULT fox_mamba_master_ep6 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 11 (0.688/min)
RESULT fox_mamba_master_ep6 offstage eps/min 5.563  recovery rate 0.862  neutral 0.371  repeat_prev 0.745  max frozen run 558 f  (18.6s wall)
RESULT fox_mamba_master_ep6: 36 cases x 8 trials  recovery 0.813  died 0.188  timeout 0.0  never 0/36  always 7/36
RESULT fox_mamba_master_ep6 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.784 (11 cases)  4+ routes: 0.818 (24 cases)
RESULT fox_mamba_master_ep6 recovery means: 280 offstage episodes  mismatch 0.061 (expert split-half 0.084)  means_js 0.199 (floor 0.174)  return 0.721 (expert 0.911)
RESULT fox_mamba_master_ep6 named defects model|expert: side_b_low 0.143|0.107  airdodge_with_jump 0.018|0.006  nothing_died 0.051|0.175
RESULT fox_mamba_master_ep6 timing model|expert: side_b fired from low/deep 0.067|0.182  first-means latency (median f) 10|11  side_b latency 0|13
RESULT fox_mamba_master_ep6 edge SDs model|expert: carried-off share 0.104|0.055  carried-off died 0.655|0.163  decided trips return 0.765 (n=251)|0.915  by move: attack 13(4†) side_b 8(8†) airdodge 5(5†) up_b 3(2†) | attack 78(5†) up_b 11(4†) side_b 5(3†) airdodge 4(4†)
RESULT fox_mamba_master_ep6 by height n/return/first-means model | expert: high: 215/0.735/jump 0.46,attack 0.31 | 1218/0.952/jump 0.38,attack 0.34  ledge: 23/0.696/jump 0.35,drift 0.22 | 225/0.902/jump 0.42,drift 0.19  low: 34/0.735/jump 0.35,up_b 0.35 | 298/0.836/jump 0.37,up_b 0.34  deep: 8/0.375
RESULT fox_mamba_master_ep6 first means: jump 118  attack 74  drift 33  up_b 29  side_b 15  airdodge 7  none 4
fox_mamba_master_ep6                     2.977   280   214      0.743  29       0.655   8        0.279  -            ret✗ sideB✗ cdied✗  
### fox_mamba_v3s906_ep3_null9o
RESULT fox_mamba_v3s906_ep3_null9o vs idle: 16.0 env-min  SD/min 1.0  deaths/min 1.063  kills/min 1.125  dmg dealt/taken per min 40.859/0.063
RESULT fox_mamba_v3s906_ep3_null9o deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 16 (1.0/min)
RESULT fox_mamba_v3s906_ep3_null9o offstage eps/min 4.625  recovery rate 0.764  neutral 0.4  repeat_prev 0.746  max frozen run 745 f  (19.5s wall)
RESULT fox_mamba_v3s906_ep3_null9o: 36 cases x 8 trials  recovery 0.698  died 0.302  timeout 0.0  never 0/36  always 2/36
RESULT fox_mamba_v3s906_ep3_null9o by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.693 (11 cases)  4+ routes: 0.693 (24 cases)
RESULT fox_mamba_v3s906_ep3_null9o recovery means: 318 offstage episodes  mismatch 0.047 (expert split-half 0.084)  means_js 0.2 (floor 0.174)  return 0.739 (expert 0.911)
RESULT fox_mamba_v3s906_ep3_null9o named defects model|expert: side_b_low 0.026|0.107  airdodge_with_jump 0.015|0.006  nothing_died 0.012|0.175
RESULT fox_mamba_v3s906_ep3_null9o timing model|expert: side_b fired from low/deep 0.0|0.182  first-means latency (median f) 12|11  side_b latency 0|13
RESULT fox_mamba_v3s906_ep3_null9o edge SDs model|expert: carried-off share 0.119|0.055  carried-off died 0.474|0.163  decided trips return 0.768 (n=280)|0.915  by move: attack 25(6†) side_b 7(7†) up_b 4(3†) airdodge 2(2†) | attack 78(5†) up_b 11(4†) side_b 5(3†) airdodge 4(4†)
RESULT fox_mamba_v3s906_ep3_null9o by height n/return/first-means model | expert: high: 257/0.751/jump 0.48,attack 0.26 | 1218/0.952/jump 0.38,attack 0.34  ledge: 23/0.739/attack 0.3,jump 0.3 | 225/0.902/jump 0.42,drift 0.19  low: 24/0.75/up_b 0.5,jump 0.38 | 298/0.836/jump 0.37,up_b 0.34  deep: 14/
RESULT fox_mamba_v3s906_ep3_null9o first means: jump 141  attack 77  drift 52  up_b 32  side_b 11  airdodge 4  none 1
fox_mamba_v3s906_ep3_null9o              3.0826  318   227      0.718  38       0.474   7        0.261  -            ret✗ sideB✗ cdied✓  

SESSION DONE (00:35) — table: node scripts/pq_digest.js pw_base pw_base906 pw_pw2 pw_pw5 pw_pw10

## READ (Claude, 00:40)
Press-window (MAX_FILES 1500, seed floor base/base906: fam ±0.1, joint ±0.2, AUC ±0.03, SD/min ±0.13):
  dose 2 / 5: nothing moves. dose 10: the first commitment signal — exact-frame same-family press agreement
  special 10.6 % (base 0), attack 2.2 % (0), press-edge change recall 32 vs 28, families 22.7 vs 21.7
  (+1.0, beyond the pair spread), false press 0.4 % (bar <= 1), SD/min flat (1.13 vs 1.19). Jump still 0 %
  and AUC jump 0.71 vs 0.73/0.75 (inside the floor). Read: the set objective does what the weights could not,
  but at dose 10 on a weak 1500-file model the effect is small; the dose and the slice go up next.
Master arm closed loop (fused off): SD/min 0.688 = candidate's 0.688, deaths 0.75 = 0.75, kills 2.25 vs 1.5,
  damage dealt 89/min vs 65. Less passive, same self-destructs, more offense. Null arm: SD/min 1.0 (worse than
  its parent), kills 1.1 — the null mix cost a little in play even though its loss did not move.
Build note: everything in this session is on Axon 0.9 / Polaris 0.2 with the fused custom call OFF.

# 10-28 follow-up: press-window dose + slice (2026-10-28 00:36)

### arm pw_pw20 (files 1500) — --press-window-weight 20 (00:36)
val_loss=3.3012
RESULT pw_pw20: frames=30053 change_recall=0.306 hold_agreement=0.604 repeat_prev=0.785 (expert 0.818) neutral=0.263 (expert 0.336)
RESULT pw_pw20 press edges/min model:expert  A 24.68:9.58  B 16.05:17.97  X 9.22:27.91  Y 33.78:18.69  Z 2.99:11.86  L 13.18:14.25  R 32.58:14.25
RESULT pw_pw20 jab1 episodes=3 A-edges/episode model=1.0 expert=0.333
RESULT pw_pw20 change events recall/precision  ±0: press 0.013/0.011 release 0.015/0.013 stick 0.017/0.017  |  ±2: press 0.046/0.041 release 0.05/0.043 stick 0.079/0.081  |  ±5: press 0.083/0.08 release 0.097/0.087 stick 0.164/0.17
RESULT pw_pw20 vs idle: 16.0 env-min  SD/min 2.188  deaths/min 2.313  kills/min 0.375  dmg dealt/taken per min 40.467/0.125
RESULT pw_pw20 deaths by verdict at the decision frame: kill 2 (0.125/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 35 (2.188/min)
RESULT pw_pw20 offstage eps/min 8.125  recovery rate 0.711  neutral 0.222  repeat_prev 0.694  max frozen run 127 f  (16.4s wall)
RESULT pw_pw20: 36 cases x 8 trials  recovery 0.358  died 0.642  timeout 0.0  never 5/36  always 1/36
RESULT pw_pw20 by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.159 (11 cases)  4+ routes: 0.427 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 9.4%  buttons 64.2%  families 22.6%  main dir 55.4% (bucket 44.2%, ±1 49.0%)  c dir 95.2%  trigger 92.6%  jump 85.3%  attack 96.3%  special 94.6%  surprise 4.178  change recall 37.9%  false change %
RESULT frames with a real press edge: n 481  joint 0.8%  buttons 8.9%  families 18.9%  main dir 86.1% (bucket 76.1%, ±1 83.6%)  c dir 99.8%  trigger 93.8%  jump 58.6%  attack 90.6%  special 87.1%  surprise 6.31  change recall 31.2%  false change %
RESULT press calibration jump: AUC 0.742  mean P on press frames 0.06 vs no-press 0.027  press frames above the no-press median 79.9%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.875  mean P on press frames 0.062 vs no-press 0.009  press frames above the no-press median 97.8%  (n 46 / 14647)
RESULT press calibration special: AUC 0.831  mean P on press frames 0.128 vs no-press 0.011  press frames above the no-press median 89.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 0.3% of 14428
RESULT STANDING change frames: n 60 change recall 36.7%  | press edges same-family: jump 0% attack 2.2% special 7.6% grab 0% trigger 79.5%
RESULT pw_pw20 walk-off origins: 117 trips in 16.0 env-min (7.31/min)
RESULT pw_pw20   jump_off      n  57  3.56/min  died 0.28  stick toward edge (10 f before) 0.41  full-out 0.0  jumps left 0.53  last ground action [["KNEE_BEND", 36], ["LANDING_SPECIAL", 2], ["SHIELD_START", 1]]
RESULT pw_pw20   special_off   n  10  0.63/min  died 0.9  stick toward edge (10 f before) 0.86  full-out 0.0  jumps left 0.1  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 4], ["FIREFOX_GROUND", 3], ["KNEE_BEND", 1]]

### arm pw_pw40 (files 1500) — --press-window-weight 40 (00:51)
val_loss=3.3901
RESULT pw_pw40: frames=30053 change_recall=0.263 hold_agreement=0.512 repeat_prev=0.789 (expert 0.818) neutral=0.278 (expert 0.336)
RESULT pw_pw40 press edges/min model:expert  A 32.22:9.58  B 42.05:17.97  X 10.42:27.91  Y 43.72:18.69  Z 11.38:11.86  L 12.58:14.25  R 35.7:14.25
RESULT pw_pw40 jab1 episodes=3 A-edges/episode model=0.667 expert=0.333
RESULT pw_pw40 change events recall/precision  ±0: press 0.016/0.01 release 0.009/0.006 stick 0.019/0.022  |  ±2: press 0.051/0.032 release 0.074/0.046 stick 0.09/0.107  |  ±5: press 0.107/0.073 release 0.126/0.082 stick 0.174/0.215
RESULT pw_pw40 vs idle: 16.0 env-min  SD/min 1.375  deaths/min 1.375  kills/min 0.25  dmg dealt/taken per min 6.091/0.0
RESULT pw_pw40 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 22 (1.375/min)
RESULT pw_pw40 offstage eps/min 9.688  recovery rate 0.855  neutral 0.257  repeat_prev 0.734  max frozen run 205 f  (15.9s wall)
RESULT pw_pw40: 36 cases x 8 trials  recovery 0.344  died 0.656  timeout 0.0  never 9/36  always 3/36
RESULT pw_pw40 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.227 (11 cases)  4+ routes: 0.37 (24 cases)
RESULT CHANGE frames (real input changed): n 2544  joint 9.4%  buttons 64.0%  families 22.6%  main dir 55.7% (bucket 44.2%, ±1 48.8%)  c dir 95.2%  trigger 92.3%  jump 85.3%  attack 96.3%  special 94.4%  surprise 4.258  change recall 37.0%  false change %
RESULT frames with a real press edge: n 481  joint 3.5%  buttons 12.3%  families 21.2%  main dir 86.1% (bucket 76.5%, ±1 83.2%)  c dir 100.0%  trigger 93.6%  jump 58.6%  attack 91.5%  special 88.1%  surprise 6.179  change recall 34.7%  false change %
RESULT press calibration jump: AUC 0.725  mean P on press frames 0.071 vs no-press 0.031  press frames above the no-press median 79.4%  (n 194 / 13868)
RESULT press calibration attack: AUC 0.874  mean P on press frames 0.095 vs no-press 0.012  press frames above the no-press median 95.7%  (n 46 / 14647)
RESULT press calibration special: AUC 0.836  mean P on press frames 0.237 vs no-press 0.019  press frames above the no-press median 89.4%  (n 66 / 14249)
RESULT false press (model press edge on a frame with no real press): 1.1% of 14428
RESULT STANDING change frames: n 60 change recall 50%  | press edges same-family: jump 0% attack 10.9% special 19.7% grab 0% trigger 78.8%
RESULT pw_pw40 walk-off origins: 153 trips in 16.0 env-min (9.56/min)
RESULT pw_pw40   jump_off      n 101  6.31/min  died 0.13  stick toward edge (10 f before) 0.25  full-out 0.12  jumps left 0.62  last ground action [["KNEE_BEND", 65], ["DASHING", 2], ["LANDING_SPECIAL", 2]]
RESULT pw_pw40   special_off   n   5  0.31/min  died 0.8  stick toward edge (10 f before) 0.6  full-out 0.2  jumps left 0.2  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 3], ["KNEE_BEND", 2]]

### arm pw_base3k (files 3000) —  (01:07)
val_loss=3.2589
RESULT pw_base3k: frames=32394 change_recall=0.282 hold_agreement=0.572 repeat_prev=0.788 (expert 0.764) neutral=0.262 (expert 0.303)
RESULT pw_base3k press edges/min model:expert  A 19.34:13.22  B 27.56:11.0  X 24.89:30.12  Y 28.56:19.89  Z 4.67:10.0  L 15.67:26.34  R 19.45:8.33
RESULT pw_base3k jab1 episodes=4 A-edges/episode model=0.25 expert=0.25
RESULT pw_base3k change events recall/precision  ±0: press 0.011/0.01 release 0.007/0.006 stick 0.015/0.017  |  ±2: press 0.048/0.04 release 0.038/0.033 stick 0.08/0.092  |  ±5: press 0.121/0.102 release 0.096/0.082 stick 0.143/0.172
RESULT pw_base3k vs idle: 16.0 env-min  SD/min 2.75  deaths/min 3.063  kills/min 0.75  dmg dealt/taken per min 36.179/0.438
RESULT pw_base3k deaths by verdict at the decision frame: kill 6 (0.375/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 43 (2.688/min)
RESULT pw_base3k offstage eps/min 9.063  recovery rate 0.64  neutral 0.373  repeat_prev 0.81  max frozen run 616 f  (18.2s wall)
RESULT pw_base3k: 36 cases x 8 trials  recovery 0.448  died 0.552  timeout 0.0  never 4/36  always 1/36
RESULT pw_base3k by routes: 1 routes: 0.5 (1 cases)  2-3 routes: 0.17 (11 cases)  4+ routes: 0.573 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 15.0%  buttons 68.9%  families 35.1%  main dir 63.7% (bucket 46.6%, ±1 54.6%)  c dir 96.2%  trigger 93.2%  jump 86.8%  attack 96.1%  special 97.0%  surprise 3.724  change recall 50.7%  false change %
RESULT frames with a real press edge: n 533  joint 0.8%  buttons 14.8%  families 12.9%  main dir 86.7% (bucket 79.4%, ±1 85.6%)  c dir 100.0%  trigger 85.6%  jump 56.8%  attack 88.6%  special 90.2%  surprise 5.485  change recall 28.1%  false change %
RESULT press calibration jump: AUC 0.767  mean P on press frames 0.024 vs no-press 0.022  press frames above the no-press median 86.9%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.869  mean P on press frames 0.046 vs no-press 0.008  press frames above the no-press median 90.2%  (n 61 / 15748)
RESULT press calibration special: AUC 0.864  mean P on press frames 0.084 vs no-press 0.009  press frames above the no-press median 88.5%  (n 52 / 15723)
RESULT false press (model press edge on a frame with no real press): 0.0% of 15547
RESULT STANDING change frames: n 93 change recall 29%  | press edges same-family: jump 0% attack 0% special 1.9% grab 0% trigger 52.5%
RESULT pw_base3k walk-off origins: 154 trips in 16.0 env-min (9.63/min)
RESULT pw_base3k   jump_off      n  56  3.5/min  died 0.45  stick toward edge (10 f before) 0.45  full-out 0.01  jumps left 0.7  last ground action [["KNEE_BEND", 32]]
RESULT pw_base3k   special_off   n  23  1.44/min  died 0.83  stick toward edge (10 f before) 0.58  full-out 0.04  jumps left 0.04  last ground action [["KNEE_BEND", 5], ["FIREFOX_GROUND", 1], ["NEUTRAL_B_FULL_CHARGE_AIR", 1]]

### arm pw_pw10_3k (files 3000) — --press-window-weight 10 (01:30)
val_loss=3.2326
RESULT pw_pw10_3k: frames=32394 change_recall=0.275 hold_agreement=0.535 repeat_prev=0.795 (expert 0.764) neutral=0.221 (expert 0.303)
RESULT pw_pw10_3k press edges/min model:expert  A 34.67:13.22  B 28.89:11.0  X 18.23:30.12  Y 13.89:19.89  Z 3.33:10.0  L 14.56:26.34  R 23.56:8.33
RESULT pw_pw10_3k jab1 episodes=4 A-edges/episode model=0.5 expert=0.25
RESULT pw_pw10_3k change events recall/precision  ±0: press 0.019/0.016 release 0.011/0.01 stick 0.016/0.019  |  ±2: press 0.059/0.051 release 0.047/0.041 stick 0.067/0.085  |  ±5: press 0.105/0.092 release 0.086/0.075 stick 0.138/0.178
RESULT pw_pw10_3k vs idle: 16.0 env-min  SD/min 1.75  deaths/min 1.813  kills/min 0.563  dmg dealt/taken per min 31.805/0.125
RESULT pw_pw10_3k deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 28 (1.75/min)
RESULT pw_pw10_3k offstage eps/min 6.438  recovery rate 0.713  neutral 0.35  repeat_prev 0.79  max frozen run 577 f  (17.7s wall)
RESULT pw_pw10_3k: 36 cases x 8 trials  recovery 0.472  died 0.528  timeout 0.0  never 3/36  always 3/36
RESULT pw_pw10_3k by routes: 1 routes: 0.625 (1 cases)  2-3 routes: 0.307 (11 cases)  4+ routes: 0.542 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 16.5%  buttons 68.4%  families 37.1%  main dir 65.4% (bucket 48.1%, ±1 56.1%)  c dir 96.2%  trigger 93.7%  jump 86.8%  attack 96.1%  special 97.0%  surprise 3.745  change recall 53.2%  false change %
RESULT frames with a real press edge: n 533  joint 1.9%  buttons 12.9%  families 18.8%  main dir 88.9% (bucket 79.5%, ±1 87.1%)  c dir 100.0%  trigger 89.9%  jump 56.8%  attack 89.3%  special 91.0%  surprise 5.524  change recall 34.5%  false change %
RESULT press calibration jump: AUC 0.767  mean P on press frames 0.037 vs no-press 0.021  press frames above the no-press median 86.5%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.865  mean P on press frames 0.159 vs no-press 0.011  press frames above the no-press median 93.4%  (n 61 / 15748)
RESULT press calibration special: AUC 0.867  mean P on press frames 0.169 vs no-press 0.011  press frames above the no-press median 90.4%  (n 52 / 15723)
RESULT false press (model press edge on a frame with no real press): 0.5% of 15547
RESULT STANDING change frames: n 93 change recall 32.3%  | press edges same-family: jump 0% attack 6.6% special 9.6% grab 0% trigger 68.8%
RESULT pw_pw10_3k walk-off origins: 95 trips in 16.0 env-min (5.94/min)
RESULT pw_pw10_3k   jump_off      n  55  3.44/min  died 0.2  stick toward edge (10 f before) 0.2  full-out 0.04  jumps left 0.82  last ground action [["KNEE_BEND", 33]]
RESULT pw_pw10_3k   special_off   n   5  0.31/min  died 1.0  stick toward edge (10 f before) 0.74  full-out 0.0  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 5]]

### arm pw_pw20_3k (files 3000) — --press-window-weight 20 (01:54)
val_loss=3.2773
RESULT pw_pw20_3k: frames=32394 change_recall=0.298 hold_agreement=0.588 repeat_prev=0.814 (expert 0.764) neutral=0.349 (expert 0.303)
RESULT pw_pw20_3k press edges/min model:expert  A 25.0:13.22  B 15.11:11.0  X 21.78:30.12  Y 22.23:19.89  Z 6.33:10.0  L 14.22:26.34  R 20.89:8.33
RESULT pw_pw20_3k jab1 episodes=4 A-edges/episode model=0.5 expert=0.25
RESULT pw_pw20_3k change events recall/precision  ±0: press 0.014/0.013 release 0.01/0.01 stick 0.016/0.019  |  ±2: press 0.05/0.049 release 0.048/0.045 stick 0.077/0.092  |  ±5: press 0.103/0.099 release 0.089/0.086 stick 0.154/0.189
RESULT pw_pw20_3k vs idle: 16.0 env-min  SD/min 1.938  deaths/min 1.938  kills/min 0.688  dmg dealt/taken per min 42.224/0.063
RESULT pw_pw20_3k deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 31 (1.938/min)
RESULT pw_pw20_3k offstage eps/min 6.938  recovery rate 0.713  neutral 0.41  repeat_prev 0.809  max frozen run 609 f  (18.0s wall)
RESULT pw_pw20_3k: 36 cases x 8 trials  recovery 0.417  died 0.583  timeout 0.0  never 6/36  always 1/36
RESULT pw_pw20_3k by routes: 1 routes: 0.875 (1 cases)  2-3 routes: 0.17 (11 cases)  4+ routes: 0.51 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 15.6%  buttons 68.4%  families 35.5%  main dir 63.5% (bucket 46.5%, ±1 54.4%)  c dir 96.1%  trigger 93.7%  jump 86.8%  attack 96.1%  special 97.0%  surprise 3.81  change recall 52.1%  false change %
RESULT frames with a real press edge: n 533  joint 2.1%  buttons 13.1%  families 18.6%  main dir 87.6% (bucket 78.6%, ±1 85.7%)  c dir 100.0%  trigger 89.9%  jump 56.8%  attack 89.5%  special 90.8%  surprise 5.793  change recall 34.0%  false change %
RESULT press calibration jump: AUC 0.707  mean P on press frames 0.044 vs no-press 0.022  press frames above the no-press median 77.0%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.874  mean P on press frames 0.165 vs no-press 0.012  press frames above the no-press median 91.8%  (n 61 / 15748)
RESULT press calibration special: AUC 0.852  mean P on press frames 0.153 vs no-press 0.011  press frames above the no-press median 86.5%  (n 52 / 15723)
RESULT false press (model press edge on a frame with no real press): 0.7% of 15547
RESULT STANDING change frames: n 93 change recall 34.4%  | press edges same-family: jump 0% attack 8.2% special 7.7% grab 0% trigger 68.1%
RESULT pw_pw20_3k walk-off origins: 114 trips in 16.0 env-min (7.13/min)
RESULT pw_pw20_3k   jump_off      n  61  3.81/min  died 0.31  stick toward edge (10 f before) 0.39  full-out 0.0  jumps left 0.74  last ground action [["KNEE_BEND", 33], ["LANDING", 1], ["LANDING_SPECIAL", 1]]
RESULT pw_pw20_3k   special_off   n   6  0.38/min  died 1.0  stick toward edge (10 f before) 0.47  full-out 0.0  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 6]]

SESSION 2 DONE (02:17) — table: node scripts/pq_digest.js pw_base pw_base906 pw_pw2 pw_pw5 pw_pw10 pw_pw20 pw_pw40 pw_base3k pw_pw10_3k pw_pw20_3k

## READ 2 (Claude, 02:25) — press-window dose + slice
3,000-game slice (seed floor there ±0.4 fam / ±0.6 joint): base3k fam 35.1 joint 15.0 recall 50.7, press attack 0 / special 1.9,
SD/min 2.75 | pw10_3k fam 37.1 (+2.0) joint 16.5 (+1.5) recall 53.2, press attack 6.6 / special 9.6, false press 0.5 %, AUC jump
0.767 = base, SD/min 1.75, special-off trips 1.44 -> 0.31/min, STANDING 32 vs 29 | pw20_3k smaller gains and AUC jump 0.707 (down).
1,500-game slice: dose 20 ~ dose 10; dose 40 commits hardest (attack 10.9, special 19.7, STANDING 50) but false press 1.1 % (over
the bar) and val up. => DOSE 10 PASSES every bar on the standard slice: commitment up, families up beyond noise, false press <= 1,
AUC flat, SDs not up. Jump presses still 0 % at every dose (jump onsets need something else — the double-jump timing is the one
family the window does not fix). PROMOTE: full-scale A/B from master_ep6 (the better data): ep7 with --press-window-weight 10 vs
plain ep7, same resume / lr / seed, fused custom call OFF (EXLA operand_layouts), light readout + agreement + walk-off each.
