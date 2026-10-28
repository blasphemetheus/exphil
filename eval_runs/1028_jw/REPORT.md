
# jump-window testbed queue (2026-10-28 11:33)

### arm jw_base3k —  (11:33)

# jump-window queue via pq_multi (2026-10-28 11:49)
== multi-arm training: 5 arms (11:49)
scripts/pq_multi.sh: line 30: 2589974 Killed                     $run scripts/train_fox_mamba.exs "${shared[@]}" --arms "$spec_p" > "eval_runs/1001_queue/${prefix}_multi/train.log" 2>&1
TRAIN_FAILED jw (see eval_runs/1001_queue/jw_multi/train.log)
    (elixir 1.18.4) lib/enum.ex:2546: Enum."-reduce/3-lists^foldl/2-0-"/3
    (exphil 0.1.0) lib/exphil/training/callback.ex:138: ExPhil.Training.Callback.run/3
    (exphil 0.1.0) lib/exphil/training/trainer.ex:445: anonymous fn/2 in ExPhil.Training.Trainer.train_epoch_standard/3
    (elixir 1.18.4) lib/stream.ex:1093: Stream.do_transform_each/3
    (elixir 1.18.4) lib/stream.ex:1761: anonymous fn/3 in Enumerable.Stream.reduce/3
JW QUEUE DONE (12:11)

# jump-window queue via pq_multi (2026-10-28 12:12)
== multi-arm training: 5 arms (12:12)
Chunks ready: 18 in 930 s (10374753 frames)
[32m✓ ARM jw_base3k done in 528 s: val_loss=3.221582539876302[0m
[32m✓ ARM jw_jw5 done in 501 s: val_loss=3.2745867411295575[0m
[32m✓ ARM jw_jw10 done in 507 s: val_loss=3.3041966756184897[0m
[32m✓ ARM jw_jw20 done in 511 s: val_loss=3.344096374511719[0m
[32m✓ ARM jw_pw10_jw10 done in 503 s: val_loss=3.318372599283854[0m
[32m✓ Multi-arm run complete[0m
jw_base3k val_loss=3.221582539876302
RESULT jw_base3k: frames=32394 change_recall=0.283 hold_agreement=0.544 repeat_prev=0.782 (expert 0.764) neutral=0.229 (expert 0.303)
RESULT jw_base3k press edges/min model:expert  A 20.0:13.22  B 21.0:11.0  X 26.34:30.12  Y 23.78:19.89  Z 4.33:10.0  L 15.0:26.34  R 21.78:8.33
RESULT jw_base3k jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT jw_base3k change events recall/precision  ±0: press 0.021/0.018 release 0.013/0.012 stick 0.015/0.015  |  ±2: press 0.056/0.05 release 0.046/0.041 stick 0.073/0.073  |  ±5: press 0.112/0.103 release 0.1/0.092 stick 0.157/0.163
RESULT jw_base3k vs idle: 16.0 env-min  SD/min 1.563  deaths/min 1.625  kills/min 0.313  dmg dealt/taken per min 21.616/0.063
RESULT jw_base3k deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 25 (1.563/min)
RESULT jw_base3k offstage eps/min 6.125  recovery rate 0.729  neutral 0.396  repeat_prev 0.792  max frozen run 1533 f  (21.3s wall)
RESULT jw_base3k: 36 cases x 8 trials  recovery 0.378  died 0.622  timeout 0.0  never 4/36  always 1/36
RESULT jw_base3k by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.148 (11 cases)  4+ routes: 0.469 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 17.0%  buttons 68.9%  families 37.3%  main dir 65.7% (bucket 48.6%, ±1 56.5%)  c dir 96.2%  trigger 93.3%  jump 86.8%  attack 96.1%  special 97.0%  surprise 3.728  change recall 54.1%  false change %
RESULT frames with a real press edge: n 533  joint 0.8%  buttons 14.6%  families 13.1%  main dir 87.1% (bucket 78.8%, ±1 85.9%)  c dir 100.0%  trigger 86.1%  jump 56.8%  attack 88.6%  special 90.1%  surprise 5.531  change recall 28.9%  false change %
RESULT press calibration jump: AUC 0.762  mean P on press frames 0.027 vs no-press 0.023  press frames above the no-press median 84.7%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.878  mean P on press frames 0.056 vs no-press 0.009  press frames above the no-press median 93.4%  (n 61 / 15748)
RESULT false press (model press edge on a frame with no real press): 0.1% of 15547
RESULT STANDING change frames: n 93 change recall 29%  | press edges same-family: jump 0% attack 0% special 0% grab 0% trigger 54.4%
RESULT jw_base3k walk-off origins: 86 trips in 16.0 env-min (5.38/min)
RESULT jw_base3k   jump_off      n  54  3.38/min  died 0.22  stick toward edge (10 f before) 0.2  full-out 0.09  jumps left 0.61  last ground action [["KNEE_BEND", 23], ["DASHING", 4]]
RESULT jw_base3k   special_off   n   6  0.38/min  died 0.83  stick toward edge (10 f before) 0.85  full-out 0.15  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 1]]
jw_jw5 val_loss=3.2745867411295575
RESULT jw_jw5: frames=32394 change_recall=0.284 hold_agreement=0.552 repeat_prev=0.789 (expert 0.764) neutral=0.271 (expert 0.303)
RESULT jw_jw5 press edges/min model:expert  A 19.11:13.22  B 12.78:11.0  X 28.89:30.12  Y 37.23:19.89  Z 4.67:10.0  L 13.45:26.34  R 20.78:8.33
RESULT jw_jw5 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT jw_jw5 change events recall/precision  ±0: press 0.008/0.007 release 0.01/0.009 stick 0.017/0.018  |  ±2: press 0.047/0.04 release 0.048/0.041 stick 0.085/0.094  |  ±5: press 0.097/0.088 release 0.085/0.076 stick 0.155/0.182
RESULT jw_jw5 vs idle: 16.0 env-min  SD/min 1.063  deaths/min 1.063  kills/min 0.188  dmg dealt/taken per min 16.559/0.0
RESULT jw_jw5 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 17 (1.063/min)
RESULT jw_jw5 offstage eps/min 4.375  recovery rate 0.754  neutral 0.561  repeat_prev 0.838  max frozen run 1693 f  (20.9s wall)
RESULT jw_jw5: 36 cases x 8 trials  recovery 0.469  died 0.531  timeout 0.0  never 2/36  always 2/36
RESULT jw_jw5 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.182 (11 cases)  4+ routes: 0.578 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 15.4%  buttons 68.9%  families 35.5%  main dir 63.9% (bucket 47.0%, ±1 55.1%)  c dir 96.1%  trigger 93.1%  jump 87.0%  attack 96.1%  special 96.9%  surprise 3.805  change recall 52.1%  false change %
RESULT frames with a real press edge: n 533  joint 0.6%  buttons 15.4%  families 13.7%  main dir 87.2% (bucket 79.2%, ±1 85.7%)  c dir 100.0%  trigger 84.8%  jump 58.5%  attack 88.6%  special 90.1%  surprise 5.732  change recall 28.7%  false change %
RESULT press calibration jump: AUC 0.707  mean P on press frames 0.086 vs no-press 0.031  press frames above the no-press median 74.8%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.895  mean P on press frames 0.035 vs no-press 0.008  press frames above the no-press median 95.1%  (n 61 / 15748)
RESULT false press (model press edge on a frame with no real press): 0.3% of 15547
RESULT STANDING change frames: n 93 change recall 28%  | press edges same-family: jump 4.5% attack 0% special 0% grab 0% trigger 50.6%
RESULT jw_jw5 walk-off origins: 92 trips in 16.0 env-min (5.75/min)
RESULT jw_jw5   jump_off      n  44  2.75/min  died 0.16  stick toward edge (10 f before) 0.33  full-out 0.02  jumps left 0.57  last ground action [["KNEE_BEND", 24], ["LANDING", 1]]
RESULT jw_jw5   special_off   n  13  0.81/min  died 0.92  stick toward edge (10 f before) 0.64  full-out 0.08  jumps left 0.0  last ground action [["KNEE_BEND", 2], ["NEUTRAL_B_FULL_CHARGE_AIR", 2]]
jw_jw10 val_loss=3.3041966756184897
RESULT jw_jw10: frames=32394 change_recall=0.258 hold_agreement=0.478 repeat_prev=0.744 (expert 0.764) neutral=0.215 (expert 0.303)
RESULT jw_jw10 press edges/min model:expert  A 18.89:13.22  B 15.0:11.0  X 13.45:30.12  Y 59.34:19.89  Z 4.89:10.0  L 17.11:26.34  R 29.67:8.33
RESULT jw_jw10 jab1 episodes=4 A-edges/episode model=0.0 expert=0.25
RESULT jw_jw10 change events recall/precision  ±0: press 0.014/0.011 release 0.009/0.007 stick 0.022/0.02  |  ±2: press 0.061/0.046 release 0.049/0.037 stick 0.093/0.089  |  ±5: press 0.103/0.081 release 0.095/0.072 stick 0.191/0.193
RESULT jw_jw10 vs idle: 16.0 env-min  SD/min 1.375  deaths/min 1.438  kills/min 0.313  dmg dealt/taken per min 30.078/0.063
RESULT jw_jw10 deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 22 (1.375/min)
RESULT jw_jw10 offstage eps/min 6.688  recovery rate 0.781  neutral 0.308  repeat_prev 0.762  max frozen run 802 f  (23.1s wall)
RESULT jw_jw10: 36 cases x 8 trials  recovery 0.413  died 0.587  timeout 0.0  never 1/36  always 1/36
RESULT jw_jw10 by routes: 1 routes: 0.25 (1 cases)  2-3 routes: 0.159 (11 cases)  4+ routes: 0.536 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 15.0%  buttons 68.7%  families 35.1%  main dir 63.3% (bucket 46.3%, ±1 54.4%)  c dir 96.1%  trigger 93.5%  jump 86.7%  attack 96.1%  special 96.9%  surprise 3.855  change recall 52.6%  false change %
RESULT frames with a real press edge: n 533  joint 1.5%  buttons 15.4%  families 16.7%  main dir 87.1% (bucket 78.8%, ±1 86.1%)  c dir 100.0%  trigger 86.5%  jump 60.2%  attack 88.6%  special 90.1%  surprise 5.98  change recall 32.1%  false change %
RESULT press calibration jump: AUC 0.737  mean P on press frames 0.136 vs no-press 0.047  press frames above the no-press median 80.6%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.882  mean P on press frames 0.031 vs no-press 0.009  press frames above the no-press median 95.1%  (n 61 / 15748)
RESULT false press (model press edge on a frame with no real press): 0.9% of 15547
RESULT STANDING change frames: n 93 change recall 28%  | press edges same-family: jump 9.5% attack 0% special 0% grab 0% trigger 55.6%
RESULT jw_jw10 walk-off origins: 88 trips in 16.0 env-min (5.5/min)
RESULT jw_jw10   jump_off      n  60  3.75/min  died 0.23  stick toward edge (10 f before) 0.21  full-out 0.03  jumps left 0.73  last ground action [["KNEE_BEND", 33]]
RESULT jw_jw10   special_off   n   7  0.44/min  died 0.86  stick toward edge (10 f before) 0.86  full-out 0.0  jumps left 0.0  last ground action [["KNEE_BEND", 2], ["NEUTRAL_B_FULL_CHARGE_AIR", 1]]
jw_jw20 val_loss=3.344096374511719
RESULT jw_jw20: frames=32394 change_recall=0.271 hold_agreement=0.492 repeat_prev=0.73 (expert 0.764) neutral=0.207 (expert 0.303)
RESULT jw_jw20 press edges/min model:expert  A 15.78:13.22  B 22.34:11.0  X 25.56:30.12  Y 88.79:19.89  Z 5.22:10.0  L 23.23:26.34  R 32.56:8.33
RESULT jw_jw20 jab1 episodes=4 A-edges/episode model=0.5 expert=0.25
RESULT jw_jw20 change events recall/precision  ±0: press 0.02/0.011 release 0.01/0.006 stick 0.018/0.016  |  ±2: press 0.087/0.05 release 0.076/0.043 stick 0.089/0.079  |  ±5: press 0.146/0.088 release 0.132/0.078 stick 0.186/0.178
RESULT jw_jw20 vs idle: 16.0 env-min  SD/min 0.75  deaths/min 0.75  kills/min 0.125  dmg dealt/taken per min 17.257/0.0
RESULT jw_jw20 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 12 (0.75/min)
RESULT jw_jw20 offstage eps/min 8.0  recovery rate 0.905  neutral 0.255  repeat_prev 0.731  max frozen run 375 f  (20.5s wall)
RESULT jw_jw20: 36 cases x 8 trials  recovery 0.375  died 0.625  timeout 0.0  never 7/36  always 1/36
RESULT jw_jw20 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.114 (11 cases)  4+ routes: 0.479 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 14.1%  buttons 68.7%  families 35.4%  main dir 62.8% (bucket 44.9%, ±1 53.5%)  c dir 96.2%  trigger 93.8%  jump 86.7%  attack 96.1%  special 96.9%  surprise 3.93  change recall 51.9%  false change %
RESULT frames with a real press edge: n 533  joint 2.3%  buttons 16.5%  families 18.6%  main dir 87.4% (bucket 78.2%, ±1 85.9%)  c dir 100.0%  trigger 87.8%  jump 61.5%  attack 88.6%  special 90.1%  surprise 5.887  change recall 35.1%  false change %
RESULT press calibration jump: AUC 0.746  mean P on press frames 0.188 vs no-press 0.062  press frames above the no-press median 83.3%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.828  mean P on press frames 0.015 vs no-press 0.008  press frames above the no-press median 91.8%  (n 61 / 15748)
RESULT false press (model press edge on a frame with no real press): 1.1% of 15547
RESULT STANDING change frames: n 93 change recall 31.2%  | press edges same-family: jump 11.7% attack 0% special 0% grab 0% trigger 61.3%
RESULT jw_jw20 walk-off origins: 121 trips in 16.0 env-min (7.56/min)
RESULT jw_jw20   jump_off      n  79  4.94/min  died 0.1  stick toward edge (10 f before) 0.12  full-out 0.04  jumps left 0.59  last ground action [["KNEE_BEND", 34], ["LANDING", 1]]
RESULT jw_jw20   special_off   n   9  0.56/min  died 1.0  stick toward edge (10 f before) 0.89  full-out 0.14  jumps left 0.0  last ground action [["KNEE_BEND", 2], ["FIREFOX_GROUND", 1], ["FIREFOX_WAIT_GROUND", 1]]
jw_pw10_jw10 val_loss=3.318372599283854
RESULT jw_pw10_jw10: frames=32394 change_recall=0.259 hold_agreement=0.483 repeat_prev=0.725 (expert 0.764) neutral=0.193 (expert 0.303)
RESULT jw_pw10_jw10 press edges/min model:expert  A 21.56:13.22  B 21.67:11.0  X 52.45:30.12  Y 42.45:19.89  Z 3.11:10.0  L 20.0:26.34  R 65.23:8.33
RESULT jw_pw10_jw10 jab1 episodes=4 A-edges/episode model=0.25 expert=0.25
RESULT jw_pw10_jw10 change events recall/precision  ±0: press 0.017/0.009 release 0.008/0.004 stick 0.022/0.02  |  ±2: press 0.079/0.043 release 0.069/0.037 stick 0.099/0.091  |  ±5: press 0.156/0.088 release 0.136/0.076 stick 0.175/0.171
RESULT jw_pw10_jw10 vs idle: 16.0 env-min  SD/min 1.375  deaths/min 1.375  kills/min 0.563  dmg dealt/taken per min 44.664/0.0
RESULT jw_pw10_jw10 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 22 (1.375/min)
RESULT jw_pw10_jw10 offstage eps/min 10.125  recovery rate 0.863  neutral 0.287  repeat_prev 0.686  max frozen run 403 f  (21.6s wall)
RESULT jw_pw10_jw10: 36 cases x 8 trials  recovery 0.375  died 0.625  timeout 0.0  never 7/36  always 2/36
RESULT jw_pw10_jw10 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.125 (11 cases)  4+ routes: 0.474 (24 cases)
RESULT CHANGE frames (real input changed): n 3326  joint 14.8%  buttons 68.3%  families 36.1%  main dir 63.4% (bucket 45.9%, ±1 54.3%)  c dir 96.2%  trigger 93.4%  jump 86.9%  attack 96.0%  special 97.2%  surprise 3.832  change recall 52.6%  false change %
RESULT frames with a real press edge: n 533  joint 1.9%  buttons 12.8%  families 21.2%  main dir 87.4% (bucket 78.8%, ±1 85.9%)  c dir 100.0%  trigger 90.1%  jump 60.0%  attack 88.6%  special 91.0%  surprise 5.629  change recall 35.5%  false change %
RESULT press calibration jump: AUC 0.702  mean P on press frames 0.108 vs no-press 0.041  press frames above the no-press median 76.6%  (n 222 / 14700)
RESULT press calibration attack: AUC 0.872  mean P on press frames 0.082 vs no-press 0.009  press frames above the no-press median 91.8%  (n 61 / 15748)
RESULT false press (model press edge on a frame with no real press): 1.2% of 15547
RESULT STANDING change frames: n 93 change recall 32.3%  | press edges same-family: jump 8.6% attack 0% special 7.7% grab 0% trigger 70%
RESULT jw_pw10_jw10 walk-off origins: 154 trips in 16.0 env-min (9.63/min)
RESULT jw_pw10_jw10   jump_off      n 109  6.81/min  died 0.29  stick toward edge (10 f before) 0.32  full-out 0.03  jumps left 0.75  last ground action [["KNEE_BEND", 81], ["DASHING", 2], ["EDGE_TEETERING_START", 1]]
RESULT jw_pw10_jw10   special_off   n   3  0.19/min  died 1.0  stick toward edge (10 f before) 1.0  full-out 0.0  jumps left 0.0  last ground action [["NEUTRAL_B_FULL_CHARGE_AIR", 2], ["KNEE_BEND", 1]]
== multi-arm done (13:37) — table: node scripts/pq_digest.js jw_base3k jw_jw5 jw_jw10 jw_jw20 jw_pw10_jw10 
JW QUEUE DONE (13:37)

## READ (Claude, 13:45) — jump-family window, 3,000-game slice, embed-once driver (all five arms from one embedding, same build)
arm         val    fam%  joint  pressJ  falseP  AUCj   SD/min
base3k      3.222  37.3  17.0    0.0    0.1    0.762  1.56
jw5         3.275  35.5  15.4    4.5    0.3    0.707  1.06
jw10        3.304  35.1  15.0    9.5    0.9    0.737  1.38
jw20        3.344  35.4  14.1   11.7    1.1    0.746  0.75
pw10_jw10   3.318  36.1  14.8    8.6    1.2    0.702  1.38   (+ special 7.7)
The jump family now responds — jump press agreement 0 -> 4.5 / 9.5 / 11.7 % with dose — but every jump arm pays ~2 points
of option-family agreement, 1.6-2.9 joint, a worse loss and more false changes; false press crosses 1 % at dose 20.
The term is blunter than the per-button one (P_any over two buttons AND the window grows faster), so dose 5 is already
heavy. CAVEAT on the floor: this base3k reads fam 37.3 vs 35.1 for yesterday's base3k on the streaming path / old build
(same seed, same slice) — cross-run drift of ~2 points, so the ±0.4 "seed floor" only holds WITHIN one queue and path.
NEXT queue (now): base, base906 (within-queue floor), jw1, jw2, jw5 — find the dose that moves jump without the cost.
