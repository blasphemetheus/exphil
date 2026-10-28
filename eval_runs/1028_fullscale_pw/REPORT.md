
# full-scale press-window A/B from master_ep6 (2026-10-28 02:18)

## master ep7 pw10 (02:18)
TRAIN_FAILED fox_mamba_master_ep7_pw10 (see eval_runs/1001_queue/fox_mamba_master_ep7_pw10/train.log)

## master ep7 plain (02:23)

# full-scale press-window A/B from master_ep6 (2026-10-28 02:24)

## master ep7 pw10 (02:24)
val_loss=2.8894
RESULT fox_mamba_master_ep7_pw10: frames=30583 change_recall=0.257 hold_agreement=0.578 repeat_prev=0.748 (expert 0.665) neutral=0.264 (expert 0.275)
RESULT fox_mamba_master_ep7_pw10 press edges/min model:expert  A 20.01:19.19  B 30.13:18.72  X 20.6:30.96  Y 37.31:26.72  Z 9.3:4.36  L 23.66:33.31  R 20.48:7.89
RESULT fox_mamba_master_ep7_pw10 jab1 episodes=6 A-edges/episode model=0.667 expert=0.333
RESULT fox_mamba_master_ep7_pw10 change events recall/precision  ±0: press 0.033/0.029 release 0.036/0.031 stick 0.034/0.037  |  ±2: press 0.249/0.222 release 0.158/0.139 stick 0.181/0.199  |  ±5: press 0.327/0.293 release 0.27/0.241 stick 0.284/0.317
RESULT fox_mamba_master_ep7_pw10 vs idle: 16.0 env-min  SD/min 0.625  deaths/min 0.625  kills/min 2.188  dmg dealt/taken per min 89.801/0.0
RESULT fox_mamba_master_ep7_pw10 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 10 (0.625/min)
RESULT fox_mamba_master_ep7_pw10 offstage eps/min 4.375  recovery rate 0.851  neutral 0.387  repeat_prev 0.739  max frozen run 409 f  (18.1s wall)
RESULT fox_mamba_master_ep7_pw10: 36 cases x 8 trials  recovery 0.795  died 0.205  timeout 0.0  never 0/36  always 9/36
RESULT fox_mamba_master_ep7_pw10 by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.693 (11 cases)  4+ routes: 0.833 (24 cases)
DONE fox_mamba_master_ep7_pw10
RESULT ALL frames: n 39430  joint 75.4%  buttons 93.0%  families 86.0%  main dir 93.1% (bucket 86.7%, ±1 90.0%)  c dir 99.2%  trigger 97.7%  jump 97.3%  attack 98.9%  special 98.8%  surprise 3.009  change recall 72.5%  false change 6.0%
RESULT CHANGE frames (real input changed): n 11028  joint 27.4%  buttons 77.3%  families 54.1%  main dir 75.7% (bucket 52.8%, ±1 64.5%)  c dir 97.1%  trigger 93.7%  jump 90.6%  attack 97.0%  special 96.6%  surprise 3.183  change recall 72.5%  false change %
RESULT frames with a real press edge: n 1314  joint 4.0%  buttons 22.7%  families 11.9%  main dir 90.2% (bucket 79.5%, ±1 86.2%)  c dir 99.5%  trigger 78.8%  jump 57.6%  attack 88.4%  special 86.3%  surprise 4.798  change recall 31.7%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 42.5%  buttons 81.0%  families 64.6%  main dir 79.4% (bucket 60.2%, ±1 70.0%)  c dir 99.2%  trigger 96.5%  jump 92.1%  attack 97.5%  special 97.1%  surprise 2.324  change recall 73.5%  false change 5.6%
RESULT press calibration jump: AUC 0.877  mean P on press frames 0.086 vs no-press 0.022  press frames above the no-press median 97.3%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.936  mean P on press frames 0.186 vs no-press 0.011  press frames above the no-press median 98.2%  (n 165 / 38183)
RESULT press calibration special: AUC 0.953  mean P on press frames 0.255 vs no-press 0.012  press frames above the no-press median 97.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.913  mean P on press frames 0.371 vs no-press 0.034  press frames above the no-press median 97.4%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 52.6% families 42.3%
EPOCH 7 DONE light (06:29)
RESULT fox_mamba_master_ep7_pw10 walk-off origins: 88 trips in 16.0 env-min (5.5/min)
RESULT fox_mamba_master_ep7_pw10   walk_off      n   9  0.56/min  died 0.11  stick toward edge (10 f before) 0.78  full-out 0.44  jumps left 1.0  last ground action [["DASHING", 5], ["WALK_MIDDLE", 2], ["RUNNING", 1]]
RESULT fox_mamba_master_ep7_pw10   jump_off      n  58  3.63/min  died 0.1  stick toward edge (10 f before) 0.29  full-out 0.03  jumps left 0.67  last ground action [["KNEE_BEND", 27], ["DASHING", 4], ["LANDING_SPECIAL", 2]]
RESULT fox_mamba_master_ep7_pw10   special_off   n   5  0.31/min  died 0.6  stick toward edge (10 f before) 0.16  full-out 0.04  jumps left 0.6  last ground action [["KNEE_BEND", 2], ["CROUCH_START", 1], ["NEUTRAL_B_FULL_CHARGE_AIR", 1]]

## master ep7 plain (06:30)

## master ep7 plain — RESUMED from chunk 49/123 (2026-10-28 08:09)

## master ep7 plain — RESUMED from chunk 49/123 (2026-10-28 08:49)
val_loss=2.8689
RESULT fox_mamba_master_ep7_plain: frames=30583 change_recall=0.272 hold_agreement=0.615 repeat_prev=0.739 (expert 0.665) neutral=0.268 (expert 0.275)
RESULT fox_mamba_master_ep7_plain press edges/min model:expert  A 18.6:19.19  B 18.72:18.72  X 23.42:30.96  Y 42.61:26.72  Z 5.53:4.36  L 28.96:33.31  R 17.77:7.89
RESULT fox_mamba_master_ep7_plain jab1 episodes=6 A-edges/episode model=0.333 expert=0.333
RESULT fox_mamba_master_ep7_plain change events recall/precision  ±0: press 0.038/0.035 release 0.041/0.037 stick 0.043/0.044  |  ±2: press 0.3/0.276 release 0.172/0.156 stick 0.181/0.189  |  ±5: press 0.372/0.343 release 0.298/0.271 stick 0.309/0.334
RESULT fox_mamba_master_ep7_plain vs idle: 16.0 env-min  SD/min 0.625  deaths/min 0.688  kills/min 2.063  dmg dealt/taken per min 73.842/0.063
RESULT fox_mamba_master_ep7_plain deaths by verdict at the decision frame: kill 1 (0.063/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 10 (0.625/min)
RESULT fox_mamba_master_ep7_plain offstage eps/min 4.688  recovery rate 0.853  neutral 0.377  repeat_prev 0.732  max frozen run 668 f  (17.8s wall)
RESULT fox_mamba_master_ep7_plain: 36 cases x 8 trials  recovery 0.788  died 0.212  timeout 0.0  never 0/36  always 10/36
RESULT fox_mamba_master_ep7_plain by routes: 1 routes: 1.0 (1 cases)  2-3 routes: 0.682 (11 cases)  4+ routes: 0.828 (24 cases)
DONE fox_mamba_master_ep7_plain
RESULT ALL frames: n 39430  joint 76.1%  buttons 93.5%  families 86.4%  main dir 93.1% (bucket 86.7%, ±1 90.0%)  c dir 99.2%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.993  change recall 71.9%  false change 5.0%
RESULT CHANGE frames (real input changed): n 11028  joint 27.3%  buttons 77.2%  families 53.9%  main dir 75.7% (bucket 52.8%, ±1 64.6%)  c dir 97.1%  trigger 93.8%  jump 90.6%  attack 97.1%  special 96.5%  surprise 3.159  change recall 71.9%  false change %
RESULT frames with a real press edge: n 1314  joint 1.7%  buttons 20.1%  families 9.2%  main dir 90.0% (bucket 79.3%, ±1 86.2%)  c dir 99.5%  trigger 78.8%  jump 57.6%  attack 86.8%  special 85.2%  surprise 4.768  change recall 29.1%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 42.5%  buttons 80.9%  families 64.6%  main dir 79.5% (bucket 60.2%, ±1 70.1%)  c dir 99.2%  trigger 96.6%  jump 92.1%  attack 97.5%  special 97.1%  surprise 2.304  change recall 72.9%  false change 5.0%
RESULT press calibration jump: AUC 0.88  mean P on press frames 0.09 vs no-press 0.022  press frames above the no-press median 97.3%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.941  mean P on press frames 0.104 vs no-press 0.009  press frames above the no-press median 98.8%  (n 165 / 38183)
RESULT press calibration special: AUC 0.961  mean P on press frames 0.197 vs no-press 0.009  press frames above the no-press median 98.6%  (n 207 / 37830)
RESULT press calibration trigger: AUC 0.911  mean P on press frames 0.368 vs no-press 0.033  press frames above the no-press median 97.2%  (n 387 / 32618)
RESULT STANDING change frames: n 234 change recall 50.9% families 40.6%
EPOCH 7 DONE light (11:18)
RESULT fox_mamba_master_ep7_plain walk-off origins: 64 trips in 16.0 env-min (4.0/min)
RESULT fox_mamba_master_ep7_plain   walk_off      n   5  0.31/min  died 0.0  stick toward edge (10 f before) 0.7  full-out 0.18  jumps left 1.0  last ground action [["DASHING", 4], ["RUNNING", 1]]
RESULT fox_mamba_master_ep7_plain   jump_off      n  49  3.06/min  died 0.06  stick toward edge (10 f before) 0.36  full-out 0.12  jumps left 0.57  last ground action [["KNEE_BEND", 24], ["DASHING", 1], ["LANDING_SPECIAL", 1]]
RESULT fox_mamba_master_ep7_plain   special_off   n   2  0.13/min  died 0.5  stick toward edge (10 f before) 0.05  full-out 0.0  jumps left 0.5  last ground action [["DOWN_B_GROUND_START", 1]]

## mini league (11:19)
RESULT pair 1/6 s906_ep2 vs master_ep6: 10-22-0 (0/32 finished)  63 s  1804 game-frames/s
RESULT pair 2/6 s906_ep2 vs master_ep7_plain: 13-18-1 (0/32 finished)  59 s  1926 game-frames/s
RESULT pair 3/6 s906_ep2 vs master_ep7_pw10: 18-14-0 (0/32 finished)  59 s  1923 game-frames/s
RESULT pair 4/6 master_ep6 vs master_ep7_plain: 14-16-2 (0/32 finished)  60 s  1912 game-frames/s
RESULT pair 5/6 master_ep6 vs master_ep7_pw10: 18-14-0 (0/32 finished)  60 s  1892 game-frames/s
RESULT pair 6/6 master_ep7_plain vs master_ep7_pw10: 17-14-1 (0/32 finished)  60 s  1903 game-frames/s
RESULT standings (Elo, Bradley–Terry over 192 games):
RESULT league -> eval_runs/1028_fullscale_pw/league/league.json
RESULT standing 1538 master_ep6 win 0.573
RESULT standing 1527 master_ep7_plain win 0.552
RESULT standing 1470 master_ep7_pw10 win 0.443
RESULT standing 1464 s906_ep2 win 0.432
RESULT matrix master_ep6 - 0.469 0.563 0.688
RESULT matrix master_ep7_plain 0.531 - 0.547 0.578
RESULT matrix master_ep7_pw10 0.438 0.453 - 0.438
RESULT matrix s906_ep2 0.313 0.422 0.563 -

FULLSCALE A/B DONE (11:25)

## A/B READ (Claude, 11:30) — master ep7 + press-window 10 vs plain master ep7 (same resume / lr / seed / batch 64 / fused off)
                     fam%   joint  press attack  press special  false press  false change  SD/min  deaths/min  dmg/min  val     league
master ep6 (parent)  53.4   26.4     3.0            7.7           0.2          8.2          0.688   0.75        89       2.977   1538
ep7 plain (control)  53.9   27.3     2.4            7.2           0.2          5.0          0.625   0.688       74       2.869   1527
ep7 pw10             54.1   27.4    15.2           15.9           0.8          6.0          0.625   0.625       90       2.889   1470
Against the bars: press commitment UP (attack x6, special x2 — the control did not move) PASS; false press 0.8 <= 1 PASS;
SD/min equal PASS; option families +0.2 over the control (floor ±0.4) FAIL — the +0.7 over ep6 was the extra epoch; league:
pw10 lost to the control 45/55 and to ep6 44/56 over 32 games each (n too small to call), beat the candidate 56/44 — NOT
clearly worse, not better. Jump presses untouched (0.4 %). Walk-offs: pw10 5.5 trips/min vs plain 4.0 (more jump-offs,
same per-trip death rate); special-off 0.31 vs 0.13/min.
VERDICT: the press-window objective does the one thing it was built for at full scale — the model commits to attacks and
specials on the frame — without a self-destruct cost, but it buys no agreement beyond the epoch and no strength. Keep it as
the commitment ingredient (the engine/decode needs committed presses), not as a strength lever. NEXT: a jump-specific
window (jump is the family it does not reach and the one the deaths hinge on); dose 5 at full scale to see if the play
dip is dose-related; the master arm itself (ep7 plain) is the new strongest imitation model by Elo and agreement.
