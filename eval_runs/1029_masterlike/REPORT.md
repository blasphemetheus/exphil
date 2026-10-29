
# Master-like data arm — 3 epochs (2026-10-29 08:48); corpus 11823 files

## epoch 1 (lr 0.0002) (08:48)
val_loss=3.1567
RESULT fox_mamba_mlike_ep1: frames=37786 change_recall=0.334 hold_agreement=0.633 repeat_prev=0.734 (expert 0.672) neutral=0.253 (expert 0.269)
RESULT fox_mamba_mlike_ep1 press edges/min model:expert  A 22.77:24.58  B 15.53:18.2  X 18.29:17.24  Y 26.1:46.4  Z 5.05:1.43  L 31.06:46.87  R 16.01:4.57
RESULT fox_mamba_mlike_ep1 jab1 episodes=12 A-edges/episode model=0.25 expert=0.5
RESULT fox_mamba_mlike_ep1 change events recall/precision  ±0: press 0.035/0.041 release 0.026/0.03 stick 0.036/0.037  |  ±2: press 0.184/0.221 release 0.138/0.165 stick 0.172/0.179  |  ±5: press 0.263/0.307 release 0.217/0.26 stick 0.293/0.315
RESULT fox_mamba_mlike_ep1 vs idle: 16.0 env-min  SD/min 0.75  deaths/min 0.75  kills/min 1.438  dmg dealt/taken per min 80.218/0.0
RESULT fox_mamba_mlike_ep1 deaths by verdict at the decision frame: kill 0 (0.0/min)  checkmate 0 (0.0/min)  forced 0 (0.0/min)  mixup 12 (0.75/min)
RESULT fox_mamba_mlike_ep1 offstage eps/min 3.813  recovery rate 0.8  neutral 0.348  repeat_prev 0.728  max frozen run 506 f  (21.1s wall)
RESULT fox_mamba_mlike_ep1: 36 cases x 8 trials  recovery 0.656  died 0.344  timeout 0.0  never 0/36  always 1/36
RESULT fox_mamba_mlike_ep1 by routes: 1 routes: 0.75 (1 cases)  2-3 routes: 0.591 (11 cases)  4+ routes: 0.682 (24 cases)
DONE fox_mamba_mlike_ep1
RESULT ALL frames: n 39589  joint 70.8%  buttons 92.3%  families 82.7%  main dir 91.8% (bucket 85.7%, ±1 88.9%)  c dir 98.7%  trigger 96.1%  jump 96.8%  attack 98.7%  special 99.0%  surprise 2.748  change recall 70.0%  false change 8.8%
RESULT CHANGE frames (real input changed): n 11578  joint 21.5%  buttons 74.1%  families 46.2%  main dir 72.3% (bucket 51.5%, ±1 62.1%)  c dir 96.0%  trigger 91.0%  jump 89.0%  attack 95.8%  special 96.7%  surprise 3.777  change recall 70.0%  false change %
RESULT frames with a real press edge: n 1613  joint 1.7%  buttons 18.2%  families 14.3%  main dir 88.3% (bucket 77.3%, ±1 83.6%)  c dir 99.2%  trigger 82.6%  jump 57.5%  attack 84.5%  special 88.0%  surprise 5.617  change recall 36.0%  false change %
RESULT DECISION frames (stick head trained here): n 13322  joint 35.4%  buttons 77.5%  families 56.9%  main dir 75.8% (bucket 57.6%, ±1 66.9%)  c dir 98.7%  trigger 94.0%  jump 90.4%  attack 96.3%  special 97.1%  surprise 2.979  change recall 70.6%  false change 9.4%
RESULT press calibration jump: AUC 0.814  mean P on press frames 0.054 vs no-press 0.021  press frames above the no-press median 91.0%  (n 645 / 35953)
RESULT press calibration attack: AUC 0.912  mean P on press frames 0.1 vs no-press 0.012  press frames above the no-press median 98.3%  (n 236 / 38037)
RESULT press calibration special: AUC 0.956  mean P on press frames 0.162 vs no-press 0.008  press frames above the no-press median 99.0%  (n 192 / 37599)
RESULT press calibration trigger: AUC 0.905  mean P on press frames 0.542 vs no-press 0.049  press frames above the no-press median 96.1%  (n 537 / 32050)
RESULT STANDING change frames: n 204 change recall 54.4% families 42.2%
EPOCH 1 DONE light (13:20)
### agreement on the master run's 16 held-out games (8 used) (13:20)
RESULT ALL frames: n 39430  joint 76.1%  buttons 93.4%  families 85.4%  main dir 92.1% (bucket 85.8%, ±1 89.0%)  c dir 99.1%  trigger 97.7%  jump 97.3%  attack 99.2%  special 99.0%  surprise 2.449  change recall 70.3%  false change 4.1%
RESULT CHANGE frames (real input changed): n 11028  joint 24.8%  buttons 76.8%  families 50.5%  main dir 72.2% (bucket 49.7%, ±1 60.9%)  c dir 96.9%  trigger 93.9%  jump 90.5%  attack 97.0%  special 96.3%  surprise 3.287  change recall 70.3%  false change %
RESULT frames with a real press edge: n 1314  joint 0.5%  buttons 19.0%  families 8.6%  main dir 89.4% (bucket 78.4%, ±1 85.8%)  c dir 99.5%  trigger 79.8%  jump 57.6%  attack 86.7%  special 84.1%  surprise 5.121  change recall 28.6%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 40.6%  buttons 80.6%  families 61.8%  main dir 76.4% (bucket 57.5%, ±1 67.0%)  c dir 99.1%  trigger 96.8%  jump 92.0%  attack 97.5%  special 96.9%  surprise 2.487  change recall 71.2%  false change 4.8%
RESULT press calibration jump: AUC 0.851  mean P on press frames 0.056 vs no-press 0.019  press frames above the no-press median 95.2%  (n 523 / 35276)
RESULT press calibration attack: AUC 0.929  mean P on press frames 0.096 vs no-press 0.009  press frames above the no-press median 99.4%  (n 165 / 38183)
RESULT press calibration special: AUC 0.941  mean P on press frames 0.12 vs no-press 0.008  press frames above the no-press median 96.6%  (n 207 / 37830)

## epoch 2 (lr 0.0002) (13:21)

### epoch 1 read (13:30) — on the master split (same 11,028 change frames)
| | CHANGE joint | CHANGE fam | DECISION joint | DECISION fam |
|---|---|---|---|---|
| master ep1 | 23.8 | 50.8 | 40.0 | 62.3 |
| master ep2 | 25.8 | 52.3 | 41.5 | 63.4 |
| **mlike ep1** | 24.8 | 50.5 | 40.6 | 61.8 |
Level with master ep1 within the seed floor, despite 43 % more gradient steps per epoch (172,893 batches vs ~121k).
Not a lift at equal epochs; not yet a fail — the plateau question is ep3. Epoch ~4.5 h, not the 3.5 estimated.

PAUSED 15:14 by Bradley (machine needed) at epoch 2 chunk 75/185 (batch ~75k/180k; step rate had already fallen 58 -> 200 ms/it under his load). Resume: scripts/launch_unit.sh exphil-mlike-1029 'scripts/session_1029_masterlike_resume.sh'
