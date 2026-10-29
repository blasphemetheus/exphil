
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
