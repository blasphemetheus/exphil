
# Per-player pilot [314] — fine-tune master_ep7_plain, 3 epochs (2026-10-31 02:17); corpus 281 files

## epoch 1 (lr 0.0001) (02:17)
val_loss=4.8113
RESULT fox_mamba_p314_ep1 vs idle: 16.0 env-min  SD/min 2.563  deaths/min 2.75  kills/min 0.063  dmg dealt/taken per min 13.847/0.188
DONE fox_mamba_p314_ep1
RESULT ALL frames: n 62336  joint 74.7%  buttons 92.7%  families 80.1%  main dir 87.5% (bucket 83.4%, ±1 85.7%)  c dir 98.8%  trigger 96.5%  jump 97.4%  attack 99.0%  special 98.9%  surprise 2.042  change recall 43.1%  false change 2.0%
RESULT CHANGE frames (real input changed): n 16016  joint 7.3%  buttons 72.4%  families 27.0%  main dir 52.7% (bucket 37.0%, ±1 45.7%)  c dir 95.2%  trigger 88.9%  jump 90.7%  attack 96.0%  special 95.8%  surprise 5.347  change recall 43.1%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 27.8%  buttons 76.4%  families 42.8%  main dir 58.6% (bucket 45.2%, ±1 52.6%)  c dir 98.6%  trigger 94.1%  jump 92.0%  attack 96.6%  special 96.4%  surprise 4.082  change recall 43.1%  false change 6.1%
EPOCH 1 DONE light (02:28)

## epoch 2 (lr 0.0001) (02:28)
val_loss=4.6912
RESULT fox_mamba_p314_ep2 vs idle: 16.0 env-min  SD/min 2.188  deaths/min 2.25  kills/min 0.063  dmg dealt/taken per min 6.154/0.063
DONE fox_mamba_p314_ep2
RESULT ALL frames: n 62336  joint 75.7%  buttons 92.6%  families 82.2%  main dir 89.5% (bucket 84.4%, ±1 87.3%)  c dir 98.8%  trigger 97.0%  jump 97.6%  attack 99.0%  special 98.7%  surprise 2.168  change recall 43.7%  false change 1.4%
RESULT CHANGE frames (real input changed): n 16016  joint 9.7%  buttons 72.3%  families 34.4%  main dir 59.7% (bucket 40.2%, ±1 51.1%)  c dir 95.2%  trigger 90.9%  jump 90.6%  attack 96.0%  special 95.9%  surprise 5.096  change recall 43.7%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 30.5%  buttons 76.4%  families 49.8%  main dir 65.3% (bucket 48.5%, ±1 57.9%)  c dir 98.6%  trigger 95.8%  jump 92.0%  attack 96.6%  special 96.4%  surprise 3.88  change recall 43.9%  false change 3.6%
EPOCH 2 DONE light (02:40)

## epoch 3 (lr 0.0001) (02:40)
val_loss=4.2588
RESULT fox_mamba_p314_ep3 vs idle: 16.0 env-min  SD/min 1.688  deaths/min 1.813  kills/min 0.188  dmg dealt/taken per min 26.061/0.188
DONE fox_mamba_p314_ep3
RESULT ALL frames: n 62336  joint 76.1%  buttons 92.8%  families 83.3%  main dir 90.8% (bucket 85.0%, ±1 88.8%)  c dir 98.7%  trigger 97.2%  jump 97.6%  attack 99.0%  special 98.9%  surprise 1.72  change recall 36.3%  false change 1.0%
RESULT CHANGE frames (real input changed): n 16016  joint 9.9%  buttons 72.4%  families 37.8%  main dir 64.2% (bucket 41.9%, ±1 56.4%)  c dir 95.2%  trigger 91.5%  jump 90.7%  attack 96.0%  special 95.8%  surprise 4.595  change recall 36.3%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 31.3%  buttons 76.5%  families 53.2%  main dir 69.5% (bucket 50.5%, ±1 62.9%)  c dir 98.5%  trigger 96.3%  jump 92.1%  attack 96.6%  special 96.4%  surprise 3.477  change recall 35.3%  false change 1.1%
EPOCH 3 DONE light (02:52)

## CONTROL: generalist master_ep7_plain on 314's 16 held-out games (02:52)
RESULT ALL frames: n 62336  joint 77.8%  buttons 92.9%  families 84.7%  main dir 92.2% (bucket 87.6%, ±1 90.1%)  c dir 98.9%  trigger 97.1%  jump 97.6%  attack 99.0%  special 99.0%  surprise 3.339  change recall 62.9%  false change 2.1%
RESULT CHANGE frames (real input changed): n 16016  joint 19.6%  buttons 72.6%  families 43.1%  main dir 69.7% (bucket 51.9%, ±1 61.6%)  c dir 95.8%  trigger 91.2%  jump 90.6%  attack 96.1%  special 95.9%  surprise 4.151  change recall 62.9%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 38.8%  buttons 76.7%  families 57.4%  main dir 74.2% (bucket 59.0%, ±1 67.3%)  c dir 98.8%  trigger 96.0%  jump 92.0%  attack 96.7%  special 96.6%  surprise 2.852  change recall 63.6%  false change 1.6%
## fine-tuned ep1 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 7.3%  buttons 72.4%  families 27.0%  main dir 52.7% (bucket 37.0%, ±1 45.7%)  c dir 95.2%  trigger 88.9%  jump 90.7%  attack 96.0%  special 95
RESULT DECISION frames (stick head trained here): n 18849  joint 27.8%  buttons 76.4%  families 42.8%  main dir 58.6% (bucket 45.2%, ±1 52.6%)  c dir 98.6%  trigger 94.1%  jump 92.0%  attack 96.6%  sp
## fine-tuned ep2 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 9.7%  buttons 72.3%  families 34.4%  main dir 59.7% (bucket 40.2%, ±1 51.1%)  c dir 95.2%  trigger 90.9%  jump 90.6%  attack 96.0%  special 95
RESULT DECISION frames (stick head trained here): n 18849  joint 30.5%  buttons 76.4%  families 49.8%  main dir 65.3% (bucket 48.5%, ±1 57.9%)  c dir 98.6%  trigger 95.8%  jump 92.0%  attack 96.6%  sp
## fine-tuned ep3 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 9.9%  buttons 72.4%  families 37.8%  main dir 64.2% (bucket 41.9%, ±1 56.4%)  c dir 95.2%  trigger 91.5%  jump 90.7%  attack 96.0%  special 95
RESULT DECISION frames (stick head trained here): n 18849  joint 31.3%  buttons 76.5%  families 53.2%  main dir 69.5% (bucket 50.5%, ±1 62.9%)  c dir 98.5%  trigger 96.3%  jump 92.1%  attack 96.6%  sp

## FORGETTING: fine-tuned ep3 on the MASTER split (master_ep7_plain there: CHANGE fam 53.9 / joint 27.3) (02:54)
RESULT CHANGE frames (real input changed): n 11028  joint 9.8%  buttons 76.4%  families 40.0%  main dir 62.3% (bucket 34.3%, ±1 47.4%)  c dir 96.2%  trigger 93.2%  jump 90.3%  attack 97.0%  special 96.3%  surprise 4.894  change recall 38.8%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 28.9%  buttons 80.2%  families 53.6%  main dir 68.3% (bucket 44.8%, ±1 55.7%)  c dir 98.8%  trigger 96.3%  jump 91.8%  attack 97.5%  special 96.9%  surprise 3.835  change recall 38.4%  false change 4.0%

## YARDSTICK: p314 ep3 vs slippi-ai imitation Fox d18, 3 headless games (02:55)
[Match] game 1: slippi_ai — exphil 0 stocks / 81%, slippi-ai 4 stocks / 22%, 49.0s
[Match] game 2: slippi_ai — exphil 0 stocks / 118%, slippi-ai 3 stocks / 0%, 81.3s
[Match] game 3: slippi_ai — exphil 0 stocks / 6%, slippi-ai 4 stocks / 10%, 56.2s
[02:59:21]   exphil 0 – 3 slippi-ai (3 games, 0 incomplete)

P314 DONE (02:59)
