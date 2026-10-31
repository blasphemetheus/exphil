
# Per-player pilot [314] — fine-tune master_ep7_plain, 3 epochs (2026-10-31 03:04); corpus 281 files

## epoch 2 (lr 0.0001) (03:04)
val_loss=3.3589
RESULT fox_mamba_p314_ep2 vs idle: 16.0 env-min  SD/min 1.063  deaths/min 1.063  kills/min 1.688  dmg dealt/taken per min 96.349/0.0
DONE fox_mamba_p314_ep2
RESULT ALL frames: n 62336  joint 78.9%  buttons 93.0%  families 84.9%  main dir 92.3% (bucket 87.8%, ±1 90.3%)  c dir 98.9%  trigger 97.3%  jump 97.6%  attack 99.0%  special 99.0%  surprise 3.27  change recall 62.4%  false change 0.9%
RESULT CHANGE frames (real input changed): n 16016  joint 20.7%  buttons 72.6%  families 43.7%  main dir 69.9% (bucket 52.6%, ±1 62.1%)  c dir 95.8%  trigger 91.7%  jump 90.6%  attack 96.1%  special 96.0%  surprise 4.007  change recall 62.4%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 39.8%  buttons 76.8%  families 57.9%  main dir 74.4% (bucket 59.7%, ±1 67.8%)  c dir 98.8%  trigger 96.4%  jump 92.1%  attack 96.7%  special 96.6%  surprise 2.733  change recall 63.1%  false change 0.9%
EPOCH 2 DONE light (03:15)

## epoch 3 (lr 0.0001) (03:15)
val_loss=3.3302
RESULT fox_mamba_p314_ep3 vs idle: 16.0 env-min  SD/min 0.5  deaths/min 0.5  kills/min 2.063  dmg dealt/taken per min 95.251/0.0
DONE fox_mamba_p314_ep3
RESULT ALL frames: n 62336  joint 79.0%  buttons 93.0%  families 84.9%  main dir 92.2% (bucket 87.8%, ±1 90.3%)  c dir 98.9%  trigger 97.3%  jump 97.6%  attack 99.0%  special 99.0%  surprise 3.256  change recall 62.5%  false change 0.9%
RESULT CHANGE frames (real input changed): n 16016  joint 20.9%  buttons 72.7%  families 43.8%  main dir 69.8% (bucket 52.7%, ±1 62.2%)  c dir 95.7%  trigger 91.9%  jump 90.7%  attack 96.2%  special 96.0%  surprise 3.967  change recall 62.5%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 39.9%  buttons 76.8%  families 57.9%  main dir 74.3% (bucket 59.8%, ±1 67.9%)  c dir 98.8%  trigger 96.5%  jump 92.1%  attack 96.7%  special 96.6%  surprise 2.699  change recall 63.2%  false change 0.9%
EPOCH 3 DONE light (03:26)

## epoch 4 (lr 0.0001) (03:26)
val_loss=3.3174
RESULT fox_mamba_p314_ep4 vs idle: 16.0 env-min  SD/min 0.688  deaths/min 0.813  kills/min 2.0  dmg dealt/taken per min 91.46/0.125
DONE fox_mamba_p314_ep4
RESULT ALL frames: n 62336  joint 79.0%  buttons 93.0%  families 84.9%  main dir 92.2% (bucket 87.9%, ±1 90.3%)  c dir 98.9%  trigger 97.3%  jump 97.6%  attack 99.0%  special 99.0%  surprise 3.25  change recall 62.5%  false change 0.9%
RESULT CHANGE frames (real input changed): n 16016  joint 21.0%  buttons 72.8%  families 43.8%  main dir 69.8% (bucket 52.8%, ±1 62.3%)  c dir 95.7%  trigger 91.9%  jump 90.7%  attack 96.2%  special 96.0%  surprise 3.95  change recall 62.5%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 40.1%  buttons 76.9%  families 58.0%  main dir 74.3% (bucket 59.8%, ±1 67.9%)  c dir 98.8%  trigger 96.6%  jump 92.1%  attack 96.7%  special 96.6%  surprise 2.684  change recall 63.2%  false change 1.0%
EPOCH 4 DONE light (03:38)

## CONTROL: generalist master_ep7_plain on 314's 16 held-out games (03:38)
RESULT ALL frames: n 62336  joint 77.8%  buttons 92.9%  families 84.7%  main dir 92.2% (bucket 87.6%, ±1 90.1%)  c dir 98.9%  trigger 97.1%  jump 97.6%  attack 99.0%  special 99.0%  surprise 3.339  change recall 62.9%  false change 2.1%
RESULT CHANGE frames (real input changed): n 16016  joint 19.6%  buttons 72.6%  families 43.1%  main dir 69.7% (bucket 51.9%, ±1 61.6%)  c dir 95.8%  trigger 91.2%  jump 90.6%  attack 96.1%  special 95.9%  surprise 4.151  change recall 62.9%  false change %
RESULT DECISION frames (stick head trained here): n 18849  joint 38.8%  buttons 76.7%  families 57.4%  main dir 74.2% (bucket 59.0%, ±1 67.3%)  c dir 98.8%  trigger 96.0%  jump 92.0%  attack 96.7%  special 96.6%  surprise 2.852  change recall 63.6%  false change 1.6%
## fine-tuned ep2 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 20.7%  buttons 72.6%  families 43.7%  main dir 69.9% (bucket 52.6%, ±1 62.1%)  c dir 95.8%  trigger 91.7%  jump 90.6%  attack 96.1%  special 9
RESULT DECISION frames (stick head trained here): n 18849  joint 39.8%  buttons 76.8%  families 57.9%  main dir 74.4% (bucket 59.7%, ±1 67.8%)  c dir 98.8%  trigger 96.4%  jump 92.1%  attack 96.7%  sp
## fine-tuned ep3 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 20.9%  buttons 72.7%  families 43.8%  main dir 69.8% (bucket 52.7%, ±1 62.2%)  c dir 95.7%  trigger 91.9%  jump 90.7%  attack 96.2%  special 9
RESULT DECISION frames (stick head trained here): n 18849  joint 39.9%  buttons 76.8%  families 57.9%  main dir 74.3% (bucket 59.8%, ±1 67.9%)  c dir 98.8%  trigger 96.5%  jump 92.1%  attack 96.7%  sp
## fine-tuned ep4 on 314's held-out games (from its epoch readout):
RESULT CHANGE frames (real input changed): n 16016  joint 21.0%  buttons 72.8%  families 43.8%  main dir 69.8% (bucket 52.8%, ±1 62.3%)  c dir 95.7%  trigger 91.9%  jump 90.7%  attack 96.2%  special 9
RESULT DECISION frames (stick head trained here): n 18849  joint 40.1%  buttons 76.9%  families 58.0%  main dir 74.3% (bucket 59.8%, ±1 67.9%)  c dir 98.8%  trigger 96.6%  jump 92.1%  attack 96.7%  sp

## FORGETTING: fine-tuned ep4 on the MASTER split (master_ep7_plain there: CHANGE fam 53.9 / joint 27.3) (03:40)
RESULT CHANGE frames (real input changed): n 11028  joint 27.0%  buttons 77.2%  families 52.6%  main dir 74.0% (bucket 52.2%, ±1 63.2%)  c dir 97.0%  trigger 94.2%  jump 90.6%  attack 97.1%  special 96.5%  surprise 3.179  change recall 71.8%  false change %
RESULT DECISION frames (stick head trained here): n 13176  joint 42.4%  buttons 81.0%  families 63.5%  main dir 78.0% (bucket 59.7%, ±1 69.0%)  c dir 99.1%  trigger 96.9%  jump 92.1%  attack 97.5%  special 97.1%  surprise 2.325  change recall 72.7%  false change 4.3%

## YARDSTICK: p314 ep4 vs slippi-ai imitation Fox d18, 3 headless games (03:41)
[Match] game 1: slippi_ai — exphil 0 stocks / 27%, slippi-ai 4 stocks / 48%, 75.2s
[Match] game 2: slippi_ai — exphil 0 stocks / 100%, slippi-ai 2 stocks / 18%, 120.8s
[Match] game 3: slippi_ai — exphil 0 stocks / 46%, slippi-ai 3 stocks / 12%, 86.8s
[03:46:50]   exphil 0 – 3 slippi-ai (3 games, 0 incomplete)

P314 DONE (03:46)

## VERDICT (03:50) — per-player fine-tune at this dose is a NULL
(First launch at 02:17 trained from scratch — full_epoch_recipe.sh passes --resume only for N>1 — voided as
checkpoints/void_p314_fresh_ep*, REPORT_void_fresh_init.md. Relaunch 03:04 resumed master_ep7_plain: start loss 6.9.)

On [314]'s 16 held-out games (16,016 change frames), change-frame families / joint:
| | fam | joint |
|---|---|---|
| generalist master_ep7_plain (control) | 43.1 | 19.6 |
| fine-tuned ep2 / ep3 / ep4 (265 games × 3 epochs, LR 1e-4) | 43.7 / 43.8 / 43.8 | 20.7 / 20.9 / 21.0 |
Forgetting on the master split: 52.6 / 27.0 (generalist 53.9 / 27.3). Yardstick vs slippi-ai imitation Fox: 0–3
(they lost 1, 2, 1 stocks — the only games so far where they lost stocks; 3 games, noise-level).

Reads: (1) +0.7 fam / +1.4 joint on his own games is inside the ~2-point floor: 12k steps at LR 1e-4 on 281 games
did not make the generalist play more like [314]. (2) The generalist is far less predictable on this player (fam 43)
than on masters (54): a local-scene player's choices are harder to call, and the per-player bot must close THAT gap.
(3) Nothing was forgotten, so the dose was too small to move anything. A real per-player arm needs either 10x his
games, a higher LR / more epochs with the forgetting check as the guard, or a player token conditioned at train
time on the generalist corpus (slippi-ai's name_map approach) rather than post-hoc fine-tuning.
