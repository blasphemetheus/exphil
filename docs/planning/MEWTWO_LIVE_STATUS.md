# Mewtwo imitation live status

Updated automatically: 2026-09-24T06:03:42.642598+00:00.

Phase: **imitation_complete**. Service: `exphil-mewtwo-il-v1` (inactive).
Epoch: 12/30.
Latest completed epoch: train loss 2.311, validation loss 3.1513.

Supervisor PID: 114799; latest child PID: finished.
Log: `/home/blewf/git/exphil/checkpoints/mewtwo_il_v1_20260924/train.log`. These PIDs describe this update and may exit afterward.

Corpus: 418 games, 5.24M frames; 334 train / 42 validation / 42 test.
Model: 2×1024 GRU, 10.46M parameters; up to 30 epochs with early stopping.
Preflight passed: 250 updates, train 4.2093 / validation 4.343, policy exported.
Best full-fit policy: `checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin`
(available after the first completed full-fit epoch).

PPO has not started. Final held-out testing, Dolphin play and Mewtwo sim checks remain.
While training is active, do not run Mix or edit training library files.
Detailed recipe and provenance: [Mewtwo campaign](MEWTWO_IMITATION_2026-09-24.md).
Error, if any: none reported.
