# Mewtwo opponent-pool experiment

Updated: 2026-09-25T18:58:58.672004+00:00. Phase: **complete**.

Unit: `exphil-mewtwo-pool-v2`. Log: `logs/exphil-mewtwo-pool-v2.log`.

One serialized GPU job: training → evaluation/export. No other training is launched by this chain.

Pool: frozen prior + v1 heads70/150/300; v1 head100 excluded for evaluation. KL0.01 unchanged,200 iterations maximum. This tests within-family opponent diversity, not broad character/generalist strength. No promotion.

Graceful training stop: create `eval_runs/0925_mewtwo_pool/v2/STOP`; evaluation then proceeds on the saved head. Stop the systemd unit to cancel the entire chain.

Design and completed review: [September25 resume](PPO_RESUME_2026-09-25.md).
