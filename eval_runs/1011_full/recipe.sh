# FROZEN recipe for the full Fox Mamba run (2026-10-11 01:20). Sourced by
# scripts/full_mamba_epoch.sh — every epoch of the run trains on exactly
# these flags, so the run is a reproduction of the testbed winner, not a
# reconstruction. Do NOT edit while an epoch is alive; a recipe change is a
# new run name (RUN below).
#
# Provenance: model/driver = fox_mamba_v1 (HANDOFF §7f/§7i: 512,512,256
# Mamba, lr 2e-4, ~8.5 h per epoch on the full v2_filtered Fox corpus,
# val 2.39 -> 2.35); input recipe = the input-coherence queue-40 winner on
# the MinGRU testbed (INPUT_COHERENCE_2026-10-01.md: prev-action quantized,
# button/stick events + event context, chunk horizon 8, offstage weight 3,
# stick duration 8, onset weight 5) — the only testbed arm that held the
# band, the closed loop, the aim and the jump rows together on two seeds.
#
# What the testbed could NOT validate and this run measures (HANDOFF §8e):
# whether capacity + data + epochs move the recovery death counts that no
# per-frame lever moved (carried side-B 13–22 / run, decided-trip return
# 0.38–0.60 vs the expert's 0.916). The per-epoch readout writes one row to
# eval_runs/1011_full/curve.txt; the curve, not the final number, is the
# product.
RUN=fox_mamba_v3                      # run name; epoch dirs checkpoints/coh_${RUN}_epN
CORPUS=replays/erickfm_ranked/v2_filtered
model=(--backbone mamba --stage-internals --hidden-sizes 512,512,256 --batch-size 128 --precision f32
  --window-size 80 --stride 5 --dropout 0.0)
data=(--replays "$CORPUS" --train-character fox --select-character-port
  --stream-chunk-size 64 --no-cache-streaming --label-delay 0 --no-cache --no-register)
loss=(--head autoregressive --save-best --save-every-batches 25000 --label-smoothing 0.0 --no-focal-loss
  --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0
  --neutral-weight 1.0 --stick-edge-weight 1.0)
# the queue-40 winner's input recipe (eval_runs/1001_queue/queue40_winner.sh)
pq=(--prev-action --prev-action-dropout 0.0 --prev-action-quantize)
ev=(--button-events --stick-events --event-context)
rec0=(--chunk-horizon 8 --offstage-weight 3 --stick-duration 8)
knob=(--onset-weight 5)
# LR: 2e-4 was fox_mamba_v1's; §7i: a 3rd epoch at constant LR is not the
# lever — step down (LR=0.0001) from epoch 3. Set per launch, logged per row.
LR_DEFAULT=0.0002
SEED_DEFAULT=905
# Expert-labeled DAgger mix (×1) for epoch >= 2: a round rolled from THIS
# run's previous epoch (its own states), labeler v4 :wide, gated, split —
# the queue-40 chain (scripts/coherence_queue40.sh steps 1–2) with
# --policy checkpoints/coh_${RUN}_ep$((N-1))/model_best_policy.bin. Empty =
# no mix (epoch 1 always). Gate first: dagger_set_drift_cell.exs dose +
# expert_index_drift_cell.exs label-vs-own-input (feedback 10-10).
MIX=${MIX:-}
