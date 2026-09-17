# Fox V3 — long-training handoff for Claude Fable

**Status: MECHANICAL GO (2026-09-17)** — see the GO / NO-GO section at the end of
[FOX_V3_PREFLIGHT.md](FOX_V3_PREFLIGHT.md). Launch only from a tree that includes
the 09-17 fixes (arithmetic stamp, tag normalization, export `precision`). The user assigned the long training job to Fable;
no full run has been started during preflight.

## What to train

Fresh generalist Fox, not a continuation of the multishine specialist.
Data recipe as of 09-17 late: `--player-tag-map` (STYLE_IDENTITY.md S3-S5) so
matched real tags and `~cNN` style clusters condition the name channel;
registry keeps the 111 most-played identities. Temporal
GRU, 512 hidden units, two recurrent layers, autoregressive action head,
approximately 2.901 million parameters, 264 input channels. BPTT unroll 80,
overlap **0**, batch 128, active dropout 0.1, causal label delay 0, player styles
with anonymous ID 0 reserved. Preserve F32 tensors **and** highest EXLA arithmetic.
The Peppi source cannot supply projectiles; its projectile block stays disabled.

## Evidence and source state

Evidence root: `eval_runs/0915_fox_v3_preflight/`.

- `styled_seed_905/`, `styled_seed_906/`: the matched two-epoch trials that cleared
  the gates (highest arithmetic, live style tags), with `parity.json`, `heldout.json`,
  `launch.json`. `matched_seed_*` = same before the tag fix; `verified_seed_905` =
  09-15 run of unrecorded arithmetic. All kept, none promoted.
- `verified_sources/provenance.json`, source archives and patches: all four repos.
  These are shared dirty working trees, not clean release commits. Preserve other
  Fox/Mewtwo work. Do not reset, clean, or silently update dependencies.
- `corpus.json`: frozen short-trial train/validation manifest with source hashes.
- `full_corpus.json`: eventual full-corpus inventory, when completed.
- `repairs/expanded_contracts_fixed.log`: expanded regression results.
- `live/`: sampled local-play args, logs, session reports and finalized replays.

Do not use `seed_905/`, `repaired_seed_905/`, or `highest_seed_905/` as completed
training evidence. They preserve failed or deliberately interrupted attempts.
The earlier `repaired_seed_905_gpu70/` completed under default GPU arithmetic
and the older colliding style registry; it is superseded for the launch decision.

## Launch recipe — prepared, not executed

Run only after preflight GO and after verifying that source/corpus changes since
the recorded evidence do not invalidate it. Confirm disk space, no existing
training/Dolphin session, and no competing GPU job. The 0.15 GPU allocation used
for play cannot fit this trainer; use 0.70 on this 32GB RTX 5090.

```bash
cd /home/blewf/git/exphil
train_run="$PWD/checkpoints/fox_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$train_run"
systemd-run --user --unit=exphil-v3-full --collect \
  --working-directory="$PWD" \
  -p "StandardOutput=append:$train_run/train.log" \
  -p "StandardError=append:$train_run/train.log" \
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 \
  EXPHIL_EXLA_PRECISION=highest mix run scripts/train.exs \
  --backbone gru --temporal --stage-internals \
  --hidden-sizes 512,512,256 --batch-size 128 --dropout 0.1 --precision f32 \
  --bptt --unroll 80 --bptt-overlap 0 --bptt-val-files 16 \
  --learn-player-styles --stream-chunk-size 200 \
  --replays replays/erickfm_ranked/v2_filtered \
  --train-character fox --select-character-port --label-delay 0 \
  --player-tag-map eval_runs/0917_style_identity/player_tag_map.json \
  --epochs 8 --seed 905 --head autoregressive --save-best \
  --save-every-batches 10000 --label-smoothing 0.0 --no-focal-loss \
  --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 \
  --entropy-weight 0.0 --neutral-weight 1.0 --stick-edge-weight 1.0 \
  --no-register --checkpoint "$train_run/model.axon"
```

Record the resolved path and actual service exit status. Do not launch this a
second time because the terminal returns: `systemd-run` detaches deliberately.
Do not add gradient accumulation, mixed precision, overlapping unrolls, previous
actions, a different data source, or a specialist warm start to this recipe.
Any such change needs its own relevant checks.

## Monitoring and recovery

- Watch finite train/validation loss, optimizer steps, memory, disk and saved
  checkpoint modification times. The progress bar's estimated batch count is
  inaccurate for padded BPTT; it is not a reliable ETA.
- Each replay chunk must finish parsing/embedding, including the small final
  chunk. BPTT embedding caching is disabled to avoid filling the filesystem.
- Check periodic checkpoints, best checkpoint **and best policy**, final
  checkpoint/policy, config JSON and `model_players.json`. Do not rename a best
  policy into a final-policy filename and lose its provenance.
- For a controlled stop, send **SIGTERM to the training BEAM process**, allowing
  its current batch and checkpoint/export cleanup to finish. The subprocess
  probe and real full-size interruption passed. SIGKILL/power loss cannot save
  an in-flight update; atomic checkpoint publication protects the previous file.
- Resume from an `.axon` checkpoint, not an inference `.bin`. Use a fresh output
  directory, the frozen source/corpus/recipe, and `--resume <checkpoint>` plus an
  explicit `--checkpoint <new-directory>/model.axon`. Preserve earlier bests.
  Weights, optimizer and dropout state restore; loader cursors, recurrent carry,
  epoch progress and best-validation history do not. The resumed fit starts the
  data at a fresh boundary; `--epochs N` means N additional epochs. Count actual
  completed epochs and repeated partial-epoch exposure in the report.
- A completed fit is not promotion. Run export parity, held-out evaluation and
  sampled live play on the chosen best artifact before calling it playable.

## Deployment semantics

Use `scripts/play_dolphin.exs`, `--stateful-step --live-af --reaction-delay 0
--temperature 1.0`, and `EXPHIL_EXLA_PRECISION=highest`. Local reaction delay 0
means the normal next-frame controller application (measured latency 1), not a
modification that removes Melee's native mechanics. Do not use `--deterministic`
to conceal weak sampled behavior. Anonymous style is ID 0; selecting a named
style also requires the matching `--player-registry` JSON.

## Limits to retain in the final report

Two small-data epochs test the harness, not generalist strength. A 16-game
validation set is a narrow training monitor, not proof across characters/stages.
The 16 validation games carry NO player tag (09-17 inventory), so anonymous and
registry-conditioned held-out scores on the trained artifact will be identical —
expected; conditioning liveness needs a separate tagged eval sample
(STYLE_IDENTITY.md, "Name-channel verification still owed").
Byte hashes detect identical files, not different encodings of the same game.
Masked tails preserve coverage but produce variable valid-frame counts per
optimizer step. Full-corpus timings and learning behavior may differ from the
bounded subset. Report actual training duration and live behavior. The measured estimate is
≈ 6.3 h/epoch, ≈ 50 h for 8 epochs (FOX_V3_PREFLIGHT.md, "Measured cost
estimate"; parse-bound, subset-scaled); do not promise strong play in advance.
