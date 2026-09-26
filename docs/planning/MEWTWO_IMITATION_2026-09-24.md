# Mewtwo full-match imitation → PPO

## September 24 human playtest: recovery failure

User reports frequent self-destructs and poor ability to take stocks. Scanned
the two saved games under `eval_runs/mewtwo_live_20260924_112353` with the
existing `sd_scan.exs` (CPU-only, no training launched): bot port1 lost8 stocks;
7 had no detected hit in the previous90 frames. Opponent lost1 stock, also
flagged by that heuristic. Zero bot deaths matched the scanner's walk-off
definition. These are heuristic SD labels, not verified causes; a missed
recovery after an older hit can also qualify. Report:
`eval_runs/0924_mewtwo_il/live_sd_scan.md`.

Recommended next experiment: verify live/sim Mewtwo recovery behavior, then
PPO emphasizing varied recovery starts alongside full matches, with stock-loss
and stock-taking metrics. Do not optimize against only an equally self-destructive
Mewtwo prior. Allow a looser imitation anchor while retaining stable PPO updates.
Current `ppo_r3.exs` still hardcodes Fox/Fox; it requires explicit Mewtwo setup
and a newly fitted critic before use. No Mewtwo PPO has been launched.

User request, September 23: build the highest practical scale Mewtwo replay
imitation model, then use it as the PPO base. Permit more departure from a
weaker Mewtwo prior than from the stronger Fox prior, subject to measured
improvement. This supersedes the old drill-only Mewtwo scope for this campaign.

## Live state

**Completed September 24, 01:03 CDT.** Early stopping ended training after
12 epochs / 26,388 optimizer updates, following six epochs without validation
improvement. Best validation loss: 3.1043; final train/validation: 2.311/3.1513.
Best playable policy exists at the intended path below. Registered run:
`golden_tipper` (`U0rAWUETOQo`). Training and its status watcher have exited;
PPO, untouched-test scoring and Dolphin evaluation have not run.

The selected epoch-6 policy is now independently registered as
`mewtwo-gru-il-v1-best-ep6`, with policy/trainer SHA256 hashes, corpus manifest,
selection epoch and the training-run ID. The original `golden_tipper` entry
continues to describe the final epoch-12 run checkpoint.

See [automatically updated live status](MEWTWO_LIVE_STATUS.md) for the current
phase, epoch, loss and running process record. Preflight passed at 23:50 CDT:
250 updates, train loss 4.2093, validation 4.343, saved policy; full fit launched.

Launched September 23 at 23:48 CDT. Unit: `exphil-mewtwo-il-v1`.
Machine-readable current phase/PIDs/log: `eval_runs/0924_mewtwo_il/status.json`.
The supervisor runs a preflight, then a full fit only after a finite validation
result and a successful policy export. It records failure rather than proceeding
after a bad preflight. No automatic PPO launch or production promotion.

- Preflight log: `eval_runs/0924_mewtwo_il/preflight/train.log`.
- Full-fit log: `checkpoints/mewtwo_il_v1_20260924/train.log` (created when started).
- Intended playable result: `checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin`.
- Supervisor: `scripts/mewtwo_campaign.py`; exact commands saved in each run's `launch.json`.
- `systemctl --user status exphil-mewtwo-il-v1` and the phase log show progress.
- Stop the unit with `systemctl --user stop exphil-mewtwo-il-v1`; prefer a saved
  periodic/epoch checkpoint for resume and inspect the log for shutdown saves.
- Do not run Mix or edit training library files while this staged job is active.

## Corpus and split

CPU inventory inspected 210,184 local replay paths across Mewtwo, HuggingFace,
ranked/partner, Greg and Yeti collections. Found 381 distinct Mewtwo file hashes;
two unparseable files were recorded. Public source: the MEWTWO directory of
`erickfm/slippi-public-dataset-v3.7`; 261 files (~694 MB), revision pinned in
`replays/mewtwo_public_20260924/source_manifest.json`, every download SHA256-checked.

Combined audit: 773 relevant/error rows → 426 unique candidates → 418 kept.
Require two human players, competitive stage, at least 3,600 frames; existing
deep quality filter requires damage, sane percents and a stock-result signal.
Eight candidates failed its `no_winner` check. Rejections and duplicate aliases
remain recorded, and source files are unchanged.

| Split | Games | Frames |
| --- | ---: | ---: |
| Training | 334 | 4,204,223 |
| Validation | 42 | 515,654 |
| Test | 42 | 524,543 |

`eval_runs/0924_mewtwo_il/corpus.json` records exact source paths, aliases, hashes,
ports, metadata, and splits. Split seed 924; whole games only; identical hashes
and exact match signatures cannot cross splits. Train/validation are symlinked
into `corpus/` with validation lexically last, matching the trainer's actual
BPTT split. Test files live in a separate directory excluded from training.
All six competitive stages are represented. Players/sessions may overlap splits;
this is a game-held-out measurement, not unseen-player validation.

The preflight uses 20 training-split games (16 fit, 4 temporary validation),
leaving full-run validation and test games untouched. Full training starts fresh.

## Model and training

GRU, 2 recurrent layers, hidden size 1024; hidden sizes `1024,1024,512`,
autoregressive controller heads, f32, highest EXLA arithmetic precision,
stateful BPTT unroll 80, batch 64, stream chunks 64, dropout 0.1.
AdamW recipe: LR 1e-4 constant, weight decay 0.05, gradient norm cap 0.5.
Plain imitation loss settings follow the repaired Fox recipe: causal label
delay 0, no focal loss/smoothing, unit neutral/stick weights, no oversampling.
`--select-character-port` explicitly selects Mewtwo demonstrations, including
when Mewtwo is not port 1. No inferred player-style conditioning.

Full fit: up to 30 epochs, early stopping after 6 non-improving validation epochs,
best validation export retained, periodic trainer checkpoint every 2,000 updates.
GPU reservation 80% of the local 32 GB card. This is a larger model candidate,
not an established optimal size; preflight checks feasibility. Data coverage,
generalization and eventual real-time inference remain practical limits.
The normal training completion callback registers the run in the model registry.

Known input limitation: the current replay parser does not provide projectile
observations, so that block is disabled consistently in training/inference.
This matters to Mewtwo/Shadow Ball and projectile matchups. No claim of complete
Mewtwo observation coverage or simulator parity is made by launching this fit.

## Gates before PPO

1. Finish imitation; select by validation, score the untouched test once.
2. Check stateful inference speed and train/live embedding parity; play full
   matches in Dolphin including recovery, disadvantage and varied opponents.
3. Verify Mewtwo-specific simulator mechanics relevant to the policy, especially
   teleport, double jump and projectiles, against Dolphin.
4. Fit a critic for the new Mewtwo trunk. Fox's critic and head-only checkpoint
   are not a drop-in Mewtwo prior.
5. Compare conservative vs looser prior-KL settings, using separate evaluation
   seeds/opponents. A looser imitation anchor and PPO's per-update clipping/KL
   guard are different controls: do not disable update stability to seek novelty.
   Consider trunk adaptation after establishing a head-only baseline.
6. Accept larger behavioral changes when held-out/Dolphin play improves;
   reject simulator-only gains, opponent overfitting and reward loopholes.

## Audit checks and tooling

The initial full-parser metadata scan was replaced by a header prefilter because
Peppi's metadata NIF parses all frames. Offsets come from installed Peppi 2.1.2
source; unknown formats fall through to full parsing. Compared with full Peppi
metadata on 161 sample files: 159 exact classifications, 2 corrupt files.
Unicode player tags initially exposed `IO.write`'s latin1 file device; switched
the JSONL writer to `IO.binwrite` and reran the complete scan successfully.

Scripts: `mewtwo_download.py`, `mewtwo_corpus_audit.exs`,
`mewtwo_build_corpus.py`, `mewtwo_campaign.py`. Audit logs originated in
`/tmp/exphil-mewtwo-*.log`; corpus manifests and training logs are persistent.
