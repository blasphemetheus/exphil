# Real-game coverage round after G26

Prepared while Claude investigates the loss guard. This is a collection and
analysis plan; no games or training jobs were launched as part of preparing it.

## Gate before collection

Choose the G26 snapshot after Claude's guard investigation and the existing
coverage/stand/CPU readout. Record its hash, selection evidence, and measured
reaction delay. Retain g23a ep57 as the technique reference and g24a ep55 as the
previous coverage reference. Do not equate the final export with epoch 60.

If G26 does not restore technique, resolve that before increasing rollout data.
If technique returns but opponent-context weaknesses remain, run the following
small real-game matrix. Synthetic mirror/context augmentation is not part of
this experiment.

## Collection matrix

| Factor | Initial values |
| --- | --- |
| Opponent | Fox, Falco, Marth |
| CPU level | 1 |
| Stage | Final Destination |
| Starting side | Fox left of opponent; Fox right of opponent |
| Repetitions | 2 development games per cell per policy |
| Policies | Selected G26, g23a ep57, g24a ep55 |
| Duration | 90 seconds per game |
| Decode | Sampling, temperature 1.0 and buttons temperature 1.0 |
| Physical reaction delay | 4; record the latency probe result |

This is 12 games per policy, 36 games total, 54 minutes of game time plus setup.
Rotate policy order between cells to reduce machine/time ordering effects.
Keep CPU level 3 and a larger roster for a later expansion, after this matrix
shows a reproducible benefit. A port swap does not prove a side swap: record
both players' initial positions and facing, and verify the requested side from
the observed state. Record unexpected assignments and failures explicitly.

Run the normal async runner directly for the requested opponent and port pair;
the old `gate_cpu.sh` fixes the opponent to Fox and is not the varied-character
collector. Use `--reaction-delay 4` and let the checkpoint determine its delay-id.
The current runner exposes `--dummy cpu`, `--dummy-character`,
`--dummy-cpu-level`, `--port`, and `--opponent-port`. Verify startup metadata and
the latency probe before accepting a game's intended conditions.

## Split and provenance

Assign development and confirmation IDs before collecting. Generate another
two games per cell for confirmation after candidate selection, for the selected
candidate and whichever reference its claim is against. Keep those recordings
out of training, snippet mining, expert-table building, and snapshot selection.
Hold out entire replay files, never adjacent windows from the same game.

Each run record needs policy/replay hashes, code identity, runner and decode
settings, intended and actual subject/opponent ports and side, character/CPU
level, measured latency, duration, exit status, and original logs. Record a seed
only if the runner actually controls it. Retain short games, deaths, truncated
recordings, and invalid controls as explicit outcomes. Never silently replace
failed games with successful retries.

## Learning experiment

Use development rollouts from the selected G26 to mine failure and recovery
spans. Preserve complete source segments and `label_source` through exports.
Audit labels at every configured physical training delay, including 4. A
shift-zero audit is insufficient. Enable strict provenance validation at
boundaries that guarantee tagged exports; legacy untagged recordings require
an explicit ingestion decision.

Compare two otherwise identical training arms, with equal update counts:

- Control: the corrected G26 pool, resampled for the same training budget.
- Coverage: retain 50% of sampled windows from the original pool and draw 50%
  from new verified recovery/entry spans, balanced across collection cells.

The 50/50 split is an initial experimental choice, not an established optimum.
Record effective sampling counts; raw file counts are not mixture weights.
Keep the original clean-cycle windows represented in both arms. Do not change
architecture, learning rate, delay set, or loss weighting in this comparison.
Off-loop `label_ahead` currently holds the teacher's present commitment; test
that behavior under the actual delay before treating those targets as verified
future trajectories. The existing twelve correction cases cover a limited
region of gameplay, not every new opponent state.

## Decision rule

Before training, write down a tolerated stationary-technique regression and a
required improvement in moving-opponent re-entry, using fresh reference-run
variability to choose meaningful margins. Do not invent those thresholds after
seeing the candidate. Report all cells and runs, including readiness censoring,
stock losses, and unsupported/unverified states. A gain against one opponent
does not establish general coverage.

Use the combined evidence report to identify missing measurements for each
exact checkpoint. Promote only after fresh confirmation shows the recovery
gain while retaining the agreed technique floor. Bradley's local session is
the final practical check.
