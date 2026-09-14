# Combined multishine evidence report

The report reads existing artifacts with Python's standard library. It does
not import ML libraries, run Mix, poll continuously, or start Dolphin.

```bash
python3 scripts/multishine_readout.py \
  benchmarks/multishine/g26_readout.json /tmp/g26-readout-new.json
```

Use a new output filename each time. The supplied manifest follows the queued
G26 readout's actual selections: maps for final/30/45, stand for 40/50/60, CPU
for 45/60. It intentionally does not attach one epoch's map to another epoch's
gameplay. Missing files are pending; files changing during reading are marked
changing; malformed JSON is invalid. Already-readable logs can still be
incomplete while their producer is running. Re-run after that producer ends.

All paths inside the manifest are relative to the repository root (override
with `--root`). Top-level fields are `version: 1` and a nonempty `checkpoints`
list. Each checkpoint needs a unique `id`, `policy`, and an explicit integer
`epoch` for gate tables. Optional evidence paths are:

| Key | Input and interpretation |
| --- | --- |
| `audit` | Pool auditor text; retains shift/conflict/ambiguity excerpts |
| `coverage` | Coverage-map JSON; policy, baseline, weakest cells, offset and delay-id |
| `stand` | `sweep_table.txt`; only matching `epN:` rows, with the original reported rate |
| `cpu` | `cpu_table.txt`; only matching epoch's per-run frames, counts, self/min and chain |
| `corrections` | Scenario-suite JSON; summary, divergences, errors and delay fields |
| `recovery` | Replay-benchmark JSON; matching policy's per-run recovery events |
| `execution` | Runner log; latency/history/handoff/error excerpts |

Coverage and correction artifacts naming a different policy are rejected as a
mismatch. Duplicate CPU run IDs or duplicate stand rows are ambiguous, so an
appended rerun cannot masquerade as independent confirmation. Unsupported table
formats remain visible as no matching epoch. No missing section is interpreted
as passing. The report has no aggregate score or automatic ranking.

Input artifacts and checkpoint files are hashed as read. Historical artifact
formats generally name a policy path without the policy's creation-time hash;
therefore the report cannot prove that a reused checkpoint filename produced
an older result. Likewise, execution-log excerpts are evidence to review, not
an automatic assertion that latency was correct or history was warmed. A log
containing an audit summary is not proof its producer exited successfully.

This is a practical evidence inventory, not completion of R5's full evaluation
identity system. The remaining durable fix is for every producer to emit its
checkpoint/content identities, protocol, exit status, and control results.

See [the real-game collection plan](../planning/MULTISHINE_COVERAGE_ROUND.md)
for the next experiment.
