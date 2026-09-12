# Multishine benchmark, recovery measurements, and teacher audit

The first version is an **offline replay benchmark**, not a Dolphin launcher.
It does not train, change policies, kill processes, or collect new games.

```bash
mix run scripts/benchmark_multishine.exs \
  benchmarks/multishine/0911_pilot.json /tmp/new-multishine-report.json
```

Use a new output path: existing reports are never overwritten. The committed
pilot output is `eval_runs/0911_benchmark/pilot.json`. Checkpoint and replay
binaries must exist locally; the manifest deliberately fails if they are absent
or differ from their frozen hashes.

## Frozen inputs

The JSON manifest supplies `version: 1`, `teacher_fixture`, its
`teacher_fixture_sha256`, a `protocol`, and nonempty `runs`. Paths are relative
to the manifest. Protocol fields are runner, frame_delay, delay_id, temperature,
buttons_temperature, stage, and seconds (a number or a map by scenario).
Each run has a unique id, scenario (`stand`, `cpu`, or `human`), explicit Fox
subject port, policy/replay paths, and both SHA-256 hashes. Duplicate replay
contents are rejected even when their paths differ.

The report includes source, manifest, fixture, checkpoint, and replay hashes.
Protocol settings are **declared, not independently verified** from the replay.
Hashing a policy does not prove it produced the replay. Keep original execution
logs with future runs; CPU level, opponent identity, stage, runner health, seed
when supported, and collection settings belong in the protocol record. Null seed
in the historical pilot means unknown, not reproducible game randomness.

This is not yet R5's comprehensive evaluation identity/collection system. Do not
compare reports with differing protocol or metric/source identities as though
they are interchangeable. The runner reports individual games, not a leaderboard
or an automatic promotion decision.

## Metrics

- Grounded shine onsets per observed gameplay minute, plus self-initiated versus
  recent-hit onsets. Recent-hit means reported hitstun within the preceding 30
  frames; it is an association, not proof the hit caused the shine.
- ShineChain v3 lengths, maximum chain, and ending categories (`empty_hop`,
  prolonged airborne shine, aerial jump, other action, end of input).
- Completed cycles: two grounded reflector segments connected by a valid v3
  bridge with an observed aerial reflector. One isolated shine or a jumpsquat
  without an aerial shine never counts as successful re-entry.
- Stock losses, observed versus expected duration, and a short-recording flag
  below 80% of expected duration. Short games remain visible, not discarded.
- Individual recovery episodes after reported hitstun, leaving the loop, or
  returning from a failed bridge. Each records its start, readiness, outcome,
  and successful completion frame, or last observed frame when censored.

**Readiness is a conservative proxy**, not a verified first-actionable frame:
grounded locomotion (actions 14–23) or grounded reflector, with zero reported
hitstun. Report both disruption-to-cycle and ready-to-cycle durations. This
excludes pre-readiness time from the latter but does not establish exact input
acceptance, hitlag, airborne recovery, or animation lockout. Do not describe it
as exact reaction latency. Failed bridges detected on landing start at that
landing; long airborne time is not retrospectively counted in those episodes.

Gaps and stock losses reset chain state. Hits cannot continue a chain. Repeated
interruptions, deaths, gaps, and end-of-replay censor unfinished episodes rather
than silently dropping them or recording zero latency. Durations of successful
episodes alone are selection-biased: always read completed and censored counts
together. Never treat an end-of-replay chain as an observed technique failure.
Frame numbering, not wall-clock replay playback speed, defines durations (60 Hz).

These definitions differ from older scripts: pregame frames are excluded,
hitstun interrupts chains, and total versus self-initiated onsets are separate.
Do not directly compare the new rate with old `self/min` or old chain records.

## Teacher validation: what this does and does not prove

The audit builds the expert from the exact frozen fixture. For each contiguous
same-stock successor pair, it compares the issued-input teacher label's seven
main buttons with the next recorded controller. The current raw controller is
the most recently landed input supplied to the expert's recovery rules. No
training-delay shift is applied: this audits the base issued-input teacher,
not the drill's full delayed/queued target construction.

It records fixture-table versus fallback coverage, disagreement by
`{action, action_frame, grounded}`, fallback labels during hitstun, and the
first 100 fallback/disagreement examples per run with frame numbers.
These examples can be located in the source replay for manual review.

Disagreement on policy rollouts is expected for DAgger. Agreement is not evidence
that a correction is good; fallback coverage is not a failure rate. Analog-stick
agreement is not scored. The audit does **not** execute counterfactual teacher
inputs and cannot establish a correct closed-loop teacher.

## Next experiment: preregister before collecting

1. Freeze g23a ep57 and g24a ep55 as references. The seven-game historical pilot
   is development evidence, not held-out confirmation; it helped select snapshots.
2. Collect a balanced set of fresh stand and moving-CPU games per reference,
   with identical protocol and independent repetitions. Keep failures and health
   logs. Do not pick the winner on the same games used to confirm it.
3. Add human interruption sessions: approach, hit, knock airborne, and interrupt
   near landing. Record opponent and intended scenario rather than inferring them
   from a filename. Hits themselves are not technique failures.
4. Review sampled failure states and labels. Test the scripted teacher from
   matched controlled states, including previous-input history and deployment
   delay, before accepting its corrections as recovery demonstrations.
5. Only then compare a fixed clean-loop/recovery mixture with an increased
   verified-recovery mixture. Keep architecture, optimization budget, and source
   data otherwise fixed. Gate moving-opponent recovery and stationary precision
   together. Choose quantitative acceptance thresholds before inspecting results.

## Implementation status

- [x] Integrate R1/R2 and rerun their CPU/native contracts.
- [x] Frozen replay manifest and machine-readable per-run benchmark.
- [x] Recovery timing with explicit readiness proxy and censoring.
- [x] Teacher coverage/disagreement audit and real-fixture floor tests.
- [x] Retrospective g23a/g24a replay pilot.
- [ ] Fresh balanced held-out gameplay collection with execution provenance.
- [ ] Closed-loop teacher validation on controlled failure states.
- [ ] Exact controllability/first-actionable-frame instrumentation.
