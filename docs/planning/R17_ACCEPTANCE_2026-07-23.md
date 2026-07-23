# r17 Acceptance Battery (pre-registered 2026-07-23)

Pre-registered BEFORE launching r17 so the run answers a question instead
of inviting interpretation. r16's verdict: BC did NOT move gate-10 (armed
approaches/min 0.17, target ≥1.0) — initiation is not coming from more
imitation. r17 is the ESCALATE branch, and this session built the levers
to attack initiation directly. This doc states what "r17 worked" means,
in numbers, and what each recipe must record.

## The one question r17 answers

**Does targeting the going-in DECISION (opener-weighting + curated
go-in data + a live probe-reg + style separation) move gate-10 off the
floor?** r16 lifted conversions but not initiation; r17 either moves
armed-approaches/min or tells us imitation-shaping is exhausted and the
next step is RL/PPO approach-shaping (the other r17-plan branch).

## Primary gate (decides the branch)

| Metric | r16 (baseline) | r17 PASS | r17 STRONG |
|--------|----------------|----------|------------|
| **armed approaches/min** (report_card / coach_report, mean over the probe fan-out) | **0.17** | **≥ 0.5** | **≥ 1.0** (the standing target) |

- **≥ 0.5** = the levers moved the needle → imitation-shaping is a live
  path; iterate (heavier opener-weight, more curated data).
- **≥ 1.0** = initiation essentially solved from imitation → fold in #38
  conversion seeds, move to matchup breadth.
- **< 0.5 (still ~floor)** = imitation-shaping is exhausted → escalate to
  PPO approach-shaping (prereq: the PPO smoke, still UNRUN). This is the
  decisive negative result, and it's worth just as much as a pass.

Read the primary gate from the SAME source r16 used: the launcher's
report_card fan-out (`[FAIL] armed approaches/min` lines) AND
`coach_report.exs` armed/min (they cross-validated exactly on r16).

## Secondary gates (health — must not regress while chasing initiation)

Pushing initiation can trade away hygiene; these guard against it. From
the StyleCard (`ExPhil.Interp.StyleCard`, mewtwo pack) + report_card:

| Gate | r16 | r17 must hold |
|------|-----|---------------|
| report-card avg (10 gates) | 6.8/10 | ≥ 6.5/10 |
| SDs per game | ≤ 2 obs | ≤ pack.sd_max |
| shield occupancy % | ~6% (FAIL vs ≤4) | not worse than r16 |
| conversions (total, fan-out) | 6/119 (5%) | ≥ r16 rate (don't lose the r15/r16 gain) |
| **opener entropy (bits)** (NEW gate C2) | 0.501 (FAIL, <1.0) | **↑ toward 1.0** — more varied neutral |
| **top-opener share** (NEW gate C2) | 0.914 (FAIL, >0.6) | **↓ toward 0.6** — less one-note |

The opener-entropy / top-share gates are the direct readout of the Fox
"couple different things in neutral" goal; for Mewtwo r17 they're a
diversity check (the bot was 91% one opener category on r16).

## Scenario-suite gates (situational competence)

Run `scenario_suite.exs` on the r17 best checkpoint (deterministic,
probe recipe). r16-era r-line profile for reference (r10 baseline):
opponent_behind 4/4, idle_deadlock 3/3, getup 1/1, edgeguard 1/3,
tech_chase 0/4. r17 must **not regress** any type's pass rate, and the
initiation levers should specifically help:

| Scenario type | must hold | initiation-lever target |
|---------------|-----------|-------------------------|
| tech_chase | ≥ r16 mean_score | ↑ (opener-weighting helps go-in after knockdown) |
| idle_deadlock | ≥ r16 pass rate | break deadlocks FASTER (armed approach out of Wait) |
| edgeguard | ≥ r16 | — |
| opponent_behind / getup | no regression | — |

## Data-composition stamp (every recipe records this)

Each r17 recipe MUST log and archive its data composition so results link
to what the data contained (r16 couldn't — random BC sample, dead
probe-reg). Capture in the run's handoff:

- **BC set**: curated (`curate_bc.exs`) vs random; selected count; the
  set's mean armed/min (r16 random ≈ 3.9; curated top-20 ≈ 5.54).
- **Opener weighting**: `--opener-weight` value; openers found; % frames
  upweighted; opener distribution (the "Opener weighting: N openers ...
  dist %{...}" log line).
- **Conversion weighting**: value; % upweighted.
- **Probe-reg**: value + refit stats — **verify `refit: true` with
  shield_rows ≈ 5-6% of the sample** (r16 was silently `refit: false,
  shield_rows: 4` — a dead lever; the fix is validated but CONFIRM per
  run from the log).
- **Style conditioning**: `--learn-player-styles` on/off; distinct
  players in the registry; the `.players.json` path.
- Pool frames, backbone, epochs, LR.

## Recommended r17a recipe (heavier-BC / imitation-shaping branch)

Not a launch instruction (Bradley launches), just the pre-registered
config the gates above assume:

```
--expert mewtwo_combo --backbone mamba_2 --max-epochs 100
--conversion-weight 3.0 --opener-weight 3.0
--probe-reg 0.1 --probe-reg-every 5
--bc-replays "$(curate_bc top-N high-armed/min)" --learn-player-styles
--rollouts <r13/r14 probes + human demos>
```

Deltas from r16: `--opener-weight` (NEW lever), curated `--bc-replays`
(was random `--bc-sample 20`), `--learn-player-styles` (NEW), and the
probe-reg fix makes `--probe-reg 0.1` actually apply. Everything else
matches r16 so the comparison is clean.

## Decision procedure (post-run)

1. Read primary gate (armed/min) from report_card + coach_report.
2. PASS (≥0.5) → record which levers correlated (compare the data stamp);
   plan r17b iterating the strongest. STRONG (≥1.0) → BC+#38 combined.
3. FAIL (<0.5) → the imitation-shaping conclusion is: exhausted. Escalate
   to PPO approach-shaping. Run the PPO smoke first.
4. Either way: run the scenario suite + secondary gates; a hygiene
   regression is a stop-and-fix even if the primary gate passed.
5. Stamp the result into a dated handoff with the full data composition.
