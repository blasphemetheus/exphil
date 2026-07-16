# Scenario Evaluation Suite

Targeted "what does the policy do HERE?" probes (task #18). Instead of
scoring whole games, the suite drops the policy into curated pathology
moments mined from real probe games and scores its first response:

| Type | Situation at handoff | Pass criterion (default window 300f) |
|------|---------------------|--------------------------------------|
| `opponent_behind` | Fox within 30 units behind Mewtwo's back, both grounded | turned to face (or up/down-smashed) within 60f |
| `tech_chase` | Fox enters knockdown (missed tech or tech) | within 25 units at Fox's first actionable frame AND attack/grab within 30f after |
| `edgeguard` | Fox offstage on FD, Mewtwo onstage | moved ≥ 10 units toward the ledge within 60f (+ score bonus if Fox loses the stock) |
| `getup` | Mewtwo knocked down (missed tech) | tech or getup option within 40f of first actionable frame |
| `idle_deadlock` | both characters in Wait ≥ 90f | left Wait within 300f AND moved ≥ 5 units toward Fox |

Scores are graded 0..1 (faster/cleaner responses score higher), so
checkpoints can be ranked even at equal pass rates. The scoring math is
pure and unit-tested: `ExPhil.Eval.ScenarioScore`.

## How it works: virtual savestates (input-prefix replay)

Dolphin savestates are build-version-locked (GUI netplay 3.6.4 vs headless
ExiAI 3.5.1) and cannot be created headlessly. Instead, the suite replays
the RECORDED inputs of BOTH ports from game start up to the handoff frame
— Melee is deterministic enough for Mewtwo-vs-Fox on FD (no items, no
hazards) that this reaches the recorded state exactly in most cases — then
hands port 1 to the policy.

**Drift check:** at the handoff frame the live P1/P2 x/y/action are
compared against the replay's recorded values (tolerance ±3 units,
actions must match modulo the shield family 178–182). A mismatch marks
the run DIVERGED — it is reported and excluded from the summary, never
scored silently. The curated manifest replays 15/15 with drift exactly
(0.0, 0.0) on both ports.

**Frame/input alignment:** libmelee reports frame `f` (post-simulation);
the inputs sent afterwards are consumed for frame `f+1`. So after seeing
frame `f`, the suite sends the replay's *pre-frame inputs of frame `f+1`*
(`--input-offset`, default 1 — validated by the exact-zero drift checks).
Peppi already normalizes recorded sticks to the bridge's 0..1 range and
they round-trip bit-exactly through libmelee.

**Analog-trigger caveat (GOTCHAS #66):** the ExiAI headless build ignores
analog trigger pipe commands, so recorded analog trigger holds ≥ 0.31 are
converted to digital L/R presses. Powershield subtype and shield-slide
micro-physics can differ as a result: prefixes that cross a shield-HIT or
missed-tech interaction may deterministically diverge (2/16 curated
candidates did; they were replaced/dropped). When curating, prefer
handoff moments not preceded by shield-hit interactions.

**P2 after handoff is deterministic, NOT reactive:** the opponent keeps
replaying the source game's inputs — it responds to the ghost of the
original game, not to the policy. Scores measure the policy's *initial
response to a situation*, not a full adaptive interaction. (When the
recording runs out, P2 goes neutral.)

**Agent state at handoff:** the policy's temporal buffer starts cold at
the handoff frame and pads with copies of it — the same regime as a
windowed game start (GOTCHAS #65), so it is on-distribution.

## Scanning for candidates

```bash
devenv shell -- bash -c 'mix run scripts/scan_scenarios.exs \
  --out /tmp/candidates.json ~/Slippi/Game_2026*.slp'
```

Detectors live in `ExPhil.Eval.ScenarioScan` (pure, unit-tested).
Candidates are bounded to ≥ 300 frames in (room for a prefix) and ≤ 2/3
of the game (room for the response window), and same-type candidates are
spaced ≥ 240 frames apart. Curate by hand — read the notes (distance,
entry class, position), don't take the first hit.

The curated manifest is `scenarios/manifest.json`; its source .slp files
are copied into `scenarios/replays/` (committed) so entries reference
stable relative paths. Current coverage: opponent_behind ×4,
tech_chase ×4, edgeguard ×3, getup ×1, idle_deadlock ×3. Getup is thin
because the tech_random dummy never attacks — policy knockdowns are
self-inflicted and rare (4 candidates in 55 games); expect more once a
fixture/dummy that attacks exists.

## Running the suite

```bash
# Deterministic scoreboard for a checkpoint (the probe recipe defaults:
# deterministic + press/release hysteresis 0.45/0.3)
EXLA_MEMORY_FRACTION=0.15 devenv shell -- bash -c \
  'mix run scripts/scenario_suite.exs \
     --policy checkpoints/mewtwo_combo_newera_r10_policy.bin'

# Subsets / distributional scoring
...scenario_suite.exs --types tech_chase,getup
...scenario_suite.exs --only 0,5,9
...scenario_suite.exs --runs 5 --temperature 0.8   # sampled; runs>1 is
                                                   # pointless when deterministic
```

One headless Dolphin boots per run (`--slippi-port` base 51480, +1 per
run; per-run `--replay-dir` under `logs/scenario_runs/<ts>/`). A full
15-entry deterministic pass takes ~6 minutes wall (~17–39 s per run,
dominated by prefix length — headless runs ~6-7x realtime).

Output: per-run lines (score, pass, drift, details), per-type summary,
and a JSON scoreboard at `logs/scenario_scores_<ts>.json`:

```json
{
  "summary": {"tech_chase": {"runs": 4, "pass_rate": 0.0, "mean_score": 0.125, ...}},
  "runs": [{"type": "tech_chase", "score": 0.5, "pass": false,
            "drift": {...}, "details": {...}, "p1_actions_rle": [[14,30],...]}],
  "diverged_runs": 1
}
```

`p1_actions_rle` is the policy's response window as `[action, run_length]`
pairs — read it to see *what it actually did* without re-parsing replays.
Diverged runs are counted and carried in `runs` but excluded from
`summary`.

## Composing with the report card

The report card (`ExPhil.Interp.ReportCard`) gates full-game mechanics
health (jump discipline, shield occupancy, deadlocks); the scenario suite
scores situational competence. They answer different questions and
complement each other in round selection: a checkpoint should pass the
report card's hygiene gates AND raise scenario pass rates. Both emit
loop-consumable JSON, so a training loop can select on e.g.
`report_card.passed >= 6 and scenarios.summary.tech_chase.mean_score`.

## Debugging divergence

- `--trace-all` records per-frame drift + live/ref actions during the
  prefix (`drift_trace` in the JSON); `first_drift` marks where each port
  first exceeded 0.5 units. Frame-alignment bugs show up at frame ~0;
  event nondeterminism shows up at a specific interaction.
- `--input-offset N` changes the recorded-input frame offset (default 1).
- The run's own .slp lands in its `--replay-dir`, but is truncated
  (unfinalized) if Dolphin is killed mid-game — use `--trace-all` instead
  of trying to parse it.
- r10 baseline (2026-07-16, `logs/scenario_scores_r10_baseline.json`):
  opponent_behind 4/4 pass (mean 0.715), idle_deadlock 3/3 (1.0),
  getup 1/1 (1.0), edgeguard 1/3 (0.333), tech_chase 0/4 (0.125) —
  matches the known r-line profile (turnaround fixed, tech chase still
  the weakness).
