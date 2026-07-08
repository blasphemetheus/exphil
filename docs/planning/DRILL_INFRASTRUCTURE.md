# Drill Infrastructure Roadmap

Status of the scripted-expert DAgger drill system and the infrastructure worth
building around it. Born 2026-07-08 out of the multishine probe (see
`scripts/dagger_drill.exs`, `lib/exphil/agents/multishine_expert.ex`,
HANDOFF_2026-07-08.md).

## The core insight

**Anything a human can record, we can drill.** A drill expert is built FROM a
recording: table of modal inputs per game-state key + hand-written recovery
rules for off-script states. Nobody needs to write down frame data — the
fixture IS the spec. Slippi replays of live play ARE on-policy DAgger
rollouts, so an iteration is just: play → relabel with expert → retrain
(`scripts/dagger_loop.sh`).

Validated on Fox multishine: jump-cancels per solo game across iterations
15 → 39 → 69 → 119 → 172. Key lessons baked into the protocol:

- Prev-action channel must carry the policy's ACTUAL press, not the expert's
  correction (`:prev_controller` override) — else recovery is unlearnable.
- Recovery taps must alternate against the previously-landed input (press
  EDGES; Melee ignores held buttons). Never key alternation on action_frame —
  it freezes on repeated frames and its embedding saturates (af/60 cap 2.0).
- Button hysteresis decode (`--press-threshold/--release-threshold`)
  complements this at inference time.
- val_loss is meaningless across drills/iterations; score from replays
  (`scripts/trace_multishine.exs`, `scripts/trace_mewtwo_fair.exs`).

## Drill menu

### Built / in flight
| Drill | Expert | Status |
|---|---|---|
| Fox multishine | `MultishineExpert` (table + JC/shine recovery) | iterating, 172 JC/game |
| Mewtwo fair chains (SH/FH/DJ fair + L-cancel) | `MewtwoFairExpert` (physics-keyed table: jumps_left + height bucket) | drafted, iteration 1 pending |

### Next up (roughly ordered)
1. **Wavedash / waveland** — Mewtwo has a top-tier wavedash; rhythmic, solo,
   FD-only. Cheapest next drill; forces the DrillExpert behaviour extraction.
   Most transferable drill (same skill for every character).
2. **DJC aerials** — Mewtwo's double jump cancels into aerials; the fast
   version of its offense. Timing-critical → exercises delay-2 + prev-action.
3. **Teleport ledge-cancels** (Taj warps) — up-B onto platform edges cancels.
   Needs Battlefield → first stage-geometry-aware drill.
4. **Under-stage teleport / ledge mixups** — ledge-drop → DJ → teleport
   across/under; Confusion ledge shenanigans (needs dummy for the grab
   variants). User records the fixture — exact mechanics don't need to be
   known in advance.
5. **Shadow ball management** — charge, JC-charge, B-reverse, wavebounce.
   Teaches a resource concept (charge state) nothing else does.
6. **Ledgedash (galint)** — frame-perfect; save as the stress test of the
   whole delay/prev-action stack.
7. **Tech-chase vs random-tech dummy** — first REACTION drill (reads the
   opponent) rather than rhythm drill. Gated on dummy infra.

## Infrastructure worth building (priority order)

1. **Scripted port-2 dummy** ⏳ NEXT (after Mewtwo fair proves out —
   user-confirmed order). libmelee can drive both ports; a scripted dummy
   (stand / shield pattern / jump pattern / random tech / walk) unlocks all
   interactive drills: shield pressure, tech chases, Confusion grabs,
   fair-combo at rising percent. Touches `priv/python/melee_bridge.py`
   (port-2 controller), bridge config, play scripts (`--dummy MODE`).
2. **DrillExpert behaviour + registry** — two experts share ~70% structure;
   the third justifies extraction: `from_fixture/2`, `label/3`, `score/1`,
   name→module registry so `dagger_drill.exs`/`dagger_loop.sh` never need
   editing for new drills. Include a fixture-quality gate (refuse to train
   below a button-agreement bar — catches bad recordings early).
3. **Stage geometry constants** — platform heights/edges for BF/YS/DL so
   table keys and recovery rules can be platform-aware (drills 3-4).
4. **Position decorrelation** — solo drills live at center stage; a scripted
   random-movement prelude (~5s per rollout) spreads states across the stage.
5. **Scoreboard JSONL** — trace scripts append structured rows
   (drill, iteration, metrics) for cross-iteration plots instead of log greps.
6. **Curriculum mixing** — the end-game: drill datasets as weighted auxiliary
   data in main training (`train.exs --mix-drills`), so E2-class models get
   drilled execution AND replay-corpus decisions. Where drills hand off:
   execution comes from drills, decisions (when to SH vs FH vs DJ) come from
   the general corpus.

### Smaller QoL
- Fixture recorder helper: launch a 2-human-port recording session, then
  auto-copy newest replay into `test/fixtures/replays/` with a census
  printout (ports, characters, action states).
- Multi-fixture experts: `from_fixture/2` accepting globs — several
  recordings average out human timing variance.
- Hitlag awareness: when drills connect (dummy era), hitlag shifts timing;
  consider an in-hitlag key feature.

## Operational notes

- One drill per day per replay dir, or pass ROLLOUTS explicitly —
  `dagger_loop.sh` auto-seeds from today's >500KB replays and will happily
  ingest another drill's games (or your unrelated matches).
- Never run `mix` while a loop's training beam is live (second EXLA client
  SIGSEGVs it — learned twice). Inference-only sessions coexist fine.
- Loop hygiene: `--on-game-end stop` self-terminates and frees the GPU
  (fixed 2026-07-08); Dolphin opens floating 1280x1056 on Hyprland ws 9.
