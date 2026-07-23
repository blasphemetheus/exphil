# Data Flywheel Design (2026-07-23)

**Status: Priority 1 IMPLEMENTED (2026-07-23, untested against mix —
r16 held the GPU). Priorities 2-5 still design-only.** Every feature
below includes its implementation spec.

### Implementation log

**Priority 1 (2026-07-23) — DONE, parse-checked + standalone-logic-verified,
NOT yet `mix test`-run (r16 mix-lock):**
- `lib/exphil/eval/gap_ledger.ex` + `test/exphil/eval/gap_ledger_test.exs`
- `lib/exphil/eval/failure_scan.ex` + `test/exphil/eval/failure_scan_test.exs`
- `scripts/auto_bookmarks.exs`, `scripts/coach_report.exs`
- 20 detector/ledger logic assertions verified standalone (pure funcs run
  outside the project — see the scratchpad harness note in the handoff).

Two deviations from the spec below, both deliberate:
1. **Bot-port detection by connect code is NOT possible** as written —
   the native Peppi parser exposes `player.name_tag` (the 4-char in-game
   tag), NOT the netplay connect code (EXPH#288). `coach_report.exs`
   therefore uses `--bot-port` as the robust path (default 1 = probe
   convention) and `--bot-tag` as a name-tag fallback. Extracting the
   netplay code needs a native change (deferred; note in B2 too).
2. **`passivity_window`** uses a stricter *consecutive* close-and-passive
   run (≥300f) instead of "300-of-600 window" — cheaper, and a contiguous
   span is what a drill wants. Documented in the module.

**Priority 2 (2026-07-23) — DONE, parse-checked + standalone-logic-verified,
NOT yet `mix test`-run:**
- `lib/exphil/eval/coverage.ex` + `test/exphil/eval/coverage_test.exs` —
  5-feature situation buckets, occupancy ledger, Laplace-smoothed diff
- `scripts/coverage_ledger.exs` — bot-vs-corpus occupancy diff → coverage
  gaps (source "coverage", slp/frame null, bucket key = type). Supports
  `--corpus-ledger`/`--save-corpus` so the corpus side is computed once.
- `scripts/find_situations.exs` — bucket-grep retrieval: `--like SLP:FRAME`
  / `--bucket KEY` / `--gap ID` → corpus matches (JSON + optional
  manifest), marks the source gap "mined". `--char-filter` via metadata.
- 16 Coverage logic assertions PASS standalone.

**Priority 3 + 4 (2026-07-23 ~03:00-04:15) — DONE, tested + live-smoked:**
- **P3a style tags:** audit found tag→id→embedding fully wired for training
  (`--player-registry`, PlayerRegistry, always-present 112-dim name one-hot)
  but NOT inference. Wired: Agent `style_id`/`style_tag`+registry opts →
  `embed(..., name_id:)` (embed-parity tests green — id 0 default preserves
  existing behavior); `--style-id/--style-tag/--player-registry` CLI flags;
  train.exs persists `<checkpoint>.players.json` after fit.
- **P3b NeutralScan:** `lib/exphil/eval/neutral_scan.ex` (+11 tests) —
  opener taxonomy (jab/dash_attack/tilt/smash/aerial/grab/special; laser+
  projectile fold into :special v1, drift-in dropped — no velocity in slim
  frames), entropy/top-share; StyleCard gains 2 appended gates + pack
  thresholds + `openers` in the result. REAL-DATA VALIDATION: r16 probe =
  35 openers, top_share 0.914, entropy 0.501 bits → both gates FAIL —
  exactly the one-note neutral the Fox program targets.
- **P3c sparring ingest:** `scripts/ingest_sparring.exs` — copy + merged
  session.json sidecar (human/bot ports per game, checkpoint) + nested
  coach_report per bot port. Smoked end-to-end on probe stand-ins.
- **P4 seeded self-play:** `scripts/selfplay_rollouts.exs` — both ports
  live policies from a both-neutral prefix handoff; finalize + seed_meta
  (same contract as build_seed_dir). LIVE-VERIFIED: 6484-frame parseable
  rollout, 27 distinct action states per port in the window (both live).
  RL/PPO consumption only — that arm stays gated on the PPO smoke.
- **P4 preference pairs:** human_drill keeps ≥min-length discards in
  `<out>.discards.frames` with per-attempt list_file/list_index in the
  sidecar (pairs recoverable by slot).
- **P4 uncertainty logging:** Agent `:uncertainty_log` buffers the per-head
  confidences sampling ALREADY computes, JSONL-flushes every 600f + on
  terminate; `--uncertainty-log` in CLI/play_dolphin_async/selfplay;
  consumer `scripts/uncertainty_gaps.exs` (bottom-percentile clusters →
  ledger). LIVE-VERIFIED end-to-end on a self-play run.
- **#38 finalize** also live-confirmed this session (parse OK, Mewtwo SDs)
  — with one caveat observed: `finalize_timeout` from one entry's position
  (the float/DJ save case the #38 note predicted); works from others. The
  unfinalized-run drop path held (0 bad seeds).
- Full sweep after all of it: 100/100 tests.

**Priority 5 (2026-07-23 morning) — DONE, tested + live-validated. The
flywheel build-order table is COMPLETE.**
- The #33 gate dissolved on inspection: P5 needs a memory-flat embedding
  pass, not the dagger training wiring. `ExPhil.Data.SituationIndex`
  (+7 tests) IS that pass — per-file streaming build (parse → embed →
  f16 shard → discard; idempotent extend; `--char` port autodetect) with
  brute-force GPU cosine top-k + temporal spacing on query (spacing added
  after the first live query showed one held situation flooding the
  top-k with consecutive frames).
- **Memory flatness PROVEN** (the DATA_SCALING step-2 criterion, embed
  pass): self-reported peak VmHWM 1.48 / 1.55 / 1.77 GB at 65k / 415k /
  1.77M frames — flat at 27x scale, ~11x under the 19.9 GB all-in-RAM
  point. greg corpus extrapolates to ~2 GB RAM.
- Live: mewtwo archive index (127 files / 1.77M frames / 142s build);
  r16 passivity gap g_54affc1c queried → 40 diverse human instances at
  0.999 cosine across 6+ games → marked mined.
- `scripts/situation_index.exs` build/query (query by `--like SLP:FRAME`
  or `--gap ID`; `--manifest-out` for drill manifests).
- **Still open on the data-scaling side (tracked there, not here):** the
  dagger_drill shard-streaming wiring — compact targets in shards,
  relabel first-pass, shuffle buffer — to be landed WITH a full training
  smoke.

### ✅ VERIFICATION DONE (2026-07-23 02:47-02:48, after r16 finished)

All P1+P2 `mix test`-green and smoke-validated on real r16 probe replays:
- **`mix test` 30/30 green** (fixed one test-only assertion bug: the
  `flip/1` test checked the pre-swap side).
- **auto_bookmarks** on r16 probes: passivity_window 125 / dropped_punish
  42 / death_sequence 3 / neutral_loss 0 → **live Peppi frames DO carry
  stock/percent/facing** (detectors gave sane counts, risk resolved).
  neutral_loss 0 is correct: the tech_random probe dummy doesn't attack.
- **coach_report**: armed/min 0.17 **exactly cross-validates the launcher's
  own report_card** (independent code path) — bot-port orientation confirmed.
- **coverage_ledger**: 76 bot vs 464 corpus buckets, coherent Laplace diff.
  Caveat found: probe games skew the bot's percent low (passive dummy), so
  coverage is most meaningful on SPARRING data, not probe fan-outs.
- **find_situations**: 15 correctly-spaced human retrievals of a bucket.
- Graceful skip on 1 truncated .slp across all scripts.
- Thresholds (dropped 15% / passivity 300f / min_corpus_frac) produced
  sensible counts; no tuning needed yet. `--char-filter` NOT smoke-tested
  (deferred — needs a Fox corpus + `PlayerMeta.character_name` check).

The original pre-run checklist (kept for reference):

The gate-10 verdict comes FIRST (that's the whole run's purpose). Then:

```
# 1. Unit tests — all pre-verified standalone, should be green first try:
mix test test/exphil/eval/gap_ledger_test.exs \
         test/exphil/eval/failure_scan_test.exs \
         test/exphil/eval/coverage_test.exs

# 2. FailureScan real-replay smoke (expect nonzero passivity_window +
#    dropped_punish on a gate-10≈0 bot; neutral_loss on any probe game):
mix run scripts/auto_bookmarks.exs 'probes/newera8/r16/**/*.slp'
cat scenarios/gaps.json   # should have detector gaps

# 3. Coach report smoke (bot is P1 in probe games):
mix run scripts/coach_report.exs 'probes/newera8/r16/**/*.slp'
cat logs/coach/*/report.md

# 4. Coverage: bot probes vs the archive corpus, cache the corpus ledger:
mix run scripts/coverage_ledger.exs --corpus 'corpus/archive/mewtwo/*.slp' \
  --save-corpus cache/coverage_corpus_mewtwo.json \
  --bot 'probes/newera8/r16/**/*.slp'

# 5. Retrieval: pick a coverage gap id from gaps.json, mine the corpus:
mix run scripts/find_situations.exs --gap <id> --corpus 'corpus/archive/mewtwo/*.slp'
```

Things a real run may expose that standalone logic can't:
- **Slim-frame field drift:** `ScenarioScan.player_summary` must actually
  yield `percent`/`stock`/`facing` on live Peppi frames (verified present
  in the module, not on a real parse). If `stock`/`percent` come back nil,
  `death_sequence`/`dropped_punish`/coverage pct-bands degrade — check the
  smoke output for sane counts, not just non-crash.
- **coach_report** calls `ReplayStats.approach_stats/2` + `conversion_stats/2`
  with the bot's sub-map — confirm the arg orientation gives the BOT's
  numbers (swap-args path for bot-port 2 is untested).
- **find_situations `--char-filter`** depends on `PlayerMeta.character_name`
  being populated by the native parser — confirm on one Fox replay.
- Tune `dropped_punish`'s 15% / `passivity_window`'s 300f / coverage
  `min_corpus_frac` against real distributions; the defaults are guesses.

---
 Companion to DRILLS_DESIGN_2026-07-23.md (the
drill trio: seeded rollouts / human_drill / manual bookmarks — written,
untested). This doc is the pipeline AROUND those tools: automatically
noticing what training data is missing when the bot plays, acquiring it
cheaply, and making it teach the right thing. End goals include a Fox
bot with genuinely mixed neutral and a G&W bot (see Character Programs).

## Implementation constraints (read first)

- **Mix rule:** if a training run is live on nixos_slanka, no `mix`
  anything (CLAUDE.md). While the r16 LAUNCHER (incl. post-training
  fan-out) is alive, these files are invoked with `--no-compile` and
  must not be edited: `scenario_suite.exs`, `report_card.exs`,
  `trace_tech_chase.exs`, `play_dolphin_async.exs`,
  `interp_p3_case3.exs`, `priv/python/melee_bridge.py`. Once it exits,
  everything unlocks.
- Targeted tests only (CLAUDE.md test table). New lib modules need unit
  tests over synthetic frame lists — the ScenarioScan pattern: detectors
  are pure functions over `[%{frame:, p1:, p2:}]` slim frames
  (`ScenarioScan.load/1` + `player_summary/1` produce that shape; reuse
  them, don't reinvent).
- Action-state ID sets must mirror `ExPhil.Interp.ReplayStats` (the
  ScenarioScan convention) so numbers stay comparable across the
  toolkit.
- Keep `docs/guides/` updated per script; follow Output-module logging
  standards.

## Shared currency: the gap ledger

Everything in stage A produces, and everything in stage B consumes,
one file: **`gaps.json`** (repo root or `scenarios/gaps.json`). Schema:

```json
{ "gaps": [ {
  "id": "g_<8hex>",                  // hash of (source,slp,frame,type)
  "source": "manual|detector|coverage|uncertainty|coach",
  "type": "neutral_loss|dropped_punish|... or bucket key or scenario type",
  "slp": "path or null (coverage gaps have no single moment)",
  "frame": 4180,
  "note": "human-readable evidence",
  "status": "new|mined|drilled|verified",
  "created": "2026-07-23",
  "evidence": {}                     // source-specific extras
} ] }
```

A tiny module `ExPhil.Eval.GapLedger` (load/merge/append/dedupe-by-id,
pure + unit-tested) is used by every producer script. Manual bookmarks
(`scan_bookmarks.exs`, already written) get a thin follow-up flag
`--gaps PATH` appending to the same ledger. Status transitions are made
by the tools that do the work (build_seed_dir → `drilled`, scenario
battery pass → `verified`), not by hand.

---

# Stage A — noticing gaps automatically

## A1. FailureScan: detectors over the bot's own games  [PRIORITY 1]

`ScenarioScan` finds *situations* (well-posed "what now?" moments).
`FailureScan` finds *outcomes that indicate a gap* — run it over every
probe game, exhibition, and sparring set; each hit becomes a gap-ledger
entry with a handoff frame for drilling.

**New module `lib/exphil/eval/failure_scan.ex`** — same shape as
ScenarioScan (`types/0`, `detect/2`, `scan/2` with min_frame/gap
curation, consumes `ScenarioScan.load/1` frames). Port convention: P1 =
bot. Detectors (constants from ScenarioScan: `@hitstun`, `@lifecycle`,
`@idle 14`, `@fd_edge 85.5`):

- **`:neutral_loss`** — P1 enters hitstun from true neutral: in the 60
  frames before the hit, NEITHER player was in hitstun, the knockdown
  lifecycle, or shieldstun, and then P1's action enters `@hitstun`.
  Candidate frame = hit frame; the emitted gap's `frame` (drill
  handoff) = hit − 90, clamped ≥ 300 (the opening began before the
  hit). Note records p2's action at the hit (what move opened P1 up)
  and the distance 90f earlier.
- **`:dropped_punish`** — P1 lands a hit from neutral (P2 enters
  `@hitstun`, symmetric precondition to the above) but within the next
  120 frames P2 returns to actionable (not hitstun/lifecycle) without
  taking another hit, and P2's total percent gain over the window is
  < 15%. Candidate at the first hit — the conversion drill starts
  there. This is gate-10's sibling: initiation that goes unrewarded.
- **`:death_sequence`** — P1 loses a stock. Walk backwards from the
  death to the most recent frame where both players were neutral
  (neither in hitstun/lifecycle for 30 consecutive frames); that frame
  is the candidate — drill the entry-point of the sequence that killed,
  not the death itself. Note records the elapsed frames and P2's
  opener action.
- **`:passivity_window`** — a rolling 600-frame window in which the
  players were within 40 x-units for ≥ 300 frames but P1 never entered
  an attack/grab action state. Candidate at window start. This
  localizes the gate-10 pathology to specific drillable moments
  (attack-state ID set: mirror ReplayStats' attack classes).

Unit tests: synthetic frame lists per detector (the ScenarioScan test
file is the template).

**New script `scripts/auto_bookmarks.exs`** — CLI mirror of
`scan_bookmarks.exs`: replay globs → FailureScan.scan → gap-ledger
append (`--gaps`, default `scenarios/gaps.json`) + optional
`--manifest-out` for classified drillable entries. `--types` subset
flag. Detector hits on P2's port (`--port 2` semantics: flip the
port convention to scan the bot when it's P2 in a sparring replay).

## A2. Post-set coach report  [PRIORITY 1, same PR as A1]

**New script `scripts/coach_report.exs`** — input: the .slp files of
one set (globs) + which port is the bot. Bot-port autodetect: Peppi
metadata netplay codes — the bot is **EXPH#288** (fallback `--bot-port`
when codes are absent, e.g. local probe games where the bot is P1 by
convention). Output: one markdown report + JSON to
`logs/coach/<ts>/`, and gap-ledger appends.

Per game and aggregated per set:
- openings for/against (FailureScan `:neutral_loss` run in both port
  orientations), opener-action histogram against the bot
- conversion rate after openings (`:dropped_punish` complement),
  damage-per-opening both ways
- deaths: `:death_sequence` entries with elapsed-frames (how long from
  neutral loss to death — punish resistance)
- passivity windows count + total armed-approach rate (reuse the
  StyleCard gate-10 computation — `ExPhil.Interp.StyleCard`)
- top N (default 10) gap candidates appended to the ledger, deduped
- a "drills that already exist for these gaps" section: cross-reference
  `scenarios/manifest.json` types + existing seed dirs

This is the artifact to read after every Yeti exhibition / Direct
sparring night, and the primary LLM-coach ingest.

## A3. Coverage ledger: bot visitation vs corpus visitation  [PRIORITY 2]

Where does human play go that the bot never goes? (The r15→r16 lesson —
"converts fine, never initiates" — restated as measurable occupancy.)

**New module `lib/exphil/eval/coverage.ex`:**
- `bucket(p1_summary, p2_summary) :: String.t()` — coarse situation
  key. v1 features (deliberately cheap, no NN embedding):
  `dist:{0-15|15-30|30-50|50+}` × `p1zone:{center|mid|edge|offstage}`
  (FD |x| bands 0-30/30-60/60-85.5/85.5+) × `actionable:{both|p1|p2|none}`
  × `p1pct:{0-40|40-80|80-120|120+}` × `facing:{toward|away}`.
  Key = joined string, e.g. `"d15-30|zmid|aboth|p40-80|ftoward"`.
- `ledger(frames) :: %{key => count}`; `merge/2`;
  `diff(bot, corpus, opts) :: [%{key:, bot_frac:, corpus_frac:, ratio:}]`
  with Laplace smoothing, sorted by under-representation; `save/load`
  (JSON).

**New script `scripts/coverage_ledger.exs`** — `--bot` globs vs
`--corpus` globs (or a cached corpus ledger JSON — corpus side is
computed once and reused), writes the diff report + appends the top-K
under-visited buckets to the gap ledger (`source: "coverage"`,
`slp: null`). Corpus frames come via `ScenarioScan.load` (cheap slim
frames — no embedding pass needed, so this does NOT wait on #33).

## A4. Policy-uncertainty logging  [PRIORITY 4; touches locked files]

Per-frame, the agent already computes head distributions; log
`{frame, entropy_per_head, logprob_chosen}` as JSONL next to each probe
replay when `EXPHIL_LOG_UNCERTAINTY=1`. High-entropy clusters =
states the training data doesn't cover — a gap stream that fires
*before* failures do. Implementation: `lib/exphil/agents/agent.ex`
(get_controller path) + a consumer script that maps top-percentile
frames back to (slp, frame) gap entries. **Deferred behind the file
lock** (`play_dolphin_async.exs` wiring) and needs a perf check —
logging must not break the 60fps budget (buffer in memory, flush at
game end).

---

# Stage B — acquiring matching data cheaply

## B1. Situation retrieval over the corpora  [PRIORITY 2 (v1) / gated (v2)]

Given a gap, find k human-played instances of *the same situation* and
what they did next. This is the Fox unlock: ~95k CC0 replays
(slippi-public-dataset-v3.7) of world-class Fox exist; the bottleneck
is finding the right 200 frames, not data volume.

**v1 — bucket-grep (build now):** `scripts/find_situations.exs` —
`--like SLP:FRAME` (bucket the moment via `Coverage.bucket`) or
`--bucket KEY` or `--gap ID`; scan corpus globs for frames in the same
bucket (plus optional `--char-filter fox`); emit `[{slp, frame}]`
manifest-style JSON, capped + spaced like ScenarioScan curation. Output
feeds: BC oversampling lists, TM-CE practice imports, drill manifests.
Marks the gap `mined`.

**v2 — embedding ANN (gated on #33 streaming embed):** per-frame
embeddings sharded to disk during the streaming-embed pass (same pass,
second consumer — design the shard format together with
DATA_SCALING_2026-07-22.md work); query = brute-force GPU cosine top-k
(fine to ~10M frames on the 5090), HNSW later only if needed.
Interface to build against now: `SituationIndex.build(globs, opts)` /
`query(state_embedding, k)`.

## B2. Sparring capture  [PRIORITY 3]

Every human-vs-bot game is (a) a labeled record of where the bot fails
against real humans, and (b) on-distribution human corrections —
currently these games evaporate. **New script
`scripts/ingest_sparring.exs`:** takes a session's .slp files, detects
the bot port (EXPH#288), then:
1. copies to `corpus/sparring/<date>/` with a sidecar
   (`{human_port, human_tag, bot_checkpoint}` — checkpoint from
   `--checkpoint`, it is not recorded in the .slp),
2. runs `coach_report.exs` on the set,
3. registers the HUMAN port for BC ingestion (tag `human_blewf` — the
   #30 corpus). **Never BC on the bot's port** — those are policy
   actions.
Requires no new training code: sparring replays enter the pool like any
BC replay; the sidecar's human_port drives `--port` selection at
ingestion (small dagger_drill/train flag: honor sidecar port if
present — same auto-detect slot as the seed_meta.json hook).

## B3. Seeded self-play for neutral  [PRIORITY 4]

Ghost-P2 drills can't teach neutral (neutral is interactive). **New
script `scripts/selfplay_rollouts.exs`**, scenario_suite-derived:
prefix-replay to a both-neutral handoff (idle_deadlock candidates or
Coverage-selected neutral buckets), then hand BOTH ports to policies —
port 2 via a second `Agent` (the bridge already dual-drives:
`send_controller(bridge, Map.put(input, :port, 2))`, dummy_mode
"external", cpu_level 0). Variants: mirror (same checkpoint, sampled),
league (current vs r13/r14/r15 checkpoints — `ExPhil.League` exists),
cross-char later. `--finalize` + seed_meta as in the drill pipeline.
**Data use: RL/PPO only, not BC** (self-play trajectories reinforce the
policy's own habits; the PPO smoke test — currently UNRUN, task on the
board — is the prereq). Reward shaping for neutral: openings-for minus
openings-against from FailureScan logic applied online, plus the
approach-shaping candidates in the r17 escalation plan.

## B4. Preference pairs from the drill recorder  [PRIORITY 4, tiny now]

`human_drill.exs` save/discard verdicts are labeled preferences on the
same scenario. **Now (one small edit to the already-written, untested
script):** also write discarded attempts to `<out>.discards.frames` and
give the sidecar a `pair_id` per (slot, consecutive attempts) so pairs
are recoverable. **Later (research task):** trajectory-level DPO-style
loss — `-log σ(β[(logπ(chosen) - logπ_ref(chosen)) - (logπ(rejected) -
logπ_ref(rejected))])` summed over frames, π_ref = pre-finetune
checkpoint. Even without that, discards are hard negatives for eval.

---

# Stage C — making small data teach the right thing

## C1. Style tags → "a couple different things in neutral"  [PRIORITY 3]

BC on mixed players *averages* multi-modal neutral into mush. The
counter is conditioning: `Peppi.to_training_frames` already attaches
`player_tag` per frame (`get_player_tag`) — **first task is an audit:
grep player_tag through training/embedding to establish how far the
conditioning actually goes** (frame → dataset → embedding → policy
input); wire the missing links (likely: a learned tag-embedding
analogous to `character_mode: :learned`, a `--style-tag` inference
flag, and tag vocabulary persistence in the export). Then:
- corpus manifest maps tag → replays (3-5 distinct high-level Foxes
  with different neutral profiles, from the public dataset),
- train tagged; at inference sample/rotate tags, or expose the tag as
  a coach-controllable knob per game.
Fallbacks if conditioning underdelivers: separate fine-tunes per style
+ checkpoint rotation (crude but trivially works).

## C2. Neutral-opener taxonomy + diversity gates  [PRIORITY 3, same PR]

Make "does a couple different things" measurable BEFORE training toward
it (the gate-10 lesson: name the number first).

**New module `lib/exphil/eval/neutral_scan.ex`:**
- `opener_events(frames)` — for each neutral→engagement transition
  (both-neutral ≥ 60f, then P1 enters attack/grab/aerial-drift-in),
  classify the opener: `dash_attack | grab | aerial_approach | laser |
  projectile | other` (action-state IDs mirrored from ReplayStats;
  laser = Fox/Falco blaster states; projectile covers G&W bacon;
  aerial_approach requires airborne + x-velocity toward opponent).
- `distribution(events)`, `entropy_bits(dist)`.

**StyleCard char packs** gain two gates (existing `gate/4` pattern):
`opener_entropy >= H_min` and `top_opener_share <= S_max`, thresholds
per char pack (Fox pack: entropy ≥ 1.2 bits over ≥ 8 openers,
provisional). Report_card composition unchanged — these ride the
existing gates list.

---

# Character programs

**Fox** (data-rich): retrieval (B1) + style tags (C1) + opener gates
(C2) are the program. Scenario machinery ports directly; regenerate
manifests per matchup — `gen_scenario_manifest.exs` currently
hardcodes FD-Mewtwo(p1)-vs-Fox(p2): parameterize `--p1-char/--p2-char`
(small edit, it's an unlocked file).

**G&W** (data-poor, #23): pull the greg corpus from B2
(`scripts/pull_replays.sh` reaches exphil-replays-blewfargs), add own
play via B2-sparring + human_drill, filter the public dataset for G&W
games (rare — expect hundreds, not thousands). **RNG caveat: Judge
(side-B) is RNG and input-prefix replay does NOT restore the RNG seed —
prefixes crossing a Judge will diverge (drift check catches and
excludes, shrinking the pool). Savestate-based human_drill is immune
(savestates capture RNG state); prefer it and self-play for G&W, or
investigate seeding headless RNG later.** StyleCard needs a gnw char
pack (bacon in the projectile opener class).

---

# LLM coach (consumes everything above)

v0 is a workflow, not code: a Claude session (or skill) that reads
`gaps.json` + latest coach reports + scenario scoreboards + StyleCard,
proposes a diagnosis and a drill/mining manifest (which gaps, which
acquisition route — find_situations / human_drill / seeded rollouts —
what oversampling weight), and Bradley approves before anything trains.
Every stage-A/B tool above emits/consumes JSON specifically so this
loop needs no new plumbing. Automate only after the manual loop has
been run a few times and the prompt stabilizes. (A `.claude/skills/`
coach skill checklist is the natural v1 packaging.)

---

# Build order

| # | What | Depends on | Files |
|---|------|-----------|-------|
| 1 | GapLedger + FailureScan + auto_bookmarks + coach_report (A1+A2) | nothing | new lib ×2, new scripts ×2, tests |
| 2 | Coverage + coverage_ledger + find_situations v1 (A3+B1v1) | GapLedger | new lib, new scripts ×2, tests |
| 3 | Style-tag audit→wiring + NeutralScan + StyleCard gates + sparring ingest (C1+C2+B2) | 1 (coach_report for B2) | audit first; new lib, style_card edit, new script |
| 4 | Seeded self-play, preference pairs, uncertainty logging (B3+B4+A4) | PPO smoke (B3); file locks (A4) | scripts + agent edit |
| 5 | Retrieval v2 (B1) | #33 streaming embed | index shards + query |

Test order per feature is standard: unit tests on synthetic frames for
every detector/bucket/opener classifier (pure functions), then one
real-replay smoke against a probe game (expected: nonzero neutral_loss
and dropped_punish counts on any r-line probe replay; gate-10 ≈ 0 bots
should produce passivity_window hits), then the script CLI end-to-end.
All of it gated on r16's launcher exiting.
