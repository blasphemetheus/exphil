# Drill Data & Bookmarks Design (2026-07-23)

Built while r16 trained (nothing below has been executed — see Test Plan).
Goal: turn the UnclePunch practice loop (find a moment → snapshot → drill
it repeatedly) into training data and evaluation for the bot, plus a way
to mark moments live during play. The portable unit everywhere is
**(source .slp, handoff frame)** — TM-CE consumes it as a savestate for
human practice; ExPhil consumes it via input-prefix replay
(scenario_suite mechanics, SCENARIOS.md).

Three features:

| Feature | Script(s) | Status |
|---|---|---|
| A. Bot drill rollouts (extends #38) | `scenario_suite.exs` (exists) + `build_seed_dir.exs` (new) + `dagger_drill.exs` slicing (edited) | written, untested |
| B. Human drill recorder | `human_drill.exs` (new) | written, untested (needs hardware loop) |
| C. Input-signature bookmarks | `scan_bookmarks.exs` (new) | written, untested |

## A. Bot drill rollouts — the policy practices one situation N times

Generator already exists: `scenario_suite.exs --only <entry> --runs N
--temperature T --finalize` boots one headless Dolphin per attempt,
prefix-replays to the handoff, lets the policy play the response window
(sampled — deterministic runs would produce N identical attempts), then
`--finalize` SDs to force GAME! so the .slp parses (#38/GOTCHA #73).

New pieces:

1. **`scripts/build_seed_dir.exs`** — reads scenario scoreboard JSON(s),
   collects each clean run's .slp (skips errored/diverged, and
   non-finalized unless `--allow-unfinalized`), copies them into a seed
   dir as `<type>_<ts>_s<seq>_r<run>.slp`, and writes/merges
   `seed_meta.json` mapping basename → `{handoff, window, type, score,
   pass, source_slp}`. `--min-score` filters weak attempts (or invert:
   keep weak ones if the goal is expert-relabel correction — the flag
   supports both directions via `--max-score`).

2. **`dagger_drill.exs` slicing** — at rollout load, if a
   `seed_meta.json` sits next to the rollout, frames outside
   `[handoff, handoff+window]` are dropped BEFORE relabeling (the
   `recorded` map for prev-action lookups is built from the unsliced
   frames, so relabel context at the slice boundary stays correct).
   Without this, N attempts from one handoff would inject N identical
   copies of the prefix plus N junk SD tails into the pool. The existing
   `NEWERA8_SEED_ROLLOUTS_DIR` launcher wiring (find *.slp) picks the
   sliced behavior up automatically since the meta file rides in the dir.

Flow: `gen_scenario_manifest.exs` / bookmarks (C) → curate manifest →
`scenario_suite --finalize --runs N --temperature 0.8` →
`build_seed_dir.exs` → `NEWERA8_SEED_ROLLOUTS_DIR=seeds/...` launch.

## B. Human drill recorder — `scripts/human_drill.exs`

You mark a moment (from a bot game or your own), the tool puts you IN
that moment on real hardware, and every attempt you save becomes
training data in the exact format the trainer eats.

Design constraints that shaped it:
- Savestates CAN be created on the GUI netplay build (they're only
  build-locked and headless-impossible), and Wayland `wtype` can press
  Dolphin's hotkeys when the window is focused.
- The bridge serializes every port's `controller_state` live — but only
  `d_up` of the D-pad (types.ex/melee_bridge.py). So live markers must
  use d_up-visible chords; full D-pad is offline-only (Peppi has all 4).
- Dolphin pad config can be changed mid-game via the Controllers dialog,
  but not scripted — one manual swap per session is the v1 cost.
- The session's own Slippi .slp is garbage across savestate loads, so
  episodes are logged live from the gamestate stream instead.

One windowed netplay-build Dolphin session, two phases:

**Phase 1 (automated):** parse source .slp, boot via MeleePort
(windowed, netplay build, speed 1.0), prefix-replay BOTH ports exactly
like scenario_suite (same input-offset/drift-check/pipe-shim logic). At
each requested handoff frame (up to 8 per session, all from the same
source game, sorted ascending) it drift-checks and presses
Shift+F<slot> via wtype → savestate slots 1..8. Prefix runs at realtime
in v1 (you get to watch the lead-up; unthrottled prefix would leave the
whole session unthrottled since emulation_speed is boot-time config).

**Swap:** script prompts you to open Dolphin's controller config and
switch Port 1 from the pipe pad to your GC adapter, then play.

**Phase 2 (recording):** the script keeps stepping, observing:
- **Ghost P2:** keeps sending the source game's recorded P2 inputs
  indexed by the live frame number — savestate loads restore the frame
  counter, so the ghost re-aligns automatically after every reset.
- **Attempt segmentation:** F<slot> load = frame counter jumps backward
  → attempt starts (slot matched by nearest handoff). Markers, chosen to
  be live-visible through the bridge:
  - **taunt (d_up)** = end attempt + SAVE
  - **L+R+taunt** = end attempt + DISCARD (shield eats the taunt, so
    nothing happens in-game)
  - loading a state mid-attempt = implicit DISCARD
  Marker frames and the first 3 post-load frames are stripped from the
  saved episode.
- **Episodes:** frames logged as `%{game_state, controller,
  prev_controller, player_tag: "human_blewf"}` (controller = your live
  `controller_state` on port 1), written after every save (crash-safe)
  to `drills/human/<name>.frames` — the `term_to_binary` payload
  `%{expert, exported_at, frame_lists}` that `train.exs --mix-frames`
  already loads — plus a `.json` sidecar (per-attempt slot/frames/kept).

Session ends at GAME! (match timer or stocks) or Ctrl-C; both leave the
.frames file valid. Known v1 limits: realtime prefix, ≤8 slots/session,
one manual pad swap, 8-minute match timer caps a session, wtype needs
Dolphin focused (fallback: press Shift+F<slot> yourself when prompted —
a savestate a few frames late is still the same situation for human
drilling).

Deferred (small, do after r16): serialize the full D-pad in
melee_bridge.py + types.ex (then d_down = discard directly, matching the
D-pad-down-discard / D-pad-up-save scheme you described); extract the
prefix-replay code shared with scenario_suite.exs into
`ExPhil.Drills.PrefixReplay` (couldn't touch scenario_suite tonight —
the r16 launcher's fan-out still runs it `--no-compile`).

## C. Input-signature bookmarks — `scripts/scan_bookmarks.exs`

Mark moments DURING play with an input that does nothing in-game but is
recorded in the .slp: **D-pad down** (also d_left/d_right via
`--signature`; d_up is taunt and excluded). Works identically on
netplay and console/tournament Slippi — the bookmark travels inside the
replay. No live capture layer needed.

Scanner: Peppi-parses each .slp, finds rising edges of the signature
button held ≥ `--min-hold` (default 2) frames on `--port` (default: all
occupied ports), clusters marks within `--cluster` (default 120) frames,
and emits each as handoff = mark − `--lookback` (default 120, clamped to
≥ 300 for prefix room — the situation started before you reacted).

Output `bookmarks.json`: `[{slp, port, mark_frame, frame, note}]`.
With `--classify`, each bookmark is matched against `ScenarioScan.scan`
candidates within `--classify-window` (default 240) frames; matched ones
also land in a `--manifest-out` file in manifest-entry format, directly
usable by scenario_suite / TM-CE curation. Unmatched bookmarks keep
`type: null` (future: a `:freeform` scenario type with neutral scoring
so unclassified moments can still seed drills).

rwing already has hotkey → .gci export for the human/TM-CE side;
whether to build on it depends on source access (Patreon binary —
check with the author before planning contributions).

## LLM coach layer (sketch — not built)

The three features give an LLM coach its verbs; the coach is an
orchestration loop, no new ML:

1. **Ingest:** bookmarks + notes, scenario scoreboards, report_card,
   gate/StyleCard numbers, Peppi stats.
2. **Diagnose:** name the weakness (the r15→r16 "conversions up,
   initiation ≈ 0" analysis, mechanized). Mine more instances via
   ScenarioScan / bookmark history.
3. **Prescribe:** emit a drill manifest — which handoffs, who generates
   data (human_drill / seeded rollouts / archive mining), what
   oversampling weight.
4. **Verify:** rerun the scenario battery + human-baseline percentiles
   (score your own attempts from B with the same ScenarioScore), decide
   the next round.

Human-guided at first: Bradley approves diagnosis + manifest before
anything trains.

## Test plan (in order, once the r16 launcher fully exits)

1. **#38 finalize test** (already queued): one scenario `--finalize`,
   confirm Peppi parses the .slp and Mewtwo actually SDs (GOTCHA #73).
2. `build_seed_dir.exs` on that scoreboard → verify seed dir +
   `seed_meta.json`.
3. `dagger_drill.exs --rollouts <seed dir slp>` dry-run → verify the
   "sliced" log line and frame counts (window+1 expected).
4. `scan_bookmarks.exs` on a fresh netplay game where you press d_down
   a few times (no existing replay has the signature); unit-test
   clustering separately if needed.
5. `human_drill.exs` phase 1 only on one manifest entry: verify netplay
   build boots windowed under the bridge, wtype reaches Dolphin
   (focused), savestate lands near the handoff, drift ≈ 0.
6. Full mini-session: 2 slots, ~5 attempts, pad swap, taunt-save →
   inspect the .frames payload loads under `ExPhil.Training.MixFrames`.
