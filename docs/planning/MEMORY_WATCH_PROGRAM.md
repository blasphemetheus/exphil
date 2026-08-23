# Memory-Watch Program — living plan

Started 2026-08-22 (Bradley: "build dolphin memory watches, cause in
similar problems in the future we'll have this capability to lean
on"). UPDATE THIS DOC as items land — it is the program's checklist
and log.

## Capability, proven

Dolphin's MemoryWatcher (present AND functional in the mainline-beta
netplay build — spike 2026-08-22, 1,165 datagrams/20s): drop a
`User/MemoryWatcher/Locations.txt` of hex address chains (single =
direct read; multiple = pointer chase), bind a Unix datagram socket at
`User/MemoryWatcher/MemoryWatcher`, and dolphin sends
`"<line>\n<hexvalue>\0"` on every change, polled 600/s. Read-only,
build-agnostic (Ishiiruka source in ~/git/slippi-Ishiiruka is the
reference implementation).

## Big picture

The Slippi stream gives us a FIXED SCHEMA someone else chose; RAM has
everything the game knows. MemoryWatcher turns "the stream doesn't
carry X" from a wall into a config line. Four postures it enables:

1. **Observability where the stream is blind** — menus (the 08-22
   frozen-CSS saga), stage internals, engine state.
2. **Ground truth for verification** — RAM as the arbiter when
   parsers/streams disagree (the parity-differential method, live).
3. **Richer training signal** — labels/events the replay schema
   cannot express, feeding AWBC channels and Situations.
4. **Diagnosis speed** — "is X true in the game right now" becomes a
   one-line watch instead of a human session (today's five-relaunch
   CSS hunt would have been 10 minutes).

## Build checklist (core plumbing)

- [x] Protocol understood (Ishiiruka MemoryWatcher.cpp).
- [x] Capability spike on mainline beta (tmp/mw_spike.exs) — WORKS.
- [x] `Melee.MemoryWatcher` GenServer (2026-08-22): writes
      Locations.txt + binds socket in init (single "start before
      dolphin" contract); receiver task; named watches incl. pointer
      chains; get/get_f32/snapshot/subscribe; junk-frame-tolerant
      parser. 8 unit tests incl. loopback socket end-to-end.
- [x] `:memory_watch` option on Melee.Dolphin.launch (2026-08-22):
      watcher started between prepare_home and spawn, pid on the
      Dolphin struct (`:memory_watcher`), stopped in Dolphin.stop.
- [x] Named-address layer `Melee.MemoryMap` (2026-08-22): classic
      libmelee locations.csv provenance, 0x80-prefixed for mainline.
      VERIFIED LIVE: rng_seed / menu_frame / menu_state (packed word,
      decode TBD). STALE: all per-port CSS fields — re-derivation is
      the active thread (HANDOFF_2026-08-22 attack plan).
- [x] Tests (2026-08-22): 26-test battery — HtDP input-class grammar
      tests for the MAINLINE composite format (the trailing-newline
      format bug these tests caught cost the afternoon), total-f32
      decode (NaN crash caught), subscriptions, lifecycle, external
      streaming. Suite 500/0.
- [x] `menu_state` scene-word decoder (2026-08-22b):
      `MemoryMap.decode_scene/1` + `scene_name/1` — the packed word is
      the scene controller `<<major, pending, previous, minor>>`;
      stream scene = `(minor <<< 8) ||| major`, so the decode reuses
      `Events.Menu.scene_name/1`'s whole taxonomy as offline test
      vectors (live word 0x02020200 @ VS CSS = the confirming read).
      OWED: a live transition trace pinning byte-1 vs byte-2
      (pending vs previous) — mw_verify now prints decoded scenes.
- [x] Capability reference doc (2026-08-22b):
      `libmelee_ex/docs/memory-watch.md` — contract, datagram grammar
      data definition, semantics, address book + provenance, the
      address-hunt playbook, gotcha table.
- [x] Test battery round 2 (2026-08-22b): MemoryMap invariants
      (MEM1-virtual range, chain-offset sanity, full dolphin-echo
      round trip name recovery), parse_datagram TOTALITY fuzz (200
      random binaries + adversarial shapes — a raise in the receive
      path kills the watcher mid-session), scene-word input classes
      (settled/transitioning/boundary + all known scenes).
- [ ] exphil bridge: MeleePort accepts watcher values and merges them
      into menu-scene GameStates (fixes GOTCHA #101 properly; the
      blind CSS fallback becomes the no-watcher fallback).
- [x] Address-hunt kit (2026-08-22b): `Melee.MemoryHunt` — pure,
      18-test class enumeration: candidates/3 (region -> watch batch,
      MEM1-range guarded — silent out-of-range watches read 0 and
      poison the differential), changed/2 (snapshot diff; appearance
      IS change under on-change semantics), correlated/2
      (driven-minus-idle — kills always-ticking counters),
      f32_class/1 (zero/denormal/normal/nan triage), tracks?/2
      (commanded-coordinate closing argument; no evidence != pass).
      Live driver: `examples/memory_hunt_css.exs` (HUNT_BASE/COUNT/
      STRIDE env). Ready for the CSS re-derivation runs.

## Application list (ordered; check off / date as done)

1. [~] **Live online-CSS state** — CURSORS RE-DERIVED 2026-08-22
       (libmelee_ex 60b20a8): the classic 4-port block relocated
       intact by +0x17200, stride 0xB80 preserved; P1 bit-exact
       verified via MemoryWatcher, P2-P4 delta-derived. Method that
       won: PARK-AND-SCAN (examples/memory_scan_css.exs — read
       /proc/<emulator>/mem as its ancestor, scan MEM1 for the
       stream-reported f32 bits, intersect across parks); the
       differential hunt proved the classic region dry but couldn't
       find the new one (settled CSS memory is STATIC — even the RNG
       doesn't tick there; traffic deltas are the only liveness).
       LATER SAME NIGHT — hover byte CONFIRMED LIVE: classic STATIC
       803F0E0A responds per-portrait on mainline (ids tracked 3
       hovers + verified fox=10 at park; static region survived, only
       the heap moved). Status byte 803F0E08 plausibly intact
       (constant 0=HMN). DEAD: classic coin chain (804A0BC0 stale
       pointer) AND the offline stream's coin_down (false throughout
       a navigate-picked CSS — GOTCHA #101's offline sibling).
       SELECTION STATE SOLVED 2026-08-22c (grim-screenshot ground
       truth + A/B toggle experiment, libmelee_ex
       tmp/mw_select_toggle.exs + tmp/mw_stride.exs):
       `css_pN_selected` = u32 at 0x8043208C + 8·(N−1), value = the
       port's locked-in EXTERNAL character id, 0x21 = none; flips on
       A-select, back on B-deselect, untouched by hovering; parallel
       copy at +0x54. P1 verified fox (0x21->0x02), P2 falco
       (0x21->0x14), both through the live MemoryWatcher path; in
       MemoryMap.menu() (replaces the dead coin chain) with
       css_selected/1 decoder, test-pinned. The earlier candidates
       were red herrings born of TWO false premises: navigate! never
       pre-picked fox (its default until = CSS ARRIVAL), and probe
       tap! presses land nondeterministically against a free-running
       dolphin (step-counted holds can be sub-frame wall time — use
       wall-clock press/sleep/release). 0x80444964 falsified
       (2->16->0 across select/deselect), 0x80479C58 falsified.
       REMAINING for #1: confirm 0x8043208C family reads correctly at
       the ONLINE CSS (static region, expected to survive — next
       Direct session), then feedback menuing replaces the blind
       fallback's guesswork.
2. [~] **Menu-scene ground truth for the watchdog** — diagnosis slice
       SHIPPED 2026-08-22b: MeleePort's MENU STUCK report/log now
       carries `ram_scene` (SceneView — names the actual screen where
       the stream says 255) and `ram_traffic_delta` (250ms window:
       zero = core wedged, positive = core fine / menuing stuck).
       OPEN: using the signal to SUPPRESS false alarms during
       legitimate holds (matchmaking waits), not just label them.
3. [ ] **Parser parity, live**: cross-check stream-parsed player state
       vs RAM reads during a headless game — extends the peppi
       differential method to the live bridge (catches GOTCHA
       #81-class action_frame divergences at the source).
4. [ ] **Scenario-farm verification**: Improoover/scenario setups
       assert exact game state (percent, position, RNG seed) before
       recording a rep — removes inference from drill farms.
5. [ ] **RNG seed capture per game** -> determinism audits and
       exact-replay debugging for the eval protocol.
6. [ ] **Stage internals live**: FoD platform heights / PS
       transformation state on ANY build or era (the stream only has
       these >=3.18) — feeds StageCollision consumers in live play.
7. [ ] **Richer outcome channels** (with the tech-drill program):
       L-cancel flags, hitlag/hitstun counters, knockback vectors
       read directly — new AWBC/GameEvents signal without waiting for
       stream schema support.
8. [ ] (later, netplay era) lobby automation: read connect/matchmaking
       state for unattended session management.
9. [x] **Scene-transition early warning** — CODE SHIPPED 2026-08-22b
       (LIVE VALIDATION OWED, next Direct session):
       `Melee.MemoryMap.scene_view/1` (SceneView data definition:
       settled | leaving) + `ExPhil.Bridge.BlindCss` (pure decision
       table, 15-test class enumeration + whole-trace tests) wired
       into MeleePort's blind CSS fallback. Semantics: START-pulse
       phase hands back EARLY on confirmed departure; still settled
       at the CSS after the window -> bounded A-press retry (2);
       unmapped scenes classify :unknown (never close the loop on
       noise); no watcher = exactly the validated open-loop timings.
       Watcher default-ON for online launches
       (EXPHIL_MEMORY_WATCH=1 forces any session, =0 disables).
       Menu watchdog consumer still open.
10. [ ] **Delay-regime measurement** (feeds LATENCY_ARCHITECTURE's
       ping-table thread, the 22→4 chain-gap question): RAM frame
       counter read at datagram arrival vs the stream event's frame
       stamp = a direct, per-session measure of the local pipeline's
       contribution to effective delay. Same method against a netplay
       session decomposes effective delay into local vs network parts
       — today it's inferred from qtrace lag peaks after the fact.
11. [ ] **In-game player ground truth** (percent/stock/action/position
       per port from RAM): classic addresses exist, need the same
       0x80-virtual re-verification as CSS. Superset of application
       #3 (parity) and prerequisite for #4 (scenario asserts) and #7
       (outcome channels) — verify once, three consumers.
12. [x] **Liveness ratchet** — SHIPPED 2026-08-22b, simpler than
       planned: `MemoryWatcher.traffic/1`, a monotone
       PARSE-INDEPENDENT datagram count. Key wire fact (pinned by the
       grammar battery): dolphin sends a bare-NUL empty-step datagram
       EVERY step, so raw arrival — no address needed at all — is
       ground truth for "core running and paced"; liveness =
       `delta > 0` between two reads. First consumer: the stuck
       report's `ram_traffic_delta` (#2). Open: adopt in launchers/
       harnesses beyond MeleePort.

## Non-goals / boundaries

- WRITE access: MemoryWatcher is read-only. Write-side control stays
  with the Improoover savestate (.gci) route.
- Not a replay-era tool: replays have no RAM; everything here is
  live-session only. Replay-based pipelines keep the Peppi path.
- Address book is NTSC 1.02 + this dolphin family; version-guard the
  map file.

## Operational notes

- Datagrams flow only while GAME FRAMES ADVANCE: a bare
  Dolphin.launch with no console attached leaves the core unpaced
  (observed 2026-08-22: zero traffic, game parked at the Slippi
  online login scene) — every real consumer (Session/Probe/MeleePort)
  paces frames, so this only bites minimal demos/tests. Watch traffic
  is therefore also a free "is the game actually running" signal.
- The mainline build launched bare lands on the Slippi online menu
  (Log-in prompt when the home has no user.json). Ignorable for
  memory-watch purposes.

## Log

- 2026-08-22: capability spike PASSED on mainline beta (RNG seed
  streaming; first run hit the SlotA=255 black-window EXI gotcha —
  `memory_card: true` required). Program doc created.
- 2026-08-22 (later): core plumbing SHIPPED + live-verified after a
  long hunt — mainline sends COMPOSITE frames `("line\nhex\n")+ NUL`
  (bare NUL = unchanged step); the Ishiiruka-format parser silently
  rejected everything. Addresses must be 0x80-virtual. get_f32 made
  total (NaN bits). Classic per-port CSS addresses stale on this
  build → next thread = address re-derivation (HANDOFF_2026-08-22).
- 2026-08-22b (foundation pass): menu_state DECODED offline — packed
  scene-controller word, cross-validated against the stream taxonomy
  (every Events.Menu scene = a test vector; live 0x02020200 @ VS CSS
  fits). Shipped: decode_scene/scene_name, MemoryMap invariant
  battery + parse totality fuzz, docs/memory-watch.md reference,
  mw_verify scene printout. Applications #9-#12 added (transition
  early-warning unblocks a semi-closed online CSS WITHOUT the per-port
  addresses; delay-regime measurement; in-game ground truth as the
  verify-once/three-consumers step; liveness ratchet). Tests 39/0 in
  the two batteries.
- 2026-08-22 EVENING: **first live netplay validation PASSED**
  (eval_runs/0822_netplay_crown addendum; ep4 chain 62 on FD =
  ALL-TIME NETPLAY RECORD en route, 87.6 self/min, lag sharp 6).
  Watcher engaged over netplay; online CSS = settled major-8 word
  (layout confirmed at a 2nd scene family). NEW FACT: the Direct
  CSS -> code-entry transition does NOT change the scene word, so
  departure is unobservable there pre-match — BlindCss retries always
  fire (spurious-but-safe: deterministic 3 A presses = odd = ends
  selected). Enriched MENU STUCK fired live and diagnosed a real bug:
  blind-CSS Process flags persisted across games -> post-game CSS
  unpicked forever (masked before by per-game relaunches). FIXED same
  night: re-arm on leaving the CSS scene + raw scene-word CHANGE
  logging (science trace: code-entry minor, byte order, match-start
  word now land in every session log).
- 2026-08-22c: **selection-state holdout CLOSED** — the visual
  ground-truth plan (grim screenshots read by the agent + Bradley
  live) exposed both premises as false in one run: fox was NEVER
  pre-picked (navigate!'s until = arrival), and the probe's A presses
  were landing nondeterministically (tap!'s step-counted hold is
  sub-frame against a free-running windowed dolphin; wall-clock holds
  fixed it). With real selects/deselects the toggle signature found
  `css_pN_selected` (0x8043208C stride 8, external id, 0x21=none;
  0x804320E0 parallel copy) — visually confirmed both directions,
  watcher-path verified, shipped in MemoryMap + tests (suite 564/0).
  Bonus corroboration: 0x80432058 holds the costume file string
  ("PlFx"/"PlFc"). Old candidates 0x80444964/0x80479C58 falsified.
- 2026-08-22b (later): application #9 SHIPPED code-side —
  `scene_view/1` SceneView (libmelee_ex) + `ExPhil.Bridge.BlindCss`
  pure decision table (HtDP: observation classes -> progress classes
  -> phase x progress table; legacy-equivalence trace test pins that
  no-watcher behavior is bit-identical to the validated 08-22
  open-loop). MeleePort launches the watcher by default for online
  sessions. OWED: live Direct-session validation (watch for the
  "departure CONFIRMED via RAM scene word" log line + whether the
  code-entry scene shows as a distinct major-8 minor — that read
  also settles the pending/previous byte order).
