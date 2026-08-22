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
- [ ] Address-hunt kit: promote the tmp/mw_verify differential-scan
      pattern into a real tool (batch candidate generation over a
      region, changed-line report across a driven state change,
      stale-read heuristics: constants/denormals/never-changes).
      The CSS re-derivation thread should build this as it goes.

## Application list (ordered; check off / date as done)

1. [ ] **Live online-CSS state** (cursor/character/coin per port from
       RAM) -> real feedback menuing on any build; delete the blind
       fallback's guesswork. First consumer, validates the plumbing.
2. [ ] **Menu-scene ground truth for the watchdog**: watch the scene
       ids -> the MENU STUCK watchdog gets real signal (no more
       false alarms during holds).
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
9. [ ] **Scene-transition early warning** (cheap; unblocked NOW, no
       CSS addresses needed): `pending_major != major` in the decoded
       scene word = a scene change committed but not yet landed —
       RAM-only signal the stream never carries. Consumers: the menu
       watchdog (act one step earlier), and a SEMI-closed online CSS
       loop today — the scene word alone confirms "we left the CSS"
       i.e. the blind fallback's selection actually took, replacing
       the open-loop hope with a checkable postcondition.
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
12. [ ] **Menu-frame liveness ratchet**: `menu_frame`/`rng_seed`
       monotonicity as the universal "session is alive and paced"
       check for every launcher/watchdog — replaces process-alive
       heuristics that can't see a wedged-but-running core.

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
