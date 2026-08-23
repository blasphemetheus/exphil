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
- [x] exphil bridge: MeleePort menu-GameState merge — SHIPPED
      2026-08-23. Pure overlay `Melee.MemoryMap.merge_css/2`
      (libmelee_ex 78dd688: cursor/hover/selected/status/ready,
      strictly additive, only observed fields substitute) applied in
      MeleePort.navigate_menus before any helper sees the gamestate.
      LIVE-VALIDATED offline (tmp/mw_merge_smoke.exs: merged
      coin_down=true + fox locked while the stream says false —
      repairs the dead offline coin_down). Policy: OFFLINE CSS always
      (when a watcher runs); ONLINE CSS behind EXPHIL_RAM_MENU=1
      until the cursor block + selected array are validated at that
      scene (owed next Direct session) — then feedback menuing
      replaces blind arithmetic and the blind fallback becomes the
      no-watcher fallback. Id-space lesson: the selected words hold
      the GAME-external scheme (Slippi/engine, fox=2 falco=20), a
      THIRD space vs internal ids and CSS-grid ids —
      `Character.from_game_external/1` is the converter.
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
       ONLINE VALIDATED 2026-08-23 (live Direct session,
       eval_runs/0823_live_observability): :none -> {:character, 2}
       across the pick AT the online CSS; selection-skip fired on
       every retry; post-game CSS RETAINS the pick (zero A presses on
       rematch cycles). EXPHIL_RAM_MENU now defaults ON online for
       static fields (merge_css fields: :static); cursors stay off
       online until the heap block is park-and-scanned there. #1 is
       CLOSED. The direct-code keyboard fell same night (5-round hunt,
       tmp/mw_codebuf*.exs): typed-text buffer at **0x804A0740,
       STATIC** — 3 bytes/char (SJIS-fullwidth pair + NUL pad),
       NUL-terminated; `MemoryMap.direct_code()` watch set +
       `decode_direct_code/1`; MeleePort logs the readback on change.
       Facts: the field opens AUTOFILLED and the first keystroke
       REPLACES it; the stream's menu_selection IS frozen at the
       keyboard (blind-typing confirmed — the readback is the fix);
       the online CSS hand SPAWNS OFF-GRID (left edge; A there picks
       nothing, START no-ops — rounds 2-3 failed exactly there);
       selection-word entry value can be garbage (26) — only
       :none -> {:character, c} transitions count. Keyboard-CURSOR RAM
       state still unknown — optional (readback + clear/retype covers
       unattended reconnects). ONLINE CSS CURSOR: hunt INCONCLUSIVE
       (offline heap block frozen/denormal online; movement-diff
       candidates weak, logged in tmp/mw_visual/online_hunt.log) —
       DEPRIORITIZED: the adopted pattern is selection-word-feedback
       steering (heading -> A -> read the selected word), which the
       hunt harnesses used successfully.
2. [x] **Menu-scene ground truth for the watchdog** — COMPLETE
       2026-08-23. Diagnosis slice (22b): the MENU STUCK report
       carries `ram_scene` + `ram_traffic_delta`. Suppression slice
       (`ExPhil.Bridge.StuckPolicy`, pure verdict table, 7-test class
       enumeration): a report is SUPPRESSED (info log, no notify) only
       on positive evidence on BOTH axes — traffic > 0 AND a known
       legitimate hold ({:leaving,..} transition in flight, or settled
       online CSS with the blind fallback DONE = post-pick opponent/
       code-entry wait). A suppressed verdict re-arms the helper's
       stuck detector (stalled_frames/stuck_reported reset), so the
       hold is re-evaluated every stuck window — a hold that decays
       into a core wedge alarms one window later. Pinned regression:
       the bot14 wedge class (online CSS, fallback NOT done, healthy
       traffic) still alarms. Matchmaking minors are not in the scene
       taxonomy yet → they classify :unknown → alarm (never suppress
       on noise); extend the table when the scene-word change log
       captures them.
3. [x] **Parser parity, live** — VERIFIED 2026-08-22c (the quartet
       run, libmelee_ex tmp/mw_quartet.exs: one solo CPU game on FD,
       900 arrival rows): stream-vs-RAM bit-exact wherever the value
       did not change across the arrival boundary (stock 100%, action
       94-96%, x/y exact at rest); continuously-moving fields read ~1
       frame FRESHER from RAM (first x mismatch = exactly one frame of
       walk speed) — within-frame phase, not lag (the frame counter
       read +123 on every row). Standing per-frame parity harness =
       re-run the quartet script.
4. [ ] **Scenario-farm verification**: Improoover/scenario setups
       assert exact game state (percent, position, RNG seed) before
       recording a rep — removes inference from drill farms.
5. [x] **RNG seed capture per game** — VERIFIED 2026-08-22c (quartet):
       `rng_seed` (804D5F90) readable at match start and ticking EVERY
       frame in-game (900 distinct/900 rows; the "canary doesn't tick"
       caveat is settled-menus-only). Ready for determinism audits;
       wiring into the eval protocol is a consumer task.
6. [~] **Stage internals live** — PIGGYBACK DONE 2026-08-24: the
       quartet re-run on FoD and PS (tmp/mw_quartet_fod/_ps.exs)
       confirms the in-game player truth, parity profile, and the
       +123 delay constant generalize to the moving-geometry stages
       (FoD: 899/900 at 123 + one 122 — the first jitter sample ever,
       benign boundary race; PS: 900/900). OPEN refinement: the
       stage-internal ADDRESSES themselves (FoD platform-height f32s,
       PS transformation state word) — hunt method: scan MEM1 for the
       stream-reported platform-height bit pattern (>=3.18 streams
       carry it) or slow-drift differential; feeds StageCollision in
       live play on any build/era.
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
10. [~] **Delay-regime measurement** — METHOD VALIDATED 2026-08-22c
       (quartet): `ram_frame = slippi_frame + 123` dead constant across
       all 900 arrival rows (zero jitter) in the local sync harness —
       0x80479D60 counts from scene start, Slippi stamps from -123, so
       any drift from +123 at arrival IS pipeline lag, measured
       per-arrival. NETPLAY SLICE DONE 2026-08-23 (live Direct
       session): 48/48 in-game samples read EXACTLY 123, zero jitter —
       the local pipeline contributes ZERO frames of observation lag
       over netplay; the effective-delay gap (22→4) is entirely
       input-delay regime + rollback/network. Probe is a standing
       MeleePort log line (5s cadence in-game); [x] as a capability.
11. [x] **In-game player ground truth** — VERIFIED 2026-08-22c
       (quartet): the classic locations.csv player block survived
       mainline INTACT (base 0x80453080, stride 0xE90; x/y/facing/
       percent/stock static, action/action_frame via the entity
       pointer at base+0xB0). Shipped as `MemoryMap.game()` (+
       percent/1, stock/1 decoders; libmelee_ex 10f8a17), test-pinned.
       Consumer gotcha: watches read :unknown until first CHANGE
       (percent silent until first damage). #4 (scenario asserts) and
       #7 (outcome channels) are now unblocked.
12. [x] **Liveness ratchet** — SHIPPED 2026-08-22b, simpler than
       planned: `MemoryWatcher.traffic/1`, a monotone
       PARSE-INDEPENDENT datagram count. Key wire fact (pinned by the
       grammar battery): dolphin sends a bare-NUL empty-step datagram
       EVERY step, so raw arrival — no address needed at all — is
       ground truth for "core running and paced"; liveness =
       `delta > 0` between two reads. First consumer: the stuck
       report's `ram_traffic_delta` (#2). Open: adopt in launchers/
       harnesses beyond MeleePort.

## Late-night arc 2026-08-24 (the online-CSS closed loop, condensed)

Full detail in the commit messages (libmelee_ex 7b92410/4035cbf,
exphil 1c18781/7fe20f8) and JIT_WARMUP.md. The laws it minted:

- **The A-press probe is the position sensor** at the online CSS
  (hover byte = stale local residue there; helper steering = frozen-
  snapshot luck). Probe-steering: sweep -> tap -> the selection word
  names the portrait -> B real mis-picks (bounded) -> converge.
- **Per-frame /proc pread beats the watcher** for on-change-hostile
  words (selection): initial values never sent, correction datagrams
  droppable, every seed/staleness heuristic timing-fragile. MEM1
  found once async; 4 preads/frame = microseconds of always-truth.
- **The online CSS's unselected sentinel is 26** (Master Hand — no
  roster mapping); ids without a CSS-roster mapping classify :none.
- **One bounded watcher call per menu frame**: ~10 synchronous calls
  per frame starved the spectator socket (dolphin_disconnected);
  so did an inline /proc scan retry loop. Anything slow runs OFF the
  frame loop.
- **Unknown submenu = WAIT, never B** (transient values during scene
  loads; blind B backed out of Direct in a loop).
- New addresses: `online_menu_selection` 0x804D7788 (Ranked0/
  Unranked1/Direct2/Teams3/Party4; merged onto main-menu gamestates —
  main menu -> online CSS now 0.5s, was ~6s) and `online_menu_depth`
  0x804060E0 (CSS=2, keyboard=3 — the scene-word-invisible
  transition; consumer queued: instant pulse exit + keyboard-strand
  detection).
- XLA executable cache: CONVICTED live (inference hang mid-game,
  inputs latched) — default-off; JIT_WARMUP.md tracks it. Warmup
  stage instrumentation added (the temporal branch holds the whole
  ~20s; breakdown logs on next boot).
- Frozen warmup steer budget: probes own the JIT window; picks land
  MID-WARMUP (g17/g19); post-JIT CSS->typed-code ≈ 4s.

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
- 2026-08-23 (validation session 2, eval_runs/0823_live_observability
  RESULTS §2): early handback 7.0s ×3 cycles; typing readback +
  verified confirm; rematch = zero presses; **GHOST-TEXT STRAND fixed
  live** (AUTO FILL suggestion sits in the RAM buffer but only Z
  commits it — buffer==code can't distinguish; Z interleaved every
  4th confirm press, libmelee_ex c22eb4f; disconnect→re-search
  reconnects cleanly); #10 netplay again 57/57 flat 123. NEXT
  SLIVERS: RAM marker for "at the keyboard" (stream submenu frozen);
  EVENT-DRIVEN blind-CSS timeline (hover/selection/buffer readbacks
  all online-validated → replace the 480/600/900 frame timers;
  ~15s→4-5s post-JIT).
- 2026-08-23 (latest): **local 1v1 profiled; three menu bugs fixed
  same night** (eval_runs/0823_menu_profile/RESULTS.md; Bradley's live
  narration drove every diagnosis). (1) METASTABLE PICK TOGGLE:
  2-frame A-edges outrun the coin readback and A over the placed coin
  reclaims it — a select/deselect war on both ports; the LYING stream
  coin_down had accidentally damped it, honest RAM truth exposed it →
  20f press debounce (A-pick/B-reclaim/box-click). (2) CSS hands
  COLLIDE: p1's warmup wiggle on the grid knocked the dummy's picks
  around → warmup animation parks below the grid. (3) SSS deadzone
  freeze (single-axis 08-09 residue) → 45f stall detection + full-tilt
  unstick burst. Result: local menu overhead ~3-4s beyond JIT (was
  30s-to-never). LAW: any press that toggles game state must be
  debounced against its readback latency — honest readbacks expose
  latch races that lying streams damp.
- 2026-08-23 (late): **code-entry blindness CLOSED + menu-time
  profiler shipped.** Typed-code buffer 0x804A0740 (static, 3
  bytes/char SJIS+NUL) found by dump-diff, confirmed by
  scan-for-visible-text across a replacing keystroke (screenshot
  ground truth caught two failed rounds where the keyboard never
  opened — the hand spawns off-grid at the online CSS).
  MemoryMap.direct_code()/decode_direct_code/1; MeleePort watches +
  logs the readback. Bradley's ask "record where menu time goes"
  shipped as scripts/analyze_menu_time.exs — first run on the
  0823 session log flagged 14s of retry windows after RAM confirmed
  the pick; fixed same hour (BlindCss window-end
  selection-confirmed -> :handback; ~14s saved per CSS cycle, safe
  because the RAM merge lets the helper press START itself). Online
  CSS cursor hunt: INCONCLUSIVE, deprioritized in favor of
  selection-word-feedback steering.
- 2026-08-23 (live session, eval_runs/0823_live_observability):
  **everything validated in production in one Direct session vs
  Bradley** — online selection array (:none -> {:character,2}),
  selection-skip on every retry, post-game pick RETENTION (zero
  A presses on rematch), StuckPolicy suppressing a real 30s connect
  hold (game started 2.4s later), game-end re-arm x2, #10 netplay =
  flat 123 x48 (local pipeline contributes zero), new scene word
  0x08080101 (post-game online SSS flash). EXPHIL_RAM_MENU flipped
  DEFAULT ON online (static fields). NEW open sliver: the direct-code
  keyboard is the last blind menu (details at #1); teardown must also
  kill `[s]lippi-dolphin-bot` (orphan searched a truncated code).
  Ops note: observability sessions MUST launch with --verbose
  (default verbosity sets Logger :warning and eats the science trace).
- 2026-08-23 (later): **#2 watchdog suppression SHIPPED** —
  ExPhil.Bridge.StuckPolicy verdict table + MeleePort wiring with
  detector re-arm (details at #2 above). Program applications now
  fully closed except: #6 (FoD/PS quartet re-run), #10-netplay, #4/#7
  (consumers of #11), #8 (netplay era), plus the online-CSS address
  validation gating EXPHIL_RAM_MENU.
- 2026-08-23: **MeleePort menu-GameState merge SHIPPED + live-validated**
  (libmelee_ex 78dd688 merge_css/2 + from_game_external/1; exphil
  wiring in navigate_menus). Smoke proof: merged coin_down=true with
  fox locked while the stream reports false. Online CSS application
  gated on EXPHIL_RAM_MENU=1 pending next-Direct-session address
  validation. Remaining core plumbing: NONE — the build checklist is
  complete; open program items are #2 (watchdog suppression), #6
  (stage internals via a FoD/PS quartet re-run), #10-netplay, #4/#7
  consumers, #8.
- 2026-08-22c (later): **THE QUARTET RAN — #11/#3/#5 closed, #10
  method validated** in one solo CPU game on FD (900 arrival rows;
  details on each item above; libmelee_ex 10f8a17 ships
  MemoryMap.game()). Casting note: the run played cptfalcon vs
  young_link because MenuHelper `:character` is the INTERNAL id (fox
  0x01) and the script passed external ids — Bradley's live ID caught
  it; verification is character-agnostic. #6 (stage internals) still
  owed — piggyback on a FoD/PS re-run of the same script.
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
