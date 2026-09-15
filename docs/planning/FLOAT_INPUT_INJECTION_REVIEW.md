# Float input injection: review and implementation contract

2026-09-15. Companion to `FLOAT_INPUT_INJECTION_BRIEF.md`.

## Assessment

The diagnosis and separate byte/processed paths are sound. The brief is a
good direction, but was not yet an implementation-ready plan. In particular,
the proposed six-float/16-bit-button record would discard information that
Slippi playback actually restores. The bridge integration is also larger
than described. Resolve the issues below before calling the feature done.

## Verified corrections

1. **Correct assembly source branch.** `project-slippi/slippi-ssbm-asm`
   master has playback code but no `AI/OverwriteInputs` directory. The
   source fork is `vladfi1/slippi-ssbm-asm`, branch `ai-inputs-rebase`,
   commit `0ca638e6a2ca5d97fe3d62d15be86f5f8b402550`. It is now checked
   out at `../slippi-ssbm-asm`. Its older raw hook differs from the shipped
   hook in allocation/debug details; preserve the shipped raw Gecko code.
   Playback was also inspected at upstream
   `fcf47f10dc244152c2ebaa3a9dec142ea42243b7`.
2. **The target is fighter data, not merely a processed PAD struct.**
   Playback restores main X/Y at `0x620/0x624`, C-stick X/Y at
   `0x638/0x63C`, one processed trigger at `0x650`, and a **32-bit**
   processed button word at `0x65C`, relative to fighter data.
   Its hook is `0x8006B0DC`, immediately before recording at `0x8006B0E0`
   and subsequent button processing. Copy input restoration only; do not
   restore positions, damage, or action state to conceal drift. The broader
   test later established that recorded RNG restoration is also necessary
   for idle animations and Peach item outcomes (see work log).
3. **Physical inputs are distinct.** Existing Peppi `l_trigger/r_trigger`
   come from `pre.triggers_physical`, and boolean buttons from
   `pre.buttons_physical`. Neither substitutes for the processed trigger
   or button word. Playback also restores raw stick bytes in the five-slot
   circular buffer at `0x8046B108`, indexed via `0x804C1F78`, for UCF.
   Retain missing raw fields as missing; do not invent zeroes for old replays.
4. **Avoid a lossy round trip through normalized axes.** Original f32
   values promoted to f64 retain their bits, including negative zero and
   tiny values. `(value + 1) / 2` can lose those. Keep original inputs in
   a separate `Peppi.ProcessedInput` struct; leave training fields unchanged.
5. **MeleePort does not use the Bot/Session transport setup.** The suite
   currently uses ENet and pipes, with no numbered direct-pad registration.
   Add explicit direct transport, raw-event protocol, Dolphin flags, port
   registration, and processed-code activation there. Preserve default
   pipe behavior. Add `--direct-inputs` for byte transport controls and
   `--float-ports 1,2` for explicitly selected recorded ports.
6. **A frame must be atomic.** Separate 0x01 and 0x02 commits can release
   lockstep after only one port arrives. Use a complete mixed-port packet.
   A second EXI query must read the snapshot already latched by D9; it
   must never wait for or consume another frame. Validate full packets
   before changing state; test partial/coalesced receives, duplicate ports,
   disconnects, menu transitions and float-to-byte handoff. The original
   parser's byte flags were persistent, not cleared each frame as stated.
7. **Do not allow pipe fallback for a float port.** In D9, float ports
   explicitly produce nine zero bytes before considering the pipe pads.
   DA provides the 44-byte record for the later fighter hook. Restore the
   physical fields separately so re-recorded inputs can be audited too.
8. **Exactness needs a stronger gate.** The old comparison rounded sticks
   to four decimals, positions to two, truncated action frames, ignored
   missing data, and always exited successfully. Compare unrounded exposed
   fields, fail missing frames/players, and return a nonzero exit status.
   Compare only through the last prefix frame: handoff 344 means last 343,
   not 400. Even this proves exposed-state equality, not all emulator memory.
9. **Recorded RNG is necessary too.** The v1 coverage test established that
   CPU-to-human slot changes affect outcomes despite exact inputs: Marth's
   idle-animation frame counter and Peach's item outcomes differed. v2
   restores the recorded pre-frame RNG seed at the same hook, before the
   recorder. It does not restore positions, damage, or action state. The
   failing Marth and Peach prefixes then matched exactly in `smoke11`.
   Reject unsupported follower cases until explicitly tested.
10. **Build and launch checks matter.** The existing fork has a duplicate
    `m_slippiRngSeed` member declaration, preventing compilation; remove
    the duplicate only. The CMake target is `dolphin-nogui` and its output
    is `build/Binaries/dolphin-emu-nogui`. On this NixOS host use that
    headless build in a separate install rather than the GUI AppImage
    packaging script. Record binary and Gecko hashes and reject a stale
    or unsupported install before sending mixed frames.

## Protocol v2

All multibyte fields are big-endian. The legacy 0x01 byte encoder remains
unchanged. The suite's explicit `--direct-inputs` and `--float-ports`
sessions use 0x03 for complete frames, including byte-only menus and the
post-handoff response. Both enable the processed hook; byte-only records
leave its per-port flags disabled. Default pipe sessions retain their path.

Direct packet: `03 count {port mode data}...`, count 1..4, unique ports 1..4.
Mode 0 carries the existing 8 raw bytes; mode 1 carries the record below.

| Offset | Size | Field |
| --- | --- | --- |
| 0, 4, 8, 12 | 4 each | Main X/Y, C X/Y, original game-unit f32 |
| 16 | 4 | Processed trigger f32 |
| 20 | 4 | Processed button word |
| 24 | 2 | Physical button word |
| 26 | 2 | Reserved, zero |
| 28, 32 | 4 each | Physical L/R trigger f32 |
| 36..39 | 1 each | Signed raw main X/Y, C X/Y |
| 40 | 4 | Original pre-frame RNG seed |

EXI command DA takes no payload and returns four records of
`u32 enabled + 44 bytes`, 192 bytes total. It never drains the socket.
D9 alone commits a new frame. Unspecified ports in a mixed packet have
no direct override. A controller's processed override expires when the
console snapshots it; `release_all` also clears it.

The new hook is assembled from source, with an independent ELF relocation
check. This matters: Gecko's assemble/objcopy workflow can silently turn
an unresolved symbol into a zero address. `FN_ShouldRecord` requires
`Recording/Recording.s`, in addition to `Common/Common.s`.
The 192-byte DMA buffer is explicitly aligned to 32 bytes and occupies
separate cache lines from the stack back-chain and saved registers.

## Verification order

1. Parser bit-preservation tests; legacy packing tests; mixed-record tests;
   malformed/fragmented packet tests; override lifecycle tests; exact-audit
   tests including structs, missing data and signed zero.
2. Build a separate `exi-ai-float` installation. Verify its binary/Gecko
   hash manifest before float-mode launch. Preserve `exi-ai-flush`.
3. Byte controls on the new binary, first with original pipe settings,
   then explicit direct byte transport. Establish frame timing independently
   of float precision. Use the existing delay-4 candidate, not retraining.
4. Raw CPU Marth handoffs 344 and 1026, then additional CPU sources.
   Require exact inputs and exposed state throughout each prefix plus
   zero suite drift. Check game-start and game-end transitions, and
   processed-to-byte release at the response boundary.
5. Cold/warm 12/12 regression controls, existing parity/timing checks.
6. Only after these pass, rerun coverage from raw rollouts in a fresh
   directory. Split held-out data by source game to avoid prefix leakage;
   preserve old manifests/candidate hashes and compare the same held-out
   situations where possible. The existing 24/24 and 22/24 are comparison
   points, not guaranteed scores for a different dataset.

## Work log

Implementation and smoke artifacts: `eval_runs/0915_float_input/`.
Failed or interrupted attempts are retained in separate directories.

- Rust parser tests: 2 passed. Focused ExPhil tests: 11 passed, including
  bit preservation, unchanged training conversion, hash checks, strict
  comparison, and asynchronous replay-write retry.
- libmelee_ex: 119 doctests, 3 properties, 574 tests, zero failures,
  71 excluded. Standalone C++ decoder tests passed with warnings as errors.
- `smoke07`: raw CPU Marth handoffs 344 and 1026 passed. Strict comparison
  matched both ports' complete inputs (including raw bytes) and all exposed
  player state on all 383 and 1065 prefix frames, respectively. These v1 checks did
  not include RNG in the audit and did not restore RNG or positions.
- `controls01` pipe cold/warm and `controls02` mixed direct byte cold/warm:
  12/12 each, zero drift/errors/invalid timing. All four sets produced chains
  `[14,14,14,14,14,14,13,13,13,13,13,13]` on the existing delay-4 candidate.
- The original 0x01 direct mode failed during startup in `controls01`.
  Its encoder/receiver semantics remain unchanged; it is not the suite's
  new direct transport. The successful direct controls use atomic 0x02.
- Startup must retain lockstep, without an extra commit on GAME_START.
  After GAME_END, mixed-mode polls release lockstep because scene transitions
  may produce no frame events. The console also flushes at GAME_END.
- `coverage01` was stopped before training: the immediate teacher audit
  raced Dolphin's asynchronous replay writer and rejected incomplete files.
  Completed files audited exactly. `smoke08` isolated this failure;
  `smoke09` passed with the bounded 20 × 50 ms retry now implemented.
  Missing, corrupt, or ambiguous evidence still fails closed.
- `coverage02` completed 77 raw-source teachers: all inputs matched, but
  only 69 prefixes matched every exposed state field. Four exclusions were
  Marth/Peach RNG cases; four human-source prefixes differed in the sign of
  a zero velocity. The run exported qualified clips, then deliberately
  halted before training to address RNG.
- **v2 adds recorded RNG.** Each float record appends `pre.random_seed` and
  the hook writes it to `0x804D5F90` at the pre-frame boundary. No other
  state is restored. `smoke11` matched both ports' inputs, RNG, and exposed
  state through Marth r2 handoff 1454 and Peach r2 handoffs 2580/2633.
- `smoke10` exposed an assembly DMA-buffer alignment bug when the record
  grew. The corrected hook reserves extra stack space and aligns its
  buffer to 32 bytes, isolating cache invalidation from saved stack data.
- `coverage03` repeats transport controls and the fresh raw-source pipeline
  on v2. Its source-disjoint split remains declared in `SPLIT.md` before
  outcomes; only exact, valid teacher runs qualify for training.
  All 48 transport controls passed with unchanged chains and zero drift,
  errors, or timing failures. All **71/71 CPU-source prefixes** matched
  inputs, RNG, and every exposed player-state field. Overall **73/77**
  prefixes matched, with zero emulator errors and 69 qualified teachers.
  The four exclusions are the same human-source `speed_y_attack` signed
  zeros at frame 783; v2 preserves the strict bit comparison.
- `smoke12`: mixed byte/float mode (`--float-ports 2`) passed both Marth r1
  handoffs 344 and 1026, including normal game finalization. The byte port
  remains subject to the full exposed-state check.
- v2 training: 43 training handoffs and 26 source-disjoint held-out
  handoffs; 41,219 supervised targets including canonical and original
  clips. The unchanged 21-epoch recipe finished at loss `3.84974e-4`.
  All 104 training clips passed the frozen early-fit gate. Held-out
  first-18 conditional accuracy averaged 85.7%, minimum 22.2% (52 clips).
  Original cold/warm controls passed 12/12 each with unchanged chains;
  original interruptions reached chain ≥10 in 6/6 runs. New training
  handoffs reached chain ≥10 in 81/86 runs, with 86/86 exact prefixes.
  Held-out neutral-opponent runs reached chain ≥10 in 52/52 runs, with
  52/52 exact prefixes. These stages had no errors, rejected prefixes,
  or invalid timing. Held-out pressure evaluation finished at **42/52**
  chains ≥10, also with 52/52 exact prefixes and no errors or invalid timing.

## Final coverage result

The input-injection implementation passes its gates: 71/71 raw CPU prefixes
are exact, all 48 transport controls pass, and all 190 new coverage policy
runs retain exact prefixes. The unchanged policy recipe remains imperfect
under pressure. Its weaker cases are now measured from verified starting
states rather than obscured by replay divergence.

| Gate | Previous 09-14 coverage | Raw-source v2 coverage |
| --- | --- | --- |
| Original cold / warm controls | 12/12 / 12/12 | 12/12 / 12/12 |
| Original interruptions, chain ≥10 | 6/6 | 6/6 |
| Training handoffs, chain ≥10 | 75/78 | 81/86 |
| Held-out neutral opponent, chain ≥10 | 24/24 | 52/52 |
| Held-out replay opponent, chain ≥10 | 22/24 | 42/52 |

These are different datasets and splits: the previous set used regenerated
ghost games and held out handoffs within games; v2 uses raw games and holds
out each roster's entire r2 game. The table does not establish a causal
performance improvement or regression. All 220 final policy runs had valid
timing, no emulator errors, and no rejected prefixes. The 30 original
regression runs retain the legacy pipe/tolerance checks; the 190 new runs
use the strict float/RNG audit. The chain threshold is the historical
coverage metric, distinct from the suite's `pass` score.

Under pressure, chain misses were Falco r2 frames 1219 (one run) and 2179
(two); Marth r2 1454 and 1551 (two each); Peach r2 1014 (two); and Samus r2
309 (one). Of 26 hit-start trials, 18 re-entered during the initial hit
episode and eight were interrupted again; 15 resumed within 60 frames of
the grounded, zero-hitstun readiness proxy. This proxy is not an exact
actionable-time measurement. Training-set misses were Peach r1 2750 (two),
the human source at 302 (one), and the older Fox r1 source at 439 (two).

The next policy experiment should target sustained pressure and the
near-edge/stalled re-entry cases, with a new held-out set; do not tune this
candidate on the held-out cases above and continue claiming they are unseen.
The declared 21-epoch budget was preserved. No candidate was installed as
the default policy.

Artifacts: `eval_runs/0915_float_input/coverage03/summary.json`, generated
by its Elixir `summarize.exs`, contains denominators and every chain miss.
Launchers, source/checkpoint hashes, per-run audits, and recovery reports
are in the same directory. Candidate:
`coverage03/round21/candidate.bin`, SHA256
`366800e253762bac3a5f6c55f89713e715545d79073eb7881581e537568776db`.

## Build and use

### Commit review (2026-09-15)

Reviewed all four repositories and fixed a disconnect edge case: Dolphin's
raw-pad reader now rejects a disconnected client, matching the processed
reader. Rebuilt Dolphin and reran the focused ExPhil tests (11 passed),
full libmelee suite (574 tests, 119 doctests, 3 properties; zero failures,
71 excluded), Rust tests (2 passed), and standalone C++ decoder test.

The reviewed dependency commits are libmelee `2fcc935`, Dolphin `893d910dc`,
and ASM `f392bde`. The ASM fork is now `blasphemetheus/slippi-ssbm-asm`,
branch `ai-inputs-rebase`; the upstream remote remains available.
The rebuilt installation is `~/.local/share/slippi/exi-ai-float-review-v2`.
Its binary SHA-256 is
`26d8cf78683828f0feb32f20c1ee35b295d473f255b0d68c86253a3b8b7f89f5`;
the Gecko and ASM hashes are unchanged. `review_smoke.sh` reproduces its
Marth/Peach checks. Older installations and experiment outputs are retained.
Reports larger than 5 MB remain local, with checksums in
`eval_runs/0915_float_input/large_reports.sha256`; compact summaries,
prefix audits, manifests, and launchers are committed.

Build the fork's `dolphin-nogui` target and run `build-float-inputs.sh` in
the ASM checkout. Install using Elixir (1.18+, for its standard JSON module):

```sh
elixir scripts/install_float_dolphin.exs \
  --dolphin ../slippi-Ishiiruka --asm ../slippi-ssbm-asm \
  --out /absolute/path/to/a/new/float-install
```

The installer refuses an existing destination, copies the Sys tree, replaces
only the named processed-input Gecko code, and records binary/INI/source
hashes. The suite verifies the binary and INI hashes before direct launch.
This is a host-local Nix build with build-tree runtime dependencies, not a
portable AppImage. The manifest `float-input-v2.json` records the build hashes. The installer
also requires the binary's v2 capability marker and the v2 hook description.

On this host, `~/.local/share/slippi/exi-ai-float` points to the validated
`exi-ai-float-rng-v2b`. All earlier installs are preserved; `exi-ai-flush`
is unchanged. The `coverage03` launcher pins this new v2 install directly.

Add `--float-ports 1,2 --no-pipe-shim --dolphin PATH` to the suite to replay
both recorded ports exactly; use `--float-ports 2` for a mixed byte/float
prefix. Each completed float run writes `prefix_audit.json` and rejects its
score on audit failure. Raw stick-byte and recorded RNG equality are checked only for float ports.
Byte ports may reconstruct different raw pairs with identical processed axes
and do not explicitly restore RNG; all their other input/state fields are checked.

Current supported scope is two-player Final Destination sources with raw
stick fields, without Ice Climbers/followers. Unsupported sources fail
upfront. Equality covers the input and state fields exposed by Peppi;
it does not assert that all emulator memory matches.

## Focused verification commands

Run ExPhil's Mix commands only when no training/evaluation BEAM is active.

```sh
EXLA_TARGET=host mix test test/exphil/eval/replay_prefix_audit_test.exs \
  test/exphil/eval/float_input_build_test.exs test/exphil/data/processed_input_test.exs
cargo test --manifest-path native/exphil_peppi/Cargo.toml --offline
(cd ../libmelee_ex && mix test)
c++ -std=c++11 -Wall -Wextra -Werror \
  ../slippi-Ishiiruka/Tools/tests/processed-pad.cpp -o /tmp/processed-pad-test
/tmp/processed-pad-test
```
