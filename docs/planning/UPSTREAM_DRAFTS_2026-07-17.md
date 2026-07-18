# Upstream contribution drafts — 2026-07-17 (REVIEW BEFORE POSTING)

Status: DRAFTS ONLY per Bradley. Posting decision gated on the mainline
headless test (task WS5b) — if mainline dissolves our need, the issue is
still community-valuable but reframe from "blocking us" to "FYI + repro".

---

## Draft 1 — Issue for vladfi1/slippi-Ishiiruka

**Title:** ExiAI build: analog triggers dropped on the pipe path; expressed
but never released via EXI input overrides

**Body:**

Build: exi-ai-0.2.0 release AppImage (Ishiiruka 3.5.1 / Slippi 3.19.0),
Linux (NixOS, extracted AppImage), driven via libmelee 0.47.2,
gfx_backend=Null, audio disabled, blocking pipes.

Two related behaviors around analog triggers, with a reproducible
input-replay methodology (we replay both ports' recorded inputs from a
.slp frame-by-frame and diff live state against the recording — exact
divergence detection):

**1. Pipe path: analog `SET L/R` silently ignored.**
Digital `PRESS L/R` works; main/c-stick analog round-trips bit-exactly;
analog trigger values never register in-game (recorded analog shield
holds produce no shield; replay diverges at the shield frame). 13 of 15
recorded sequences containing analog trigger holds diverge exactly at
their first analog-trigger frame; converting those holds to digital
presses makes all 15 replay with zero drift.

**2. EXI input overrides (`use_exi_inputs=True`): analog presses express,
but releases appear to LATCH.**
With the "Allow Bot Input Overrides" gecko active, the same recorded
analog holds DO produce shields — but a 20-40 frame analog hold then
never releases: the character rides shield to a full shield break +
dizzy, every stock, in every game we ran. Hypothesis: zero/neutral analog
is treated as "no override present" rather than "override to zero", so a
release cannot be expressed through the override path. (Sticks do not
show this — neutral stick returns fine — which would fit a
nonzero-sentinel check applied per-field to triggers.)

Happy to provide .slp pairs (recorded source vs replayed output), the
replay-diff harness, or to test candidate fixes. Is behavior (2) known /
is there an explicit override-clear we should be sending?

**[NOTE to Bradley: if the mainline test shows mainline pipes carry
analog fine, add a line: "For our use we've moved headless to Slippi
mainline + Null video, so this isn't blocking us — filed for other bot
developers on the ExiAI build."]**

---

## Draft 2 — libmelee PR: don't raise on unknown install-dir names

`_is_mainline(path)` (console.py:85) raises `ValueError("Unknown path")`
for any install dir whose name lacks "netplay" — e.g. a directory named
`exi-ai/` holding the ExiAI build. The information it guesses from the
dirname is derivable from the exe itself (get_dolphin_version already
classifies builds). Proposed change: on unknown dir names, warn and fall
back to exe-based detection instead of raising; keep the dirname fast
path. ~5 lines + a test. Workaround exists (pass the exe file path
directly), but the error message doesn't hint at it.

## Draft 3 — libmelee docs PR: NixOS + headless notes

README additions:
- NixOS: AppImages can't run directly (no FUSE); `--appimage-extract`
  + a wrapper exporting the five missing libs via nix-ld
  (alsa-lib, libglvnd, libusb1, zlib, gcc-lib) works; point libmelee's
  `path` at the wrapper FILE to bypass the dirname heuristic.
- Headless timing caveat: with Null video + disabled audio the game runs
  unthrottled, paced only by blocking input. SYNCHRONOUS agents (respond
  to each console.step before the next) are timing-exact at any speed;
  ASYNCHRONOUS agent loops quantize their inputs by wall-clock and will
  mistime frame-precise actions — pace your feed loop (e.g. 60Hz) or go
  synchronous.

---

Posting checklist when approved: issue → vladfi1/slippi-Ishiiruka; PRs →
fork altf4/libmelee as blasphemetheus, branches `fix/unknown-path-fallback`
and `docs/nixos-headless`; all under Bradley's account, he reviews final
text first.
