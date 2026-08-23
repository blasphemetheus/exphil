# 0823_menu_profile — local 1v1 menu-time profiling + three menu fixes

Bradley's ask: "record where menu time goes, compare to where it
should be, kill the unnecessary waits" — netplay was profiled earlier
tonight (eval_runs/0823_live_observability: 14s retry waste fixed);
this run did the LOCAL 1v1 side and surfaced two live bugs plus one
design flaw, all fixed same night. Bradley's live narration drove
every diagnosis.

## Runs

- local.log — baseline: CSS resolved after ~30s of select/deselect
  "war", then WEDGED at stage select forever (deadzone freeze,
  cursor parked just below BF = FD's fine zone). MENU STUCK fired.
- local2.log — wedged AT the CSS (war never converged; MENU STUCK).
- css_repro (libmelee_ex tmp/mw_local_css_repro.exs) — instrumented
  probe repro: both ports' selected words genuinely cycling 33<->2;
  stream coin_down spuriously TRUE at scene entry even on a fresh
  home (the merge correctly overrides it).
- local3 — off-grid warmup park only: war persisted (P1 in an
  infinite select/deselect loop) -> the collision was aggravation,
  not cause.
- local4 — ALL fixes: clean run to game.

## Root causes (three, stacked)

1. **Metastable pick toggle** (the war): helper A-edges re-fire every
   2 frames — faster than the coin/readback settles — and A over the
   just-placed coin picks it back UP. Self-sustaining
   select/deselect loop on BOTH ports; the status-box click has the
   same structure (cycles HMN->CPU->closed past the readback).
   Diagnosed from Bradley's "dropping the cursor then immediately
   picking it up again". IRONY: the pre-merge code never looped
   because the mainline stream's LYING coin_down=true damped it —
   honest RAM truth exposed the race.
   FIX: `@css_press_cooldown_frames 20` debounce on A-pick /
   B-reclaim / box-click (menu_helper).
2. **Hand collision entropy** (Bradley's spot): CSS hands nudge each
   other; P1's warmup wiggle sat ON the portrait grid, fighting P2's
   working hand for the whole JIT window.
   FIX: local warmup animation parks BELOW the grid first
   (melee_port steer-down before wiggling).
3. **Stage-select deadzone freeze** (single-axis residue of the
   2026-08-09 class): fine tilt 0.22 inside the deadzone just outside
   tolerance, frozen forever next to FD.
   FIX: freeze detection (45 stalled frames) -> full-tilt unstick
   burst (12 frames), steer_toward gains `min_tilt:`.

## local4 profile (scripts/analyze_menu_time.exs, local mode)

    launch -> connected      1.0s
    connected -> local CSS   1.8s
    CSS phase               18.9s   (JIT 19.8s overlaps ~17s of it;
                                     pick landed ~2s after warmup)
    stage select             2.1s   (was: infinite)
    CSS+stage -> in-game    20.9s total, 19.8s of it XLA compile

**Menu overhead beyond JIT: ~3-4s** (was 30s-to-never).

## Standing lesson

Honest readbacks EXPOSE latch races that lying streams accidentally
damp. Any helper press that toggles game state must be debounced
against its readback latency — grep for other 2-frame press edges
before trusting them with merged gamestates.
