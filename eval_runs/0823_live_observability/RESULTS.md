# 0823_live_observability — Direct session vs DBTD#411 (ep4, d4-id3)

Purpose: the "next Direct session bundle" from HANDOFF_2026-08-22c —
online-CSS address validation, netplay #10 delay decomposition, live
exercise of everything shipped 08-22c/08-23. NOT a performance rung
(2 casual games, no chain scoring claimed).

Log: g1b.log (--verbose; g1.log = 40s false start at default
verbosity, which sets Logger to :warning and eats every science line —
CLI.setup_verbosity default; always launch observability sessions with
--verbose).

## Findings

1. **Online-CSS selection array VALIDATED** (the EXPHIL_RAM_MENU
   gate's question): `css_p1_selected` (0x8043208C) read `:none` at
   the press point, `{:character, 2}` (fox, game-external) on every
   retry after the press landed — both semantic directions, live at
   the online CSS. The BlindCss selection-skip fired on every retry
   ("SKIPPING A press") — retries are now pure START re-pulses.
2. **Post-game CSS RETAINS the pick**: first rematch press point read
   `{:character, 2}` — the bot pressed A ZERO times that cycle. The
   old open-loop arithmetic would have "re-picked" (= deselected fox)
   here. Note the tension with the 08-22 "post-game CSS unpicked"
   incident: RAM says retained in this flow; reading beats assuming
   in both worlds.
3. **#10 delay over netplay: ram_frame − stream_frame = 123 on 48/48
   samples, zero jitter** — identical to the local-sync baseline. The
   local pipeline contributes ZERO frames of observation lag over
   netplay; the effective-delay gap (the 22→4 chain question) is
   entirely input-delay regime + rollback/network.
4. **StuckPolicy suppression validated live**: 30s hold at the
   connect wait → "menu hold (:online_wait) … watchdog re-armed"
   (info, no notify) → game started 2.4s later. Correct verdict on a
   genuinely legitimate hold.
5. **Game-end re-arm** (RAM word transition) fired cleanly on both
   game ends.
6. **Scene taxonomy additions**: boot chain ({:leaving, :boot,
   :press_start} → main_menu → online CSS), online in-game
   0x08080104 re-confirmed, and NEW: post-game online SSS flash
   **0x08080101** (major 8 minor 1) ~70ms between game-end and the
   rematch CSS. Zero-word transients appeared twice (now labeled in
   the log).

## Open item found: code entry is the last blind menu

After Bradley exited, the bot re-entered the direct-code keyboard,
re-typed "DB…" of DBTD#411, and the teardown killed the beam
mid-typing — the ORPHANED dolphin (pipe gone, last input latched)
confirmed and sat searching "DB". Two lessons:

- **Teardown pattern gap**: this bot's dolphin (EXPHIL_NETPLAY_HOME)
  matches NEITHER "libmelee_" nor "play_dolphin_async" — kill
  `[s]lippi-dolphin-bot` too.
- **enter_direct_code steers by the STREAM's menu_selection at an
  online scene** — same GOTCHA-#101 exposure class as the CSS. It
  visibly live-steered tonight (typed D, B correctly), so the field
  may be honest on this build, but it is unverified and the code-entry
  scene has no RAM coverage at all. Next memory-watch sliver: derive
  the code-entry keyboard state (park-and-scan or dump-diff at the
  keyboard) so unattended reconnects are closed-loop.

## Shipped as a result (same night)

- `EXPHIL_RAM_MENU` default flipped ON for the online CSS,
  **static fields only** (`merge_css/3 fields: :static` — selection/
  hover/status/ready; cursors stay off online until the heap block is
  validated there; `=full` opts in, `=0` disables).
- Zero-word scene-log annotation ("load noise").
