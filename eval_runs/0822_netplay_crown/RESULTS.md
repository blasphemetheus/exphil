# ep4 netplay validation session (2026-08-22 afternoon)

ms_g19_ep4 over Slippi Direct (loopback-grade), --frame-delay 4
--delay-id-override 3, vs Bradley (DBTD#411). NOT a blind A/B — a
single-arm validation that turned into a menu-infrastructure repair
session (see "Infrastructure" below).

## Scores (replays = ground truth; Bradley-side ~/Slippi/2026-08/)

| game | bot port | shines/min | max chain | note |
|---|---|---|---|---|
| 16:25 (18.9k frames) | 2 | 22.3 | **23** | Bradley's session |
| 16:35 (7.2k frames) | 1 | 45.4 | **12** | POKEMON STADIUM — multishines on a transform stage at netplay delay |

qtrace: lag peak SHARP 6 (99.3/99.4%) = d4 + intrinsic 2, both
sessions — first row of the ping->knob table (loopback-grade ping).
qtrace applied-B runs read 212/245 — COMMANDED cycles, wildly above
landed shines; the "score chains from replays, never qtrace presses"
rule vindicated again.

## Verdict

ep4 is NETPLAY-VIABLE: no collapse, correct lag regime, chains 23/12
(inside g15's typical netplay distribution {3,1,46,6}) — but n=2 and
no record; TODAY DOES NOT DECIDE THE CROWN. The crown match = blind
netplay A/B (ep4 vs g15, 0809-style), now cheap to run with the
repaired flow. ms_g15 remains champion-of-record.

## Infrastructure repaired en route (the session's real yield)

1. **Online-CSS state feed: DIAGNOSED (GOTCHA #101).** Raw-dump solo
   repro: the menu payload streams but only a frame counter changes —
   the CSS field region is a one-shot scene-entry snapshot holding the
   account's PREVIOUSLY SELECTED character. The build (same
   netplay-beta as 08-09) never streamed live CSS state; historic
   "feedback" menuing was the timed make-do branches muddling through,
   and it wedged today because the bot's own past fox picks poisoned
   the snapshot into reading fox pre-pick. Not a parser regression;
   the 08-17 session is exonerated. Blind fallback is now DEFAULT ON
   for online; real feedback would need dolphin memory watches (not
   currently justified).
2. Blind CSS fallback (EXPHIL_CSS_BLIND_FALLBACK=1): helper steers
   (parks on target vs frozen feedback) -> ONE A press (A toggles the
   pick — even counts deselect, learned live) -> START pulses ~5s ->
   hand back to the helper for the code/name-entry scene. Worked
   end-to-end twice.
3. Loading-animation warmup hold at the CSS (Bradley UX): menus
   navigate immediately; during JIT the cursor plays
   EXPHIL_LOADING_ANIM (infinity = two cardinal-leg diamond lobes —
   sinusoids drift under the nonlinear cursor response; square-wave
   segments close exactly; r=0.18 to stay off portraits). Candidates
   wiggle/circle/nod await Bradley's pick.
4. menu_helper online-CSS "locked in" branch now requires a selection
   signal (coin_down or ready banner) — the hover byte alone read as
   "selected" and mashed START forever.
5. Known traps re-hit: pipe latches last stick state (center before
   blind presses); pkill self-match in compound commands; Logger :info
   suppressed by default verbosity ate two debug attempts.

## ADDENDUM: evening session (bot12, first memory-watch netplay run)

ms_g19_ep4, same knobs (d4/id3), vs DBTD#411. Two games, replays
Bradley-side (Game_20260822T190919 / T191049).

| game | frames | shines (self) | self/min | max chain |
|---|---|---|---|---|
| 19:09 | 4,951 | 47 (43) | 31.3 | 4 |
| 19:10 FD | 28,924 | 712 (704) | **87.6** | **62** |

**CHAIN 62 = ALL-TIME NETPLAY RECORD** (old: 46, ms_g15 08-09; ep4
prior best 23), on FD. qtrace lag peak sharp 6 @ 99.5% (d4+2, correct
regime). Single-arm — the crown still awaits the blind A/B, but ep4
now holds the netplay record outright.

### Memory-watch wiring: first live validation (the session's purpose)

1. Watcher engaged over netplay (Locations.txt + socket in the bot
   home; scene classifications flowed all session).
2. Online CSS reads a SETTLED major-8 scene word — scene-controller
   layout confirmed at a second scene family, live.
3. BlindCss retries (2) fired and were SPURIOUS-BUT-SAFE: the pick had
   landed; **the CSS -> code-entry transition does NOT change the RAM
   scene word** (departure unobservable at the Direct CSS pre-match).
   Deterministic consequence: always 3 A presses, odd count, ends
   selected — game 1 started 9s after handback. Table behavior
   retained; noise, not damage.
4. **Enriched MENU STUCK fired live**: ram_scene={:settled,
   :slippi_online_css} ram_traffic_delta=15 (core healthy @60/s,
   screen named, genuinely menu-stuck) — and diagnosed a REAL BUG:
   css_blind_done/n/retries persist across games, so the post-game
   CSS sits unpicked forever (masked yesterday by per-game bot
   relaunches). Fix: reset blind-CSS state on scene re-entry.
5. No raw scene words were logged (only classifications) — byte-order
   + code-entry-minor science still owed; add hex logging pre-next
   session.
