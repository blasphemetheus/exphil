# Yeti Debut — someone at Yeti Weekly plays the bot (target: August 2026)

The commitment: by August, a human at Yeti Weekly (Thursdays) plays the bot.
Route: **Slippi Direct netplay, Phillip-style** — the bot runs at home on the
5090; the venue side is any ordinary Slippi netplay client (the Latitude
suffices). The bot never travels.

## Why this route

- Production-proven: vladfi's Phillip ships exactly this way.
- Dissolves the hardware question — no porting, no ONNX-on-laptop, no
  hauling the desktop.
- The bot can also play anyone, anywhere, any time — friends can test
  remotely before the debut.

## Schedule

> **Status 2026-08-03:** technically READY — every gate below the
> account/couch rows is green and the model side is done (ms_g6_sp1).
> Remaining are the two human-scheduled items: a remote friend session
> (doubles as the task-#12 ping measurement; ACAB#182 is owed a match)
> and the debut date itself. The original dates below have slipped;
> the plan holds.

| When | Milestone | Owner |
|---|---|---|
| Week of 07-14 | Netplay plumbing: `--connect-code` through bridge + play script; bot Slippi account created | Claude / Bradley (account) |
| Week of 07-14 | E3 trained with netplay delay profile (`--online-robust`, delay ~18 like Phillip; + curriculum mixing + prev-action-dropout 0.3) | Claude (overnight run) |
| Week of 07-21 | **Couch test**: Bradley on the Latitude direct-connects to the bot at home; feel + latency verdict | Bradley |
| Week of 07-21 | Fixes from couch test; friend remote-tests | both |
| Week of 07-28 | Dress rehearsal (one more couch session with the final checkpoint) | Bradley |
| First Thursday of August | **Yeti debut** — friendlies setup, Latitude + connect code | Bradley |

## Technical checklist (updated 2026-08-03)

- [x] Bridge: `connect_code` config → libmelee Direct navigation, account
      home via `EXPHIL_NETPLAY_HOME`, `online_delay` passed to Console —
      DONE and verified live (2026-08-01 Direct loopback games).
- [x] Play script: `--connect-code CODE#123`; dummies auto-disabled for
      online. DONE.
- [x] Bot Slippi account: EXPH#288 (verified in the 08-01 games).
- [x] Delay-robust model: **the 18-21-frame Phillip target is RETIRED**
      (2026-08-03 ladder verdict: SS-on-queue doesn't ladder past d4,
      and realistic Direct = 2-4 frame buffer + intrinsic 2). The
      production policy is `ms_g6_sp1` (multi-delay {2,3}): d2 434.5
      c434 / d3 413.4 c409 / d4 332.4 c313 with `--delay-id-override 3`.
      Launch recipe: HANDOFF_2026-07-31 ops + `--frame-delay 3`,
      qtrace on, `analyze_qtrace.exs` between games (rung = peak − 2).
- [ ] Robustness pass: matchup variety (corpus is all-character ✓), no
      SD pathologies (recovery mixing), sane behavior on non-FD stages
      (drills are FD-only; corpus isn't — verify live).
- [x] Couch test: run 2026-08-02 (Bradley vs then-champion mdq_ss) —
      verdict "very cool"; observation: two modes vs humans (sparse
      shines vs entering the loop). Replays LOST to the temp-dir trap —
      future human sessions MUST pass --replay-dir (GOTCHA #84).
      Remaining before debut: one session vs ms_g6_sp1 + a REMOTE
      friend session for the real ping measurement (task #12).
- [ ] Failure fallbacks for the night: pre-verified connect code, phone
      hotspot if venue wifi is hostile, and a recorded exhibition replay as
      plan C.

## Open questions

- Slippi Direct from the venue: most weeklies have netplay laptops; confirm
  Yeti's setup or bring the Latitude.
- Which character to debut: Fox (best drills) vs Mewtwo (the project's
  heart, crowd appeal, weaker model). Could offer both.
- Netplay stage selection flow under libmelee (Direct mode striking) —
  verify during couch test.
