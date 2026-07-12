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

| When | Milestone | Owner |
|---|---|---|
| Week of 07-14 | Netplay plumbing: `--connect-code` through bridge + play script; bot Slippi account created | Claude / Bradley (account) |
| Week of 07-14 | E3 trained with netplay delay profile (`--online-robust`, delay ~18 like Phillip; + curriculum mixing + prev-action-dropout 0.3) | Claude (overnight run) |
| Week of 07-21 | **Couch test**: Bradley on the Latitude direct-connects to the bot at home; feel + latency verdict | Bradley |
| Week of 07-21 | Fixes from couch test; friend remote-tests | both |
| Week of 07-28 | Dress rehearsal (one more couch session with the final checkpoint) | Bradley |
| First Thursday of August | **Yeti debut** — friendlies setup, Latitude + connect code | Bradley |

## Technical checklist

- [ ] Bridge: `connect_code` config → libmelee `menu_helper_simple(connect_code:)`
      navigates Slippi Direct; `copy_home_directory` so the temp User dir
      carries the logged-in Slippi account; `online_delay` actually passed to
      `Console` (currently read but unused).
- [ ] Play script: `--connect-code CODE#123` flag; dummy/port-2 logic bypassed
      for online (opponent is remote).
- [ ] Bot Slippi account (needs email verification — Bradley). Suggested tag
      with style: distinct from SPKN so opponents know it's the bot.
- [ ] Delay-robust model: netplay ≈ 6–18 effective delay frames vs our local
      delay-2 training. E3 flags: `--online-robust` / high `--action-delay`
      (Phillip: 18–21). This IS the planned E3 — netplay just fixes its
      delay target.
- [ ] Robustness pass: matchup variety (corpus is all-character ✓), no
      SD pathologies (recovery mixing), sane behavior on non-FD stages
      (drills are FD-only; corpus isn't — verify live).
- [ ] Couch test protocol: 3 games, note lag feel, exploits found, and the
      "is this fun to play against?" verdict.
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
