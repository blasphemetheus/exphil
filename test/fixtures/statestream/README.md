# State-stream reconciliation pairs (task #8 / GOTCHAS #81)

Two recordings of the multishine teacher where BOTH sides of the same game
survive:

- `*.slp` — the Slippi replay Dolphin wrote (what **Peppi** parses; the
  coordinates every trained policy learned in).
- `*.live-trace.log` — the recorder's own per-frame observations through the
  libmelee bridge as the game ran (`MULTISHINE_TRACE=1`; the coordinates a
  policy actually receives at inference). Grep `\[trace\]`; fields:
  `f<in-game-frame> act= af= gnd= y= vy=`. 300 trace lines each (5s runs).

| pair | technique | why it's here |
|------|-----------|---------------|
| `fox_ms_float` | pre-fix teacher: aerial shine on airborne frame 2 → 22-frame float | long, varied air states |
| `fox_ms_frame1` | final teacher: frame-1 shine, 9-frame TAS cycle | the tight loop where every af matters |

Purpose: diff the two streams frame by frame to derive the EXACT
parsed↔live mapping. **Done** — see `mix run scripts/diff_state_streams.exs`
and the table in GOTCHAS #81. `action_frame` is the only field that shifts.

Alignment hint: the in-game frame counter in the trace (`f###`) and the
replay's frame index count the same game, but confirm alignment on an
unambiguous event (the first jumpsquat entry) rather than assuming f0
matches parsed frame 0 — menu frames differ between the two.

Do not regenerate casually: a pair is only valid if the .slp and the trace
come from the SAME run (record with `MULTISHINE_TRACE=1` and keep both).

## Recording NEW pairs (the coverage problem)

These two pairs only cover the Fox multishine loop, so the derived table
covers 9 of 399 action states. Measured share of frames it can normalize:
~77% on the multishine fixture but only **~8-12% on Mewtwo** — the commonest
unmeasured states are everyday ones (14 Wait, 20, 66, 90). Broadening that is
the bottleneck on making the phase-2 fix useful for Mewtwo.

Any live script can now record the live half, not just the multishine
recorder:

```bash
EXPHIL_STATE_TRACE=1 mix run scripts/<any live script>.exs ... \
  > mewtwo_pair.live-trace.log 2>&1
```

Then pair it with the `.slp` Dolphin wrote for that SAME run (newest file in
your Slippi replay dir) and check it:

```bash
mix run scripts/diff_state_streams.exs \
  --slp mewtwo_pair.slp --trace mewtwo_pair.live-trace.log
```

A good pair reports **100% agreement on action / on_ground / y**. Anything
less means the two halves are not the same run (or the port is wrong — set
`EXPHIL_STATE_TRACE_PORT=N`), and the mapping it produces is garbage.

What makes a pair VALUABLE is action-state coverage, not good play: aim for
ordinary Mewtwo movement, shielding, aerials, getting hit, ledge and
recovery — states 14/20/66/90 and friends. Several short varied sessions beat
one long repetitive one. `ActionFrameConvention.unknown_actions/1` tells you
what a recording still leaves unmeasured.

Note: pairs recorded via `EXPHIL_STATE_TRACE=1` carry the TRUE in-game frame
(so they align at offset 0), while the two committed fixtures used a
recorder-local counter (offset -123). Both are fine — the differ anchors on
the first jumpsquat rather than assuming a numbering.
