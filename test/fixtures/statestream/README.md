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
parsed↔live mapping (known so far: parsed jumpsquat af 0,1,2 vs live 1,2,3;
action 365 agrees; the offset varies per action — and check whether other
fields shift too). This is the first experiment of task #8 and needs no
Dolphin and no GPU — it is deliberately laptop-sized.

Alignment hint: the in-game frame counter in the trace (`f###`) and the
replay's frame index count the same game, but confirm alignment on an
unambiguous event (the first jumpsquat entry) rather than assuming f0
matches parsed frame 0 — menu frames differ between the two.

Do not regenerate casually: a pair is only valid if the .slp and the trace
come from the SAME run (record with `MULTISHINE_TRACE=1` and keep both).
