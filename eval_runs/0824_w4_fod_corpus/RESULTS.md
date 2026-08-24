# 0824 W4 rung 1 — FoD platform heights: the champion is STAGE-BLIND

INTERP_ROADMAP_V2 W4 (customer #33), first probe, enabled by the
08-24 stage-address work (RAM heights live + peppi per-frame labels).

**Question**: can a linear probe read the current FoD platform heights
from ms_g19_ep4's trunk state? Heights are NOT inputs, so decodability
would mean accumulated inference from indirect cues.

**Method**: 6 FoD replays (3 recorded for this probe + 3 smoke games;
34.8k window-aligned frames, heights spanning the full −1..35 range),
6 equal-width buckets per side, by-replay split (last replay held
out), `Probe.fit_eval`. Controls: shuffled-label floor (GOTCHA #79)
and an INPUT probe (`Activations.input_trunk` — identity over the raw
current-frame embedding).

| probe | left bal_acc | right bal_acc |
|---|---|---|
| trunk | 0.130 | 0.142 |
| input (leakage floor) | **0.273** | 0.151 |
| shuffle floor | 0.117–0.174 | — |
| majority | 0.167 | 0.167 |

**Verdict: STAGE-BLIND, both platforms.** Trunk = shuffle floor,
below majority. The sharp detail: the instantaneous input embedding
DOES leak left-platform height (0.273 — own-y while standing on it),
and the trunk sits BELOW that — the recurrence actively discards the
correlate rather than accumulating it.

**Caveats**: multishine-centric self-play corpus (platform contact is
sparse; cues are scarcer than in platform-heavy play) — but the input
probe proves cues existed in-distribution and the trunk still reads
none. Representational claim only (knowing-side); no causal claim
needed since the finding is absence-at-floor.

**Consequence for #33**: FoD-specific behavior is played blind to
platform state. Concrete feature candidate for the next training
line: append the two height f32s to the embedding — training labels
exist in every FoD replay (peppi fod_platform_*), and live values are
already merged into local gamestates from RAM (netplay pending the
probe verification). Script: scripts/interp_w4_stage_height.exs.
