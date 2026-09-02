# Live look — fox_gen_v1.3_ARrefit (Bradley, 2026-09-01 night)

Setup: FD, local d0, T=0.5/0.5; replays in `2026-09-Mainline/`.

## Impressions

- "Feels a little bit different." Neutral: **starting to do okay things
  sometimes** — good sign, but not often.
- **First observed real punish structure: up-throw → up-air.**
- Shield-grab spam persists but has structure now: shield-grab maybe
  twice, then **rolls back** (a defensive reset instead of pure spam).
- Survival mixed: sometimes survives, sometimes airdodges offstage,
  sometimes side-Bs offstage (consistent with the score's ambiguous
  died% read — neither v1.2-clean nor regressed-clearly).
- **THE HEADLINE ASK: punishes cap at 1–2 hits.** The corpus contains
  deep 10-hit combos (people going for guaranteed things); the bot
  never continues. Platform-fox artifacts EXIST in its behavior (SH
  dair, SH multishine fragments) but are never USED as tools —
  "whatever is being learned is a shallower layer than what is there.
  How do we show it the deeper layers?"
- Bradley also asked to re-turn the decode knobs (mode-of-N, critic
  selector) on the fixed trunk.

## Instruments launched in response (unit v13-knobs)

1. **F4 combo-depth probe** (`scripts/combo_depth_probe.exs`) — match
   rate on expert states stratified by hits-into-punish; flat = the
   continuations are learned and the gap is state-visitation (drills /
   DAgger lever); collapse-with-depth = BC never learned the deep layer
   (curation / AWBC lever).
2. **Decode-knob ladder rerun on v1.3-ARrefit** with a critic retrained
   on CLEAN extracts (all prior ladder numbers were dirty-trunk and/or
   dirty-instrument) → `eval_runs/0901_critic_v13/`.

## Mode-of-N live session (same night, ~23:30)

- `--mode-of-n 16` and `8` on v1.3-ARrefit (VERIFIED: live path uses the
  AR-aware coherent sampler — sample_autoregressive_from_features honors
  mode_of_n with joint draws; this was NOT chimera-voting).
- **Verdict: argmax-class live collapse.** Runs off the stage a lot,
  rolls to the stage edge repeatedly, "pretty boring," "resolves to the
  same behavior as argmax where it just kills itself a lot."
- L9 confirmed again: offline mode-of-16 was the biggest free win
  (18.3 vs 5.2) and it is live-dead. Mode-of-N FAILS the live gate;
  the critic selector remains the only untested-live knob (different
  failure profile — learned score, not frequency vote — but tonight
  lowers expectations).
