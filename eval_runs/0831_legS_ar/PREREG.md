# PRE-REGISTRATION — Leg S on the AR head (0831_legS_ar)

**Written 2026-08-31, before any arm runs.** Bradley's direction: "Leg S
rerun and critic follow-ups, coming from a good place with the
autoregressive head."

## The question

How much of the "selection is the ceiling" verdict (Leg S, 08-28: joint
pass@1 14.5 → pass@16 43.7, headroom +29) was actually the **independent
factorization** of the action space, not selection?

The AR head models the joint action; a sampled action is coherent. If a
chunk of the old headroom was "the right components each existed but never
co-occurred in one sample," AR should raise pass@1 and shrink the
headroom without any selector.

## Why this needed instrument work first (done, this session)

`interp_passk` drew all six components INDEPENDENTLY from one forward's
logits. On an AR checkpoint that (a) ignores the conditioning and (b)
reuses one sampled path's conditional logits — it would score the AR head
as if it were independent. New path: `Sampling.sample_autoregressive_n`
(the fused mode-of-N machinery minus the vote) via
`Agent.get_action_samples` (debounce-free, side-effect-free). Test:
key-reproducible, n distinct coherent samples.

## Arms (all three in one launch, same knobs, same corpus, same seed)

| arm | checkpoint | sampling path |
|---|---|---|
| ep10 | `fox_gen_v1_20260825_210355_ep10.bin` | independent (existing) |
| INDhead | `fox_gen_v1.1_INDhead_policy.bin` | independent (existing) |
| ARhead | `fox_gen_v1.1_ARhead_policy.bin` | sequential (new) |

INDhead is the load-bearing control: same frozen trunk, same refit data
and loss recipe as ARhead — the ARhead−INDhead delta isolates the head.
ep10 re-baselines the instrument same-day (and, with per-file fox port
detection, doubles as the fox-only re-read of the 08-28 Leg S numbers,
which used a corpus later shown 43% non-fox on port 1 — E1).

## Knobs

- corpus `replays/erickfm_ranked/FOX/extracted/*.slp`, auto port by
  character (fox, dittos skipped) — NOT `--port 1` (E1)
- n=16, `--temperature 0.5 --buttons-temperature 0.5` (deploy decode)
- `--limit-frames 2000 --limit-files 20 --seed 20260831`

## Pre-registered readings

1. **Joint pass@1: ARhead vs INDhead.** ≥3 pts higher = the conditioning
   shows up in open-loop match (expected direction; the 0.5-nat val gap
   predicts it). Within ±3 pts = unresolved. LOWER would be a surprise
   worth its own investigation (sampling-path bug first — L1).
2. **Headroom (pass@16 − pass@1): ARhead vs INDhead.** Markedly smaller
   (≥5 pts) = part of the old "selection gap" was factorization, and the
   selector program's upper bound shrinks accordingly. Similar = the
   selection question survives intact on the AR base; critic follow-ons
   keep their priority.
3. **ep10 vs the 08-28 numbers** (14.5 / 43.7): drift here is
   instrument/corpus-handling drift (fox-only ports, new day), NOT a
   model claim. Recorded for calibration only.

Declared: no formal cross-checkpoint noise floor exists for this
instrument (seed replication was 0.1 pt on ONE checkpoint); the ±3 pt
threshold is a judgment call made before the numbers. Standing caveats
travel: pass@k is mode-seeking (L9 — never rank a decode on it), an upper
bound, open-loop, and a master-match — only differences and gaps mean
anything.

## Follow-on (decided by reading 2)

Critic follow-ons run AFTER this lands, on the AR base, with sampling via
the same coherent-sample path: if headroom survives, the next arm is the
CRITIC_D2_DESIGN NULL-branch (V + short CycleSim rollouts) over
`sample_autoregressive_n` candidates; if headroom collapses, the selector
program is demoted and the unfreeze (plan 8/9) carries the line alone.
