# 2026-09-10 — multishine under the causal recipe, gated under SAMPLING

Bradley's direction: "best version we can of the multishine bot, local
play, no delay", with v3's recipe. Prereg: `run_g20.sh` (arms a/b),
`../0910_g21_sharp/run_g21.sh` (arms a/b). All numbers: stand-fox FD,
headless sync runner, 60 s games, ShineChain v3; T=1.0 unless marked.

## 1. The g19 record was an argmax artifact

| policy | decode | rung | shines/min | max chain |
|---|---|---|---|---|
| ms_g19_ep4 (the card) | argmax | d3 id3 | 437.4 | 438 |
| ms_g19_ep4 | T=1.0 (x2) | d3 id3 | 44.9 / 42.9 | 3 / 2 |
| ms_g19_ep40 | T=1.0 | d3 id3 | 101.9 | 9 |
| ms_g19_ep58 | T=1.0 | d3 id3 | 126.8 | 12 |

A 438-chain is ~4,000 frame-exact inputs; only a near-deterministic
distribution survives that under sampling. Per-state probe
(`scripts/probe_ms_state_confidence.exs`): g19_ep4 sits at 0.77-0.94 on
the loop decisions, ep58 at 0.90-1.00. Bradley (12:xx): "relying on
deterministic was a silly thing to do" -> `gate_sweep.sh` now samples at
T=1.0 by default (`GATE_TEMP=0` restores argmax). **The peak-at-ep4 /
decay-after lore (0820) was the argmax metric**: under sampling the LATE
epochs are better, monotone with loss (0.21 -> 0.010).

## 2. g20 arms (g19 recipe + causal labels + clean loss; rungs {0,1,2,3}, 24 ep)

| arm | head | SS | d1 sweep (24 ep) | argmax ep | notes |
|---|---|---|---|---|---|
| g20a | autoregressive | no | 79-100/min, chain 1-4, FLAT | ep11 99.9 | rung-insensitive (d1/d2/d3 all ~80-95), temperature-insensitive (T=1/0.5/0.25 ~83; argmax WORSE 56 c1) |
| g20b | independent | 0.5/10 | 78-103/min, chain 1-3, FLAT | ep13 102.9 | offline fixture agreement 0.888 @offset 2 (g19: 0.887 @5; g20a 0.777 @4) |

Trace (`trace_multishine.exs`, g20a ep11 vs g19 ep4 at T=1): JC 39 vs 29,
shines-from-jumpsquat 19 vs 9, empty hops 35 vs 26 — both jump-cancel
fine; the cycle breaks AFTER the jump.

Per-state confidence (issued-input semantics; "fixture B@off" = the
fixture's issued B rate at that offset; the policies track offset 2):

| state | g20a p(B) | g20b p(B) | g19_ep4 p(B) | g19_ep58 p(B) | fixture B@2 |
|---|---|---|---|---|---|
| ground reflector af1 | 0.37 | 0.91 | 0.87 | 1.00 | 1.0 |
| ground reflector af2 (release) | 0.38 | 0.02 | 0.82 | 0.90 | 0.0 |
| jumpsquat af0/1/2 | 0.97 | 1.00/0.93/0.96 | 0.83-0.87 | 0.94-0.99 | 1.0 |
| aerial reflector af1/2/3 | 0.33-0.37 | 0.99 | 0.89/0.94/0.77 | 0.96/0.98/0.91 | 1.0 |
| reflector-open af0 | 0.22 | 1.00 | 0.13 | 0.06 | 1.0 |

- **g20a (AR head)** genuinely fails: p(hold B) ~0.33 through the aerial
  reflector -> releases mid-air -> the 18-frame wind-down. Argmax makes
  it worse (release whenever p<0.5): 56/min c1. AR head is OUT for the
  ms line.
- **g20b (independent)** is decisive and correct (0.91-1.00) and STILL
  chains 1-3: a 9-frame cycle at ~0.95/frame chains ~3. Chain length
  under sampling ~ 1/(1-p_cycle); a real multishiner needs ~0.999/frame,
  i.e. a loss floor near 1e-3, not 1e-2.

## 3. Where the loss floor comes from (label conflicts)

`scripts/probe_ms_label_dilution.exs` over the 27-replay rollout pool
(expert-relabeled, issued semantics): aerial reflector af1-3 -> B=1.0
(7,380 frames); reflector-open af0 -> B=1.0 (2,414); reflector-open
af1..12+ -> B=0.0 (**32,000 frames** — the bot FLOATS in rollouts). Two
conflicts follow:
1. reflector-open af0 (hold) vs af1+ (release) differ by ONE frame; the
   1/60-scaled action_frame scalar cannot separate them -> a blend
   (g20a 0.22, g19 0.06-0.13 there). The bucketized action-frame
   embedding (`--action-frame-buckets`, wired into dagger_drill today)
   is the direct fix — arm g21b.
2. Multi-rung shifted labels: at base+2..5 the label for an aerial-shine
   state reaches into the float (release) for the higher rungs and stays
   hold for the lower -> the same state is labeled both ways across ids,
   and the id channel is weak -> g20a's 0.33. Single rung = arm g21a.

## 3b. CORRECTION (13:05): every "d1" gate above ran delay-id 1, not 0

The sync runner (`scripts/play_dolphin.exs`) still copied `--frame-delay`
into `delay_id` (the async script was fixed 09-09; this one was not).
So g20a/g20b "d1" = live d1 at id 1 (label offset 3); "d2" = id 2
(offset 4); "d3" = id 3 (offset 5). g21a (trained id 0 only) was REFUSED
by the untrained-id guard on its first sweep — which is how this
surfaced. Fixed (the runner passes nil; the Agent derives id N-1 for
causal checkpoints). The per-state probe used `--delay-id 0` explicitly
and is unaffected. g20b ep13 is being re-gated at the true id 0.

## 4. g21 (single rung 0, 60 ep, gated late under sampling)

- g21a (g20b recipe, --multi-delay "0"): loss 0.17 (ep20) -> 0.08 (ep40)
  -> 0.017 (ep56) -> 0.029 (ep60): NOT below g19's 0.010 (a quarter of
  the pool; SS + AWBC make the loss noisy). Sweep at the TRUE id 0, d1,
  T=1.0: ep8..60 all **chain 1** (86-114/min; argmax ep48 113.9). g20b
  ep13 at true id 0: 89.9/min chain 1 (vs chain 3 at id 1). VERDICT: the
  single-rung quarter-pool arm converged worse and does not chain; the
  local rung is not obviously id 0 either (both ids chain <=3, not
  decisive). g21b NOT run.
- **g22** (`../0910_g22_converge/run_g22.sh`, 13:10): g20b recipe (4
  rungs, full pool) to 60 epochs — the direct causal successor of the
  g19 late epochs; gate late at d3 id 3 against g19_ep58 (127/min c12)
  and at d1. Arm b adds --action-frame-buckets 24.

## 5. Harness (GOTCHA #114)

Every gate before 11:27 ran the NETPLAY AppImage headless (global
`DOLPHIN_DIR`); it needs a display and died when X went stale. Sweeps
now FORCE the exi-ai headless build, kill TERM-immune survivors per
gate, and run as `systemd-run --user` units (no display needed; retires
the tool-shell 10-min/memory-kill chunking).

## Open

- Whether the loss floor is reachable by data cleanliness (fixture vs
  expert-table one-frame disagreement; float-heavy rollouts:
  `--rollout-cap-per-state` exists) or needs a sharpening term.
- Local look (Bradley) at the best sampled snapshot at `--frame-delay 1`.
- The Agent derives id N-1 at live N for causal checkpoints; for the
  drill line g20b tracks offset 2 at id 0 and physically d1 needs offset
  2 — consistent. g19 (legacy) at d3 uses id 3 (offset 4).
