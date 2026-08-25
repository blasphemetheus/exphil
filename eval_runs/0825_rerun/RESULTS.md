# 0825 rerun — both arms clean

## A1: stage-internals shakeout — WIRED, NULL-PASS, canary at the new layout

Checkpoint metadata proves the fix: **stage_internals=true,
canary=344 dims** (336 + 7 + alignment), train_delays [2,3,4]. The
canary validated on every sweep snapshot load at the new layout.

Behavior: argmax **ep2 at 439.4/min c440** — ceiling fox retained;
the near-inert channel changed nothing on the fox axis (the pass
condition). Epoch lottery grows: {4, 16, 45, 4, 59, 2}. Mewtwo
confirm drew **0.0 c0** — a new low for the transfer lottery (prior
draws 24-256); one draw, noted not interpreted (if a second
stage-internals run also zeroes, revisit whether the channel widens
the transfer lottery).

`ms_g20si2_ep2.bin` = the first stage-internals-capable checkpoint
(features live only on FoD/PS; behaviorally inert in this pool by
design).

## A2: GENERALIST PILOT — first master-corpus checkpoint EXISTS

20/20 epochs on 300 master-master games (GRU-60, --stage-internals,
overnight's embedding cache reused): best val_loss **4.694**,
descending curve, full artifact set incl.
`checkpoints/fox_gen_pilot1_20260825_102456_best_policy.bin`
(11.9MB — larger net than the specialist line). The non-streaming
scale wall sits between 300 files (fits) and 1,000 (dies at val
placement) — full-corpus runs need --stream-chunk-size or chunked
training.

The generalist line is OPEN: pipeline proven end-to-end on master
data. Next reads (not run here): behavior look vs CPU, Situations
profile, then the scaling question (300 -> 7,911 via streaming).

## Program notes
Guard #9 (unknown-flag abort) is live in the drill — the flag that
was silently dropped overnight now either lands or halts the run.

## Live behavior look (Bradley, 0825_gen_pilot_live)

Played the pilot locally (fixed.bin): **mostly stands idle; crouches
when approached; occasional jab / dtilt / short-hop; jumps sometimes
when airborne; sometimes runs offstage.** Textbook small-corpus BC
mode-averaging (the modal master-Fox frame is neutral/idle-adjacent),
with real context reactions already visible (crouch-on-approach).
The scale-up + (properly wired) features are the response, not recipe
surgery. NOTE: the pilot actually trained WITHOUT stage internals —
train.exs's data pipeline drops the flag (third standalone
config-build; fix owed) — and its metadata claimed otherwise
(canary/params width divergence -> guard #6 spec).
