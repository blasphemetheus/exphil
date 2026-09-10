# v2 prep list

The v2 recipe (Bradley-approved direction, 09-04): **BPTT x full corpus
x name conditioning**, GRU, days-scale. This is the checklist between
here and launch. Architecture stays GRU unless a named trigger fires
(BPTT_LOADER_DESIGN / lever discussion 09-04: infeasible wall-clock ->
Mamba's parallel scan; or plateau far below slippi-ai parity).

## Done
- [x] Contiguous-BPTT training path (loader/carry/loss/val/inference
      zero-init) — BPTT_LOADER_DESIGN.md, all tested.
- [x] Name conditioning (filename tags -> registry -> name one-hot;
      cache-key guard) — 150e83c.
- [x] Corpus downloaded: FOX 7,911 + MARTH 29,356 + FALCO 42,547 +
      ZELDA_SHEIK 21,535 (~101k games on disk).
- [x] Style-fingerprint instruments + ditto measurement (69.2% ->
      positional tags rejected) — 927bdc5, STYLE_IDENTITY.md.
- [x] `ExPhil.Data.SubjectResolver` — the port-vs-role chokepoint
      (explicit/identity/character ladder, loud failures, provenance;
      pipeline migrated). **The law: ports exist only at the parse
      boundary; everything downstream speaks subject/opponent.**

## Throughput verdict (09-04 profiling session — REPLACES the fused-kernel priority)

The "~3.3k frames/s, 5 days/epoch, sequential unroll dominates" premise
was FALSIFIED by stage profiling (`EXPHIL_BPTT_PROFILE=1`, GOTCHA #112):

- The jitted train step (fwd+bwd+optimizer, AR heads, Axon-unrolled
  GRU) was only **18.6ms** at B=128 T=80. The isolated carry backbone
  value_and_grad is 22.7ms (~450k frames/s) — the unroll was never the
  bottleneck.
- **98.5% of each step (1.9s)** was `TrajectoryCursors.next_batch`
  slicing the chunk embedding at 128 varying offsets — eager EXLA
  compiles one executable per distinct slice start (recompiles every
  batch). Fixed with one `Nx.take` gather (offsets as runtime data):
  0.4ms/step, ~60x whole-step speedup, loss trajectory identical
  (val 8.1083 vs 8.1088, seed 905, 200-file probe).
- Post-fix steady state: **~30ms/step ≈ 337k supervised frames/s**.
  New epoch-wall dominator: per-chunk parse+embed (~110s per 184-file
  chunk; steps for that chunk total ~6s). Full-fox-corpus epoch ≈
  ~1.3h, not 5 days.
- **Next throughput lever = chunk prep, not kernels**: overlap
  parse+embed of chunk k+1 with training on chunk k (ChunkPipeline
  does this for the windowed path; the bptt branch bypasses it), or
  cache embedded chunks (disk-size math needed at full corpus).
  The fused-GRU-h0 CUDA work (bhn operand + h0 threading + H>256 fix,
  landed in edifice 09-04) is now a LATER optimization — it can only
  shave part of ~19ms/step; the custom-call tier also still needs the
  EXLA-fork defimpl + kernel link to exist at all (native_impl? is
  false — every "fused" scan in a defn silently runs the pure-Nx
  fallback; the Mamba trigger's "parallel scan" advantage is measured,
  see below).

## Architecture profiling pass (09-05, Bradley-directed; full data in
## logs/profile_train_step_20260905*.json + session notes)

Six backbones on the real train graph (w60/b64/h256, honest per-arch
precision). Custom-call tier REVIVED (defimpl + kernels linked into the
exla fork, opt-in via `EDIFICE_FUSED_CUSTOM_CALL=1`; f32-only; GRU fwd+
bwd verified 2.8e-6 from f64 truth at H=512, selective_scan fwd+grads
verified — `edifice test/edifice/cuda/fused_custom_call_gpu_test.exs`):

| arch | step (flag on) | compile | verdict |
|---|---|---|---|
| mlp | 0.7ms | 4.5s | reference floor |
| gru | 16.5ms (6.5 off) | 6.0s (41.5 off) | TRADE: kernel = 7x compile win, 2.5x step loss (64-block launch underfills 170 SMs). Flag ON for shakeouts/drills, OFF for long windowed runs. BPTT unaffected (carry backbone = Axon.gru). |
| xlstm | 3.1ms | 36.6s | compile-bound; slstm kernel exists but xlstm.ex never calls it + its custom_grad unwired — DEFERRED |
| mamba | 6.3ms (10.8 off) | 6.3s | PURE WIN — flag on always |
| ttt | 5.5ms | 35.0s + 34s/shape | fused kernel dead code under defaults (`output_gate: true` gates it); unstable-arch class (lr 5e-7) |
| retnet | 2.8ms | 10.0s | plain-Nx O(L²) decay matrix; cheap at w60; no work needed |

Notable: the bf16 accident measured a third GRU config — custom-grad
arm WITHOUT the kernel = 3.6ms step (best), but forfeits the compile
win (the forward unroll is the compile cost). No free lunch; documented.

**Recommendation for larger runs**: GRU stays the v2 seat (only
BPTT-carry arch; kernel trade irrelevant there). Mamba = confirmed
fallback with the flag on (6.3ms step, 6.3s compile, 8.9ms inference).
retnet = surprise dark horse for windowed experiments (2.8ms/10s, zero
custom kernels). xlstm/ttt not competitive on compile cost without
further kernel work.

## v2 READOUT (2026-09-07) — stopped at epoch 7 of 8 (plateau: val
## 1.848/1.857/1.832/1.845 over ep3-6; best = ep5 1.8317)

Checkpoint `fox_gen_v2_20260906_000910_best_policy.bin` (28,452 deduped
files, clean loss). Config sidecar SYNTHESIZED from v16a's (stopped
before end-of-run write; identical embed flags). Bradley live look (6
games, T=1.0, stateful): "multi-jab less but saw it once; plays a lot
more passively — stands there, turns away." Instruments
(eval_runs/0907_pathology/, logs/loops_0907_v2/):
- **Scale WON on rate**: jab1/min 0.74 (corpus 0.84; v16a 3.16). Spam
  stays dead (d_up 0.52/min, taunts 0.26 mean/0 median).
- **Scale LOST on passivity — 3x worse**: idle(WAIT) frac 0.169 =
  6.5x corpus 0.026 (v16a 0.058); frozen-input 0.31 (v16a 0.11);
  dominant cycle WALK_SLOW>STANDING, max 30 reps. Cascades: untouched
  deaths 2.5/game (6x corpus), recovery 0.648. This is the
  neutral_weight trade predicted in item 0, amplified by scale: WAIT
  is a sticky absorbing mode under per-frame sampling once its
  per-frame downweight is gone.
- facing-away frac 0.369 vs corpus 0.392 — turning away is
  expert-like; the pathology is turning away AND idling.
- Jab probe on v2: late window closed (0.05->0.01), chain-window
  median halved (0.13->0.06) but mean 0.125 (fat tail) and early
  bucket overshoots (0.75 vs expert 0.46) — scale learned the timing
  DIRECTION not its magnitude; bucketized action-frame embedding
  remains the targeted lever (run plateaued, more epochs won't).
- **Decision: bisect neutral_weight ONLY** — v16b shakeout
  (`exphil-v16b-neutral`, v16a twin with --neutral-weight 0.5) running
  09-07 11:30. Readout = idle/frozen/untouched vs v16a 0.058/0.11/1.33
  and d_up must stay ~0.5. If 0.5 is insufficient -> 0.25 (with the
  clean button knobs). Then v2 relaunch with the one change.
- Deferred controlled A/B: projectile-block zeroing at the live
  boundary (FIXES.md P0) — deliberately NOT applied for this readout
  to keep the v15/v16a comparison clean.

### Passivity bisect VERDICT (2026-09-08): reweighting is the WRONG lever
- v16b (neutral 0.5) vs v16a (neutral 1.0), 6 AFK-gated games vs 9:
  idle 0.062 vs 0.058, frozen 0.12 vs 0.11, recovery 0.82 vs 0.80 —
  NULL. Passivity does not manifest at the 2.5k shakeout rung at all,
  so the rung cannot bisect it.
- Corpus-composition check (pathology_scan by source dir): partner-
  derived foxes idle 0.033-0.055 vs FOX-ranked 0.026 — a 1.3-2x shift,
  explains a sliver of v2's 0.169, not the bulk.
- **`scripts/probe_wait_exit.exs`** (new): p(leave WAIT) on 946 EXPERT
  settled-standing windows. Expert per-frame leave rate 0.026. v2:
  0.037 (1.4x; f11-30 bucket 1.05x) = CALIBRATED. v16a: 0.063 (2.4x,
  over-eager). v16c (transition 3.0): 0.067 (2.55x; median 0.054 vs
  v16a 0.040 — the directed lever DID raise p(leave) as designed).
  => v2's passivity is mechanism (b): calibrated on expert contexts,
  wrong only on its own self-generated states (compounding). Scale
  actually sharpened p(leave) DOWN toward the true expert rate; v16a's
  over-eagerness was masking the problem at small scale. The entry
  INTO WAIT (via OOD drift), not the exit rate, is the fault.
- Consequence: neutral_weight / transition_weight cannot fix it (they
  only move the exit rate on in-distribution contexts, and it is
  already right). Levers that address compounding, in cost order:
  (1) AWBC (`--awbc --awbc-reward standard`, exists) — outcome-weighted
  frames, chunk-local, forces the serial path; (2) DAgger-style
  on-policy data (drill infra exists; needs a labeling source for
  general play); (3) RL fine-tune with KL-to-BC anchor (slippi-ai's
  answer; parked infra). Offstage deaths are a DIFFERENT mechanism
  (rare-state coverage) where frame upweighting CAN work — keep that
  arm.
- Standing regression check: re-run probe_wait_exit on any future
  arm; model/expert ~1 on expert contexts + high live idle = compounding
  signature, don't reach for weights.
- **REFINED 09-08 afternoon (pathology_scan WAIT entries/dwell +
  probe_wait_exit --min-af 0, aligned +-2-bucket neutral band):**
  passivity is DWELL, not ENTRY — WAIT entries/min corpus 18.6 / v16a
  16.5 / v2 18.6 (identical); dwell frames/entry corpus 5.0 / v16a
  12.5 / v2 32.8 (6.6x). On expert contexts v2's p(leave) is AT OR
  ABOVE expert in every frame-in-WAIT bucket incl. entry (f0 1.36x,
  f1-2 1.14x) — the "experts leave sharply at f0-2 and the model
  smears it" hypothesis is NOT supported. On v2's OWN replays its
  realized per-frame leave rate (0.034) ~ expert (0.036) and the
  model's p(leave) on those states is 0.051 (1.5x) — the earlier 4x
  gap was the LOOSE neutral band (center-only) overcounting model
  leaves. Remaining puzzle: per-frame leave rates match experts yet
  episodes last 6.6x longer => many bot "leave" inputs must FAIL to
  end the WAIT state (sub-walk-threshold stick, no-op presses) or the
  long-idle tail (f31+, where BOTH model and experts assign ~0.5-2%)
  is reached far more often. Next instrument: on bot replays, fraction
  of non-neutral-input WAIT frames NOT followed by a state change
  within 3 frames, vs corpus. Candidate 2 (carried state vs zero-init
  window) still untested. Frame reweighting remains ruled out.
- **FAILED-EXIT INSTRUMENT (09-08, `scripts/failed_exit_scan.exs`) —
  the pin.** During WAIT: experts give a non-neutral input on 6.6% of
  frames, the bot on 2.0-2.7%; of those attempts experts fail to exit
  within 3 frames 49.5%, the bot 70-82%. Kind mix: experts 47% mid-stick
  / 39% FULL-stick / 9% button; **v2: 99% mid-stick, ZERO full-stick in
  88 attempts across 3,314 WAIT frames** — the bot NEVER dashes out of
  standing. Then probe_wait_exit with a stick-magnitude readout: the
  model assigns p(full deflection) 0.0136/frame on expert WAIT contexts
  (expert 0.0125 — CALIBRATED, 1.09x) and 0.0115/frame on its OWN
  standing states (fresh windows) / 0.0052 under full-game carry
  (`--carry`). Realized: 0 in ~3,300 frames (P ~ 1e-7 at 0.5%/frame).
  Sampler config checked: T=1.0 every head, mode_of_n off. => The
  discrepancy is between the model's distribution and the LIVE PATH,
  not training, not carry, not the sampler knobs. Prime suspect =
  FIXES.md P0: the projectile block (zeros in training, populated
  live, every frame) — an OOD input that can plausibly push the stick
  heads toward center. **A/B switch landed: `EXPHIL_ZERO_PROJECTILES=1`
  zeroes the block at the live boundary (agent.ex embed_game_state).**
  Prediction if it's the cause: full-stick attempts out of WAIT appear
  (~1%/frame), failed-exit frac drops toward 0.5, dwell drops toward
  corpus. If not: remaining live-path candidates are the af convention
  (:live vs :parsed action_frame off-by-one) and the stateful step vs
  windowed features (parity-tested, but re-check under this lens).
- v16d (offstage_weight 4.0; `--offstage-weight` + Data.frame_offstage?
  landed 09-08) trained; readout = recovery rate / untouched deaths vs
  v16a 0.80 / 1.33 — awaiting live games.

### ROOT CAUSE FOUND (2026-09-09): delay-0 LABEL LEAK — GOTCHA #113
- Headless rig validated as a live proxy (v2 headless idle 0.188 /
  dwell 34.7 / entries 19.5 / frozen 0.30 vs live 0.169 / 32.8 / 18.6 /
  0.31). NOTE: eval_live_protocol.sh defaults to --deterministic
  (argmax collapses) — pass `--temperature 1.0` as the RUNNER option.
- Projectile zeroing (EXPHIL_ZERO_PROJECTILES=1): NO effect on
  passivity (headless A/B). Same-frame re-stepping: not real (guard
  exists; conf 0.03 = 1/31 cached-return dilution).
- `scripts/probe_sampler_wait.exs` = the real live path offline (step
  trunk + real sampler + real decode). Sampler/decode fine (dash
  contexts 99.8% full). In WAIT the model puts 99.97% of main_x on the
  exact center bucket. Then the alignment discovery: Slippi records the
  input on the frame it PRODUCED (91/91 dash inputs sit on the DASH
  frame, 0 on the WAIT frame). Successor-aligned: experts dash out of
  WAIT on 13-29% of frames, model 0.01-0.02%. v2/v16* all trained at
  frame_delay 0 = leaked labels. ms_g19 line trained {2,3} = causal.
- The failed-exit instrument's definition is INVALID (state-ending
  inputs land on the successor frame by construction); its dwell/idle
  numbers (state-based) stand. Same-frame calibration probes are
  blind to this class — use successor-aligned labels.
- **v16e VERDICT (09-09 15:40): CAUSAL LABELS FIX THE PASSIVITY
  MECHANISM.** val 2.4153 (harder target — not comparable). Successor-
  aligned sampler probe on expert WAIT frames, real live path: model
  p(dash-out) v2 0.0002 -> v16e 0.11-0.15 vs experts 0.13-0.29
  (~1000x, within 1-2x of calibrated). Headless card vs CPU9 (sampled,
  2 games/arm): idle frac v2 0.188 / v16a 0.032 / v16e 0.019-0.021
  (corpus 0.026); WAIT dwell 34.7 / 6.7 / 3.2 frames (corpus 5.0);
  successor-aligned failed-exit 0.43 / 0.20 / 0.01-0.035 with
  dash-outs now the dominant exit (68 of 104 attempts at d1); frozen
  0.30 / 0.09 / 0.045; controller changes 53% / 70% / 77%. At
  --frame-delay 1 (deploy-matched) games ran 2x longer (16.4k vs 8-10k
  frames per 2 games) — it survives. Small-n caveats: v16e d0 showed a
  taunt uptick (0.33/min mean, one game) — watch at scale.
  NOT fixed by causal labels: the jab chain — successor-aligned jab
  probe: experts re-press A from jab1 at ~1%/frame in every bucket
  (the old 46% at f0-5 was the leaked press that STARTED jab1); v16e
  predicts 0.21 in the chain window, v2 0.15 -> still ~16x. That one
  remains the action-frame-resolution lever (bucketized embedding).
  **Recommendation: relaunch v2 (28k files) with causal labels —
  v3 = v2 recipe + causal labels.** Deploy at --frame-delay 1.
  **RENUMBERED 09-09 evening (INVARIANTS item 1 structural):** the
  parser now emits causal pairs by construction and `--frame-delay` is
  reaction delay on top, so v3's training flag is the DEFAULT
  (`--frame-delay 0` == v16e's old `--frame-delay 1`); the deploy
  flag stays `--frame-delay 1` (live N = reaction N-1; the Agent
  derives delay-id 0 for a causal checkpoint at that rung).
- v16e = v16a twin + `--frame-delay 1` (causal labels), launched 09-09
  ~14:30. Readout: (1) probe_sampler_wait successor-aligned on
  expert contexts — model full-X at WAIT should jump from 1e-4 toward
  0.13+; (2) headless v CPU9 at frame-delay 0 and 1: dwell/idle,
  jab chain, dash-outs; (3) jab probe (chain-window leak should
  shrink — the A-on-jab1-f0 label was the same leak). If it lands, v2
  RELAUNCHES with --frame-delay 1 (and the whole fox_gen line's live
  looks get re-read as leak-limited).
- ms_g20 prep landed: dagger_drill.exs `--head autoregressive` and
  `--clean-loss` (opt-in; g19 recipe untouched).

## Open (ordered — reordered 09-05 after the v15full live look + taunt audit)

0. **Loss-knob ablation shakeout — VERDICT: CLEAN LOSS WINS, v2 ADOPTS
   IT (09-05 evening).** fox_gen_v16a_cleanloss (2 ep, 2,500 files,
   val 2.4165->2.2501 descending, NO button collapse). Bradley's live
   look (9 games, T=1.0, stateful): "doing some things that are pretty
   good," one taunt total. loop_report vs the morning v15full T=1.0
   bucket: d_up 80.6 -> 0.49/min mean, 0.00 median (~165x drop, now
   ~10x corpus's 0.05 instead of 1600x); taunts 0.72 -> 0.08/min
   (matches his count exactly; corpus 0.02); loops ~0; the
   GRAB_WAIT>GRAB_PUMMEL cycle GONE (top residual "cycle" =
   WALK_SLOW>STANDING x1, benign). Reports:
   logs/loops_0905_v16a/ vs logs/loops_0905_v15full/.
   ONE metric moved the other way: frozen-input frac 0.00 -> ~0.11 —
   exactly what neutral_weight 0.25 was fighting (passivity). Watch it
   at v2 scale; if passivity emerges, neutral_weight is the single
   candidate to partially restore — NOT the button knobs.
   v2 loss config = label-smoothing 0 / no-focal / pos-weight 1s /
   oversample 1.0 / entropy 0 / neutral 1.0 / stick-edge 1.0.
   Original item text (audit numbers + mechanism) follows:
   the taunt audit (09-05): corpus taunts
   0.02/min vs bot 0.5-0.7/min (~30x), corpus d_up 0.05/min vs bot
   ~82/min (~1600x). Mechanism: the v1-era anti-collapse stack is all
   still DEFAULT (`label_smoothing 0.1`, `button_pos_weight :auto`,
   `focal_loss`, `action_oversample 3.0`, plus `entropy_weight 0.01`,
   `neutral_weight 0.25`) — every knob deliberately inflates
   rare-button probability, and 60 draws/s turns any floor into spam.
   REFINED 09-05: label smoothing was ALREADY disabled for the button
   head (networks/policy/loss.ex ~:300 — the smoothing x pos_weight
   optimum-shift pathology was found before; buttons opt out), so the
   live suspects are pos_weight (:auto = sqrt((1-p)/p) capped at 30;
   d_up sits AT the cap, and weighted BCE shifts the rare-button
   optimum by ~w — the ~30x executed-taunt factor matches), focal,
   oversample, entropy. Bonus fix landed: explicit
   `--button-pos-weight 1,1,...` lists NEVER worked (raw Elixir list
   reached defn — latent since the flag existed); normalize_pos_weight
   in imitation/loss.ex tensorizes them now; targeted tests green.
   Arm = v15 shakeout twin (same resume/corpus/seed/epochs 2) with the
   FULL clean loss: smoothing 0, no focal, pos-weight 1s, oversample 1,
   entropy 0, neutral 1.0, stick-edge 1.0. Readout = behavioral
   (d_up/min + taunts/min via loop_report; val loss NOT comparable
   across loss functions). Caveat recorded: the twin resumes from v1
   (10 epochs WITH the stack) — a big d_up drop is decisive; no drop is
   ambiguous (baked-in init) -> from-scratch pair next. If clean is
   stable (no button collapse at BPTT scale), v2 adopts it; if it
   collapses, bisect the stack.
1. **v1.5 shakeout readout** (relaunch post-fix): stability + steps/sec
   at batch 128 -> the v2 wall-clock budget; carry-threaded val
   descending. (Partially superseded: the 09-04/05 profiling already
   fixed throughput; the cleanloss shakeout (item 0) doubles as this.)
1b. **BPTT chunk prep overlap/cache — WIRED 09-05 (smoke pending).**
   The bptt branch in Pipeline now routes through
   ChunkPipeline.stream_prepared_chunks when `pipeline_chunks` (default
   on; escape hatch = the existing flag) — parse+embed of chunk k+1
   overlaps training on chunk k, order preserved so cursor/carry
   semantics unchanged; `--cache-streaming` rides along. Serial
   fallback kept for awbc / flag-off.
2. **Corpus quality-filter + dedupe pass — DONE 09-05.**
   `scripts/filter_corpus.exs` (tier A metadata + sha256 dedupe; --deep
   = damage>=100, has-winner, percent sanity; symlink out-dir +
   append-only manifest.jsonl, resumable). Full-corpus run
   (101,349 files, 745s): **kept 64,117; DUPLICATES 33,129 (33% of the
   corpus was double-counted!)**; rejects: no_winner 2,324, too_short
   1,127, not_1v1 580, low_damage 43, unparseable 29. Output:
   `replays/erickfm_ranked/v2_filtered/` (+ REPORT.md). v2 trains from
   this dir. The dup rate alone changes effective corpus-mix math.
3. **Migrate remaining port-assuming call sites to SubjectResolver**:
   scorecard scripts (each has bespoke picking), drill scripts
   (explicit port 1 stays but via the :explicit rung), fingerprint
   script (drop its copied character table), agent live-port discovery
   (verify the launch flag against the actual seat at game start —
   loud mismatch, not trust).
4. **Style identity plan** (STYLE_IDENTITY.md) once GPU free:
   metadata-residue probe -> fingerprint corpus -> calibration ->
   perceived-player clustering -> ditto 2-way assignment ->
   `--player-tag-map` wiring.
5. **Yeti corpus identity sort** (Bradley, 09-04): local-setup replays
   (NO connect codes/netplay names — recorded on Slippi setups).
   Identity sources there: (a) in-game 4-char NAMETAGS when players
   entered them (the SubjectResolver :identity rung already checks
   :tag), (b) otherwise fingerprint clustering — and this corpus is
   the ideal CALIBRATION/DEMO target because Bradley knows the players
   and can name clusters from a single game each (human ground-truth
   oracle). Needs: corpus location from Bradley, then
   scripts/style_fingerprint.exs over it + cluster report.
6. **v2 launch config** from the shakeout throughput: corpus mix,
   batch, epochs budget; registry over the FULL tagged corpus.
   **Tag-frequency histogram COMPUTED 09-05** (scratchpad + numbers in
   HANDOFF/status): FOX dir (7,911) is fully ANONYMOUS (hashed
   `master-master-*` filenames) — identity there only via fingerprint
   clustering (item 4 is therefore the real conditioning lever, not the
   registry cap). Tagged fox-subject games live in the partner dirs:
   178 tags / 2,102 games; 86 tags at >=5 games -> **112 slots fit with
   a min-games threshold ~5**. MARTH/ZS use the same `[TAG] Char`
   bracket convention (parser-compatible; MARTH-side 4,698 tagged, ZS
   874). In-v15-corpus per-tag exposure is tiny (top: [314] 33 games,
   [C2] 27, [EASY] 25; [INFP] 7) — v15 conditioning = sanity check
   only, [314] is the strongest-fed tag for a live A/B.
7. **FilenameTags multi-convention extension — DONE 09-05.** Measured
   grammar of FALCO's 42,547: 16,374 standard `A + B` (5,648 bracket
   tags — parseable all along), ~10.8k `vs`-forms (Game_-style paren
   tags + tournament long-form `SOPH Marth (White)` PREFIX tags with
   costume parens), 15,386 bare `Game_*.slp` (no identity, nothing
   recoverable). Parser now handles all three conventions with an
   exact-case denylist (costume colors + stage abbrevs; "(Red)" drops,
   "(RED)" keeps) and longest-first char tail-matching ("Young Link" >
   "Link"). 21 tests green (7 new).
7b. **Pathology scan verdicts (09-05 evening;
   `scripts/pathology_scan.exs`, reports in eval_runs/0905_pathology/)**
   — expert Fox corpus (345 games) vs all bot arms:
   - **Multi-jab is a REAL pathology, not undertraining noise, and the
     clean loss did NOT fix it**: experts continue jab1->jab2 12.7% of
     the time and rapid-jab 0.3% per jab; every bot arm continues
     ~88-100% and rapid-jabs ~80-100% (v16a: 0.878 / 0.805). ~7x
     continuation, ~270x rapid-jab.
     **PROBED 09-05 evening (`scripts/probe_jab_conditional.exs`,
     n=700 expert jab1 frames): mechanism = WITHIN-STATE TEMPORAL
     SMEARING, not a prev-action loop** (hypothesis died at the config:
     use_prev_action=false for the whole fox_gen line). Bucketed by
     frame-within-jab1: experts press A at 46%/frame in f0-5
     (double-tap residue, doesn't chain) then 0.0% in f6-11 (the chain
     window) and 0.0% late; the model matches f0-5 almost exactly
     (0.44 vs 0.46 — calibrated!) but bleeds 0.16 mean into the chain
     window and 0.05 late, where experts have a hard cliff to zero.
     1-(1-0.16)^6 ~ 65% chain from the window alone — reproduces the
     observed ~88% continuation. WAIT control clean (p(A)=0.006).
     The model reads "A happens during jab1" but not WHEN — the
     action_frame signal (1-2 normalized dims) is too weak to carve a
     cliff at f6. v2 levers, in order: (a) scale may sharpen it for
     free — RE-RUN THIS PROBE on the v2 checkpoint (it takes ~90s);
     (b) if not, bucketized/one-hot action-frame embedding makes the
     cliff linearly learnable (recipe change, small). Generalizes: any
     within-state mistiming (early smash releases, mistimed techs)
     shares this signature.
   - **Recovery gap confirmed**: recovery rate ~0.78-0.80 vs expert
     0.959 (5x death-per-episode), untouched deaths 0.75-1.33/game vs
     0.417. Rare-STATE coverage problem -> candidate recipe lever =
     offstage frame weighting via the existing per-frame-weight seam
     (train-side, no decode hacks).
   - away-drift frac is NOT a clean pathology metric (experts drift
     away MORE, 0.189 — they chase edgeguards); dropped from the
     watchlist.
8. **Commitment/chunking lever (NOT v2-gating; named trigger)**: if
   residual dithering survives v2 (clean loss + conditioning + scale),
   the per-frame-independent-sampling mechanism is the next candidate —
   ACT/diffusion heads (already in-repo) predict k-frame action chunks
   and execute them, replacing 60 independent draws/s with committed
   segments. slippi-ai proves per-frame CAN look coherent, so scale
   first.
