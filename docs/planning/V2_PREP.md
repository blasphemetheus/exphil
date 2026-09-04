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

## Open (ordered)
1. **v1.5 shakeout readout** (relaunch post-fix): stability + steps/sec
   at batch 128 -> the v2 wall-clock budget; carry-threaded val
   descending.
1b. **BPTT chunk prep overlap/cache** (the actual throughput lever —
   see verdict above).
2. **Corpus quality-filter + dedupe pass** (slippi-ai filter list:
   1v1 human, 8-min timer, >=1 min, >=100 dmg, has-winner, physical
   sanity) + **dedupe by CONTENT HASH** (games can appear under both
   players' character dirs; paths encode seating accidents — same law
   as SubjectResolver).
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
   batch, epochs budget; registry over the FULL tagged corpus
   (thousands of tags -> the 112 cap needs a min-games threshold or a
   bigger name slot — decide from tag-frequency histogram).
