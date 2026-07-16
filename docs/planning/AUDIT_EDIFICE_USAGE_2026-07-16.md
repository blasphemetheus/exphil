# ExPhil → Edifice Consumption Audit (2026-07-16, static analysis)

Repos: `~/git/exphil` (consumer) · `~/git/edifice` v0.2.0 (library, path dep
`../edifice` per exphil `mix.exs` `edifice_dep/0`). exphil never pins a
version — it takes whatever `../edifice` HEAD provides, so the newly-merged
manifest/Stateful/Profile/SAE surface is already on-disk and available.

Structural headline: **exphil consumes Edifice purely as an
architecture-`build/1` catalog.** It never calls the top-level `Edifice`
facade — `grep "Edifice.build|Edifice.step|Edifice.init_state|Edifice.build_with_spec|Edifice.Profile|Edifice.Stateful"`
across `lib` + `scripts` returns **zero hits**. Every consumption is a direct
alias of a concrete arch module followed by `.build(...)`. All four
recently-merged capabilities (manifest/Spec, Stateful step API, Profile
`:step`, SAE fit) are 100% unconsumed by exphil today — exactly what task
#16 flags.

## 1. Inventory of Edifice.* call sites, by subsystem

### A. Backbone catalog — `lib/exphil/networks/policy/backbone.ex` (dominant consumer)
Single file, ~90 architecture aliases, each dispatched via `<Alias>.build(...)`:

- SSM family: `Edifice.SSM.GatedSSM` (:94), `Mamba` (:97), `MambaCumsum` (:98),
  `Hybrid` (:96), `Zamba` (:998), `S5` (:1201), `S4` (:1223), `S4D` (:1245),
  `H3` (:1267), `Mamba3` (:1720), `Hyena`/`HyenaV2` (:1744/:2734),
  `Longhorn` (:1836), `Samba` (:1858), `Hymba` (:1880), `GSS` (:1902),
  `MixtureOfMamba` (:2210), `BiMamba` (:2712), `SSTransformer` (:2754),
  `MambaHillisSteele.build` (:2981), `MambaSSD.build` (:3006).
- Attention family: `Edifice.Attention.MultiHead` (:93), `Griffin` (:95, :1030,
  :1057), `RetNet`/`RetNetV2` (:1107/:2528), `RWKV` (:1131), `GLA`/`GLAv2`
  (:1153/:1946), `HGRN`/`HGRNv2` (:1179/:1968), `Performer` (:1291),
  `FNet` (:1335), `Perceiver` (:1353), plus the modern-attention wall
  (:2012–:2650): GSA, RLA, NHA, FoX, LogLinear, LASER, MoBA, TNN, Mega, Based,
  InfiniAttention, Conformer, MLA, DiffTransformer, Megalodon,
  LightningAttention, FlashLinearAttention, KDA, SigmoidAttention, SPLA,
  RNoPESWA, NSA, InfLLMV2, DualChunk, GatedAttention, MTA.
- Recurrent family: `Edifice.Recurrent` (:99), `XLSTM`/`XLSTMv2` (:1079/:2690),
  `DeltaNet`/`DeltaProduct` (:1313/:1924), `TTT`/`TTTE2E` (:1375/:1990),
  `Reservoir` (:1459), `TransformerLike` (:1600), `DeepResLSTM` (:1630),
  `MinGRU`/`MinLSTM` (:1652/:1672), `Titans` (:1768), `GatedDeltaNet` (:1792),
  `NativeRecurrence` (:1816), `MIRAS` (:2188), `Huginn` (:2232), `SLSTM` (:2670).
- Other: `Edifice.Liquid` (:1553), `Feedforward.KAN` (:1576),
  `Convolutional.TCN` (:1692), `Meta.Coconut` (:2254), `Blocks.SSMax` (:2778),
  `Blocks.Softpick` (:2798).

### B. Thin delegating wrappers (the "good" migration pattern already in place)
- `lib/exphil/networks/deep_res_lstm.ex:72,126` — aliases
  `Edifice.Recurrent.DeepResLSTM`, `build/1` translates ExPhil opt names →
  Edifice opt names and delegates.
- `lib/exphil/networks/transformer_like.ex:79,153` — same pattern for
  `Edifice.Recurrent.TransformerLike`.

### C. Interpretability — `lib/exphil/interp/`
- `lib/exphil/interp/probe.ex:26,86` — aliases
  `Edifice.Interpretability.LinearProbe`; calls `LinearProbe.build(...)` at :86
  but **discards the returned model** (`_probe_model`) and hand-rolls its own
  weight init + `train_jit`/`predict_jit` loop (:88–:110).
- `lib/exphil/interp/erase.ex:20` — references `Edifice.Interpretability.LEACE`
  in moduledoc only ("the trainable architecture; this is the exact closed
  form … Worth upstreaming"); the module hand-rolls closed-form LEACE.
- Consuming scripts: `scripts/interp_leace_fit.exs`,
  `scripts/interp_p3_followups.exs`, `scripts/interp_shield_jump_offline.exs`
  (drive `Interp.Probe`/`Interp.Erase`, not Edifice directly).

### D. CUDA / fused-scan kernels — scripts only
- `Edifice.CUDA.FusedScan.linear_scan/2` and `.custom_call_available?/0`:
  `scripts/benchmark_triton_scan.exs:83,203`, `benchmark_julia_scan.exs:228,234`,
  `benchmark_rust_scan.exs:90,194`, `benchmark_cuda_compute_scan.exs:107,254`,
  `benchmark_thunderkittens_scan.exs:101,219`,
  `benchmark_kernel_languages.exs:87,207`, `benchmark_mojo_scan.exs:203,209`,
  `benchmark_futhark_scan.exs:80,170`, `benchmark_fused_scans.exs:80,225`,
  `test_selective_scan_backward.exs:9`, `test_fused_kernels.exs:10`,
  `profile_griffin_mamba.exs:32-34`.
- `Edifice.CUDA.NIF.fused_mingru_scan/6`: `benchmark_fused_scans.exs:83-84`.
- Edifice recurrent scan primitives benchmarked directly:
  `benchmark_fused_scans.exs:276-368` → `Recurrent.MinGRU.min_gru_scan`,
  `MinLSTM.min_lstm_scan`, `NativeRecurrence.{elu_gru_scan,real_gru_scan,diag_linear_scan}`,
  `Liquid.liquid_exact_scan`, `Recurrent.DeltaNet.delta_net_sequential_scan`,
  `GatedDeltaNet.gated_delta_net_sequential_scan`,
  `SSM.Common.selective_scan_fallback` (:348),
  `Recurrent.SLSTM.slstm_scan_fallback`, `Recurrent.TTT.ttt_scan_fallback`.

### E. Direct SSM.Common / Mamba builder use — scripts only
- `scripts/bisect_model_complexity.exs:144` — `Edifice.SSM.Common.build_model`
  (the only such call in exphil).
- `scripts/profile_griffin_mamba.exs:40-135` — `Edifice.SSM.Mamba.build` and
  `Edifice.Attention.Griffin.build` across many configs.

Nothing under `lib/exphil/{training,agents,bridge,embeddings,rewards,self_play}`
references `Edifice` at all — the entire runtime/training/inference spine is
Edifice-free.

## 2. Capabilities used vs ignored

### USED
- **Architecture `build/1`** for ~90 backbones (§1A/B).
- **`Edifice.Interpretability.LinearProbe.build/1`** — nominally (model
  discarded, §1C).
- **`Edifice.CUDA.FusedScan` / `Edifice.CUDA.NIF`** — benchmark/profile
  scripts only (§1D).
- **`Edifice.SSM.Common.build_model` + `selective_scan_fallback`** — two
  scripts only (§1E).

### IGNORED (present in edifice, zero exphil consumption)

**Manifest / Checkpoint spec** — `Edifice.Checkpoint`
(`edifice/lib/edifice/checkpoint.ex`) + `Edifice.Spec` (`spec.ex`):
- `Edifice.Checkpoint.save(params, path, spec: spec)` / `load_model/2`
  (checkpoint.ex:100, :304) — self-describing checkpoints; `load_model/2`
  rebuilds from the embedded `Edifice.Spec` and calls `validate_shapes!/3`
  (:345), raising loudly on shape mismatch. Moduledoc: "Embedding an
  Edifice.Spec kills a silent-failure class: loading a checkpoint into a model
  rebuilt with default options."
- `Edifice.Spec.new/3`, `to_map/1`, `from_map/1` (spec.ex:51/73/90).
- `Edifice.Checkpoint.attach/3`, `resume/2`, `save_loop_state/2`
  (checkpoint.ex:533/506/468) — Axon.Loop integration.
- This is precisely the #16 remaining item.

**Stateful step API + JIT** — `Edifice.Stateful` (`stateful.ex`) +
`Edifice.Stateful.Ops` (`stateful/ops.ex`):
- Behaviour `init_state/2` (:57) + `step/3` (:66); dispatch helpers
  `Edifice.init_state(:arch, params, opts)` / `Edifice.step(:arch, params, state, frame)`
  (stateful.ex:40-41 moduledoc).
- `jit_step/2` (:87) caching `Nx.Defn.jit(&module.step/3)` in `:persistent_term`.
- `snapshot/1` (:129), `serialize/1` (:138), `deserialize/1` (:149) — the
  netplay rollback primitive.
- `Edifice.Stateful.Ops`: `dense/2`, `layer_norm/3`, `silu/1`, `softplus/1`,
  `conv1d_step/3`, `unwrap_params/1`, `layer_params!/2`, `zeros/2` (ops.ex:20-118).
- #21 notes the step path is already JITted in edifice at 0.054–0.101 ms/step.

**Profile `:step`** — `Edifice.Profile.run(arch, mode: :step)` (profile.ex:124,
dispatch :125-128, `run_step/1` :214). Reports `init_ms, p50_step_ms,
p95_step_ms, mean_step_ms, max_step_ms, state_bytes` for any Stateful arch
(:102-105); raises if not Stateful (:222). Also `compare/1` (:339),
`print_table/1` (:375), `memory_stats/1` (:74). exphil instead maintains its
own `scripts/profile_*.exs` fleet.

**SAE fit infrastructure** — `Edifice.Interpretability.SAETrainer.fit/3`
(sae_trainer.ex:84) + SAE family (sparse_autoencoder, batch_top_k_sae,
gated_sae, jump_relu_sae, matryoshka_sae, crosscoder, transcoder,
cross_layer_transcoder). `fit/3` provides decoder unit-norm renorm, BatchTopK
inference threshold, AuxK dead-feature revival (:15-27), returns
`%{params, model, threshold, history, dead_count, module, build_opts}`.
exphil has **zero** SAE consumption; #15's remediation built this consumer
shape for exphil's pending activation-capture port.

**Building blocks** — `Edifice.Blocks.{RMSNorm,SwiGLU,FFN,RoPE,ALiBi,
TransformerBlock,CausalMask,KVCache,DepthwiseConv,SDPA}` — all ignored;
exphil hand-rolls norms/convs.

**Whole subsystems ignored:** `Edifice.Training` +
`Training.{Adaptive,MemoryTracker,Monitor}`, `Edifice.Metrics`,
`Edifice.MixedPrecision`, `Edifice.Serving.*`,
`Edifice.RL.{GAE,PPOTrainer,DecisionTransformer,PolicyValue}` — exphil
re-implements all of these.

## 3. Things exphil hand-rolls that Edifice already provides (or should)

**a) `ExPhil.Networks.Mamba.Common` is a structural clone of
`Edifice.SSM.Common`.** Same function names throughout:
`default_hidden_size/state_size/expand_factor/conv_size/num_layers/dropout`
(exphil :39-54 vs edifice :35-55), `dt_min`/`dt_max` (:57-60 vs :59-63),
`build_model/2` (:86 vs :89), `build_block/3` (:165 vs :168),
`build_depthwise_conv1d/4` (:247 vs :251), `build_ssm_projections/2`
(:298 vs :285), `discretize_ssm/4` (:363 vs :350), `compute_ssm_output/2`
(:415 vs :410), `sequential_scan/2` (:443 vs :440), `blelloch_scan/2`
(:481 vs :478), `output_size/1` (:527 vs :527), `param_count/1` (:535 vs :535).
Only divergence: exphil's `melee_defaults/0` (:568) vs edifice's
`recommended_defaults/0` (:568). Largest duplication; clearest merge candidate.

**b) Selective-scan kernel duplication — `xla_selective_scan` vs edifice
`cuda/fused_scan`.** `lib/exphil/native/xla_selective_scan.ex` implements
`selective_scan/5` (:57) dispatching via
`Nx.Shared.optional(:fused_selective_scan, …)` to an EXLA custom call, with a
pure-Nx `selective_scan_fallback` (:77) and a custom-grad reverse sweep (:60).
Edifice owns the fused-scan surface (`Edifice.CUDA.FusedScan.linear_scan/2`,
`custom_call_available?/0`, `Edifice.CUDA.NIF`,
`Edifice.SSM.Common.selective_scan_fallback/5` at edifice ssm/common.ex:585).
exphil's benchmarks already treat edifice's as the reference. Task #13
earmarks reconciling both onto nx's sanctioned `Nx.block`/CustomCall.
Additional exphil scan re-implementations overlapping edifice:
`mamba_hillis_steele.ex:89`, `mamba_cumsum.ex:117,153`,
`mamba_ssd.ex:370,479`, `gated_ssm.ex:353`, `mamba_nif.ex` (Rust NIF vs
`Edifice.CUDA.NIF`).

**c) Checkpoint serialization.** exphil hand-rolls Erlang-term checkpoints:
`lib/exphil/training/imitation/checkpoint.ex:64,243` (`:erlang.term_to_binary`),
`lib/exphil/training/checkpoint.ex:97-108` (`binary_to_term [:safe]`,
`deserialize_trusted`), `async_checkpoint.ex:218`. Edifice's `Checkpoint`
uses `Nx.serialize` ("~3-5x faster than Erlang term serialization",
checkpoint.ex:5-8) and embeds an `Edifice.Spec` for shape-validated rebuild.
exphil's `ExPhil.Training.Checkpoint` only validates `embed_size` manually
(checkpoint.ex:16-33). Seven separate exphil checkpoint modules exist
(checkpoint, async_checkpoint, checkpoint_pruning, gradient_checkpoint,
imitation/checkpoint, config/checkpoint, callbacks/checkpoint) with no
Edifice manifest anywhere.

**d) Stateful/real-time inference.** `lib/exphil/networks/recurrent.ex:266-350`
hand-rolls `build_stateful/1` (:287) + `initial_hidden/2` (:337) — LSTM/GRU
only, external state tuples, **no JIT, no snapshot/serialize/rollback, no
arch-generic dispatch**. `Edifice.Stateful` supersedes this for the whole
linear-recurrence family with `jit_step/2` + snapshot/serialize/deserialize.
This is the GOTCHAS-#29 / netplay-rollback path in #16.

**e) LinearProbe training loop.** `lib/exphil/interp/probe.ex` builds the
edifice LinearProbe (:86), throws it away, re-implements standardization +
weighted logistic training (:88-110). Edifice's Probe/LinearProbe house
pattern (same full-batch jitted trainer SAETrainer uses) is the intended path.

**f) Closed-form LEACE.** `lib/exphil/interp/erase.ex` implements exact
closed-form LEACE; its moduledoc (:20-22) says edifice's LEACE is the
trainable variant and the closed form is "Worth upstreaming." Edifice
`leace.ex` currently only offers `fit/2` (SGD, :68) + `build_trainable/1`
(:182) — the closed form is genuinely missing upstream.

**g) Norm/block utilities.** Hand-rolled norms in `deep_res_lstm.ex,
mamba/common.ex, transformer_like.ex, gated_ssm.ex, griffin.ex` overlap
`Edifice.Blocks.RMSNorm.layer/2` (blocks/rms_norm.ex:36) and
`Edifice.Stateful.Ops.layer_norm/3`.

## 4. Build-in-edifice recommendations (prioritized)

**P0 — Adopt `Edifice.Checkpoint` + `Edifice.Spec` manifest (unblocks #16).**
Edifice side is complete. Work is exphil-side: emit an `Edifice.Spec` at
model-build time and route saves/loads through `Edifice.Checkpoint`. Call
sites to migrate: `ExPhil.Training.Checkpoint.load/2` (checkpoint.ex:63),
`.load_policy` (:85), `imitation/checkpoint.ex:64,243`,
`async_checkpoint.ex:218`, loaders in the agent + `scripts/eval_model.exs` /
`scripts/play_dolphin*.exs`. Possible edifice gap: a spec producer for
exphil's multi-head autoregressive policy bundle if `build_with_spec` doesn't
cover composite policy models.

**P1 — Adopt `Edifice.Stateful` step API for agent inference (rest of #16 +
#9 Yeti netplay).** Replace `Recurrent.build_stateful/1` + `initial_hidden/2`
with `Edifice.init_state/3` + `Edifice.step/5` (+ jit_step), use
snapshot/serialize/deserialize for rollback. Call sites:
`lib/exphil/agents/agent.ex`, `bridge/async_runner.ex`,
`scripts/play_dolphin_async.exs`, `scripts/profile_agent_inference.exs`.
If a production backbone isn't yet Stateful in edifice, that's the
build-in-edifice task.

**P2 — Collapse `ExPhil.Networks.Mamba.Common` into `Edifice.SSM.Common`.**
Add `melee_defaults` (or `defaults:` override) to edifice and delete exphil's
clone, following the deep_res_lstm/transformer_like delegation template.
Consumers: `mamba.ex, gated_ssm.ex, mamba_cumsum.ex, mamba_hillis_steele.ex,
mamba_ssd.ex, griffin.ex` + `scripts/bisect_model_complexity.exs:144`.

**P3 — Unify selective-scan kernels (#13).** One edifice-owned dispatcher on
nx's sanctioned CustomCall mechanism; migrate exphil's XLASelectiveScan and
the ~11 benchmark scripts.

**P4 — Adopt `SAETrainer.fit/3` + LinearProbe trainer (closes #15 consumer
side).** Build exphil's activation-capture → `SAETrainer.fit` bridge; migrate
`ExPhil.Interp.Probe` to use the built model. Consumers:
`scripts/interp_probe_zoo{,_v2}.exs`, `interp_capture.exs`,
`interp_leace_fit.exs`, `interp_p3_*.exs`, `interp_shield_jump_offline.exs`.

**P5 — Upstream exphil's closed-form LEACE into edifice, then consume.**
Add `LEACE.fit_closed_form/3` (or `solve/3`) to edifice from
`Interp.Erase.fit/3` (erase.ex:38), then delegate. Consumers:
`scripts/interp_leace_fit.exs`, `scripts/interp_p3_followups.exs`.

**P6 — Adopt `Edifice.Profile` (`mode: :step`) once P1 lands.** Replace
bespoke step-latency scripts with `Edifice.Profile.run(arch, mode: :step)`.
Reporting convenience, not correctness.

### Notes / caveats
- exphil pins edifice by path with no version floor — adoption should add a
  `~> 0.2` floor in `mix.exs` so Docker/GitHub builds don't silently regress.
- The delegation pattern for P2/P5 already exists (`deep_res_lstm.ex`,
  `transformer_like.ex`) — the template for opt-name translation.
