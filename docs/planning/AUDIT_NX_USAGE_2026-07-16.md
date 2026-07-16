# Nx Usage Audit — ExPhil (2026-07-16, static analysis)

nx 0.12 local checkout, EXLA/CUDA (RTX 5090). Findings ordered by likely
impact on training throughput and correctness.

## Executive summary

- The **imitation-learning path is the gold standard**
  (`training/imitation/loss.ex`): JIT built once with `compiler: EXLA,
  on_conflict: :reuse`, tensors flow as arguments, no per-batch recompiles or
  backend copies.
- The **PPO / league / lr_finder paths violate every rule that path follows**:
  bare `Nx.Defn.jit`/`value_and_grad` with no `compiler: EXLA`, closures
  rebuilt per minibatch, per-minibatch device→host→device round trips.
- A project-wide config bug (`config.ex:1627`) guarantees any bare
  `defn`/`jit` call falls back to the pure-Elixir `Nx.Defn.Evaluator`.
- **No exposure to nx#1729/#1730** (vectorized grads through
  cholesky/triangular_solve): no `Nx.vectorize` anywhere; the only LinAlg
  decompositions live in eager, non-differentiated code correctly scoped to
  `BinaryBackend`.
- `XLASelectiveScan` (custom-call + `custom_grad`) is implemented but **wired
  into zero backbones** — live Mamba/GatedSSM paths use a pure-Nx scan or a
  graph-breaking NIF.

## 1. defn vs jit vs eager — JIT boundary map

**Healthy (JIT once, reuse):**
- `training/imitation/loss.ex:196,226,257,277,348,370,395,409` — all
  loss/grad builders `Nx.Defn.jit(inner_fn, compiler: EXLA, on_conflict: :reuse)`,
  built once in `Imitation.new/1`, tensors passed as args. Dispatched via
  cached `trainer.loss_and_grad_fn` in `train_loop.ex:598-614`.
- `training/imitation.ex:328-329` — optimizer + apply_updates jitted once.
- `networks/policy/sampling.ex:154-172` — `jitted/2` caches compiled closures
  in `:persistent_term` with `compiler: EXLA`. Docstring at 151-153 explains
  the "bare defn runs on the pure-Elixir evaluator (~120ms)" gotcha.
- `inference/policy_serving.ex:120-124` — carries compiler + on_conflict.
- `interp/probe.ex:311,315` — `jit_apply(..., compiler: EXLA)`.

**Hot paths running eager / mis-compiled (throughput bugs):**
- **`training/ppo.ex:452`** — `grad_fn = Nx.Defn.jit(Nx.Defn.value_and_grad(loss_fn))`
  — no `compiler: EXLA`, and `loss_fn` is a fresh closure created inside
  `minibatch_update/7` (defined at ppo.ex:414). (a) resolves to Evaluator;
  (b) new closure per minibatch = new cache key = recompile every minibatch.
  Called per minibatch × num_epochs (ppo.ex:343,351-366).
- **`training/ppo.ex:475`** — `apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)`
  — no compiler; Evaluator.
- **`league/evolution.ex:447-457`** — entire PPO inner loop eager:
  `compute_ppo_loss_and_grad` (evolution.ex:541-542, bare value_and_grad),
  `clip_gradients`, `apply_updates` all outside jit, per minibatch per epoch.
  Same in `league/pretraining.ex:651`.
- **`training/lr_finder.ex:225`** — eager value_and_grad per LR step (short-lived).
- `scripts/train_distillation.exs:326` — odd double-wrap, no compiler, recompiles.

## 2. Backend transfers & device round trips

**Unnecessary device→host→device in a hot path:**
- **`training/ppo.ex:405-410`** — every minibatch copies states/old_log_probs/
  advantages/returns/old_values/actions to BinaryBackend (device→host), then
  the grad path pulls them back. The fix demonstrated by `loss.ex:99-127` is
  to pass tensors as JIT arguments instead.

**Correct serialization handling (gotcha-compliant):**
`async_checkpoint.ex:247`, `ppo.ex:651-674` (to_binary_backend before saves
at :627-628,:721,:745), `mmap_embeddings.ex:130-131`, `pipeline.ex:579,585`.

**Graph-breaking host transfers in scan NIFs:**
`native/selective_scan.ex:148-263`, `thunderkittens_scan.ex:74-84`,
`futhark_scan.ex:88-95`, `rust_linear_scan.ex:82-83` (+ mojo/julia/bend/
cuda_compute ports) — all `Nx.to_binary()` → NIF → `Nx.from_binary()`.
Cannot run inside defn, break autodiff. Mostly benchmark-only, EXCEPT the
mamba_nif landmine (§4).

**Eager `to_number` in the 60 fps inference path (minor):**
`agents/agent.ex:564,813-828` — per-frame to_number + Random.key sampling
eager. Acceptable at inference altitude; each is a device sync.

## 3. Recompile risks

- **PPO closure recompile** — ppo.ex:414/452 (§1).
- **`config.ex:1627`** — `Application.put_env(:nx, :default_defn_options, seed: seed)`
  OVERWRITES default_defn_options with only `[seed: ...]`, dropping any
  compiler. No config/*.exs sets `default_defn_options: [compiler: EXLA]`
  (only default_backend). Root cause amplifying all of §1. Fix: merge, don't
  replace; and/or set the compiler in config.
- **Compile-time seq_len branch** — `networks/mamba.ex:168` `if seq_len <= 32`
  picks sequential vs blelloch; concrete at trace time, so each distinct
  seq_len compiles a distinct graph. Fine while seq_len is fixed
  (mamba/common.ex:94-95 threads a concrete seq_len).
- **Unrolled scan** — `mamba/common.ex:442-452` sequential_scan unrolls via
  `Enum.reduce(1..(seq_len-1))`. Bounded to seq_len ≤ 32 by caller; do not
  raise that threshold.
- `networks/test_time_scaling.ex:514,521-537` — to_number + dynamically
  seeded Random.key per iteration. Test-time compute only.

## 4. Known-gotcha violations & near-misses

- **nx#1729/#1730: NOT EXPOSED.** No vectorize anywhere. Only decompositions:
  `interp/erase.ex:94` cholesky, `:96` triangular_solve, `:102` svd — eager,
  non-vectorized, outside grad.
- **eigh/svd XLA-compile hang: correctly handled** — erase.ex:52-74 transfers
  to BinaryBackend and scopes `Nx.default_backend` with `after` restore;
  results transferred back at :121-122. Exemplary.
- **Closure tensors in value_and_grad:** PPO applies backend_copy
  (ppo.ex:405-410) at round-trip cost — imitation path shows the correct fix
  (tensors as JIT args).
- **Autodiff through NIF (latent violation):** `networks/mamba_nif.ex:124-147`
  calls `SelectiveScan.scan` (to_binary at native/selective_scan.ex:148)
  inside an Axon.layer. mamba_nif is a selectable backbone
  (training/config.ex:44, policy/backbone.ex:587-588,868,2914). If selected
  for TRAINING, gradients cannot flow through the scan — silent zero-grad
  through the SSM core. Guard to inference-only or replace.

## 5. Correctness bug (latent, currently unused)

- **`training/gradient_checkpoint.ex:130-152`** — `scale_gradient` implements
  a mathematically wrong VJP: for non-matching output shapes it collapses the
  cotangent to `Nx.mean(g)` (:145,:150) instead of computing Jᵀg;
  `custom_grad` at :111 calls `value_and_grad(fun)` (:117), scalar-only.
  Not currently called by any backbone — dead code, but a trap. Fix to a
  real VJP or delete.

## 6. Modernization opportunities (nx 0.12)

- **Wire in `XLASelectiveScan`** (`native/xla_selective_scan.ex`): proper
  defn using `Nx.Shared.optional(:fused_selective_scan, …)` (:74) StableHLO
  custom_call, pure-Nx while fallback (:106-132), hand-written reverse-time
  custom_grad (:60-63, backward :142-223). Referenced only by
  `scripts/test_selective_scan_backward.exs`. Routing mamba.ex/mamba_nif.ex
  through it keeps data on-GPU, restores autodiff, and (per its perf table
  :15-19) cuts the scan from ~11-50 ms to ~5 ms.
- **`gated_ssm.ex:353-393` is not a real selective scan** — computes
  `sigmoid(dt) * (B*C*x)` + EMA (cumulative_ema :398); comments :362-363
  admit the approximation. Replace with the real recurrence.
- **Vectorize opportunity:** `networks/policy/sampling.ex:176-188` six
  hand-fused heads; per-head gumbel_argmax is a natural Nx.vectorize
  candidate (no LinAlg in graph — safe from nx#1729/#1730).
- **No dead workarounds found** — erase.ex scoping and serialization
  backend_copys reflect live EXLA constraints; leave them.

## Priority-ordered action list

1. **PPO throughput:** hoist loss/grad out of minibatch_update, build once
   with `compiler: EXLA, on_conflict: :reuse`, pass tensors as JIT args
   (mirror imitation/loss.ex). Fixes ppo.ex:452,475,405-410. Same for
   league/evolution.ex:447-457 and league/pretraining.ex:651.
2. **Global compiler default:** stop clobbering default_defn_options at
   config.ex:1627 (merge, don't replace); consider
   `config :nx, :default_defn_options, compiler: EXLA`.
3. **mamba_nif training landmine:** guard native scans to inference-only or
   replace with XLASelectiveScan.
4. **gradient_checkpoint VJP:** fix or delete scale_gradient before anyone
   wires it in.
5. **Replace approximate gated_ssm scan; adopt XLASelectiveScan** across
   Mamba backbones.
6. No action on nx#1729/#1730 (no exposure) or erase.ex scoping (compliant).
