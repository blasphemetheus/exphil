# Task #13 — Migration Evaluation: Selective-Scan Kernels → Nx.block / EXLA.CustomCall

2026-07-17, static analysis of local nx checkout (origin = upstream elixir-nx/nx,
branch feat/nx-fuzz-errors, nx 0.12.1 / exla 0.12.0) + exphil/edifice scan code.

## TL;DR / Decision

**Migrate, but not now — schedule after r12.** The API both current
implementations depend on, `Nx.Shared.optional/4`, **no longer exists in
upstream nx main** — replaced by the sanctioned `Nx.block`/`EXLA.CustomCall`
mechanism (#1709/#1739, both merged upstream). Migration is therefore
eventually mandatory (any future rebase breaks the fused-scan path). But the
GPU/CUDA kernel-registration half has NOT been ported to the new protocol —
it exists only on the personal fork branches — so the migration is gated on
real native work that must not run beside a live training round.

## 1. What the sanctioned mechanism provides

- `Nx.block(struct, args, output, fun)` — nx/lib/nx.ex:7251; dispatches to
  `backend.block/4` (nx.ex:7259-7260); default backends run the fallback fun.
- `EXLA.CustomCall` protocol — exla/lib/exla/custom_call.ex:1
  (`@fallback_to_any true`); `defimpl` returns `:skip` (compile default fun)
  or `{:ok, %EXLA.CustomCall.Spec{}}` → `stablehlo.custom_call`.
- `EXLA.CustomCall.Spec` (spec.ex:28-31): `call_target_name` (XLA FFI handler
  name), `attributes` (→ backend_config dict), `operand_element_types`
  (declarative dtype coercion).
- EXLA lowering: exla/lib/exla/defn.ex:777-816 (cast at :988, emit at :809).
- **Autodiff:** grad.ex:185/:322 differentiate the DEFAULT callback's
  expression — the mechanism gives free autodiff only through the pure-Nx
  fallback. A fused backward still uses `Nx.Defn.Kernel.custom_grad/3`
  (kernel.ex:271) — the existing custom_grad pattern carries over unchanged.
- **Maturity:** 20+ internal block structs (LinAlg, Cumulative*, TopK, FFT…)
  say the API is here to stay. BUT built-in CustomCall impls are HOST-ONLY
  (QR/Eigh gated `platform: :host`, custom_call.ex:69,108,115); zero GPU
  impls, tests, or :cuda gating upstream. We would be the first GPU consumer.

## 2. Gap analysis

exphil `XLASelectiveScan` (native/xla_selective_scan.ex): custom_grad wrap
(:57-63) over `Nx.Shared.optional(:fused_selective_scan, …)` (:74), pure-Nx
while fallback (:104-137), hand-written reverse-time backward (:158-224).
Zero backbone consumers; requires fork branch per its own moduledoc (:20-26).

edifice `CUDA.FusedScan` (cuda/fused_scan.ex): 3-tier dispatch (:370-384) —
(1) custom-call via `Nx.Shared.optional` + custom_grad (:2595,:2606-2610),
gated by `custom_call_available?/0` (:5167) probing the FORK-ONLY
`EXLA.MLIR.Value.custom_call_fused/4`; (2) Rust/CUDA NIF (nif.ex:86) —
breaks the XLA graph (GPU→NIF→GPU); (3) pure-Nx SSM.Common fallback.

What the protocol does NOT cover: (a) the CUDA .cu kernels + FFI
registration must be re-expressed under `call_target_name` and registered on
the CUDA platform — none of that is upstream (fork branch
feat/exla-fused-cuda-kernels has the .cu sources); (b) the tier-2 NIF
fallback + GPU memory lifecycle stay edifice's problem regardless.
What it improves: declarative dtype coercion (Spec.operand_element_types),
kernel config via attributes instead of positional hacks.

## 3. Migration design (audit P3 shape)

One edifice-owned `%Edifice.Block.SelectiveScan{}` +
`defimpl EXLA.CustomCall` returning a Spec on CUDA / `:skip` elsewhere;
`FusedScan.selective_scan/5` becomes
`Nx.block(struct, args, out, &fallback/6) |> custom_grad(...)`.
- edifice keeps: block struct + impl, SSM.Common fallback (as default fun),
  hand-written backward (inside custom_grad), tier-2 NIF arm (separate),
  `custom_call_available?` rewritten to probe protocol registration.
- Native port (the load-bearing work): fused_selective_scan*.cu +
  fused_selective_scan_backward.cu from the fork onto the protocol's FFI
  convention, validated on the 5090.
- exphil deletes native/xla_selective_scan.ex entirely (zero consumers) and
  repoints the ~11 benchmark scripts at edifice. Dovetails with the
  Mamba.Common → SSM.Common collapse (DONE 2026-07-16, commit 9021713).

Scope: Elixir-side ~1 day; CUDA/FFI port multi-day; total ~1 week dominated
by native re-wiring + gradient validation.

## 4. Risks

- nx API churn: low-moderate — protocol is stable; the RISK is that
  `Nx.Shared.optional` removal breaks us on the next upstream rebase
  (a reason to migrate, not delay indefinitely).
- GPU protocol immaturity: HIGH — first GPU consumer; expect to re-learn the
  fork's lessons (XLA 0.10 header build, OutputBuffer default-ctor fix) on
  the protocol.
- Gradient correctness: HIGH — pin the reverse-sweep backward with planted
  value-level grad pins vs Nx.Defn.grad through the pure fallback, plus
  cross-backend differential (mirror nx BUG-1740/BUG-1747 pins). Keep the
  backward f32 regardless of forward precision.
- Perf: current in-graph custom call ~5 ms vs NIF ~11 ms vs pure ~50 ms
  (xla_selective_scan.ex:15-19). If the port loses the in-graph path, ~2x
  regression — migration must keep `stablehlo.custom_call` on-GPU.

## 5. Recommendation

Migrate AFTER r12, bundled with remaining P3 cleanup. Go/no-go gate: a spike
that registers ONE GPU custom-call target through the new protocol on the
5090 and keeps it in-graph at ~5 ms. If the spike fails (protocol can't
target CUDA without upstream FFI work), keep tier-2 NIF as primary and treat
the in-graph path as fork-only until upstream grows GPU support.
