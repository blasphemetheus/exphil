#!/usr/bin/env elixir
# Benchmark: Fused CUDA scan kernels vs Nx sequential fallbacks
#
# Compares the FusedScan dispatch path (XLA custom call or NIF) against
# pure-Elixir sequential scan fallbacks for all recurrent kernels.
#
# Usage:
#   mix run scripts/benchmark_fused_scans.exs
#   mix run scripts/benchmark_fused_scans.exs --iterations 50
#   mix run scripts/benchmark_fused_scans.exs --batch 8 --seq 120 --hidden 128
#
# Kernel tiers:
#   P0 - Element-wise: MinGRU, MinLSTM, ELU-GRU, Real-GRU, DiagLinear, Liquid, Linear
#   P1 - Matrix-state: DeltaNet, GatedDeltaNet, Mamba selective scan
#   P2 - R@h matmul:   sLSTM, TTT

alias ExPhil.Training.Output

require Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      iterations: :integer,
      warmup: :integer,
      batch: :integer,
      seq: :integer,
      hidden: :integer,
      heads: :integer,
      head_dim: :integer,
      state: :integer,
      help: :boolean
    ]
  )

if opts[:help] do
  IO.puts("""
  Benchmark Fused CUDA Scan Kernels

  Options:
    --iterations N   Timed iterations per kernel (default: 30)
    --warmup N       Warmup iterations (default: 5)
    --batch N        Batch size (default: 4)
    --seq N          Sequence length (default: 60)
    --hidden N       Hidden dimension (default: 64)
    --heads N        Number of attention heads for DeltaNet/GatedDeltaNet (default: 4)
    --head-dim N     Head dimension for DeltaNet/GatedDeltaNet (default: 16)
    --state N        State dimension for Mamba (default: 16)
    --help           Show this help
  """)

  System.halt(0)
end

iterations = opts[:iterations] || 30
warmup = opts[:warmup] || 5
batch = opts[:batch] || 4
seq_len = opts[:seq] || 60
hidden = opts[:hidden] || 64
num_heads = opts[:heads] || 4
head_dim = opts[:head_dim] || 16
state_dim = opts[:state] || 16

Output.banner("Fused CUDA Scan Kernel Benchmark")

Output.config([
  {"Batch", batch},
  {"Sequence length", seq_len},
  {"Hidden dim", hidden},
  {"Heads (DeltaNet)", num_heads},
  {"Head dim (DeltaNet)", head_dim},
  {"State dim (Mamba)", state_dim},
  {"Warmup iterations", warmup},
  {"Timed iterations", iterations}
])

IO.puts("")

# Check dispatch tier availability
xla_custom_call = Edifice.CUDA.FusedScan.custom_call_available?()

nif_loaded =
  Code.ensure_loaded?(Edifice.CUDA.NIF) and
    function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 6)

cond do
  xla_custom_call ->
    Output.success("Dispatch: XLA custom call (graph-preserving, best path)")

  nif_loaded ->
    Output.warning("Dispatch: NIF bridge (graph-breaking, fused kernel)")
    Output.puts("  XLA custom call not available — install EXLA fork for best path")

  true ->
    Output.warning("Dispatch: Elixir fallback only (no fused kernels available)")
    Output.puts("  Install EXLA fork or compile Edifice NIF with CUDA for fused paths")
end

IO.puts("")

# Force EXLA backend for GPU execution
Nx.default_backend(EXLA.Backend)

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule BenchHelper do
  def bench(fun, warmup_iters, timed_iters) do
    # Warmup — JIT compile + fill caches
    for _ <- 1..warmup_iters do
      fun.() |> Nx.backend_transfer(Nx.BinaryBackend)
    end

    # Timed runs
    times =
      for _ <- 1..timed_iters do
        t0 = System.monotonic_time(:microsecond)
        fun.() |> Nx.backend_transfer(Nx.BinaryBackend)
        System.monotonic_time(:microsecond) - t0
      end

    sorted = Enum.sort(times)
    median = Enum.at(sorted, div(length(sorted), 2))
    {median, Enum.min(times), Enum.max(times)}
  end

  def try_bench(fun, warmup_iters, timed_iters) do
    try do
      {:ok, bench(fun, warmup_iters, timed_iters)}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end
end

# ============================================================================
# Generate test data
# ============================================================================

Output.puts("Generating test tensors...")
key = Nx.Random.key(42)

# P0: Element-wise scans [batch, seq_len, hidden]
{gates, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{candidates, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{forget_gates, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{input_gates, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{a_vals, key} = Nx.Random.uniform(key, -1.0, 0.0, shape: {batch, seq_len, hidden}, type: :f32)
{b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{tau, key} = Nx.Random.uniform(key, 0.5, 2.0, shape: {batch, seq_len, hidden}, type: :f32)
{activation, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)

# P1: DeltaNet/GatedDeltaNet [batch, seq_len, num_heads, head_dim]
{q_dn, key} = Nx.Random.normal(key, shape: {batch, seq_len, num_heads, head_dim}, type: :f32)
{k_dn, key} = Nx.Random.normal(key, shape: {batch, seq_len, num_heads, head_dim}, type: :f32)
k_dn =
  Nx.divide(
    k_dn,
    Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(k_dn, k_dn), axes: [-1], keep_axes: true), 1.0e-6))
  )

{v_dn, key} = Nx.Random.normal(key, shape: {batch, seq_len, num_heads, head_dim}, type: :f32)
{beta_dn, key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {batch, seq_len, num_heads, head_dim}, type: :f32)
{alpha_dn, key} = Nx.Random.uniform(key, 0.9, 1.0, shape: {batch, seq_len, num_heads}, type: :f32)

# P1: Mamba selective scan
{x_mamba, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{dt_mamba, key} = Nx.Random.uniform(key, 0.001, 0.1, shape: {batch, seq_len, hidden}, type: :f32)
a_mamba = Nx.negate(Nx.add(Nx.iota({hidden, state_dim}, type: :f32), 1.0))
{b_mamba, key} = Nx.Random.normal(key, shape: {batch, seq_len, state_dim}, type: :f32)
{c_mamba, key} = Nx.Random.normal(key, shape: {batch, seq_len, state_dim}, type: :f32)

# P2: sLSTM [batch, seq_len, 4*hidden] + R [hidden, 4*hidden]
{wx_slstm, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, seq_len, 4 * hidden}, type: :f32)
{r_slstm, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {hidden, 4 * hidden}, type: :f32)

# P2: TTT [batch, seq_len, inner_size] + W0 [inner_size, inner_size]
ttt_inner = hidden
{q_ttt, key} = Nx.Random.normal(key, shape: {batch, seq_len, ttt_inner}, type: :f32)
{k_ttt, key} = Nx.Random.normal(key, shape: {batch, seq_len, ttt_inner}, type: :f32)
{v_ttt, key} = Nx.Random.normal(key, shape: {batch, seq_len, ttt_inner}, type: :f32)
{eta_ttt, key} = Nx.Random.uniform(key, 0.0, 1.0 / ttt_inner, shape: {batch, seq_len, ttt_inner}, type: :f32)
w0_ttt = Nx.multiply(Nx.eye(ttt_inner, type: :f32), 0.01)
{ln_g, key} = Nx.Random.uniform(key, 0.9, 1.1, shape: {ttt_inner}, type: :f32)
{ln_b, _key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {ttt_inner}, type: :f32)

# Transfer to GPU
Output.puts("Transferring to GPU...")

gates = Nx.backend_transfer(gates, EXLA.Backend)
candidates = Nx.backend_transfer(candidates, EXLA.Backend)
forget_gates = Nx.backend_transfer(forget_gates, EXLA.Backend)
input_gates = Nx.backend_transfer(input_gates, EXLA.Backend)
a_vals = Nx.backend_transfer(a_vals, EXLA.Backend)
b_vals = Nx.backend_transfer(b_vals, EXLA.Backend)
tau = Nx.backend_transfer(tau, EXLA.Backend)
activation = Nx.backend_transfer(activation, EXLA.Backend)
q_dn = Nx.backend_transfer(q_dn, EXLA.Backend)
k_dn = Nx.backend_transfer(k_dn, EXLA.Backend)
v_dn = Nx.backend_transfer(v_dn, EXLA.Backend)
beta_dn = Nx.backend_transfer(beta_dn, EXLA.Backend)
alpha_dn = Nx.backend_transfer(alpha_dn, EXLA.Backend)
x_mamba = Nx.backend_transfer(x_mamba, EXLA.Backend)
dt_mamba = Nx.backend_transfer(dt_mamba, EXLA.Backend)
a_mamba = Nx.backend_transfer(a_mamba, EXLA.Backend)
b_mamba = Nx.backend_transfer(b_mamba, EXLA.Backend)
c_mamba = Nx.backend_transfer(c_mamba, EXLA.Backend)
wx_slstm = Nx.backend_transfer(wx_slstm, EXLA.Backend)
r_slstm = Nx.backend_transfer(r_slstm, EXLA.Backend)
q_ttt = Nx.backend_transfer(q_ttt, EXLA.Backend)
k_ttt = Nx.backend_transfer(k_ttt, EXLA.Backend)
v_ttt = Nx.backend_transfer(v_ttt, EXLA.Backend)
eta_ttt = Nx.backend_transfer(eta_ttt, EXLA.Backend)
w0_ttt = Nx.backend_transfer(w0_ttt, EXLA.Backend)
ln_g = Nx.backend_transfer(ln_g, EXLA.Backend)
ln_b = Nx.backend_transfer(ln_b, EXLA.Backend)

IO.puts("")

# ============================================================================
# Run benchmarks
# ============================================================================

alias Edifice.CUDA.FusedScan

sep = String.duplicate("=", 90)
IO.puts(sep)
IO.puts("Kernel                   | Dispatch (μs) | Fallback (μs) | Speedup | Tier | Path")
IO.puts(sep)

results = []

# Helper: benchmark dispatch vs fallback, handling errors gracefully
run_bench = fn name, tier, dispatch_fn, fallback_fn ->
  # Always benchmark the fallback (guaranteed to work)
  {fb_med, _fb_min, _fb_max} = BenchHelper.bench(fallback_fn, warmup, iterations)

  # Try the dispatch path (may use custom call, NIF, or fallback internally)
  case BenchHelper.try_bench(dispatch_fn, warmup, iterations) do
    {:ok, {d_med, _d_min, _d_max}} ->
      speedup = fb_med / max(d_med, 1.0)

      path =
        cond do
          xla_custom_call -> "XLA CC"
          nif_loaded and speedup > 1.5 -> "NIF"
          true -> "fallback"
        end

      name_pad = String.pad_trailing(name, 24)
      d_str = String.pad_leading("#{round(d_med)}", 10)
      fb_str = String.pad_leading("#{round(fb_med)}", 10)
      sp_str = String.pad_leading(:erlang.float_to_binary(speedup, decimals: 1) <> "x", 7)

      IO.puts("#{name_pad} | #{d_str}    | #{fb_str}    | #{sp_str} | #{tier}   | #{path}")
      {name, tier, d_med, fb_med, speedup, path}

    {:error, _reason} ->
      # Dispatch failed (e.g. NIF not loaded), report fallback-only
      name_pad = String.pad_trailing(name, 24)
      fb_str = String.pad_leading("#{round(fb_med)}", 10)

      IO.puts("#{name_pad} |     N/A     | #{fb_str}    |     N/A | #{tier}   | no fused")
      {name, tier, fb_med, fb_med, 1.0, "no fused"}
  end
end

# ============================================================================
# P0: Element-wise scans
# ============================================================================

results = [
  run_bench.("MinGRU", "P0",
    fn -> FusedScan.mingru(gates, candidates) end,
    fn -> Edifice.Recurrent.MinGRU.min_gru_scan(gates, candidates) end
  ) | results
]

results = [
  run_bench.("MinLSTM", "P0",
    fn -> FusedScan.minlstm(forget_gates, input_gates, candidates) end,
    fn -> Edifice.Recurrent.MinLSTM.min_lstm_scan(forget_gates, input_gates, candidates) end
  ) | results
]

results = [
  run_bench.("ELU-GRU", "P0",
    fn -> FusedScan.elu_gru(gates, candidates) end,
    fn -> Edifice.Recurrent.NativeRecurrence.elu_gru_scan(gates, candidates) end
  ) | results
]

results = [
  run_bench.("Real-GRU", "P0",
    fn -> FusedScan.real_gru(gates, candidates) end,
    fn -> Edifice.Recurrent.NativeRecurrence.real_gru_scan(gates, candidates) end
  ) | results
]

results = [
  run_bench.("DiagLinear", "P0",
    fn -> FusedScan.diag_linear(a_vals, b_vals) end,
    fn -> Edifice.Recurrent.NativeRecurrence.diag_linear_scan(a_vals, b_vals) end
  ) | results
]

results = [
  run_bench.("Liquid", "P0",
    fn -> FusedScan.liquid(tau, activation) end,
    fn -> Edifice.Liquid.liquid_exact_scan(tau, activation) end
  ) | results
]

results = [
  run_bench.("Linear", "P0",
    fn -> FusedScan.linear_scan(a_vals, b_vals) end,
    fn -> FusedScan.linear_scan_fallback(a_vals, b_vals) end
  ) | results
]

# ============================================================================
# P1: Matrix-state scans
# ============================================================================

results = [
  run_bench.("DeltaNet", "P1",
    fn -> FusedScan.delta_net_scan(q_dn, k_dn, v_dn, beta_dn) end,
    fn -> Edifice.Recurrent.DeltaNet.delta_net_sequential_scan(q_dn, k_dn, v_dn, beta_dn) end
  ) | results
]

results = [
  run_bench.("GatedDeltaNet", "P1",
    fn -> FusedScan.gated_delta_net_scan(q_dn, k_dn, v_dn, beta_dn, alpha_dn) end,
    fn ->
      Edifice.Recurrent.GatedDeltaNet.gated_delta_net_sequential_scan(
        q_dn, k_dn, v_dn, beta_dn, alpha_dn
      )
    end
  ) | results
]

results = [
  run_bench.("Mamba SelScan", "P1",
    fn -> FusedScan.selective_scan(x_mamba, dt_mamba, a_mamba, b_mamba, c_mamba) end,
    fn ->
      Edifice.SSM.Common.selective_scan_fallback(x_mamba, dt_mamba, a_mamba, b_mamba, c_mamba)
    end
  ) | results
]

# ============================================================================
# P2: R@h matmul scans
# ============================================================================

results = [
  run_bench.("sLSTM", "P2",
    fn -> FusedScan.slstm_scan(wx_slstm, r_slstm) end,
    fn -> Edifice.Recurrent.SLSTM.slstm_scan_fallback(wx_slstm, r_slstm) end
  ) | results
]

results = [
  run_bench.("TTT-Linear", "P2",
    fn -> FusedScan.ttt_scan(q_ttt, k_ttt, v_ttt, eta_ttt, w0_ttt, ln_g, ln_b) end,
    fn ->
      Edifice.Recurrent.TTT.ttt_scan_fallback(q_ttt, k_ttt, v_ttt, eta_ttt, w0_ttt, ln_g, ln_b)
    end
  ) | results
]

IO.puts(sep)

# ============================================================================
# Summary
# ============================================================================

results = Enum.reverse(results)

IO.puts("\nSummary by tier:")

for tier <- ["P0", "P1", "P2"] do
  tier_results = Enum.filter(results, fn {_, t, _, _, _, _} -> t == tier end)
  fused_results = Enum.filter(tier_results, fn {_, _, _, _, _, path} -> path != "no fused" end)

  if length(tier_results) > 0 do
    tier_label =
      case tier do
        "P0" -> "P0 (element-wise)"
        "P1" -> "P1 (matrix-state)"
        "P2" -> "P2 (R@h matmul)"
      end

    if length(fused_results) > 0 do
      avg_speedup =
        Enum.map(fused_results, fn {_, _, _, _, s, _} -> s end) |> Enum.sum() |> Kernel./(length(fused_results))

      best = Enum.max_by(fused_results, fn {_, _, _, _, s, _} -> s end)
      {best_name, _, _, _, best_speedup, _} = best

      IO.puts(
        "  #{tier_label}: avg #{:erlang.float_to_binary(avg_speedup, decimals: 1)}x" <>
          ", best #{best_name} (#{:erlang.float_to_binary(best_speedup, decimals: 1)}x)" <>
          " [#{length(fused_results)}/#{length(tier_results)} fused]"
      )
    else
      IO.puts("  #{tier_label}: no fused kernels available [0/#{length(tier_results)}]")
    end
  end
end

fused_results = Enum.filter(results, fn {_, _, _, _, _, path} -> path != "no fused" end)

if length(fused_results) > 0 do
  overall_avg =
    Enum.map(fused_results, fn {_, _, _, _, s, _} -> s end) |> Enum.sum() |> Kernel./(length(fused_results))

  IO.puts(
    "\n  Overall: #{:erlang.float_to_binary(overall_avg, decimals: 1)}x avg speedup" <>
      " (#{length(fused_results)}/#{length(results)} kernels fused)"
  )
end

# 60 FPS check
IO.puts("\n60 FPS feasibility (< 16.67 ms per scan):")

for {name, _tier, dispatch_us, fb_us, _, path} <- results do
  best_us = if path == "no fused", do: fb_us, else: dispatch_us
  best_ms = best_us / 1000
  status = if best_ms < 16.67, do: "YES", else: "NO "
  IO.puts("  #{status} #{String.pad_trailing(name, 20)} #{:erlang.float_to_binary(best_ms, decimals: 2)} ms")
end

IO.puts("\nDone!")
