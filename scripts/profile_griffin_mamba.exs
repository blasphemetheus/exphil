#!/usr/bin/env elixir
# Profile Griffin vs Mamba inference gap
#
# Isolates the sources of Griffin's ~2.4x slowdown vs Mamba by testing:
#   1. Layer count (6 vs 2 — Griffin default vs Mamba default)
#   2. Local attention (Griffin hybrid vs pure RG-LRU "Hawk" mode)
#   3. Expand factor (3x vs 2x)
#   4. compiler: EXLA (graph compilation vs re-tracing)
#   5. Per-component breakdown (RG-LRU scan vs attention vs MLP)
#
# Usage:
#   EDIFICE_LOCAL_NX=1 mix run scripts/profile_griffin_mamba.exs

unless "--verbose" in System.argv() do
  System.put_env("TF_CPP_MIN_LOG_LEVEL", "2")
end

defmodule GriffinMambaProfile do
  @iterations 30
  @warmup 5
  @embed_size 288
  @window_size 60

  def run do
    IO.puts(:stderr, "\n=== Griffin vs Mamba Profiling ===")
    IO.puts(:stderr, "embed_size=#{@embed_size}, window_size=#{@window_size}, iterations=#{@iterations}\n")

    is_exla = match?({EXLA.Backend, _}, Nx.default_backend())
    IO.puts(:stderr, "Backend: #{if is_exla, do: "EXLA (GPU)", else: "BinaryBackend (CPU)"}")

    # Check if fused kernels are available
    fused? = Code.ensure_loaded?(Edifice.CUDA.FusedScan) and
             function_exported?(Edifice.CUDA.FusedScan, :custom_call_available?, 0) and
             Edifice.CUDA.FusedScan.custom_call_available?()
    IO.puts(:stderr, "Fused custom calls: #{fused?}\n")

    configs = [
      # ── Baseline: original benchmark conditions ──
      {"Mamba (2L, expand=2)",
       fn -> Edifice.SSM.Mamba.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
         expand_factor: 2, state_size: 16, conv_size: 4,
         dropout: 0.0, window_size: @window_size
       ) end},

      {"Griffin (6L, expand=3, local_attn)",
       fn -> Edifice.Attention.Griffin.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 6,
         expand_factor: 3, local_attn_window: 32, num_heads: 4,
         dropout: 0.0, window_size: @window_size, seq_len: @window_size,
         use_local_attention: true
       ) end},

      # ── Factor 1: Match layer count ──
      {"Griffin (2L, expand=3, local_attn)",
       fn -> Edifice.Attention.Griffin.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
         expand_factor: 3, local_attn_window: 32, num_heads: 4,
         dropout: 0.0, window_size: @window_size, seq_len: @window_size,
         use_local_attention: true
       ) end},

      # ── Factor 2: Disable local attention (Hawk mode) ──
      {"Griffin (6L, expand=3, NO local_attn)",
       fn -> Edifice.Attention.Griffin.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 6,
         expand_factor: 3, dropout: 0.0,
         window_size: @window_size, seq_len: @window_size,
         use_local_attention: false
       ) end},

      {"Griffin (2L, expand=3, NO local_attn)",
       fn -> Edifice.Attention.Griffin.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
         expand_factor: 3, dropout: 0.0,
         window_size: @window_size, seq_len: @window_size,
         use_local_attention: false
       ) end},

      # ── Factor 3: Match expand factor ──
      {"Griffin (2L, expand=2, NO local_attn)",
       fn -> Edifice.Attention.Griffin.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
         expand_factor: 2, dropout: 0.0,
         window_size: @window_size, seq_len: @window_size,
         use_local_attention: false
       ) end},

      # ── Factor 4: Match everything (should be ~same as Mamba) ──
      {"Mamba (6L, expand=2)",
       fn -> Edifice.SSM.Mamba.build(
         embed_dim: @embed_size, hidden_size: 256, num_layers: 6,
         expand_factor: 2, state_size: 16, conv_size: 4,
         dropout: 0.0, window_size: @window_size
       ) end},
    ]

    results = Enum.map(configs, fn {label, build_fn} ->
      IO.write(:stderr, "  #{String.pad_trailing(label, 45)} ... ")

      try do
        model = build_fn.()
        {p95, param_count} = benchmark_model(model, is_exla)
        IO.puts(:stderr, "#{fmt(p95)} ms  (#{fmt_params(param_count)} params)")
        {label, p95, param_count}
      rescue
        e ->
          IO.puts(:stderr, "ERROR: #{Exception.message(e)}")
          {label, nil, 0}
      end
    end)

    # ── Also test with compiler: EXLA if available ──
    results =
      if is_exla do
        IO.puts(:stderr, "\n--- With compiler: EXLA (graph-compiled) ---\n")

        compiler_configs = [
          {"Mamba (2L) + compiler:EXLA",
           fn -> Edifice.SSM.Mamba.build(
             embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
             expand_factor: 2, state_size: 16, conv_size: 4,
             dropout: 0.0, window_size: @window_size
           ) end},

          {"Griffin (6L) + compiler:EXLA",
           fn -> Edifice.Attention.Griffin.build(
             embed_dim: @embed_size, hidden_size: 256, num_layers: 6,
             expand_factor: 3, local_attn_window: 32, num_heads: 4,
             dropout: 0.0, window_size: @window_size, seq_len: @window_size,
             use_local_attention: true
           ) end},

          {"Griffin (2L, no attn) + compiler:EXLA",
           fn -> Edifice.Attention.Griffin.build(
             embed_dim: @embed_size, hidden_size: 256, num_layers: 2,
             expand_factor: 3, dropout: 0.0,
             window_size: @window_size, seq_len: @window_size,
             use_local_attention: false
           ) end},
        ]

        compiler_results = Enum.map(compiler_configs, fn {label, build_fn} ->
          IO.write(:stderr, "  #{String.pad_trailing(label, 45)} ... ")

          try do
            model = build_fn.()
            {p95, param_count} = benchmark_model(model, is_exla, compiler: EXLA)
            IO.puts(:stderr, "#{fmt(p95)} ms  (#{fmt_params(param_count)} params)")
            {label, p95, param_count}
          rescue
            e ->
              IO.puts(:stderr, "ERROR: #{Exception.message(e)}")
              {label, nil, 0}
          end
        end)

        results ++ compiler_results
      else
        results
      end

    # ── Summary ──
    IO.puts(:stderr, "\n" <> String.duplicate("=", 75))
    IO.puts(:stderr, "RESULTS SUMMARY")
    IO.puts(:stderr, String.duplicate("=", 75))
    IO.puts(:stderr,
      String.pad_trailing("Configuration", 48) <>
      String.pad_leading("p95 (ms)", 10) <>
      String.pad_leading("Params", 10))
    IO.puts(:stderr, String.duplicate("-", 75))

    baseline_p95 = case List.first(results) do
      {_, p95, _} when is_number(p95) -> p95
      _ -> 1.0
    end

    for {label, p95, params} <- results do
      p95_str = if p95, do: fmt(p95), else: "ERROR"
      ratio = if p95 && baseline_p95 > 0, do: " (#{:erlang.float_to_binary(p95 / baseline_p95, decimals: 1)}x)", else: ""
      IO.puts(:stderr,
        String.pad_trailing(label, 48) <>
        String.pad_leading(p95_str <> ratio, 17) <>
        String.pad_leading(fmt_params(params), 10))
    end
  end

  defp benchmark_model(model, is_exla, build_opts \\ []) do
    all_opts = [mode: :inference] ++ build_opts
    {init_fn, pred_fn} = Axon.build(model, all_opts)

    template = Nx.template({1, @window_size, @embed_size}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())
    param_count = ExPhil.Training.GPUUtils.count_params(params)

    key = Nx.Random.key(42)
    {input, _} = Nx.Random.uniform(key, shape: {1, @window_size, @embed_size}, type: :f32)

    # Warmup
    for _ <- 1..@warmup do
      result = pred_fn.(params, input)
      if is_exla, do: Nx.backend_transfer(result, Nx.BinaryBackend)
    end

    # Time
    times = for _ <- 1..@iterations do
      {time_us, _} = :timer.tc(fn ->
        result = pred_fn.(params, input)
        if is_exla, do: Nx.backend_transfer(result, Nx.BinaryBackend), else: result
      end)
      time_us / 1000.0
    end

    sorted = Enum.sort(times)
    p95_idx = round(length(sorted) * 0.95) - 1
    p95 = Enum.at(sorted, max(p95_idx, 0))

    {p95, param_count}
  end

  defp fmt(val), do: :erlang.float_to_binary(val * 1.0, decimals: 1)

  defp fmt_params(n) when n >= 1_000_000, do: "#{:erlang.float_to_binary(n / 1_000_000, decimals: 1)}M"
  defp fmt_params(n) when n >= 1_000, do: "#{:erlang.float_to_binary(n / 1000.0, decimals: 1)}K"
  defp fmt_params(n), do: "#{n}"
end

GriffinMambaProfile.run()
