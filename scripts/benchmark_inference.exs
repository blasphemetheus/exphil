#!/usr/bin/env elixir
# Real-Time Inference Viability Benchmark
#
# Measures pure forward-pass latency at batch_size=1 to determine which
# architectures can sustain 60 FPS (16.67ms budget) for real-time Melee play.
# Uses random inputs — no training data required.
#
# Usage:
#   mix run scripts/benchmark_inference.exs [options]
#
# Options:
#   --only arch1,arch2    Only test specific architectures
#   --skip arch1,arch2    Skip specific architectures
#   --quick               Fewer sweep configs (16 per arch vs 72)
#   --thorough            Include more architectures + larger sweep
#   --iterations N        Timing iterations per config (default: 30)
#   --warmup N            Warmup iterations (default: 5)
#   --jit-timeout N       Max JIT compile seconds (default: 120)
#   --embed-size N        Input embedding size (default: 288)
#   --output PATH         JSON output path
#   --quiet / --verbose   Verbosity control
#
# Examples:
#   # Quick sanity check
#   mix run scripts/benchmark_inference.exs --only mlp,mamba --quick
#
#   # Full default sweep (13 architectures)
#   mix run scripts/benchmark_inference.exs
#
#   # Thorough sweep with all viable architectures
#   mix run scripts/benchmark_inference.exs --thorough
#
#   # Single architecture deep dive
#   mix run scripts/benchmark_inference.exs --only lstm --thorough

# ============================================================================
# Section 0: Early environment configuration
# ============================================================================

# Suppress XLA ptxas warnings before EXLA initializes
unless "--verbose" in System.argv() do
  System.put_env("TF_CPP_MIN_LOG_LEVEL", "2")
end

# ============================================================================
# Section 1: Helper module
# ============================================================================

defmodule BenchInference do
  @fps_threshold_ms 16.67
  @cutoff_ms 50.0

  @doc "Build backbone opts tailored for each architecture type."
  def build_opts(arch, hidden_size, num_layers, window_size) do
    base = [window_size: window_size, seq_len: window_size, dropout: 0.0]

    case arch do
      :mlp ->
        base ++ [hidden_sizes: List.duplicate(hidden_size, num_layers)]

      type when type in [:sliding_window, :attention] ->
        {nh, hd} = decompose_heads(hidden_size)
        base ++ [num_heads: nh, head_dim: hd, num_layers: num_layers]

      :lstm_hybrid ->
        {nh, hd} = decompose_heads(hidden_size)
        base ++ [hidden_size: hidden_size, num_heads: nh, head_dim: hd, num_layers: num_layers]

      :perceiver ->
        base ++ [latent_dim: hidden_size, num_latents: min(64, window_size),
                 num_layers: num_layers, num_heads: min(4, hidden_size)]

      :decision_transformer ->
        {nh, hd} = decompose_heads(hidden_size)
        base ++ [hidden_size: hidden_size, num_heads: nh, head_dim: hd, num_layers: num_layers]

      :jamba ->
        base ++ [hidden_size: hidden_size, num_layers: max(num_layers, 2),
                 attention_every: max(num_layers, 2), num_heads: min(4, hidden_size)]

      :zamba ->
        base ++ [hidden_size: hidden_size, num_layers: max(num_layers, 2),
                 attention_every: max(num_layers, 2), num_heads: min(4, hidden_size)]

      _ ->
        base ++ [hidden_size: hidden_size, num_layers: num_layers]
    end
  end

  @doc "Decompose hidden_size into (num_heads, head_dim) for attention architectures."
  def decompose_heads(hidden_size) do
    num_heads = cond do
      rem(hidden_size, 4) == 0 -> 4
      rem(hidden_size, 2) == 0 -> 2
      true -> 1
    end
    {num_heads, div(hidden_size, num_heads)}
  end

  @doc "Benchmark a single (arch, hidden, layers, window) config."
  def benchmark_config(arch, embed_size, hidden_size, num_layers, window_size, bench_opts) do
    alias ExPhil.Networks.Policy.Backbone
    alias ExPhil.Training.GPUUtils

    iterations = bench_opts[:iterations]
    warmup_n = bench_opts[:warmup]
    jit_timeout = bench_opts[:jit_timeout]
    is_exla = bench_opts[:is_exla]

    arch_opts = build_opts(arch, hidden_size, num_layers, window_size)

    # Create input tensor
    key = Nx.Random.key(42)
    {input, _} = Nx.Random.uniform(key, shape: {1, window_size, embed_size}, type: :f32)

    # Build + init + warmup inside a timeout task
    task = Task.async(fn ->
      model = Backbone.build_temporal_backbone(embed_size, arch, arch_opts)
      build_opts = if Code.ensure_loaded?(EXLA), do: [mode: :inference, compiler: EXLA], else: [mode: :inference]
      {init_fn, pred_fn} = Axon.build(model, build_opts)

      template = Nx.template({1, window_size, embed_size}, :f32)
      params = init_fn.(template, Axon.ModelState.empty())
      param_count = GPUUtils.count_params(params)

      # Warmup iterations (triggers JIT compilation)
      for _ <- 1..warmup_n do
        result = pred_fn.(params, input)
        if is_exla, do: Nx.backend_transfer(result, Nx.BinaryBackend)
      end

      # NaN check after warmup
      check_result = pred_fn.(params, input)
      check_result = Nx.backend_transfer(check_result, Nx.BinaryBackend)
      has_nan = check_result |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 1

      {pred_fn, params, param_count, has_nan}
    end)

    case Task.yield(task, jit_timeout * 1000) || Task.shutdown(task, :brutal_kill) do
      {:ok, {_pred_fn, _params, _param_count, true}} ->
        %{status: :nan, hidden_size: hidden_size, num_layers: num_layers,
          window_size: window_size, param_count: 0}

      {:ok, {pred_fn, params, param_count, false}} ->
        # Time the forward passes
        times =
          for _ <- 1..iterations do
            {time_us, _} = :timer.tc(fn ->
              result = pred_fn.(params, input)
              if is_exla, do: Nx.backend_transfer(result, Nx.BinaryBackend), else: result
            end)
            time_us / 1000.0
          end

        stats = compute_stats(times)

        Map.merge(stats, %{
          status: :ok,
          hidden_size: hidden_size,
          num_layers: num_layers,
          window_size: window_size,
          param_count: param_count,
          viable_60fps: stats.p95 < @fps_threshold_ms
        })

      nil ->
        %{status: :jit_timeout, hidden_size: hidden_size, num_layers: num_layers,
          window_size: window_size, param_count: 0}
    end
  rescue
    e ->
      msg = Exception.message(e)
      status = if String.contains?(msg, "memory"), do: :oom, else: :error
      %{status: status, hidden_size: hidden_size, num_layers: num_layers,
        window_size: window_size, param_count: 0, error: msg}
  end

  @doc "Compute timing statistics from a list of durations (ms)."
  def compute_stats(times) do
    sorted = Enum.sort(times)
    n = length(sorted)
    %{
      median: percentile(sorted, n, 50),
      p95: percentile(sorted, n, 95),
      p99: percentile(sorted, n, 99),
      min: List.first(sorted),
      max: List.last(sorted),
      mean: Enum.sum(times) / n,
      std: std_dev(times)
    }
  end

  defp percentile(sorted, n, p) do
    rank = max(1, ceil(p / 100 * n))
    Enum.at(sorted, rank - 1)
  end

  defp std_dev(values) do
    n = length(values)
    mean = Enum.sum(values) / n
    variance = Enum.sum(Enum.map(values, &((&1 - mean) * (&1 - mean)))) / n
    :math.sqrt(variance)
  end

  @doc "Format milliseconds for display."
  def fmt_ms(nil), do: "  -   "
  def fmt_ms(ms) when ms < 1.0, do: "#{Float.round(ms * 1.0, 2)}ms"
  def fmt_ms(ms) when ms < 10.0, do: "#{Float.round(ms * 1.0, 1)}ms"
  def fmt_ms(ms), do: "#{round(ms)}ms"

  @doc "Format parameter count for display."
  def fmt_params(0), do: "-"
  def fmt_params(n) when n >= 1_000_000, do: "#{Float.round(n / 1_000_000, 1)}M"
  def fmt_params(n) when n >= 1_000, do: "#{Float.round(n / 1_000, 1)}K"
  def fmt_params(n), do: "#{n}"

  @doc "Find the largest config that runs under the FPS threshold."
  def max_viable_config(configs) do
    configs
    |> Enum.filter(&(&1.status == :ok and &1.viable_60fps))
    |> Enum.sort_by(&(-&1.hidden_size * 1000 - &1.num_layers * 100 - &1.window_size))
    |> List.first()
  end

  @doc "Generate the sweep grid (sorted cheapest-first)."
  def sweep_grid(quick?) do
    {window_sizes, hidden_sizes, layer_counts} =
      if quick? do
        {[1, 10, 30, 60], [128, 256], [1, 2]}
      else
        {[1, 5, 10, 20, 30, 60], [64, 128, 256, 512], [1, 2, 4]}
      end

    for h <- hidden_sizes,
        l <- layer_counts,
        w <- window_sizes do
      {h, l, w}
    end
  end

  @doc "Apply per-architecture caps to the sweep grid."
  def apply_arch_caps(grid, arch) do
    caps = arch_caps(arch)
    max_h = caps[:max_hidden] || 99999
    max_l = caps[:max_layers] || 99999
    max_w = caps[:max_window] || 99999

    Enum.filter(grid, fn {h, l, w} ->
      h <= max_h and l <= max_l and w <= max_w
    end)
  end

  defp arch_caps(arch) do
    case arch do
      :liquid -> [max_hidden: 256, max_layers: 2]
      :kan -> [max_hidden: 256, max_layers: 2]
      :hopfield -> [max_hidden: 256, max_layers: 2]
      :jamba -> [max_hidden: 256, max_layers: 4]
      :zamba -> [max_hidden: 256, max_layers: 4]
      :perceiver -> [max_hidden: 256, max_layers: 2]
      :ttt -> [max_hidden: 256, max_layers: 2]
      :ntm -> [max_hidden: 256, max_layers: 2]
      :decision_transformer -> [max_hidden: 256, max_layers: 4]
      _ -> []
    end
  end

  @doc "Render the summary table."
  def render_summary_table(all_results) do
    IO.puts(:stderr, "")
    IO.puts(:stderr, String.duplicate("=", 90))
    IO.puts(:stderr, "SUMMARY: Max Viable Config for 60 FPS (<#{@fps_threshold_ms}ms)")
    IO.puts(:stderr, String.duplicate("=", 90))

    header = String.pad_trailing("Architecture", 22) <>
      "| " <> String.pad_trailing("Max Viable Config", 24) <>
      "| " <> String.pad_trailing("p95", 9) <>
      "| " <> String.pad_trailing("Params", 8) <>
      "| " <> String.pad_trailing("Configs", 8) <>
      "| Status"
    IO.puts(:stderr, header)
    IO.puts(:stderr, String.duplicate("-", 90))

    for {arch, configs} <- all_results do
      ok_count = Enum.count(configs, &(&1.status == :ok))
      fail_count = length(configs) - ok_count
      viable = max_viable_config(configs)

      {config_str, p95_str, param_str, status_str} =
        case viable do
          nil ->
            # No viable config — check if anything ran at all
            any_ok = Enum.find(configs, &(&1.status == :ok))
            if any_ok do
              {"(none < 16.7ms)", fmt_ms(any_ok.p95), fmt_params(any_ok.param_count), "SLOW"}
            else
              first_fail = List.first(configs)
              status = if first_fail, do: Atom.to_string(first_fail.status), else: "no data"
              {"(failed)", "  -  ", "-", String.upcase(status)}
            end

          v ->
            config = "h=#{v.hidden_size} w=#{v.window_size} L=#{v.num_layers}"
            {config, fmt_ms(v.p95), fmt_params(v.param_count), "OK"}
        end

      counts = "#{ok_count}/#{ok_count + fail_count}"

      line = String.pad_trailing("#{arch}", 22) <>
        "| " <> String.pad_trailing(config_str, 24) <>
        "| " <> String.pad_leading(p95_str, 7) <> "  " <>
        "| " <> String.pad_trailing(param_str, 8) <>
        "| " <> String.pad_trailing(counts, 8) <>
        "| " <> status_str
      IO.puts(:stderr, line)
    end

    IO.puts(:stderr, String.duplicate("-", 90))
  end

  @doc "Render ASCII heatmap showing window_size scaling at fixed h and L."
  def render_heatmap(all_results, heatmap_h, heatmap_l, window_sizes) do
    IO.puts(:stderr, "")
    IO.puts(:stderr, String.duplicate("=", 70))
    IO.puts(:stderr, "HEATMAP: Window size scaling (h=#{heatmap_h}, L=#{heatmap_l})")
    IO.puts(:stderr, String.duplicate("=", 70))

    # Header
    w_labels = Enum.map_join(window_sizes, "  ", fn w ->
      String.pad_leading("w=#{w}", 5)
    end)
    IO.puts(:stderr, String.pad_trailing("Architecture", 18) <> "| " <> w_labels)
    IO.puts(:stderr, String.duplicate("-", 18) <> "+" <> String.duplicate("-", length(window_sizes) * 7))

    for {arch, configs} <- all_results do
      cells = Enum.map_join(window_sizes, "  ", fn w ->
        config = Enum.find(configs, fn c ->
          c.hidden_size == heatmap_h and c.num_layers == heatmap_l and c.window_size == w
        end)

        case config do
          nil -> "    -"
          %{status: :ok, p95: p95} when p95 < @fps_threshold_ms -> "    *"
          %{status: :ok, p95: p95} when p95 < @cutoff_ms -> "    ."
          %{status: :ok} -> "    x"
          _ -> "    !"
        end
      end)

      IO.puts(:stderr, String.pad_trailing("#{arch}", 18) <> "| " <> cells)
    end

    IO.puts(:stderr, "")
    IO.puts(:stderr, "Legend:  * = 60 FPS (<#{@fps_threshold_ms}ms)   . = 16-50ms   x = >50ms   ! = failed   - = skipped")
  end

  @doc "Write results to JSON file."
  def write_json(all_results, metadata, output_path) do
    arch_data = Map.new(all_results, fn {arch, configs} ->
      viable = max_viable_config(configs)
      config_data = Enum.map(configs, fn c ->
        base = %{
          hidden_size: c.hidden_size,
          num_layers: c.num_layers,
          window_size: c.window_size,
          status: c.status,
          param_count: c.param_count
        }
        if c.status == :ok do
          Map.merge(base, %{
            median_ms: Float.round(c.median, 3),
            p95_ms: Float.round(c.p95, 3),
            p99_ms: Float.round(c.p99, 3),
            min_ms: Float.round(c.min, 3),
            max_ms: Float.round(c.max, 3),
            viable_60fps: c.viable_60fps
          })
        else
          base
        end
      end)

      viable_data = case viable do
        nil -> nil
        v -> %{hidden_size: v.hidden_size, num_layers: v.num_layers,
               window_size: v.window_size, p95_ms: Float.round(v.p95, 3)}
      end

      {Atom.to_string(arch), %{configs: config_data, max_viable_config: viable_data}}
    end)

    json = %{
      timestamp: DateTime.to_iso8601(DateTime.utc_now()),
      backend: metadata.backend_name,
      embed_size: metadata.embed_size,
      batch_size: 1,
      iterations: metadata.iterations,
      warmup: metadata.warmup,
      fps_threshold_ms: @fps_threshold_ms,
      architectures: arch_data
    }

    json_str = Jason.encode!(json, pretty: true)
    File.write!(output_path, json_str)
    output_path
  end
end

# ============================================================================
# Section 2: CLI parsing
# ============================================================================

alias ExPhil.Training.Output
alias ExPhil.Training.GPUUtils

require Logger

{parsed, _pos, _inv} = OptionParser.parse(System.argv(),
  strict: [
    only: :string,
    skip: :string,
    quick: :boolean,
    thorough: :boolean,
    iterations: :integer,
    warmup: :integer,
    jit_timeout: :integer,
    embed_size: :integer,
    output: :string,
    quiet: :boolean,
    verbose: :boolean
  ]
)

opts = Keyword.merge([
  iterations: 30,
  warmup: 5,
  jit_timeout: 120,
  embed_size: 288,
  quick: false,
  thorough: false
], parsed)

# Setup verbosity
cond do
  opts[:quiet] ->
    Logger.configure(level: :error)
    Output.set_verbosity(:quiet)
  opts[:verbose] ->
    Output.set_verbosity(:verbose)
  true ->
    Logger.configure(level: :warning)
    Output.set_verbosity(:normal)
end

# ============================================================================
# Section 3: Architecture tiers
# ============================================================================

tier_default = [
  :mlp, :mamba, :gated_ssm, :gru, :fnet, :hawk, :reservoir,
  :hgrn, :rwkv, :gla, :retnet, :s4d, :s5
]

tier_thorough = [
  :lstm, :mamba_ssd, :griffin, :xlstm, :s4, :h3, :performer,
  :deltanet, :sliding_window, :snn, :bayesian, :kan
]

tier_only = [
  :jamba, :zamba, :lstm_hybrid, :perceiver, :ttt, :hopfield,
  :ntm, :liquid, :decision_transformer, :min_lstm, :min_gru
]

all_known = tier_default ++ tier_thorough ++ tier_only

# Select architectures
selected_archs =
  if opts[:only] do
    requested = opts[:only] |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
    unknown = Enum.reject(requested, &(&1 in all_known))
    if unknown != [] do
      Output.warning("Unknown architectures: #{Enum.join(unknown, ", ")}")
    end
    Enum.filter(requested, &(&1 in all_known))
  else
    archs = tier_default
    archs = if opts[:thorough], do: archs ++ tier_thorough, else: archs
    archs
  end

selected_archs =
  if opts[:skip] do
    skip_set = opts[:skip] |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1))) |> MapSet.new()
    Enum.reject(selected_archs, &MapSet.member?(skip_set, &1))
  else
    selected_archs
  end

if selected_archs == [] do
  Output.error("No architectures selected. Use --only or check --skip filter.")
  System.halt(1)
end

# ============================================================================
# Section 4: Sweep grid
# ============================================================================

base_grid = BenchInference.sweep_grid(opts[:quick])
{window_sizes, _, _} =
  if opts[:quick] do
    {[1, 10, 30, 60], [128, 256], [1, 2]}
  else
    {[1, 5, 10, 20, 30, 60], [64, 128, 256, 512], [1, 2, 4]}
  end

# Heatmap baseline: use smallest hidden/layers in the grid
{heatmap_h, heatmap_l, _} = List.first(base_grid)

# ============================================================================
# Section 5: Backend detection and banner
# ============================================================================

backend = Nx.default_backend()
is_exla = match?({EXLA.Backend, _}, backend)
backend_name = cond do
  is_exla and GPUUtils.gpu_available?() ->
    case GPUUtils.device_name() do
      {:ok, name} -> "EXLA (#{name})"
      _ -> "EXLA (GPU)"
    end
  is_exla -> "EXLA (CPU)"
  true -> "BinaryBackend (CPU)"
end

Output.banner("Real-Time Inference Viability Benchmark")

Output.config([
  {"Backend", backend_name},
  {"Embed size", opts[:embed_size]},
  {"Batch size", 1},
  {"Iterations", opts[:iterations]},
  {"Warmup", opts[:warmup]},
  {"JIT timeout", "#{opts[:jit_timeout]}s"},
  {"Mode", if(opts[:quick], do: "quick", else: if(opts[:thorough], do: "thorough", else: "default"))},
  {"Architectures", "#{length(selected_archs)} (#{Enum.map_join(selected_archs, ", ", &Atom.to_string/1)})"},
  {"Configs/arch", "#{length(base_grid)} (before per-arch caps)"},
  {"60 FPS threshold", "#{BenchInference.fmt_ms(16.67)}"}
])

if is_exla do
  Output.puts(GPUUtils.memory_status_string())
end

unless is_exla do
  Output.warning("Running on BinaryBackend — results will be much slower than EXLA.")
  Output.warning("For realistic CPU inference times, configure EXLA with EXLA_TARGET=host.")
end

# ============================================================================
# Section 6: Main benchmark loop
# ============================================================================

bench_opts = %{
  iterations: opts[:iterations],
  warmup: opts[:warmup],
  jit_timeout: opts[:jit_timeout],
  is_exla: is_exla
}

total_archs = length(selected_archs)
overall_start = System.monotonic_time(:millisecond)

all_results =
  selected_archs
  |> Enum.with_index(1)
  |> Enum.map(fn {arch, arch_idx} ->
    IO.puts(:stderr, "")
    IO.puts(:stderr, "=== #{arch} (#{arch_idx}/#{total_archs}) ===")

    grid = BenchInference.apply_arch_caps(base_grid, arch)
    total_configs = length(grid)

    if total_configs == 0 do
      Output.warning("  All configs capped for #{arch}, skipping")
      {arch, []}
    else
      {configs, _early_stopped} =
        Enum.reduce_while(Enum.with_index(grid, 1), {[], false}, fn {{h, l, w}, cfg_idx}, {acc, _} ->
          # Progress indicator
          label = "[h=#{String.pad_leading("#{h}", 3)} w=#{String.pad_leading("#{w}", 2)} L=#{l}]"
          IO.write(:stderr, "  #{label}")

          result = BenchInference.benchmark_config(arch, opts[:embed_size], h, l, w, bench_opts)

          case result.status do
            :ok ->
              tag = if result.viable_60fps, do: " OK", else: " SLOW"
              IO.write(:stderr, "  #{BenchInference.fmt_ms(result.p95)}#{tag}")
              # Print newline every 3 configs or at end of line
              if rem(cfg_idx, 3) == 0, do: IO.puts(:stderr, ""), else: IO.write(:stderr, "    ")

              # Early terminate if this config exceeds cutoff and it's not the smallest
              if result.p95 > 50.0 and cfg_idx > 1 do
                IO.puts(:stderr, "")
                Output.puts("  Early stop: p95 #{BenchInference.fmt_ms(result.p95)} > 50ms")
                {:halt, {[result | acc], true}}
              else
                {:cont, {[result | acc], false}}
              end

            status ->
              IO.write(:stderr, "  #{String.upcase(Atom.to_string(status))}")
              if rem(cfg_idx, 3) == 0, do: IO.puts(:stderr, ""), else: IO.write(:stderr, "    ")

              # If the very first (smallest) config fails, skip the rest
              if cfg_idx == 1 do
                IO.puts(:stderr, "")
                Output.puts("  Smallest config failed (#{status}), skipping architecture")
                {:halt, {[result | acc], true}}
              else
                {:cont, {[result | acc], false}}
              end
          end
        end)

      # Ensure we end on a newline
      IO.puts(:stderr, "")

      # Show max viable for this architecture
      viable = BenchInference.max_viable_config(Enum.reverse(configs))
      if viable do
        Output.success("  Max viable: h=#{viable.hidden_size} w=#{viable.window_size} L=#{viable.num_layers} (#{BenchInference.fmt_ms(viable.p95)} p95)")
      else
        ok_configs = Enum.filter(configs, &(&1.status == :ok))
        if ok_configs != [] do
          best = Enum.min_by(ok_configs, & &1.p95)
          Output.warning("  No 60 FPS config found. Best: #{BenchInference.fmt_ms(best.p95)} p95")
        else
          Output.error("  All configs failed")
        end
      end

      # GC between architectures
      :erlang.garbage_collect()
      Process.sleep(100)

      {arch, Enum.reverse(configs)}
    end
  end)

overall_elapsed = System.monotonic_time(:millisecond) - overall_start

# ============================================================================
# Section 7: Summary output
# ============================================================================

BenchInference.render_summary_table(all_results)
BenchInference.render_heatmap(all_results, heatmap_h, heatmap_l, window_sizes)

# Recommendations
IO.puts(:stderr, "")
IO.puts(:stderr, String.duplicate("=", 70))
IO.puts(:stderr, "RECOMMENDATIONS")
IO.puts(:stderr, String.duplicate("=", 70))

viable_archs =
  all_results
  |> Enum.filter(fn {_arch, configs} -> BenchInference.max_viable_config(configs) != nil end)
  |> Enum.map(fn {arch, configs} ->
    v = BenchInference.max_viable_config(configs)
    {arch, v}
  end)
  |> Enum.sort_by(fn {_arch, v} -> {-v.hidden_size, -v.window_size, -v.num_layers} end)

if viable_archs == [] do
  Output.warning("No architectures achieved 60 FPS at any config on this backend.")
  Output.puts("Try: --quick with smaller sweep, or run on GPU with EXLA.")
else
  IO.puts(:stderr, "")
  IO.puts(:stderr, "Top architectures for real-time play (sorted by max viable capacity):")
  IO.puts(:stderr, "")

  viable_archs
  |> Enum.take(5)
  |> Enum.with_index(1)
  |> Enum.each(fn {{arch, v}, idx} ->
    IO.puts(:stderr, "  #{idx}. #{arch} — h=#{v.hidden_size} w=#{v.window_size} L=#{v.num_layers} (#{BenchInference.fmt_ms(v.p95)} p95, #{BenchInference.fmt_params(v.param_count)} params)")
  end)
end

IO.puts(:stderr, "")
IO.puts(:stderr, "Note: Measures backbone-only latency at batch_size=1.")
IO.puts(:stderr, "      Full pipeline (embedding + policy head) adds ~0.5-1ms constant overhead.")
IO.puts(:stderr, "")

# JSON output
output_path = opts[:output] ||
  "checkpoints/inference_viability_#{Calendar.strftime(DateTime.utc_now(), "%Y%m%d_%H%M%S")}.json"

case Code.ensure_loaded(Jason) do
  {:module, Jason} ->
    File.mkdir_p!(Path.dirname(output_path))
    metadata = %{
      backend_name: backend_name,
      embed_size: opts[:embed_size],
      iterations: opts[:iterations],
      warmup: opts[:warmup]
    }
    BenchInference.write_json(all_results, metadata, output_path)
    Output.success("Results saved to #{output_path}")

  _ ->
    Output.warning("Jason not available — skipping JSON output. Add {:jason, \"~> 1.4\"} to deps.")
end

Output.puts("Total time: #{Output.format_duration(overall_elapsed)}")
Output.success("Benchmark complete!")
