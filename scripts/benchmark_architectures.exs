#!/usr/bin/env elixir
# Benchmark different architectures on the same dataset
#
# Usage:
#   mix run scripts/benchmark_architectures.exs --replay-dir /path/to/replays
#
# This script runs each architecture preset on the same data and compares:
# - Training loss convergence
# - Validation accuracy
# - Training speed (batches/sec)
# - GPU memory usage
#
# Results are saved to checkpoints/benchmark_results.json and benchmark_report.html

defmodule Output do
  def puts(line) do
    timestamp = DateTime.utc_now() |> Calendar.strftime("%H:%M:%S")
    IO.puts(:stderr, "[#{timestamp}] #{line}")
  end
  def puts_raw(line), do: IO.puts(:stderr, line)
end

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Config, Data, GPUUtils, Imitation, Plots}
alias ExPhil.Embeddings

# Parse args
args = System.argv()

replay_dir = Enum.find_value(args, "./replays", fn
  "--replay-dir" -> nil
  "--replays" -> nil
  arg ->
    idx = Enum.find_index(args, &(&1 == "--replay-dir" || &1 == "--replays"))
    if idx && Enum.at(args, idx + 1) == arg, do: arg, else: nil
end)

# Find replay-dir value properly
replay_dir = case Enum.find_index(args, &(&1 in ["--replay-dir", "--replays"])) do
  nil -> "./replays"
  idx -> Enum.at(args, idx + 1) || "./replays"
end

max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> 30
  idx -> String.to_integer(Enum.at(args, idx + 1) || "30")
end

epochs = case Enum.find_index(args, &(&1 == "--epochs")) do
  nil -> 3
  idx -> String.to_integer(Enum.at(args, idx + 1) || "3")
end

batch_size = case Enum.find_index(args, &(&1 == "--batch-size")) do
  nil ->
    # Auto-detect: use 256 for GPU, 128 for CPU
    if System.get_env("EXLA_TARGET") == "cuda", do: 256, else: 128
  idx -> String.to_integer(Enum.at(args, idx + 1) || "128")
end

# Architectures to benchmark
architectures = [
  {:mlp, "MLP (baseline)", [temporal: false, hidden_sizes: [128, 128], precompute: true]},
  {:mamba, "Mamba SSM", [temporal: true, backbone: :mamba, window_size: 30, num_layers: 2]},
  {:jamba, "Jamba (Mamba+Attn)", [temporal: true, backbone: :jamba, window_size: 30, num_layers: 3, attention_every: 3]},
  {:lstm, "LSTM", [temporal: true, backbone: :lstm, window_size: 30, num_layers: 1]},
  {:gru, "GRU", [temporal: true, backbone: :gru, window_size: 30, num_layers: 1]},
  {:attention, "Attention", [temporal: true, backbone: :attention, window_size: 30, num_layers: 1, num_heads: 4]}
]

Output.puts("""

╔════════════════════════════════════════════════════════════════╗
║           ExPhil Architecture Benchmark                        ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Replay dir:    #{replay_dir}
  Max files:     #{max_files}
  Epochs:        #{epochs}
  Batch size:    #{batch_size}
  Architectures: #{length(architectures)}
  GPU:           #{GPUUtils.memory_status_string()}

""")

# Load replays once
Output.puts("Step 1: Loading replays...")
replay_files = Path.wildcard("#{replay_dir}/**/*.slp") |> Enum.take(max_files)
Output.puts("  Found #{length(replay_files)} replay files")

if length(replay_files) == 0 do
  Output.puts("ERROR: No replay files found in #{replay_dir}")
  System.halt(1)
end

# Parse all replays
Output.puts("\nStep 2: Parsing replays...")
all_frames = replay_files
|> Enum.with_index(1)
|> Enum.flat_map(fn {path, idx} ->
  if rem(idx, 10) == 0, do: Output.puts("  Parsing #{idx}/#{length(replay_files)}...")
  case Peppi.parse(path) do
    {:ok, replay} -> Peppi.to_training_frames(replay)
    {:error, _} -> []
  end
end)

Output.puts("  Total frames: #{length(all_frames)}")

# Create base dataset
dataset = Data.from_frames(all_frames)

# Split train/val
{train_frames, val_frames} = Enum.split(all_frames, trunc(length(all_frames) * 0.9))
train_dataset = Data.from_frames(train_frames)
val_dataset = Data.from_frames(val_frames)

Output.puts("  Train: #{train_dataset.size} frames, Val: #{val_dataset.size} frames")

# Benchmark results
results = []

Output.puts("\nStep 3: Running benchmarks...")
Output.puts_raw("─" |> String.duplicate(60))

results = Enum.map(architectures, fn {arch_id, arch_name, arch_opts} ->
  Output.puts("\n▶ Benchmarking: #{arch_name}")
  Output.puts("  #{GPUUtils.memory_status_string()}")

  # Merge with base options
  opts = Keyword.merge([
    epochs: epochs,
    batch_size: batch_size,
    hidden_sizes: [128, 128],
    learning_rate: 1.0e-4,
    warmup_steps: 10,
    val_split: 0.0,  # We handle split ourselves
    checkpoint: "checkpoints/benchmark_#{arch_id}.axon"
  ], arch_opts)

  # Build embed config (use default)
  embed_config = Embeddings.config()

  # Prepare dataset based on architecture
  prepared_train = if opts[:temporal] do
    seq_ds = Data.to_sequences(train_dataset,
      window_size: opts[:window_size] || 30,
      stride: opts[:stride] || 1)
    Data.precompute_embeddings(seq_ds)
  else
    if opts[:precompute] do
      Data.precompute_frame_embeddings(train_dataset)
    else
      train_dataset
    end
  end

  prepared_val = if opts[:temporal] do
    seq_ds = Data.to_sequences(val_dataset,
      window_size: opts[:window_size] || 30,
      stride: opts[:stride] || 1)
    Data.precompute_embeddings(seq_ds)
  else
    val_dataset
  end

  # Create trainer
  trainer = Imitation.new(
    hidden_sizes: opts[:hidden_sizes],
    embed_config: embed_config,
    temporal: opts[:temporal],
    backbone: opts[:backbone],
    window_size: opts[:window_size],
    num_layers: opts[:num_layers],
    num_heads: opts[:num_heads],
    attention_every: opts[:attention_every]
  )

  # Training loop with timing
  start_time = System.monotonic_time(:millisecond)

  {final_trainer, epoch_metrics} = Enum.reduce(1..opts[:epochs], {trainer, []}, fn epoch, {t, metrics} ->
    epoch_start = System.monotonic_time(:millisecond)

    # Create batches
    batches = if opts[:temporal] do
      Data.batched_sequences(prepared_train, batch_size: opts[:batch_size], shuffle: true) |> Enum.to_list()
    else
      Data.batched_frames(prepared_train, batch_size: opts[:batch_size], shuffle: true) |> Enum.to_list()
    end

    # Train epoch
    {updated_t, losses} = Enum.reduce(batches, {t, []}, fn batch, {tr, ls} ->
      {new_tr, m} = Imitation.train_step(tr, batch, nil)
      {new_tr, [m.loss | ls]}
    end)

    epoch_time = System.monotonic_time(:millisecond) - epoch_start
    avg_loss = Enum.sum(losses) / length(losses)
    batches_per_sec = length(batches) / (epoch_time / 1000)

    # Validation
    val_batches = if opts[:temporal] do
      Data.batched_sequences(prepared_val, batch_size: opts[:batch_size], shuffle: false) |> Enum.to_list()
    else
      Data.batched_frames(prepared_val, batch_size: opts[:batch_size], shuffle: false) |> Enum.to_list()
    end

    val_losses = Enum.map(val_batches, fn batch ->
      Imitation.evaluate(updated_t, batch).loss
    end)
    val_loss = if length(val_losses) > 0, do: Enum.sum(val_losses) / length(val_losses), else: avg_loss

    Output.puts("    Epoch #{epoch}: train=#{Float.round(avg_loss, 4)} val=#{Float.round(val_loss, 4)} (#{Float.round(batches_per_sec, 1)} batch/s)")

    epoch_entry = %{
      epoch: epoch,
      train_loss: avg_loss,
      val_loss: val_loss,
      batches_per_sec: batches_per_sec,
      time_ms: epoch_time
    }

    {updated_t, [epoch_entry | metrics]}
  end)

  total_time = System.monotonic_time(:millisecond) - start_time
  epoch_metrics = Enum.reverse(epoch_metrics)

  # Final metrics
  final_train = List.last(epoch_metrics).train_loss
  final_val = List.last(epoch_metrics).val_loss
  avg_speed = Enum.sum(Enum.map(epoch_metrics, & &1.batches_per_sec)) / length(epoch_metrics)

  Output.puts("  ✓ Complete: final_val=#{Float.round(final_val, 4)}, speed=#{Float.round(avg_speed, 1)} batch/s, time=#{Float.round(total_time/1000, 1)}s")

  %{
    id: arch_id,
    name: arch_name,
    final_train_loss: final_train,
    final_val_loss: final_val,
    avg_batches_per_sec: avg_speed,
    total_time_ms: total_time,
    epochs: epoch_metrics,
    config: Keyword.take(opts, [:temporal, :backbone, :window_size, :num_layers, :hidden_sizes])
  }
end)

Output.puts_raw("\n" <> String.duplicate("─", 60))

# Sort by validation loss
sorted_results = Enum.sort_by(results, & &1.final_val_loss)

# Print comparison table
Output.puts("\n╔════════════════════════════════════════════════════════════════╗")
Output.puts("║                    Benchmark Results                           ║")
Output.puts("╚════════════════════════════════════════════════════════════════╝\n")

Output.puts("Ranked by validation loss (lower is better):\n")
Output.puts("  Rank | Architecture    | Val Loss | Train Loss | Speed (b/s) | Time")
Output.puts("  ─────┼─────────────────┼──────────┼────────────┼─────────────┼──────")

sorted_results
|> Enum.with_index(1)
|> Enum.each(fn {r, rank} ->
  name = String.pad_trailing(r.name, 15)
  val = Float.round(r.final_val_loss, 4) |> to_string() |> String.pad_leading(8)
  train = Float.round(r.final_train_loss, 4) |> to_string() |> String.pad_leading(10)
  speed = Float.round(r.avg_batches_per_sec, 1) |> to_string() |> String.pad_leading(11)
  time = "#{Float.round(r.total_time_ms/1000, 1)}s" |> String.pad_leading(5)
  Output.puts("  #{rank}    | #{name} | #{val} | #{train} | #{speed} | #{time}")
end)

# Best architecture
best = List.first(sorted_results)
Output.puts("\n★ Best architecture: #{best.name} (val_loss=#{Float.round(best.final_val_loss, 4)})")

# Save results
results_path = "checkpoints/benchmark_results.json"
File.mkdir_p!("checkpoints")

json_results = %{
  timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
  config: %{
    replay_dir: replay_dir,
    max_files: max_files,
    epochs: epochs,
    train_frames: train_dataset.size,
    val_frames: val_dataset.size
  },
  results: sorted_results,
  best: best.id
}

File.write!(results_path, Jason.encode!(json_results, pretty: true))
Output.puts("\n✓ Results saved to #{results_path}")

# Generate comparison plot
report_path = "checkpoints/benchmark_report.html"

# Build loss comparison data
plot_data = results
|> Enum.flat_map(fn r ->
  Enum.map(r.epochs, fn e ->
    %{architecture: r.name, epoch: e.epoch, loss: e.val_loss, type: "val"}
  end)
end)

comparison_plot = VegaLite.new(width: 700, height: 400, title: "Architecture Comparison - Validation Loss")
|> VegaLite.data_from_values(plot_data)
|> VegaLite.mark(:line, point: true)
|> VegaLite.encode_field(:x, "epoch", type: :quantitative, title: "Epoch")
|> VegaLite.encode_field(:y, "loss", type: :quantitative, title: "Validation Loss", scale: [zero: false])
|> VegaLite.encode_field(:color, "architecture", type: :nominal, title: "Architecture")

# Save report
spec = VegaLite.Export.to_json(comparison_plot)

html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Architecture Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; }
    h1 { color: #333; }
    .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #333; color: white; }
    tr:nth-child(1) td { background: #d4edda; font-weight: bold; }
    .winner { color: #28a745; font-weight: bold; }
    #plot { margin: 30px 0; }
  </style>
</head>
<body>
  <h1>Architecture Benchmark Report</h1>

  <div class="summary">
    <h3>Configuration</h3>
    <p>Replays: #{max_files} files (#{train_dataset.size} train / #{val_dataset.size} val frames)</p>
    <p>Epochs: #{epochs}</p>
    <p>Generated: #{DateTime.utc_now() |> Calendar.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
  </div>

  <h2>Results (ranked by validation loss)</h2>
  <table>
    <tr><th>Rank</th><th>Architecture</th><th>Val Loss</th><th>Train Loss</th><th>Speed</th><th>Time</th></tr>
    #{sorted_results |> Enum.with_index(1) |> Enum.map(fn {r, rank} ->
      "<tr><td>#{rank}</td><td>#{r.name}</td><td>#{Float.round(r.final_val_loss, 4)}</td><td>#{Float.round(r.final_train_loss, 4)}</td><td>#{Float.round(r.avg_batches_per_sec, 1)} b/s</td><td>#{Float.round(r.total_time_ms/1000, 1)}s</td></tr>"
    end) |> Enum.join("\n")}
  </table>

  <p class="winner">★ Best: #{best.name}</p>

  <h2>Loss Curves</h2>
  <div id="plot"></div>

  <script>vegaEmbed('#plot', #{spec});</script>
</body>
</html>
"""

File.write!(report_path, html)
Output.puts("✓ Report saved to #{report_path}")

Output.puts("\nDone! Open #{report_path} in a browser to see the comparison.")
