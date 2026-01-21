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

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, GPUUtils, Imitation, Output}
alias ExPhil.Embeddings

require Output  # For timed macro

# Parse args
args = System.argv()

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

Output.banner("ExPhil Architecture Benchmark")
Output.config([
  {"Replay dir", replay_dir},
  {"Max files", max_files},
  {"Epochs", epochs},
  {"Batch size", batch_size},
  {"Architectures", length(architectures)},
  {"GPU", GPUUtils.memory_status_string()}
])

# Step 1: Load replays
Output.step(1, 3, "Loading replays")
replay_files = Path.wildcard("#{replay_dir}/**/*.slp") |> Enum.take(max_files)
Output.puts("Found #{length(replay_files)} replay files")

if length(replay_files) == 0 do
  Output.error("No replay files found in #{replay_dir}")
  System.halt(1)
end

# Step 2: Parse all replays
Output.step(2, 3, "Parsing replays")
total_files = length(replay_files)

all_frames = replay_files
|> Enum.with_index(1)
|> Enum.flat_map(fn {path, idx} ->
  Output.progress_bar(idx, total_files, label: "Parsing")
  case Peppi.parse(path) do
    {:ok, replay} -> Peppi.to_training_frames(replay)
    {:error, _} -> []
  end
end)
Output.progress_done()

Output.puts("Total frames: #{length(all_frames)}")

# Split train/val
{train_frames, val_frames} = Enum.split(all_frames, trunc(length(all_frames) * 0.9))
train_dataset = Data.from_frames(train_frames)
val_dataset = Data.from_frames(val_frames)

Output.puts("Train: #{train_dataset.size} frames, Val: #{val_dataset.size} frames")

# Step 3: Run benchmarks
Output.step(3, 3, "Running benchmarks")
Output.divider()

num_archs = length(architectures)
results = architectures
|> Enum.with_index(1)
|> Enum.map(fn {{arch_id, arch_name, arch_opts}, arch_idx} ->
  Output.section("[#{arch_idx}/#{num_archs}] #{arch_name}")
  Output.puts("#{GPUUtils.memory_status_string()}")

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
  prepared_train = Output.timed "Preparing train data" do
    if opts[:temporal] do
      seq_ds = Data.to_sequences(train_dataset,
        window_size: opts[:window_size] || 30,
        stride: opts[:stride] || 1)
      Data.precompute_embeddings(seq_ds, show_progress: false)
    else
      if opts[:precompute] do
        Data.precompute_frame_embeddings(train_dataset, show_progress: false)
      else
        train_dataset
      end
    end
  end

  prepared_val = Output.timed "Preparing val data" do
    if opts[:temporal] do
      seq_ds = Data.to_sequences(val_dataset,
        window_size: opts[:window_size] || 30,
        stride: opts[:stride] || 1)
      Data.precompute_embeddings(seq_ds, show_progress: false)
    else
      val_dataset
    end
  end

  # Create trainer
  trainer = Output.timed "Creating model" do
    Imitation.new(
      hidden_sizes: opts[:hidden_sizes],
      embed_config: embed_config,
      temporal: opts[:temporal],
      backbone: opts[:backbone],
      window_size: opts[:window_size],
      num_layers: opts[:num_layers],
      num_heads: opts[:num_heads],
      attention_every: opts[:attention_every]
    )
  end

  # Calculate batch count without materializing (memory efficient)
  num_train_samples = if opts[:temporal], do: prepared_train.size, else: prepared_train.size
  num_batches = div(num_train_samples, opts[:batch_size])
  Output.puts("#{num_batches} train batches (streaming, not pre-materialized)")

  Output.warning("First batch triggers JIT compilation (may take 2-5 min)")

  # Verify batch shape before training (diagnose data issues early)
  test_batch = if opts[:temporal] do
    Data.batched_sequences(prepared_train, batch_size: min(4, opts[:batch_size]), shuffle: false)
    |> Enum.take(1)
    |> List.first()
  else
    Data.batched_frames(prepared_train, batch_size: min(4, opts[:batch_size]), shuffle: false)
    |> Enum.take(1)
    |> List.first()
  end

  if test_batch do
    Output.puts("  Batch states shape: #{inspect(Nx.shape(test_batch.states))}")
    Output.puts("  Batch actions keys: #{inspect(Map.keys(test_batch.actions))}")
  else
    Output.error("  Failed to create test batch!")
  end

  # Training loop with timing
  start_time = System.monotonic_time(:millisecond)
  num_epochs = opts[:epochs]

  {_final_trainer, epoch_metrics} = Enum.reduce(1..num_epochs, {trainer, []}, fn epoch, {t, metrics} ->
    epoch_start = System.monotonic_time(:millisecond)

    # Create fresh batch stream each epoch (lazy, memory efficient)
    # Shuffle happens inside the batch creator via seed
    batches = if opts[:temporal] do
      Data.batched_sequences(prepared_train, batch_size: opts[:batch_size], shuffle: true, seed: epoch)
    else
      Data.batched_frames(prepared_train, batch_size: opts[:batch_size], shuffle: true, seed: epoch)
    end

    # Train epoch with progress
    {updated_t, losses} = batches
    |> Enum.with_index(1)
    |> Enum.reduce({t, []}, fn {batch, batch_idx}, {tr, ls} ->
      # Update progress bar
      Output.progress_bar(batch_idx, num_batches, label: "Epoch #{epoch}/#{num_epochs}")

      # Wrap train_step with error handling to see actual error
      try do
        {new_tr, m} = Imitation.train_step(tr, batch, nil)
        {new_tr, [m.loss | ls]}
      rescue
        e ->
          Output.progress_done()
          Output.error("Train step failed at batch #{batch_idx}")
          Output.error("Batch states shape: #{inspect(Nx.shape(batch.states))}")
          Output.error("Error: #{Exception.message(e)}")
          reraise e, __STACKTRACE__
      end
    end)
    Output.progress_done()

    epoch_time = System.monotonic_time(:millisecond) - epoch_start
    num_losses = length(losses)
    avg_loss = Enum.sum(losses) / num_losses
    batches_per_sec = num_losses / (epoch_time / 1000)

    # Validation (create batches lazily, don't materialize all at once)
    val_batches = if opts[:temporal] do
      Data.batched_sequences(prepared_val, batch_size: opts[:batch_size], shuffle: false)
    else
      Data.batched_frames(prepared_val, batch_size: opts[:batch_size], shuffle: false)
    end

    val_losses = Enum.map(val_batches, fn batch ->
      Imitation.evaluate(updated_t, batch).loss
    end)
    val_loss = if length(val_losses) > 0, do: Enum.sum(val_losses) / length(val_losses), else: avg_loss

    Output.puts("  Epoch #{epoch}: loss=#{Float.round(avg_loss, 4)} val=#{Float.round(val_loss, 4)} (#{Float.round(batches_per_sec, 1)} batch/s)")

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

  Output.success("Complete: val=#{Float.round(final_val, 4)}, speed=#{Float.round(avg_speed, 1)} batch/s, time=#{Float.round(total_time/1000, 1)}s")

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

Output.divider()

# Sort by validation loss
sorted_results = Enum.sort_by(results, & &1.final_val_loss)

# Print comparison table
Output.section("Benchmark Results")
Output.puts("Ranked by validation loss (lower is better):\n")
Output.puts("  Rank | Architecture    | Val Loss | Train Loss | Speed (b/s) | Time")
Output.puts("  -----+-----------------+----------+------------+-------------+------")

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
Output.puts("")
Output.success("Best architecture: #{best.name} (val_loss=#{Float.round(best.final_val_loss, 4)})")

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
Output.puts("Results saved to #{results_path}")

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

# Save report (use to_spec + Jason instead of deprecated Export.to_json)
spec = comparison_plot |> VegaLite.to_spec() |> Jason.encode!()

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

  <p class="winner">Best: #{best.name}</p>

  <h2>Loss Curves</h2>
  <div id="plot"></div>

  <script>vegaEmbed('#plot', #{spec});</script>
</body>
</html>
"""

File.write!(report_path, html)
Output.success("Report saved to #{report_path}")

Output.puts("")
Output.puts("Open #{report_path} in a browser to see the comparison.")
