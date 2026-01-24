#!/usr/bin/env elixir
# Model Evaluation Script: Evaluate trained models on test replays
#
# Usage:
#   mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
#   mix run scripts/eval_model.exs --policy checkpoints/model_policy.bin --replays ~/replays
#   mix run scripts/eval_model.exs --compare model1.axon model2.axon --replays ~/replays
#
# Options:
#   --checkpoint PATH   - Full checkpoint (.axon) for evaluation
#   --policy PATH       - Exported policy (.bin) for evaluation
#   --replays PATH      - Path to test replays (default: ../replays)
#   --max-files N       - Maximum replay files to use (default: 20)
#   --batch-size N      - Evaluation batch size (default: 64)
#   --player PORT       - Player port to evaluate (1-4, default: 1)
#   --character NAME    - Filter replays by character (optional)
#   --compare P1 P2 ... - Compare multiple checkpoints/policies
#   --detailed          - Show detailed per-component metrics
#   --output PATH       - Save results to JSON file
#
# Metrics Computed:
#   - Average loss (cross-entropy)
#   - Per-component accuracy (buttons, main_x, main_y, c_x, c_y, shoulder)
#   - Top-3 accuracy for stick axes
#   - Overall weighted accuracy

require Logger

alias ExPhil.Training.Output
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Config, Data, Imitation}
alias ExPhil.Networks.Policy
alias ExPhil.Embeddings

# Parse command line arguments
args = System.argv()

{opts, positional, _} = OptionParser.parse(args,
  strict: [
    checkpoint: :string,
    policy: :string,
    replays: :string,
    max_files: :integer,
    batch_size: :integer,
    player: :integer,
    character: :string,
    compare: :boolean,
    detailed: :boolean,
    output: :string,
    help: :boolean,
    # Temporal model options
    temporal: :boolean,
    backbone: :string,
    window_size: :integer
  ],
  aliases: [
    c: :checkpoint,
    p: :policy,
    r: :replays,
    m: :max_files,
    b: :batch_size,
    h: :help,
    t: :temporal
  ]
)

if opts[:help] do
  Output.puts("""
  Model Evaluation Script

  Usage:
    mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
    mix run scripts/eval_model.exs --policy checkpoints/model_policy.bin
    mix run scripts/eval_model.exs --compare model1.axon model2.axon

  Options:
    --checkpoint, -c PATH   Full checkpoint (.axon) to evaluate
    --policy, -p PATH       Exported policy (.bin) to evaluate
    --replays, -r PATH      Path to test replays (default: ../replays)
    --max-files, -m N       Max replay files (default: 20)
    --batch-size, -b N      Batch size (default: 64)
    --player PORT           Player port (1-4, default: 1)
    --character NAME        Filter by character
    --compare               Compare multiple models (pass paths as positional args)
    --detailed              Show per-component metrics
    --output PATH           Save results to JSON

  Examples:
    # Evaluate a single model
    mix run scripts/eval_model.exs -c checkpoints/mamba_model.axon

    # Evaluate with specific replays and detailed output
    mix run scripts/eval_model.exs -p model_policy.bin -r ~/test_replays --detailed

    # Compare two models
    mix run scripts/eval_model.exs --compare model_v1.axon model_v2.axon
  """)
  System.halt(0)
end

# Defaults
defaults = %{
  replays: System.get_env("REPLAYS_PATH") || Path.expand("../replays"),
  max_files: 20,
  batch_size: 64,
  player: 1,
  detailed: false,
  temporal: false,
  backbone: "mlp",
  window_size: 60
}

opts = Map.merge(defaults, Map.new(opts))

# Determine evaluation mode
model_paths = cond do
  opts[:compare] == true and length(positional) >= 2 ->
    positional
  opts[:checkpoint] ->
    [opts[:checkpoint]]
  opts[:policy] ->
    [opts[:policy]]
  length(positional) >= 1 ->
    positional
  true ->
    Output.puts("Error: Must provide --checkpoint, --policy, or paths to compare")
    System.halt(1)
end

Output.banner("ExPhil Model Evaluation")
Output.config([
  {"Replays", opts[:replays]},
  {"Max files", opts[:max_files]},
  {"Batch size", opts[:batch_size]},
  {"Player port", opts[:player]},
  {"Character filter", opts[:character] || "none"},
  {"Temporal", opts[:temporal]},
  {"Backbone", if(opts[:temporal], do: opts[:backbone], else: "n/a")},
  {"Window size", if(opts[:temporal], do: opts[:window_size], else: "n/a")},
  {"Models to evaluate", length(model_paths)}
])

# Step 1: Load test replays
Output.step(1, 4, "Loading test replays")
replay_dir = opts[:replays]

unless File.dir?(replay_dir) do
  Output.error("Replay directory not found: #{replay_dir}")
  System.halt(1)
end

replay_files = Path.wildcard(Path.join(replay_dir, "**/*.slp"))
  |> Enum.take(opts[:max_files])

if Enum.empty?(replay_files) do
  Output.error("No .slp files found in #{replay_dir}")
  System.halt(1)
end

Output.puts("  Found #{length(replay_files)} replay files")

# Parse replays
Output.step(2, 4, "Parsing replays")
parse_opts = [
  player_port: opts[:player],
  include_speeds: true
]
parse_opts = if opts[:character] do
  Keyword.put(parse_opts, :filter_character, opts[:character])
else
  parse_opts
end

parsed_replays = Peppi.parse_many(replay_files, parse_opts)

# Convert to training frames format (parse_many returns list of {:ok, replay} tuples)
frames = parsed_replays
|> Enum.flat_map(fn
  {:ok, replay} -> Peppi.to_training_frames(replay, player_port: opts[:player])
  {:error, _} -> []
end)

if Enum.empty?(frames) do
  Output.error("No frames extracted from replays")
  System.halt(1)
end

Output.puts("  Extracted #{length(frames)} frames")

# Get embedding config
embed_config = Embeddings.config(with_speeds: true)
embed_size = Embeddings.embedding_size(embed_config)

# Create dataset
Output.step(3, 4, "Creating evaluation dataset")

dataset = if opts[:temporal] do
  # Temporal mode: create sequences
  base_dataset = Data.from_frames(frames, embed_config: embed_config)
  seq_dataset = Data.to_sequences(base_dataset,
    window_size: opts[:window_size],
    stride: 1
  )
  # Precompute embeddings for faster evaluation
  Data.precompute_embeddings(seq_dataset, show_progress: true)
else
  Data.from_frames(frames,
    embed_config: embed_config,
    temporal: false
  )
end

num_examples = dataset.size
example_type = if opts[:temporal], do: "sequences", else: "frames"
Output.puts("  Dataset size: #{num_examples} #{example_type}")
estimated_batches = div(num_examples, opts[:batch_size])
Output.puts("  Creating ~#{estimated_batches} batches...")

# Batch the dataset with timing
batch_start = System.monotonic_time(:millisecond)
batches = if opts[:temporal] do
  Data.batched_sequences(dataset,
    batch_size: opts[:batch_size],
    shuffle: false,
    drop_last: false
  )
else
  Data.batched(dataset,
    batch_size: opts[:batch_size],
    shuffle: false,
    drop_last: false
  )
end
|> Enum.to_list()
batch_time = System.monotonic_time(:millisecond) - batch_start

num_batches = length(batches)
Output.puts("  ✓ Created #{num_batches} batches in #{Float.round(batch_time / 1000, 1)}s")

# Sample batches for faster evaluation (max 100 batches = 6400 frames)
max_eval_batches = 100
batches = if num_batches > max_eval_batches do
  Output.puts("  Sampling #{max_eval_batches} batches for evaluation (use --batch-size to adjust)")
  Enum.take_random(batches, max_eval_batches)
else
  batches
end
num_batches = length(batches)

# Helper functions for metrics
compute_accuracy = fn logits, targets, _num_classes ->
  predictions = Nx.argmax(logits, axis: -1)
  correct = Nx.equal(predictions, targets)
  Nx.mean(correct) |> Nx.to_number()
end

compute_top_k_accuracy = fn logits, targets, k ->
  {_, top_k_indices} = Nx.top_k(logits, k: k)
  expanded_targets = Nx.reshape(targets, {:auto, 1})
  matches = Nx.equal(top_k_indices, expanded_targets)
  any_match = Nx.any(matches, axes: [-1])
  Nx.mean(any_match) |> Nx.to_number()
end

compute_button_accuracy = fn logits, targets ->
  probs = Nx.sigmoid(logits)
  predictions = Nx.greater(probs, 0.5)
  correct = Nx.equal(predictions, targets)
  Nx.mean(correct) |> Nx.to_number()
end

# Evaluate a single model
evaluate_model = fn model_path ->
  Output.divider()
  Output.puts("Evaluating: #{Path.basename(model_path)}")

  is_policy_file = String.ends_with?(model_path, ".bin")

  # Load model
  {params, config} = if is_policy_file do
    {:ok, binary} = File.read(model_path)
    export = :erlang.binary_to_term(binary)
    {export.params, export.config}
  else
    {:ok, binary} = File.read(model_path)
    checkpoint = :erlang.binary_to_term(binary)
    {checkpoint.policy_params, checkpoint.config}
  end

  # Build policy model based on temporal mode
  model_embed_size = config[:embed_size] || embed_size
  hidden_sizes = config[:hidden_sizes] || [512, 512]

  policy_model = if opts[:temporal] do
    backbone_type = String.to_atom(opts[:backbone])
    Policy.build_temporal(
      embed_size: model_embed_size,
      backbone: backbone_type,
      hidden_size: hd(hidden_sizes),
      num_layers: config[:num_layers] || 2,
      num_heads: config[:num_heads] || 4,
      attention_every: config[:attention_every] || 3,
      axis_buckets: config[:axis_buckets] || 16,
      shoulder_buckets: config[:shoulder_buckets] || 4
    )
  else
    Policy.build(
      embed_size: model_embed_size,
      hidden_sizes: hidden_sizes,
      axis_buckets: config[:axis_buckets] || 16,
      shoulder_buckets: config[:shoulder_buckets] || 4
    )
  end

  {_init_fn, predict_fn} = Axon.build(policy_model)

  # Evaluate
  axis_buckets = config[:axis_buckets] || 16
  shoulder_buckets = config[:shoulder_buckets] || 4
  label_smoothing = config[:label_smoothing] || 0.0

  Output.puts("  Running inference on #{num_batches} batches...")
  Output.puts("  (First batch includes JIT compilation - may take 1-2 minutes)")

  {total_loss, total_batches, component_metrics} = batches
  |> Enum.with_index(1)
  |> Enum.reduce({0.0, 0, %{}}, fn {batch, batch_idx}, {acc_loss, acc_count, acc_metrics} ->
    # Show progress every 100 batches or on first batch
    if batch_idx == 1 or rem(batch_idx, 100) == 0 or batch_idx == num_batches do
      pct = round(batch_idx / num_batches * 100)
      IO.write(:stderr, "\r  Progress: #{batch_idx}/#{num_batches} (#{pct}%)    ")
    end
    %{states: states, actions: actions} = batch

    # Run inference (no backend_copy needed - tensors are already on correct backend)
    {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, states)

    logits = %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder
    }

    loss = Policy.imitation_loss(logits, actions, label_smoothing: label_smoothing)
    loss_val = Nx.to_number(loss)

    batch_metrics = %{
      button_acc: compute_button_accuracy.(buttons, actions.buttons),
      main_x_acc: compute_accuracy.(main_x, actions.main_x, axis_buckets + 1),
      main_y_acc: compute_accuracy.(main_y, actions.main_y, axis_buckets + 1),
      c_x_acc: compute_accuracy.(c_x, actions.c_x, axis_buckets + 1),
      c_y_acc: compute_accuracy.(c_y, actions.c_y, axis_buckets + 1),
      shoulder_acc: compute_accuracy.(shoulder, actions.shoulder, shoulder_buckets + 1),
      main_x_top3: compute_top_k_accuracy.(main_x, actions.main_x, 3),
      main_y_top3: compute_top_k_accuracy.(main_y, actions.main_y, 3)
    }

    new_metrics = if map_size(acc_metrics) == 0 do
      batch_metrics
    else
      Map.merge(acc_metrics, batch_metrics, fn _k, v1, v2 -> v1 + v2 end)
    end

    {acc_loss + loss_val, acc_count + 1, new_metrics}
  end)

  IO.write(:stderr, "\n")  # Clear progress line

  # Compute averages
  avg_loss = total_loss / total_batches
  avg_metrics = Map.new(component_metrics, fn {k, v} -> {k, v / total_batches} end)

  # Display results
  Output.puts("")
  Output.puts("  Loss (cross-entropy): #{Float.round(avg_loss, 4)}")
  Output.puts("")
  Output.puts("  Component Accuracy:")
  Output.puts("    Buttons:      #{Float.round(avg_metrics.button_acc * 100, 1)}%")
  Output.puts("    Main Stick X: #{Float.round(avg_metrics.main_x_acc * 100, 1)}% (top-3: #{Float.round(avg_metrics.main_x_top3 * 100, 1)}%)")
  Output.puts("    Main Stick Y: #{Float.round(avg_metrics.main_y_acc * 100, 1)}% (top-3: #{Float.round(avg_metrics.main_y_top3 * 100, 1)}%)")
  Output.puts("    C-Stick X:    #{Float.round(avg_metrics.c_x_acc * 100, 1)}%")
  Output.puts("    C-Stick Y:    #{Float.round(avg_metrics.c_y_acc * 100, 1)}%")
  Output.puts("    Shoulder:     #{Float.round(avg_metrics.shoulder_acc * 100, 1)}%")

  # Overall weighted accuracy
  overall_acc = (
    avg_metrics.button_acc * 0.3 +
    avg_metrics.main_x_acc * 0.2 +
    avg_metrics.main_y_acc * 0.2 +
    avg_metrics.c_x_acc * 0.1 +
    avg_metrics.c_y_acc * 0.1 +
    avg_metrics.shoulder_acc * 0.1
  )
  Output.puts("")
  Output.puts("  Overall Weighted Accuracy: #{Float.round(overall_acc * 100, 1)}%")

  %{
    path: model_path,
    loss: avg_loss,
    metrics: avg_metrics,
    overall_acc: overall_acc
  }
end

# Evaluate all models
Output.step(4, 4, "Evaluating models")
results = Enum.map(model_paths, evaluate_model)

# Show comparison if multiple models
if length(results) > 1 do
  Output.divider()
  Output.section("Model Comparison")

  sorted = Enum.sort_by(results, & &1.loss)

  Output.puts("Ranked by Loss (lower is better):")
  Output.puts("")

  Enum.with_index(sorted, 1)
  |> Enum.each(fn {result, rank} ->
    name = Path.basename(result.path)
    loss_str = Float.round(result.loss, 4) |> to_string()
    acc_str = Float.round(result.overall_acc * 100, 1) |> to_string()
    Output.puts("  #{rank}. #{name}")
    Output.puts("     Loss: #{loss_str} | Accuracy: #{acc_str}%")
  end)

  best = hd(sorted)
  Output.puts("")
  Output.puts("Best Model: #{Path.basename(best.path)}")
end

# Save to JSON if requested
if opts[:output] do
  json_results = Enum.map(results, fn r ->
    %{
      path: r.path,
      loss: Float.round(r.loss, 6),
      overall_accuracy: Float.round(r.overall_acc, 4),
      component_accuracy: Map.new(r.metrics, fn {k, v} -> {k, Float.round(v, 4)} end)
    }
  end)

  case File.write(opts[:output], Jason.encode!(json_results, pretty: true)) do
    :ok -> Output.puts("\nResults saved to #{opts[:output]}")
    {:error, reason} -> Output.puts("\nFailed to save results: #{inspect(reason)}")
  end
end

Output.puts("""

========================================================================
                       Evaluation Complete
========================================================================

Next steps:
  1. Test in-game: mix run scripts/play_dolphin_async.exs --policy <path>
  2. Continue training: mix run scripts/train_from_replays.exs --resume <checkpoint>
  3. Compare more models: mix run scripts/eval_model.exs --compare <paths...>

""")
