#!/usr/bin/env elixir
# Model Evaluation Script: Evaluate trained models on test replays
#
# Usage:
#   mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
#   mix run scripts/eval_model.exs --policy checkpoints/model_policy.bin --replays ~/replays
#   mix run scripts/eval_model.exs --compare model1.axon model2.axon --replays ~/replays
#
# Metrics Computed:
#   - Average loss (cross-entropy)
#   - Per-component accuracy (buttons, main_x, main_y, c_x, c_y, shoulder)
#   - Top-3 accuracy for stick axes
#   - Overall weighted accuracy

# Suppress XLA/CUDA noise
# Note: TF_CPP_MIN_LOG_LEVEL must be set BEFORE running the script:
#   TF_CPP_MIN_LOG_LEVEL=3 mix run scripts/eval_model.exs --quiet ...
# Setting it here is too late as EXLA is already loaded by `mix run`
if "--quiet" in System.argv() or "-q" in System.argv() do
  System.put_env("TF_CPP_MIN_LOG_LEVEL", "3")
  System.put_env("XLA_FLAGS", "--xla_gpu_autotune_level=0")
end

alias ExPhil.CLI
alias ExPhil.Training.Output
alias ExPhil.Data.Peppi
alias ExPhil.Training.{ActionViz, Checkpoint, Data}
alias ExPhil.Networks.Policy
alias ExPhil.Embeddings

# Parse command line arguments using CLI module
opts = CLI.parse_args(System.argv(),
  flags: [:verbosity, :replay, :checkpoint, :evaluation, :common],
  extra: [
    compare: :boolean,
    temporal: :boolean,
    backbone: :string,
    window_size: :integer
  ],
  defaults: [
    max_files: 20,
    batch_size: 64
  ]
)

# Setup verbosity BEFORE any EXLA operations
CLI.setup_verbosity(opts)

# Handle help
if opts[:help] do
  IO.puts("""
  Model Evaluation Script

  Usage:
    mix run scripts/eval_model.exs --checkpoint checkpoints/model.axon
    mix run scripts/eval_model.exs --policy checkpoints/model_policy.bin
    mix run scripts/eval_model.exs --compare model1.axon model2.axon

  Options:
#{CLI.help_text([:verbosity, :replay, :checkpoint, :evaluation])}
    --compare               Compare multiple models (pass paths as positional args)
    --temporal              Enable temporal model evaluation
    --backbone NAME         Backbone architecture (for temporal)
    --window-size N         Window size for temporal models

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

# Get positional args for --compare mode
positional = opts[:_positional] || []

# Convert to map for easier access, with defaults for script-specific options
opts = opts
  |> Keyword.put_new(:temporal, false)
  |> Keyword.put_new(:backbone, "mlp")
  |> Keyword.put_new(:window_size, 60)
  |> Map.new()

# Determine evaluation mode
model_paths =
  cond do
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
  {"Player port", opts[:player_port]},
  {"Character filter", opts[:character] || "none"},
  {"Temporal", opts[:temporal]},
  {"Backbone", if(opts[:temporal], do: opts[:backbone], else: "mlp")},
  {"Window size", if(opts[:temporal], do: opts[:window_size], else: "-")},
  {"Models to evaluate", length(model_paths)}
])

# Step 1: Load test replays
Output.step(1, 4, "Loading test replays")
replay_dir = opts[:replays]

unless File.dir?(replay_dir) do
  Output.error("Replay directory not found: #{replay_dir}")
  System.halt(1)
end

replay_files =
  Path.wildcard(Path.join(replay_dir, "**/*.slp"))
  |> Enum.take(opts[:max_files])

if Enum.empty?(replay_files) do
  Output.error("No .slp files found in #{replay_dir}")
  System.halt(1)
end

Output.puts("  Found #{length(replay_files)} replay files")

# Parse replays
Output.step(2, 4, "Parsing replays")

parse_opts = [
  player_port: opts[:player_port],
  include_speeds: true
]

parse_opts =
  if opts[:character] do
    Keyword.put(parse_opts, :filter_character, opts[:character])
  else
    parse_opts
  end

parsed_replays = Peppi.parse_many(replay_files, parse_opts)

# Convert to training frames format (parse_many returns list of {:ok, replay} tuples)
frames =
  parsed_replays
  |> Enum.flat_map(fn
    {:ok, replay} -> Peppi.to_training_frames(replay, player_port: opts[:player_port])
    {:error, _} -> []
  end)

if Enum.empty?(frames) do
  Output.error("No frames extracted from replays")
  System.halt(1)
end

Output.puts("  Extracted #{length(frames)} frames")

# Step 3: Load model config to determine embedding configuration
Output.step(3, 5, "Loading model configuration")

# Load first model to get its config (used for embedding setup)
first_model_path = hd(model_paths)

# Try to find companion config file
config_paths = [
  String.replace(first_model_path, ~r/\.(axon|bin)$/, "_config.json"),
  String.replace(first_model_path, "_policy.bin", "_config.json"),
  String.replace(first_model_path, "_best_policy.bin", "_config.json"),
  String.replace(first_model_path, ".axon", "_config.json")
] |> Enum.uniq()

config_path = Enum.find(config_paths, &File.exists?/1)

model_config =
  if config_path do
    case File.read(config_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, cfg} ->
            Output.puts("  Loaded config from #{Path.basename(config_path)}")
            cfg
          _ ->
            Output.warning("Could not parse config JSON, using current defaults")
            %{}
        end
      _ ->
        Output.warning("Could not read config file, using current defaults")
        %{}
    end
  else
    Output.warning("No config file found (tried: #{Enum.map(config_paths, &Path.basename/1) |> Enum.join(", ")})")
    Output.puts("  Using current embedding defaults")
    %{}
  end

# Build embedding config from model config
embed_opts = [with_speeds: true]

# Map config keys (could be strings or atoms after JSON round-trip)
get_cfg = fn cfg, key, default ->
  Map.get(cfg, key, Map.get(cfg, to_string(key), default))
end

embed_opts = if action_mode = get_cfg.(model_config, :action_mode, nil) do
  Keyword.put(embed_opts, :action_mode, if(is_binary(action_mode), do: String.to_atom(action_mode), else: action_mode))
else
  embed_opts
end

embed_opts = if char_mode = get_cfg.(model_config, :character_mode, nil) do
  Keyword.put(embed_opts, :character_mode, if(is_binary(char_mode), do: String.to_atom(char_mode), else: char_mode))
else
  embed_opts
end

embed_opts = if stage_mode = get_cfg.(model_config, :stage_mode, nil) do
  Keyword.put(embed_opts, :stage_mode, if(is_binary(stage_mode), do: String.to_atom(stage_mode), else: stage_mode))
else
  embed_opts
end

embed_opts = if nana_mode = get_cfg.(model_config, :nana_mode, nil) do
  Keyword.put(embed_opts, :nana_mode, if(is_binary(nana_mode), do: String.to_atom(nana_mode), else: nana_mode))
else
  embed_opts
end

embed_config = Embeddings.config(embed_opts)
embed_size = Embeddings.embedding_size(embed_config)

Output.puts("  Embedding config: #{inspect(Keyword.take(embed_opts, [:action_mode, :character_mode, :stage_mode, :nana_mode]))}")
Output.puts("  Embedding size: #{embed_size} dims")

# Create dataset
Output.step(4, 5, "Creating evaluation dataset")

dataset =
  if opts[:temporal] do
    # Temporal mode: create sequences
    base_dataset = Data.from_frames(frames, embed_config: embed_config)

    seq_dataset =
      Data.to_sequences(base_dataset,
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

batches =
  if opts[:temporal] do
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

batches =
  if num_batches > max_eval_batches do
    Output.puts(
      "  Sampling #{max_eval_batches} batches for evaluation (use --batch-size to adjust)"
    )

    Enum.take_random(batches, max_eval_batches)
  else
    batches
  end

num_batches = length(batches)

# Button labels (matches @legal_buttons in Controller)
button_labels = [:a, :b, :x, :y, :z, :l, :r, :d_up]

# Helper functions for metrics
compute_accuracy = fn logits, targets, _num_classes ->
  predictions = Nx.argmax(logits, axis: -1)
  # Handle both sparse indices and one-hot targets
  target_indices = if tuple_size(Nx.shape(targets)) > 1 do
    Nx.argmax(targets, axis: -1)
  else
    targets
  end
  correct = Nx.equal(predictions, target_indices)
  Nx.mean(correct) |> Nx.to_number()
end

compute_top_k_accuracy = fn logits, targets, k ->
  {_, top_k_indices} = Nx.top_k(logits, k: k)
  # Handle both sparse indices and one-hot targets
  target_indices = if tuple_size(Nx.shape(targets)) > 1 do
    Nx.argmax(targets, axis: -1)
  else
    targets
  end
  expanded_targets = Nx.reshape(target_indices, {:auto, 1})
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

# Per-button accuracy (returns map of button -> accuracy)
compute_per_button_accuracy = fn logits, targets ->
  probs = Nx.sigmoid(logits)
  predictions = Nx.greater(probs, 0.5)
  correct = Nx.equal(predictions, targets)

  # Compute accuracy for each button column
  button_labels
  |> Enum.with_index()
  |> Enum.map(fn {label, idx} ->
    col_correct = Nx.slice_along_axis(correct, idx, 1, axis: -1)
    acc = Nx.mean(col_correct) |> Nx.to_number()
    {label, acc}
  end)
  |> Map.new()
end

# Button prediction rates (returns {predicted_rates, actual_rates})
compute_button_rates = fn logits, targets ->
  probs = Nx.sigmoid(logits)
  predictions = Nx.greater(probs, 0.5)

  predicted_rates =
    button_labels
    |> Enum.with_index()
    |> Enum.map(fn {label, idx} ->
      col = Nx.slice_along_axis(predictions, idx, 1, axis: -1)
      rate = Nx.mean(col) |> Nx.to_number()
      {label, rate}
    end)
    |> Map.new()

  actual_rates =
    button_labels
    |> Enum.with_index()
    |> Enum.map(fn {label, idx} ->
      col = Nx.slice_along_axis(targets, idx, 1, axis: -1)
      rate = Nx.mean(col) |> Nx.to_number()
      {label, rate}
    end)
    |> Map.new()

  {predicted_rates, actual_rates}
end

# Stick confusion tracking (returns map of {predicted, actual} -> count)
update_stick_confusion = fn confusion, logits, targets, axis_buckets ->
  predictions = Nx.argmax(logits, axis: -1)
  # Handle both sparse indices and one-hot targets
  target_indices = if tuple_size(Nx.shape(targets)) > 1 do
    Nx.argmax(targets, axis: -1)
  else
    targets
  end
  pred_flat = Nx.to_flat_list(predictions)
  target_flat = Nx.to_flat_list(target_indices)

  # Only track non-neutral predictions/targets for readability
  neutral = div(axis_buckets, 2)

  Enum.zip(pred_flat, target_flat)
  |> Enum.reduce(confusion, fn {pred, actual}, acc ->
    # Only track errors (and skip neutral->neutral)
    if pred != actual and (pred != neutral or actual != neutral) do
      key = {pred, actual}
      Map.update(acc, key, 1, &(&1 + 1))
    else
      acc
    end
  end)
end

# Evaluate a single model
evaluate_model = fn model_path ->
  Output.divider()
  Output.puts("Evaluating: #{Path.basename(model_path)}")

  is_policy_file = String.ends_with?(model_path, ".bin")

  # Load model with embed size validation
  {params, config} =
    if is_policy_file do
      case Checkpoint.load_policy(model_path, current_embed_size: embed_size) do
        {:ok, export} ->
          {export.params, export.config}

        {:error, reason} ->
          Output.error("Failed to load policy: #{inspect(reason)}")
          System.halt(1)
      end
    else
      case Checkpoint.load(model_path, current_embed_size: embed_size) do
        {:ok, checkpoint} ->
          {checkpoint.policy_params, checkpoint.config}

        {:error, reason} ->
          Output.error("Failed to load checkpoint: #{inspect(reason)}")
          System.halt(1)
      end
    end

  # Build policy model based on temporal mode
  model_embed_size = config[:embed_size] || embed_size
  hidden_sizes = config[:hidden_sizes] || [512, 512]

  policy_model =
    if opts[:temporal] do
      backbone_type = String.to_atom(opts[:backbone])

      Policy.build_temporal(
        embed_size: model_embed_size,
        backbone: backbone_type,
        hidden_size: hd(hidden_sizes),
        num_layers: config[:num_layers] || 2,
        num_heads: config[:num_heads] || 4,
        head_dim: config[:head_dim] || 64,
        attention_every: config[:attention_every] || 3,
        window_size: config[:window_size] || opts[:window_size],
        state_size: config[:state_size] || 16,
        expand_factor: config[:expand_factor] || 2,
        conv_size: config[:conv_size] || 4,
        dropout: config[:dropout] || 0.1,
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

  # Initialize action visualizer and tracking state
  initial_viz = ActionViz.new()
  initial_state = %{
    total_loss: 0.0,
    loss_components: %{buttons: 0.0, main_x: 0.0, main_y: 0.0, c_x: 0.0, c_y: 0.0, shoulder: 0.0},
    total_batches: 0,
    component_metrics: %{},
    per_button_acc: %{},
    predicted_button_rates: %{},
    actual_button_rates: %{},
    main_x_confusion: %{},
    main_y_confusion: %{},
    total_inference_time_ms: 0.0,
    inference_count: 0,
    viz: initial_viz
  }

  # Helper to compute individual loss components
  compute_loss_components = fn logits, actions ->
    # Binary cross-entropy for buttons
    button_probs = Nx.sigmoid(logits.buttons)
    button_loss = Nx.mean(Nx.negate(
      Nx.add(
        Nx.multiply(actions.buttons, Nx.log(Nx.add(button_probs, 1.0e-7))),
        Nx.multiply(Nx.subtract(1, actions.buttons), Nx.log(Nx.add(Nx.subtract(1, button_probs), 1.0e-7)))
      )
    )) |> Nx.to_number()

    # Softmax cross entropy - handles both sparse indices and one-hot targets
    softmax_ce = fn logits_t, targets_t ->
      # log_softmax = log(softmax(x)) = x - log(sum(exp(x)))
      max_logits = Nx.reduce_max(logits_t, axes: [-1], keep_axes: true)
      shifted = Nx.subtract(logits_t, max_logits)
      log_softmax = Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
      # Convert sparse indices to one-hot if needed
      targets_oh = if tuple_size(Nx.shape(targets_t)) == 1 do
        logits_shape = Nx.shape(logits_t)
        num_classes = elem(logits_shape, tuple_size(logits_shape) - 1)
        batch_size = elem(Nx.shape(targets_t), 0)
        Nx.equal(Nx.iota({batch_size, num_classes}, axis: 1), Nx.reshape(targets_t, {:auto, 1}))
      else
        targets_t
      end
      Nx.mean(Nx.negate(Nx.sum(Nx.multiply(targets_oh, log_softmax), axes: [-1]))) |> Nx.to_number()
    end

    main_x_loss = softmax_ce.(logits.main_x, actions.main_x)
    main_y_loss = softmax_ce.(logits.main_y, actions.main_y)
    c_x_loss = softmax_ce.(logits.c_x, actions.c_x)
    c_y_loss = softmax_ce.(logits.c_y, actions.c_y)
    shoulder_loss = softmax_ce.(logits.shoulder, actions.shoulder)

    %{buttons: button_loss, main_x: main_x_loss, main_y: main_y_loss, c_x: c_x_loss, c_y: c_y_loss, shoulder: shoulder_loss}
  end

  final_state =
    batches
    |> Enum.with_index(1)
    |> Enum.reduce(initial_state, fn {batch, batch_idx}, state ->
      # Show progress every 100 batches or on first batch
      if batch_idx == 1 or rem(batch_idx, 100) == 0 or batch_idx == num_batches do
        pct = round(batch_idx / num_batches * 100)
        IO.write(:stderr, "\r  Progress: #{batch_idx}/#{num_batches} (#{pct}%)\e[K")
      end

      %{states: states, actions: actions} = batch

      # Track inference time (skip first batch due to JIT)
      inference_start = System.monotonic_time(:microsecond)
      {buttons, main_x, main_y, c_x, c_y, shoulder} = predict_fn.(params, states)
      inference_time_us = System.monotonic_time(:microsecond) - inference_start

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

      # Compute loss components
      batch_loss_components = compute_loss_components.(logits, actions)

      # Basic accuracy metrics
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

      # Per-button accuracy
      batch_per_button = compute_per_button_accuracy.(buttons, actions.buttons)

      # Button prediction rates
      {batch_pred_rates, batch_actual_rates} = compute_button_rates.(buttons, actions.buttons)

      # Stick confusion tracking
      main_x_confusion = update_stick_confusion.(state.main_x_confusion, main_x, actions.main_x, axis_buckets)
      main_y_confusion = update_stick_confusion.(state.main_y_confusion, main_y, actions.main_y, axis_buckets)

      # Merge metrics
      new_metrics =
        if map_size(state.component_metrics) == 0 do
          batch_metrics
        else
          Map.merge(state.component_metrics, batch_metrics, fn _k, v1, v2 -> v1 + v2 end)
        end

      new_per_button =
        if map_size(state.per_button_acc) == 0 do
          batch_per_button
        else
          Map.merge(state.per_button_acc, batch_per_button, fn _k, v1, v2 -> v1 + v2 end)
        end

      new_pred_rates =
        if map_size(state.predicted_button_rates) == 0 do
          batch_pred_rates
        else
          Map.merge(state.predicted_button_rates, batch_pred_rates, fn _k, v1, v2 -> v1 + v2 end)
        end

      new_actual_rates =
        if map_size(state.actual_button_rates) == 0 do
          batch_actual_rates
        else
          Map.merge(state.actual_button_rates, batch_actual_rates, fn _k, v1, v2 -> v1 + v2 end)
        end

      new_loss_components =
        Map.merge(state.loss_components, batch_loss_components, fn _k, v1, v2 -> v1 + v2 end)

      # Track predicted actions for visualization
      viz =
        ActionViz.record_batch(
          state.viz,
          %{
            buttons: Nx.greater(Nx.sigmoid(buttons), 0.5),
            main_x: Nx.argmax(main_x, axis: -1),
            main_y: Nx.argmax(main_y, axis: -1),
            c_x: Nx.argmax(c_x, axis: -1),
            c_y: Nx.argmax(c_y, axis: -1),
            shoulder: Nx.argmax(shoulder, axis: -1)
          },
          axis_buckets
        )

      # Only count inference time after first batch (skip JIT compilation)
      {new_inference_time, new_inference_count} =
        if batch_idx > 1 do
          {state.total_inference_time_ms + inference_time_us / 1000, state.inference_count + 1}
        else
          {state.total_inference_time_ms, state.inference_count}
        end

      %{state |
        total_loss: state.total_loss + loss_val,
        loss_components: new_loss_components,
        total_batches: state.total_batches + 1,
        component_metrics: new_metrics,
        per_button_acc: new_per_button,
        predicted_button_rates: new_pred_rates,
        actual_button_rates: new_actual_rates,
        main_x_confusion: main_x_confusion,
        main_y_confusion: main_y_confusion,
        total_inference_time_ms: new_inference_time,
        inference_count: new_inference_count,
        viz: viz
      }
    end)

  # Clear progress line
  IO.write(:stderr, "\n")

  # Extract from final state
  total_loss = final_state.total_loss
  total_batches = final_state.total_batches
  component_metrics = final_state.component_metrics
  action_viz = final_state.viz

  # Compute averages
  avg_loss = total_loss / total_batches
  avg_metrics = Map.new(component_metrics, fn {k, v} -> {k, v / total_batches} end)
  avg_per_button = Map.new(final_state.per_button_acc, fn {k, v} -> {k, v / total_batches} end)
  avg_pred_rates = Map.new(final_state.predicted_button_rates, fn {k, v} -> {k, v / total_batches} end)
  avg_actual_rates = Map.new(final_state.actual_button_rates, fn {k, v} -> {k, v / total_batches} end)
  avg_loss_components = Map.new(final_state.loss_components, fn {k, v} -> {k, v / total_batches} end)

  # Inference timing
  avg_inference_ms =
    if final_state.inference_count > 0 do
      final_state.total_inference_time_ms / final_state.inference_count
    else
      0.0
    end
  frames_per_batch = opts[:batch_size]
  ms_per_frame = if frames_per_batch > 0, do: avg_inference_ms / frames_per_batch, else: 0.0

  # Display results
  Output.puts("")
  Output.puts("  Loss (cross-entropy): #{Float.round(avg_loss, 4)}")

  # Loss component breakdown
  Output.puts("")
  Output.puts("  Loss Components:")
  Output.puts("    Buttons:      #{Float.round(avg_loss_components.buttons, 4)}")
  Output.puts("    Main Stick X: #{Float.round(avg_loss_components.main_x, 4)}")
  Output.puts("    Main Stick Y: #{Float.round(avg_loss_components.main_y, 4)}")
  Output.puts("    C-Stick X:    #{Float.round(avg_loss_components.c_x, 4)}")
  Output.puts("    C-Stick Y:    #{Float.round(avg_loss_components.c_y, 4)}")
  Output.puts("    Shoulder:     #{Float.round(avg_loss_components.shoulder, 4)}")

  Output.puts("")
  Output.puts("  Component Accuracy:")
  Output.puts("    Buttons:      #{Float.round(avg_metrics.button_acc * 100, 1)}%")

  Output.puts(
    "    Main Stick X: #{Float.round(avg_metrics.main_x_acc * 100, 1)}% (top-3: #{Float.round(avg_metrics.main_x_top3 * 100, 1)}%)"
  )

  Output.puts(
    "    Main Stick Y: #{Float.round(avg_metrics.main_y_acc * 100, 1)}% (top-3: #{Float.round(avg_metrics.main_y_top3 * 100, 1)}%)"
  )

  Output.puts("    C-Stick X:    #{Float.round(avg_metrics.c_x_acc * 100, 1)}%")
  Output.puts("    C-Stick Y:    #{Float.round(avg_metrics.c_y_acc * 100, 1)}%")
  Output.puts("    Shoulder:     #{Float.round(avg_metrics.shoulder_acc * 100, 1)}%")

  # Per-button accuracy
  Output.puts("")
  Output.puts("  Per-Button Accuracy:")
  for {btn, acc} <- Enum.sort_by(avg_per_button, fn {k, _} -> Enum.find_index(button_labels, &(&1 == k)) end) do
    btn_str = btn |> to_string() |> String.upcase() |> String.pad_trailing(5)
    Output.puts("    #{btn_str} #{Float.round(acc * 100, 1)}%")
  end

  # Button prediction rates comparison
  Output.puts("")
  Output.puts("  Button Press Rates (Predicted vs Actual):")
  for {btn, _} <- Enum.sort_by(avg_per_button, fn {k, _} -> Enum.find_index(button_labels, &(&1 == k)) end) do
    pred = avg_pred_rates[btn] || 0.0
    actual = avg_actual_rates[btn] || 0.0
    diff = pred - actual
    diff_str = if diff >= 0, do: "+#{Float.round(diff * 100, 1)}", else: "#{Float.round(diff * 100, 1)}"
    btn_str = btn |> to_string() |> String.upcase() |> String.pad_trailing(5)
    Output.puts("    #{btn_str} pred=#{Float.round(pred * 100, 1)}%  actual=#{Float.round(actual * 100, 1)}%  (#{diff_str}%)")
  end

  # Inference timing
  Output.puts("")
  Output.puts("  Inference Timing:")
  Output.puts("    Avg batch time: #{Float.round(avg_inference_ms, 2)} ms")
  Output.puts("    Per-frame time: #{Float.round(ms_per_frame, 3)} ms")
  realtime_ready = if ms_per_frame < 16.67, do: "✓ Yes", else: "✗ No (need <16.67ms)"
  Output.puts("    60 FPS capable: #{realtime_ready}")

  # Overall weighted accuracy
  overall_acc =
    avg_metrics.button_acc * 0.3 +
      avg_metrics.main_x_acc * 0.2 +
      avg_metrics.main_y_acc * 0.2 +
      avg_metrics.c_x_acc * 0.1 +
      avg_metrics.c_y_acc * 0.1 +
      avg_metrics.shoulder_acc * 0.1

  Output.puts("")
  Output.puts("  Overall Weighted Accuracy: #{Float.round(overall_acc * 100, 1)}%")

  # Stick confusion analysis (show top errors)
  Output.puts("")
  Output.puts("  Stick Confusion Analysis (top prediction errors):")

  # Helper to format bucket as direction
  format_bucket = fn bucket, num_buckets ->
    neutral = div(num_buckets, 2)
    cond do
      bucket == neutral -> "neutral"
      bucket < neutral - 2 -> "far_neg"
      bucket < neutral -> "neg"
      bucket > neutral + 2 -> "far_pos"
      bucket > neutral -> "pos"
      true -> "~neutral"
    end
  end

  # Main X confusion
  if map_size(final_state.main_x_confusion) > 0 do
    top_x_errors =
      final_state.main_x_confusion
      |> Enum.sort_by(fn {_, count} -> -count end)
      |> Enum.take(5)

    if length(top_x_errors) > 0 do
      Output.puts("    Main X errors:")
      for {{pred, actual}, count} <- top_x_errors do
        pred_dir = format_bucket.(pred, axis_buckets)
        actual_dir = format_bucket.(actual, axis_buckets)
        Output.puts("      #{actual_dir} → #{pred_dir}: #{count} frames")
      end
    end
  end

  # Main Y confusion
  if map_size(final_state.main_y_confusion) > 0 do
    top_y_errors =
      final_state.main_y_confusion
      |> Enum.sort_by(fn {_, count} -> -count end)
      |> Enum.take(5)

    if length(top_y_errors) > 0 do
      Output.puts("    Main Y errors:")
      for {{pred, actual}, count} <- top_y_errors do
        pred_dir = format_bucket.(pred, axis_buckets)
        actual_dir = format_bucket.(actual, axis_buckets)
        Output.puts("      #{actual_dir} → #{pred_dir}: #{count} frames")
      end
    end
  end

  # Show action distribution visualization
  Output.puts("")
  ActionViz.print_summary(action_viz)

  %{
    path: model_path,
    loss: avg_loss,
    loss_components: avg_loss_components,
    metrics: avg_metrics,
    per_button_acc: avg_per_button,
    button_rates: %{predicted: avg_pred_rates, actual: avg_actual_rates},
    inference_ms_per_frame: ms_per_frame,
    overall_acc: overall_acc
  }
end

# Evaluate all models
Output.step(5, 5, "Evaluating models")
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
  json_results =
    Enum.map(results, fn r ->
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
