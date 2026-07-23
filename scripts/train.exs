#!/usr/bin/env elixir
# ExPhil Training Script
#
# Usage:
#   mix run scripts/train.exs --backbone mamba --replays ./replays/huggingface
#   mix run scripts/train.exs --backbone mamba --max-files 200 --epochs 30
#   mix run scripts/train.exs --preset quick
#   mix run scripts/train.exs --resume checkpoints/model.axon --epochs 10
#
# Backbone defaults are auto-applied (temporal, precision, lr_schedule, etc.)
# Run precompute_embeddings.exs first for large datasets to avoid GPU memory fragmentation.

# Help
if "--help" in System.argv() or "-h" in System.argv() do
  IO.puts("""
  ExPhil Training Script

  Usage: mix run scripts/train.exs [options]

  Model:
    --backbone NAME     Architecture (mamba, griffin, mlp, lstm, ...) [mamba]
    --hidden-sizes S    Hidden layer sizes (comma-separated) [512,512,256]
    --num-layers N      Number of backbone layers [2]
    --precision TYPE    bf16 or f32 [auto per backbone]
    --dropout RATE      Dropout rate [auto per backbone]

  Training:
    --epochs N          Number of epochs [10]
    --batch-size N      Batch size [64]
    --accumulation-steps N  Gradient accumulation [1]
    --learning-rate LR  Learning rate [1e-4]
    --lr-schedule TYPE  constant, cosine, cosine_restarts [auto per backbone]
    --seed N            Random seed for reproducibility

  Data:
    --replays PATH      Path to replay directory [./replays]
    --max-files N       Limit replay files
    --train-character C Filter replays by character
    --dual-port         Train on both players (2x data)
    --augment           Enable mirror + noise augmentation
    --online-robust     Frame delay augmentation for online play
    --window-size N     Temporal window frames [60]
    --stride N          Sequence stride [5]

  Output:
    --name NAME         Model name for checkpoint
    --verbose           Show debug output + gradient norms
    --quiet             Suppress most output
    --log-file PATH     Write output to file
    --profile           Show timing breakdown

  Checkpointing:
    --save-best         Save best model by val_loss [on]
    --save-every N      Save every N epochs
    --save-every-batches N  Save every N batches
    --resume PATH       Resume from checkpoint

  Other:
    --neutral-weight W  Weight for neutral/idle frames (0.0-1.0) [0.25]
    --head-normalize    Normalize per-head loss contributions
    --early-stopping    Stop when val_loss plateaus
    --patience N        Early stopping patience [5]
    --ema               Enable model weight EMA
    --dry-run           Show setup info and exit
    --no-register       Skip model registry
    --preset NAME       Use preset (quick, standard, production, mewtwo, ...)
  """)
  System.halt(0)
end

# Suppress XLA logs in quiet mode (must be before EXLA loads)
if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

alias ExPhil.Training.{Config, Pipeline, Trainer, Output}
alias ExPhil.Training.Callbacks.{
  ProgressBar, Validation, Diagnostics, Checkpoint,
  EarlyStopping, EMA, GracefulShutdown, EpochSummary,
  LossPlot, Registry, PolicyExport, Profiler, TestEval
}

# Parse config (applies backbone defaults, preset overrides, CLI args)
opts = Config.parse_args(System.argv()) |> Config.validate!() |> Config.ensure_checkpoint_name()

# Verbosity
verbosity = cond do
  opts[:quiet] -> 0
  opts[:verbose] -> 2
  true -> 1
end
Output.set_verbosity(verbosity)

# Log file
if opts[:log_file], do: Output.set_log_file(opts[:log_file])

# Display config
Output.banner("ExPhil Training")
Output.config([
  {"Backbone", opts[:backbone]},
  {"Temporal", opts[:temporal]},
  {"Precision", opts[:precision]},
  {"LR", "#{opts[:learning_rate]} (#{opts[:lr_schedule]})"},
  {"Epochs", opts[:epochs]},
  {"Batch size", "#{opts[:batch_size]}#{if (opts[:accumulation_steps] || 1) > 1, do: " x#{opts[:accumulation_steps]} accum", else: ""}"},
  {"Stick edge wt", opts[:stick_edge_weight]},
  {"Focal loss", "#{opts[:focal_loss]} (gamma=#{opts[:focal_gamma]})"},
  {"Entropy weight", opts[:entropy_weight] || 0.0},
  {"Neutral weight", Keyword.get(opts, :neutral_weight, 0.25)},
  {"Head normalize", opts[:head_normalize] || false},
  {"Replays", opts[:replays]},
  {"Max files", opts[:max_files] || "all"},
  {"Checkpoint", opts[:checkpoint]}
])

if opts[:train_character], do: Output.puts("  Character: #{opts[:train_character]}")
if opts[:dual_port], do: Output.puts("  Dual-port: training on both players")
if opts[:augment], do: Output.puts("  Augmentation: mirror + noise")
if opts[:frame_delay_augment], do: Output.puts("  Frame delay augment: #{opts[:frame_delay_min]}-#{opts[:frame_delay_max]} frames")
if opts[:seed], do: Output.puts("  Seed: #{opts[:seed]}")

# Set up data pipeline
pipeline = Pipeline.setup!(opts)

# Build trainer
trainer = Trainer.new(pipeline, opts)
Output.puts("  Parameters: #{Trainer.param_count(trainer) |> div(1000)}K")

# Resume from checkpoint if specified
trainer =
  if opts[:resume] do
    case Trainer.resume(trainer, opts[:resume]) do
      {:ok, resumed} ->
        Output.puts("  Resumed from #{opts[:resume]} (step #{resumed.step})")
        resumed
      {:error, reason} ->
        Output.error("Resume failed: #{inspect(reason)}")
        System.halt(1)
    end
  else
    trainer
  end

# Show model architecture in verbose or dry-run mode
if opts[:verbose] == true or opts[:dry_run] == true do
  Output.puts("\n  Model architecture:")
  Trainer.display_model(trainer,
    temporal: opts[:temporal],
    embed_size: opts[:embed_size] || 288,
    window_size: opts[:window_size] || 60
  )
end

# Dry run — show setup info and exit
if opts[:dry_run] do
  Output.puts("\n  Dry run complete. Pipeline and model are ready.")
  Output.puts("  Estimated batches/epoch: #{pipeline.estimated_batches}")
  Output.puts("  Val batches: #{Pipeline.val_batch_count(pipeline)}")
  System.halt(0)
end

# Build callback list (order matters)
callbacks = [
  {GracefulShutdown, [checkpoint_path: opts[:checkpoint]]},
  {ProgressBar, [log_interval: opts[:log_interval] || 10]},
  {Validation, []},
  {EpochSummary, []},
  {Diagnostics, [verbose: opts[:verbose] || false]},
  {Checkpoint, [
    save_best: opts[:save_best] != false,
    save_every: opts[:save_every] || opts[:checkpoint_every],
    save_every_batches: opts[:save_every_batches],
    checkpoint_path: opts[:checkpoint],
    overwrite: opts[:overwrite] || false,
    backup: opts[:backup] || false
  ]},
  {LossPlot, [checkpoint_path: opts[:checkpoint]]},
  {PolicyExport, [checkpoint_path: opts[:checkpoint]]},
  {TestEval, []}
]

# Conditional callbacks
callbacks = if opts[:early_stopping],
  do: callbacks ++ [{EarlyStopping, [patience: opts[:patience] || 5]}],
  else: callbacks

callbacks = if opts[:ema],
  do: callbacks ++ [{EMA, [decay: opts[:ema_decay] || 0.999]}],
  else: callbacks

callbacks = if not (opts[:no_register] || false),
  do: callbacks ++ [{Registry, []}],
  else: callbacks

callbacks = if opts[:profile],
  do: callbacks ++ [{Profiler, []}],
  else: callbacks

# Train
{:ok, state} = Trainer.fit(trainer, pipeline, callbacks: callbacks)

# Persist the style-tag vocabulary next to the checkpoint (flywheel P3a):
# a policy trained with --learn-player-styles is conditioned on name ids,
# and inference (--style-tag) needs the same tag->id mapping to select one.
if pipeline.player_registry && opts[:checkpoint] do
  players_path = "#{opts[:checkpoint]}.players.json"

  case ExPhil.Training.PlayerRegistry.to_json(pipeline.player_registry, players_path) do
    :ok -> Output.puts("  Player registry: #{players_path}")
    {:error, reason} -> Output.warning("Player registry save failed: #{inspect(reason)}")
  end
end

# Summary
Output.puts("\nTraining complete!")
Output.puts("  Epochs: #{state.epoch}/#{state.epochs}")
Output.puts("  Final train_loss: #{Float.round((state.train_loss || 0.0) * 1.0, 4)}")
if state.val_loss, do: Output.puts("  Final val_loss: #{Float.round(state.val_loss * 1.0, 4)}")
if state.best_val_loss, do: Output.puts("  Best val_loss: #{Float.round(state.best_val_loss * 1.0, 4)}")
if opts[:checkpoint], do: Output.puts("  Checkpoint: #{opts[:checkpoint]}")
if state.meta[:seed], do: Output.puts("  Seed: #{state.meta[:seed]}")
