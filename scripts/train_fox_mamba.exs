# Windowed Fox Mamba using the normal trainer, with a whole-game holdout.
# Standard train.exs flags; reserve 16 games (or 10% for small smoke runs).
# Streaming's built-in holdout currently exists only for GRU BPTT.
alias ExPhil.Training.{Config, Pipeline, Trainer, Streaming, Data, Output}
alias ExPhil.Training.Callbacks.{GracefulShutdown, ProgressBar, Validation,
  EpochSummary, Checkpoint, PolicyExport, EarlyStopping, RollingCheckpoint}
{:ok, warnings} = Config.validate_args(System.argv())
if warnings != [], do: raise(Enum.join(warnings, "\n"))
opts = Config.parse_args(System.argv()) |> Config.validate!() |> Config.ensure_checkpoint_name()
unless opts[:backbone] == :mamba and opts[:stream_chunk_size] && !opts[:bptt] && !opts[:learn_player_styles],
  do: raise("requires windowed Mamba streaming without style-vocabulary learning")
ExPhil.Training.Inhibitor.hold("Fox Mamba imitation")
Output.banner("Fox Mamba: windowed imitation with game holdout")
Output.config([{"Checkpoint", opts[:checkpoint]}, {"Epochs", opts[:epochs]},
  {"Batch", opts[:batch_size]}, {"Window", opts[:window_size]},
  {"Fused scan available", Edifice.CUDA.FusedScan.custom_call_available?()}])
pipeline = Pipeline.setup!(opts)
files = Enum.sort_by(pipeline.replay_files, &:crypto.hash(:sha256, &1))
n_val = min(16, max(1, div(length(files), 10)))
{val_files, train_files} = Enum.split(files, n_val)
if train_files == [], do: raise("no training games remain")
dir = Path.dirname(opts[:checkpoint])
File.mkdir_p!(dir)
File.write!(Path.join(dir, "split.json"), Jason.encode!(%{train: train_files, validation: val_files}, pretty: true))
Output.puts("#{length(train_files)} training games; #{length(val_files)} disjoint validation games")
{:ok, val_frames, errors} = Streaming.parse_chunk(val_files, pipeline.streaming_chunk_opts)
if errors != [], do: Output.warning("Validation parser report: #{inspect(errors)}")
val_dataset = Streaming.create_dataset(val_frames, pipeline.streaming_dataset_opts)
val_batches = val_dataset |> Data.batched_sequences(batch_size: opts[:batch_size],
  window_size: opts[:window_size], stride: opts[:window_size], lazy: true,
  shuffle: false, drop_last: false, gpu: false, neutral_weight: 1.0) |> Enum.to_list()
if val_batches == [], do: raise("empty validation holdout")
pipeline = %{pipeline | replay_files: train_files,
  file_chunks: Streaming.chunk_files(train_files, opts[:stream_chunk_size]),
  val_batches: val_batches,
  estimated_batches: max(1, round(pipeline.estimated_batches * length(train_files) / length(files)))}
trainer = Trainer.new(pipeline, opts)
Output.puts("Parameters: #{Trainer.param_count(trainer)}; validation batches: #{length(val_batches)}")
trainer = if opts[:resume] do
  {:ok, restored} = Trainer.resume(trainer, opts[:resume])
  restored
else
  trainer
end
callbacks = [
  {RollingCheckpoint, [checkpoint_path: opts[:checkpoint], every: 500]},
  {GracefulShutdown, [checkpoint_path: opts[:checkpoint]]},
  {ProgressBar, [log_interval: opts[:log_interval] || 100]},
  {Validation, []}, {EpochSummary, []},
  {Checkpoint, [checkpoint_path: opts[:checkpoint], save_best: true,
    save_every: 1, save_every_batches: opts[:save_every_batches]]},
  {PolicyExport, [checkpoint_path: opts[:checkpoint]]},
  {EarlyStopping, [patience: opts[:patience] || 3]}
]
{:ok, state} = Trainer.fit(trainer, pipeline, callbacks: callbacks)
if !opts[:no_register] do
  {:ok, entry} = ExPhil.Training.Registry.register(%{
    name: opts[:name] || "fox-mamba-v1",
    checkpoint_path: String.replace_suffix(opts[:checkpoint], ".axon", "_best.axon"),
    policy_path: String.replace_suffix(opts[:checkpoint], ".axon", "_best_policy.bin"),
    training_config: Map.new(opts),
    metrics: %{best_val_loss: state.best_val_loss, epochs: state.epoch,
      steps: state.step, holdout_games: n_val, promotion_status: "not_promoted"},
    tags: ["fox", "mamba", "imitation", "candidate"]})
  Output.success("Registered candidate #{entry.name}: #{entry.id}")
end
File.write!(Path.join(dir, "completed.json"), Jason.encode!(%{
  finished_at: DateTime.to_iso8601(DateTime.utc_now()), epoch: state.epoch,
  steps: state.step, train_loss: state.train_loss, val_loss: state.val_loss,
  best_val_loss: state.best_val_loss}, pretty: true))
Output.success("Training complete: #{dir}")
