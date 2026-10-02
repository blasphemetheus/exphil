# Windowed Fox Mamba using the normal trainer, with a whole-game holdout.
# Standard train.exs flags; reserve 16 games (or 10% for small smoke runs).
# Streaming's built-in holdout currently exists only for GRU BPTT.
alias ExPhil.Training.{Config, Pipeline, Trainer, Streaming, Data, Output}
alias ExPhil.Training.Callbacks.{GracefulShutdown, ProgressBar, Validation,
  EpochSummary, Checkpoint, PolicyExport, EarlyStopping, RollingCheckpoint}
if Application.get_env(:exphil, :stick_rounding) == :nearest,
  do: Output.warning("EXPHIL_STICK_ROUNDING=nearest: stick targets round to the nearest bucket (experimental, testbed only)")
{:ok, warnings} = Config.validate_args(System.argv())
if warnings != [], do: raise(Enum.join(warnings, "\n"))
opts = Config.parse_args(System.argv()) |> Config.validate!() |> Config.ensure_checkpoint_name()
# Windowed streaming backbones this driver has been run with. :min_gru added
# 2026-10-01 as the cheap testbed for the input-coherence experiments.
unless opts[:backbone] in [:mamba, :min_gru] and opts[:stream_chunk_size] && !opts[:bptt] && !opts[:learn_player_styles],
  do: raise("requires windowed streaming (mamba or min_gru) without style-vocabulary learning")
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
# ---- stop / restart (2026-09-27) -------------------------------------------
# `--resume PATH` accepts a trainer .axon OR a checkpoint DIRECTORY; a
# directory resolves to its newest rolling/primary .axon. When the directory
# also holds progress.json (written by ChunkPipeline each time a chunk is
# handed to the trainer), the chunks before the one in progress are dropped,
# so the run continues where it stopped (the in-progress chunk is redone;
# the rolling checkpoint is <= 500 updates behind the kill, so that is the
# only repeated work). Stop with `systemctl --user stop <unit>` at any time.
# A resumed run finishes the CURRENT epoch; start the next epoch as a fresh
# `--resume <dir-of-finished-epoch>/model.axon` into a new --checkpoint dir.
all_chunks = Streaming.chunk_files(train_files, opts[:stream_chunk_size])
progress_path = Path.join(dir, "progress.json")

resume_path =
  case opts[:resume] do
    nil -> nil
    p ->
      if File.dir?(p) do
        candidates = Path.wildcard(Path.join(p, "model_resume_*.axon")) ++ [Path.join(p, "model.axon")]
        candidates
        |> Enum.filter(&File.regular?/1)
        |> Enum.max_by(&File.stat!(&1).mtime, fn -> raise("no .axon checkpoint in #{p}") end)
      else
        p
      end
  end

skip =
  case {resume_path, ExPhil.Training.ChunkPipeline.read_progress(progress_path)} do
    {nil, _} -> 0
    {_, {:ok, %{chunk: c, of: t}}} when t == length(all_chunks) and c > 1 -> c - 1
    {_, {:ok, %{of: t}}} when t != length(all_chunks) ->
      Output.warning("progress.json is for a #{t}-chunk epoch, this run has #{length(all_chunks)} — ignoring it")
      0
    _ -> 0
  end

if skip > 0, do: Output.puts("Resuming at chunk #{skip + 1}/#{length(all_chunks)} (#{skip} chunks already trained this epoch)")
if resume_path, do: Output.puts("Resuming trainer state from #{resume_path}")

pipeline = %{pipeline | replay_files: train_files,
  file_chunks: Enum.drop(all_chunks, skip),
  progress_path: progress_path,
  chunk_offset: skip,
  val_batches: val_batches,
  estimated_batches: max(1, round(pipeline.estimated_batches * (length(train_files) - skip * opts[:stream_chunk_size]) / length(files)))}
trainer = Trainer.new(pipeline, opts)
Output.puts("Parameters: #{Trainer.param_count(trainer)}; validation batches: #{length(val_batches)}")
trainer = if resume_path do
  {:ok, restored} = Trainer.resume(trainer, resume_path)
  restored
else
  trainer
end
# CALIBRATE_ONLY=1 (with --resume <dir>/model_best.axon and the run's own
# flags): no training — teacher-forced calibration + per-head loss on the
# held-out games (dense stride), written to <dir>/calibration.json
# (override with CALIBRATE_OUT). 2026-10-02.
if System.get_env("CALIBRATE_ONLY") == "1" do
  cal_batches = Data.batched_sequences(val_dataset, batch_size: opts[:batch_size],
    window_size: opts[:window_size], stride: 8, lazy: true, shuffle: false, drop_last: false,
    gpu: false, neutral_weight: 1.0)
  cal = ExPhil.Eval.Calibration.run(trainer, cal_batches)
  out = System.get_env("CALIBRATE_OUT") || Path.join(dir, "calibration.json")
  File.write!(out, Jason.encode!(cal, pretty: true))
  Output.puts("RESULT calibration samples=#{cal.samples} loss_by_head=#{inspect(cal.loss_by_head)}")
  Output.puts("RESULT calibration button ECE " <> Enum.map_join(~w(a b x y z l r), "  ", fn b ->
    v = cal.buttons[b]; "#{b} #{v.ece} (p #{v.model_mean_p} vs rate #{v.expert_rate})" end))
  Output.puts("RESULT calibration sticks " <> Enum.map_join(cal.categorical, "  ", fn {k, v} ->
    "#{k} acc #{v.accuracy} conf #{v.mean_confidence} ece #{v.ece}" end))
  if cal.buttons_given_previous do
    Output.puts("RESULT calibration P(down|prev) model:expert " <> Enum.map_join(~w(a b x y l r), "  ", fn b ->
      c = cal.buttons_given_previous[b]
      f = fn nil -> "-"; m -> "#{m.model_p_down}:#{m.expert_rate_down}" end
      "#{b} up #{f.(c.prev_up)} down #{f.(c.prev_down)}" end))
  end
  Output.success("wrote #{out}")
  System.halt(0)
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
# The epoch is complete: a later --resume of this directory must not skip chunks.
File.rm(progress_path)
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
