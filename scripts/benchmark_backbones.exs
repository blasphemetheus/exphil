#!/usr/bin/env elixir
# Benchmark multiple backbone architectures on the same data.
#
# Usage:
#   mix run scripts/benchmark_backbones.exs --replays ./replays/huggingface --max-files 50
#   mix run scripts/benchmark_backbones.exs --backbones mamba,griffin,gated_delta_net --epochs 10
#
# Produces a comparison table at the end with val_loss, button accuracy,
# stick accuracy, rare action recall, and inference speed per architecture.

alias ExPhil.Training.{Config, Pipeline, Trainer, Output}
alias ExPhil.Training.Callbacks.{ProgressBar, Validation, Diagnostics, EpochSummary}

# Parse args
args = System.argv()

backbones_str = case Enum.find_index(args, &(&1 == "--backbones")) do
  nil -> "mamba,griffin,gated_delta_net,min_gru,rwkv"
  i -> Enum.at(args, i + 1, "mamba")
end

backbones = backbones_str |> String.split(",") |> Enum.map(&String.to_atom/1)

epochs = case Enum.find_index(args, &(&1 == "--epochs")) do
  nil -> 10
  i -> String.to_integer(Enum.at(args, i + 1, "10"))
end

max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> 50
  i -> String.to_integer(Enum.at(args, i + 1, "50"))
end

replays_dir = case Enum.find_index(args, &(&1 == "--replays")) do
  nil -> "./replays/huggingface"
  i -> Enum.at(args, i + 1, "./replays/huggingface")
end

batch_size = case Enum.find_index(args, &(&1 == "--batch-size")) do
  nil -> 16
  i -> String.to_integer(Enum.at(args, i + 1, "16"))
end

Output.banner("ExPhil Backbone Benchmark")
Output.puts("  Backbones: #{Enum.join(backbones, ", ")}")
Output.puts("  Epochs: #{epochs}")
Output.puts("  Max files: #{max_files}")
Output.puts("  Batch size: #{batch_size}")
Output.puts("")

# Set up pipeline ONCE — shared across all architectures
Output.puts("Setting up shared data pipeline...")
base_opts = Config.parse_args([
  "--replays", replays_dir,
  "--max-files", to_string(max_files),
  "--batch-size", to_string(batch_size),
  "--epochs", to_string(epochs),
  "--backbone", "mamba",  # Needed for initial parse, overridden per-run
  "--temporal"
]) |> Config.validate!()

pipeline = Pipeline.setup!(base_opts)
Output.puts("  Pipeline ready: #{pipeline.train_dataset.size} train frames\n")

# Run each backbone
results =
  Enum.map(backbones, fn backbone ->
    Output.puts("\n" <> String.duplicate("=", 60))
    Output.puts("  Backbone: #{backbone}")
    Output.puts(String.duplicate("=", 60))

    # Get backbone-specific defaults
    backbone_opts = Config.backbone_defaults(backbone)
    run_opts = Keyword.merge(base_opts, backbone_opts)
      |> Keyword.put(:backbone, backbone)
      |> Keyword.put(:epochs, epochs)

    # Override the pipeline's resolved_opts for this backbone
    run_pipeline = %{pipeline | resolved_opts: run_opts}

    try do
      trainer = Trainer.new(run_pipeline, run_opts)
      param_count = Nx.Defn.Composite.count(trainer.policy_params)

      callbacks = [
        {ProgressBar, [log_interval: 50]},
        {Validation, []},
        {EpochSummary, []},
        {Diagnostics, []}
      ]

      train_start = System.monotonic_time(:millisecond)
      {:ok, state} = Trainer.fit(trainer, run_pipeline, callbacks: callbacks)
      train_time = System.monotonic_time(:millisecond) - train_start

      # Inference speed test (single batch, 100 iterations)
      {batch_stream, _} = Pipeline.batch_stream(run_pipeline, [])
      test_batch = Enum.take(batch_stream, 1) |> hd()

      # Warmup
      state.trainer.predict_fn.(state.trainer.policy_params, test_batch.states)

      # Time 100 iterations
      {inference_us, _} = :timer.tc(fn ->
        for _ <- 1..100 do
          state.trainer.predict_fn.(state.trainer.policy_params, test_batch.states)
        end
      end)
      inference_ms = inference_us / 100_000

      %{
        backbone: backbone,
        val_loss: state.val_loss,
        train_loss: state.train_loss,
        params: param_count,
        train_time_s: div(train_time, 1000),
        inference_ms: Float.round(inference_ms, 2),
        fps_ready: inference_ms < 16.67
      }
    rescue
      e ->
        Output.error("#{backbone} failed: #{Exception.message(e)}")
        %{backbone: backbone, val_loss: nil, train_loss: nil, params: nil,
          train_time_s: nil, inference_ms: nil, fps_ready: false}
    end
  end)

# Summary table
Output.puts("\n" <> String.duplicate("=", 70))
Output.puts("  BENCHMARK RESULTS")
Output.puts(String.duplicate("=", 70))

headers = ["Backbone", "Val Loss", "Train Loss", "Params", "Train(s)", "Infer(ms)", "60fps?"]
rows = Enum.map(results, fn r ->
  [
    to_string(r.backbone),
    if(r.val_loss, do: to_string(Float.round(r.val_loss * 1.0, 4)), else: "FAIL"),
    if(r.train_loss, do: to_string(Float.round(r.train_loss * 1.0, 4)), else: "FAIL"),
    if(r.params, do: "#{r.params}", else: "-"),
    if(r.train_time_s, do: to_string(r.train_time_s), else: "-"),
    if(r.inference_ms, do: to_string(r.inference_ms), else: "-"),
    if(r.fps_ready, do: "YES", else: "no")
  ]
end)

Output.puts_raw(Output.table(headers, rows))

# Best architecture
best = Enum.filter(results, & &1.val_loss) |> Enum.min_by(& &1.val_loss, fn -> nil end)
if best do
  Output.puts("\n  Best: #{best.backbone} (val_loss=#{Float.round(best.val_loss * 1.0, 4)}, #{best.inference_ms}ms/batch)")
end
