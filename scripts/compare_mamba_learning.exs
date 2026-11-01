# AI-generated. See eval_runs/mamba_learning_20261101/README.md for method and poem.
# Run in ExPhil's devenv with the existing compiled application.
Application.ensure_all_started(:crypto)
Application.ensure_all_started(:exphil)
Code.compile_file("lib/exphil/training/pipeline.ex")

Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.75]
)

Application.put_env(:exla, :default_client, :cuda)
Nx.global_default_backend({EXLA.Backend, client: :cuda})
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda, precision: :highest)
Logger.configure(level: :warning)

defmodule LearningBudget do
  use ExPhil.Training.Callback
  alias ExPhil.Training.{Imitation, Output}

  def init(opts), do: Map.new(opts) |> Map.merge(%{last: nil, elapsed: 0, seen: []})

  def on_train_begin(state, cb) do
    snapshot(state, cb, "initial")
    {:cont, state, %{cb | last: now()}}
  end

  def on_batch_end(state, cb) do
    elapsed = cb.elapsed + now() - cb.last
    cb = %{cb | elapsed: elapsed}
    cb = if state.step == 1, do: Map.put(cb, :first_step_ms, elapsed), else: cb
    labels = []
    labels = if state.step >= cb.steps, do: ["equal_steps" | labels], else: labels
    labels = if elapsed >= cb.seconds * 1000, do: ["equal_time" | labels], else: labels
    labels = labels -- cb.seen

    if rem(state.step, 50) == 0 do
      Output.puts(
        "step #{state.step}, loss #{state.batch_metrics.loss}, training #{Float.round(elapsed / 1000, 1)}s"
      )
    end

    Enum.each(labels, &snapshot(state, cb, &1))
    cb = %{cb | seen: cb.seen ++ labels, last: now()}
    if length(cb.seen) == 2, do: {:halt, state, cb}, else: {:cont, state, cb}
  end

  defp snapshot(state, cb, label) do
    Output.puts("Evaluating #{label} at step #{state.step} (first evaluation compiles)")
    metrics = Imitation.evaluate(state.trainer, state.pipeline.val_batches, max_concurrency: 1)
    true = is_number(metrics.loss)

    result = %{
      label: label,
      steps: state.step,
      training_seconds: cb.elapsed / 1000,
      first_step_seconds: Map.get(cb, :first_step_ms, 0) / 1000,
      gpu_process_memory:
        elem(
          System.cmd("nvidia-smi", [
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader"
          ]),
          0
        ),
      validation_loss: metrics.loss,
      train_loss: state.batch_metrics[:loss],
      parameters: ExPhil.Training.Trainer.param_count(state.trainer)
    }

    File.write!(Path.join(cb.out, "#{label}.json"), Jason.encode!(result, pretty: true))

    if label != "initial" do
      path = Path.join(cb.out, "#{label}.axon")
      ExPhil.Training.Callbacks.PolicyExport.on_train_end(state, %{checkpoint_path: path})
    end

    Output.success("#{label}: #{inspect(result)}")
  end

  defp now, do: System.monotonic_time(:millisecond)
end

alias ExPhil.Training.{Config, Pipeline, Trainer, Streaming, Data, Output, ChunkPipeline}

{args, [], []} =
  OptionParser.parse(System.argv(),
    strict: [
      backbone: :string,
      state_size: :integer,
      seed: :integer,
      out: :string,
      replays: :string,
      steps: :integer,
      seconds: :integer,
      files: :integer
    ]
  )

out = Keyword.fetch!(args, :out)
File.mkdir_p!(out)
if File.exists?(Path.join(out, "completed.json")), do: raise("already completed: #{out}")

argv = [
  "--backbone",
  Keyword.fetch!(args, :backbone),
  "--state-size",
  to_string(args[:state_size]),
  "--seed",
  to_string(args[:seed] || 905),
  "--replays",
  args[:replays] || "replays/erickfm_master",
  "--max-files",
  to_string(args[:files] || 32),
  "--checkpoint",
  Path.join(out, "model.axon")
]

argv =
  argv ++
    String.split("""
    --stage-internals --hidden-sizes 512,512,256 --batch-size 128 --precision f32
    --window-size 80 --stride 5 --dropout 0.0 --num-layers 2
    --train-character fox --select-character-port --stream-chunk-size 32
    --no-cache-streaming --label-delay 0 --no-cache --no-register
    --head autoregressive --label-smoothing 0.0 --no-focal-loss
    --button-pos-weight 1,1,1,1,1,1,1,1 --action-oversample 1.0 --entropy-weight 0.0
    --neutral-weight 1.0 --stick-edge-weight 1.0 --prev-action --prev-action-dropout 0.0
    --prev-action-quantize --button-events --stick-events --event-context
    --chunk-horizon 8 --offstage-weight 3 --stick-duration 8 --onset-weight 5
    --learning-rate 0.0002 --lr-schedule constant --epochs 20
    """)

{:ok, []} = Config.validate_args(argv)
opts = Config.parse_args(argv) |> Config.validate!()
Output.banner("Controlled Mamba learning pilot")
pipeline = Pipeline.setup!(opts)
files = Enum.sort_by(pipeline.replay_files, &:crypto.hash(:sha256, &1))
{val_files, train_files} = Enum.split(files, min(16, max(1, div(length(files), 10))))
true = train_files != []

File.write!(
  Path.join(out, "split.json"),
  Jason.encode!(%{train: train_files, validation: val_files}, pretty: true)
)

File.write!(Path.join(out, "arguments.json"), Jason.encode!(argv, pretty: true))
{:ok, frames, []} = Streaming.parse_chunk(val_files, pipeline.streaming_chunk_opts)
dataset = Streaming.create_dataset(frames, pipeline.streaming_dataset_opts)

val_batches =
  Data.batched_sequences(dataset,
    batch_size: opts[:batch_size],
    window_size: opts[:window_size],
    stride: opts[:window_size],
    lazy: true,
    shuffle: false,
    drop_last: false,
    gpu: false,
    neutral_weight: 1.0,
    chunk_horizon: opts[:chunk_horizon]
  )
  |> Enum.to_list()

true = val_batches != []
chunks = Streaming.chunk_files(train_files, opts[:stream_chunk_size])

prepared =
  ChunkPipeline.materialize_chunks(chunks,
    chunk_opts: pipeline.streaming_chunk_opts,
    dataset_opts: pipeline.streaming_dataset_opts,
    embed_config: pipeline.embed_config,
    cache_embeddings: false,
    show_progress: true
  )

pipeline = %{
  pipeline
  | replay_files: train_files,
    file_chunks: chunks,
    prepared_chunks: prepared,
    val_batches: val_batches,
    progress_path: nil,
    chunk_offset: 0
}

fingerprint = fn p ->
  {stream, _} = Pipeline.batch_stream(p, [])

  stream
  |> Enum.take(2)
  |> Enum.map(fn batch ->
    batch.states
    |> Nx.to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end)
end

first = fingerprint.(pipeline)
true = first != [] and first == fingerprint.(pipeline)
other = %{pipeline | resolved_opts: Keyword.put(pipeline.resolved_opts, :seed, opts[:seed] + 1)}
true = first != fingerprint.(other)
File.write!(Path.join(out, "batch_fingerprints.json"), Jason.encode!(first))
trainer = Trainer.new(pipeline, opts)

{:ok, state} =
  Trainer.fit(trainer, pipeline,
    callbacks: [
      {LearningBudget, [out: out, steps: args[:steps] || 300, seconds: args[:seconds] || 120]}
    ]
  )

true =
  File.exists?(Path.join(out, "equal_steps.json")) and
    File.exists?(Path.join(out, "equal_time.json"))

File.write!(
  Path.join(out, "completed.json"),
  Jason.encode!(%{steps: state.step, finished_at: DateTime.utc_now()}, pretty: true)
)
