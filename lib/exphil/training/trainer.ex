defmodule ExPhil.Training.Trainer do
  @moduledoc """
  High-level training orchestrator.

  Connects `Pipeline` (data) + `Imitation` (model) + `Callback` (features)
  into a clean training API.

  ## Usage

      opts = Config.from_argv(System.argv())
      pipeline = Pipeline.setup!(opts)
      trainer = Trainer.new(pipeline, opts)

      {:ok, state} = Trainer.fit(trainer, pipeline,
        callbacks: [
          {Callbacks.ProgressBar, []},
          {Callbacks.Validation, []},
          {Callbacks.Checkpoint, [save_best: true]}
        ]
      )

  ## How it Works

  `fit/3` runs the multi-epoch training loop:

  1. Initialize all callbacks (`on_train_begin`)
  2. For each epoch:
     a. Create batch stream from pipeline
     b. Process each batch (train_step + `on_batch_end`)
     c. Compute epoch stats
     d. Run `on_epoch_end` (validation, diagnostics, checkpointing, early stopping)
  3. Finalize (`on_train_end`)
  """

  alias ExPhil.Training.{Callback, TrainingState, Pipeline}
  alias ExPhil.Training.Imitation

  require Logger

  @doc """
  Count total trainable parameters in a trainer.
  """
  def param_count(%Imitation{policy_params: params}) do
    alias ExPhil.Training.Imitation.TrainLoop
    count_params(TrainLoop.get_params_data(params))
  end

  defp count_params(%Nx.Tensor{} = t), do: Nx.size(t)
  defp count_params(map) when is_map(map) and not is_struct(map) do
    map |> Map.values() |> Enum.map(&count_params/1) |> Enum.sum()
  end
  defp count_params(_), do: 0

  @doc """
  Create a new trainer from a pipeline and options.

  Builds the model, optimizer, and JIT-compiles loss functions.
  """
  @spec new(Pipeline.t(), keyword()) :: Imitation.t()
  def new(%Pipeline{} = pipeline, opts) do
    embed_size =
      if pipeline.train_dataset && pipeline.train_dataset.embedded_frames do
        elem(Nx.shape(pipeline.train_dataset.embedded_frames), 1)
      else
        opts[:embed_size] || 288
      end

    # Merge resolved opts with required defaults for Imitation.new
    # Config leaves many options as nil; Imitation needs concrete values
    defaults = [
      embed_size: embed_size,
      hidden_size: hd(opts[:hidden_sizes] || [512]),
      dropout: 0.0,
      num_heads: 2,
      head_dim: 32,
      policy_type: :autoregressive,
      focal_loss: false,
      label_smoothing: 0.0,
      button_weight: 1.0,
      stick_edge_weight: nil,
      max_grad_norm: 1.0
    ]

    # Filter nil values so defaults aren't overridden by unset config options
    non_nil = fn kw -> Enum.filter(kw, fn {_k, v} -> v != nil end) end

    # pipeline.resolved_opts has all config with :auto values resolved to tensors
    # Don't re-merge raw opts — they'd re-introduce unresolved :auto values
    trainer_opts =
      defaults
      |> Keyword.merge(non_nil.(pipeline.resolved_opts))
      |> Keyword.put(:embed_size, embed_size)

    Imitation.new(trainer_opts)
  end

  @doc """
  Display model architecture as a table.

  Shows layer names, input/output shapes, and parameter counts.
  Requires the `table_rex` dependency.
  """
  def display_model(%Imitation{policy_model: model} = trainer, opts \\ []) do
    temporal = Keyword.get(opts, :temporal, false)
    embed_size = Keyword.get(opts, :embed_size, 288)
    window_size = Keyword.get(opts, :window_size, 60)

    template =
      if temporal do
        Nx.template({1, window_size, embed_size}, :f32)
      else
        Nx.template({1, embed_size}, :f32)
      end

    try do
      table = Axon.Display.as_table(model, template)
      IO.puts(table)
    rescue
      _ -> Logger.warning("Model display requires table_rex dependency")
    end

    trainer
  end

  @doc """
  Auto-tune batch size by binary search for the largest batch that fits in GPU memory.
  Returns updated opts with the tuned batch_size.
  """
  def auto_tune_batch_size(trainer, pipeline, opts) do
    alias ExPhil.Training.BatchTuner

    {batch_stream, _} = Pipeline.batch_stream(pipeline, [])
    sample_batch = Enum.take(batch_stream, 1) |> List.first()

    if sample_batch do
      try do
        optimal = BatchTuner.find_optimal(trainer, sample_batch.states, sample_batch.actions,
          min_batch: opts[:auto_batch_min] || 8,
          max_batch: opts[:auto_batch_max] || 256,
          backoff: opts[:auto_batch_backoff] || 0.8
        )
        Logger.info("Auto batch size: #{optimal}")
        Keyword.put(opts, :batch_size, optimal)
      rescue
        e ->
          Logger.warning("Auto batch size failed: #{Exception.message(e)}")
          opts
      end
    else
      opts
    end
  end

  @doc """
  Resume a trainer from a saved checkpoint.

  Returns the trainer with restored policy_params, optimizer_state, and step count.
  """
  @spec resume(Imitation.t(), Path.t()) :: {:ok, Imitation.t()} | {:error, term()}
  def resume(trainer, checkpoint_path) do
    Logger.info("Resuming from #{checkpoint_path}")
    Imitation.load_checkpoint(trainer, checkpoint_path)
  end

  @doc """
  Run the full training loop with callbacks.

  ## Options
  - `:callbacks` — List of `{CallbackModule, opts}` tuples or bare modules
  - `:epochs` — Number of epochs (from opts or pipeline.resolved_opts)

  ## Returns
  `{:ok, %TrainingState{}}` with final trainer, history, and all metadata.
  """
  @spec fit(Imitation.t(), Pipeline.t(), keyword()) :: {:ok, TrainingState.t()}
  def fit(trainer, pipeline, fit_opts \\ []) do
    callback_specs = Keyword.get(fit_opts, :callbacks, [])
    opts = pipeline.resolved_opts
    epochs = opts[:epochs] || 10

    # Seed for reproducibility
    seed = opts[:seed] || :rand.uniform(1_000_000)
    :rand.seed(:exsss, {seed, seed, seed})
    Nx.Random.key(seed)
    Logger.info("Training seed: #{seed}")

    # Initialize callbacks
    callbacks = Callback.init_all(callback_specs)

    # Build initial state
    state = %TrainingState{
      trainer: trainer,
      pipeline: pipeline,
      epochs: epochs,
      opts: opts,
      meta: %{seed: seed}
    }

    # on_train_begin
    state = Callback.increment_event_count(state, :on_train_begin)
    {_, state, callbacks} = Callback.run(callbacks, :on_train_begin, state)

    # Epoch loop
    {final_state, final_callbacks} =
      Enum.reduce_while(1..epochs, {state, callbacks}, fn epoch, {st, cbs} ->
        st = %{st | epoch: epoch, epoch_losses: [], batch_idx: 0}
        epoch_start = System.monotonic_time(:second)

        # on_epoch_begin
        {result, st, cbs} = Callback.run(cbs, :on_epoch_begin, st)
        if result == :halt do
          {:halt, {st, cbs}}
        else
          # Get batch stream
          {batch_stream, _num_batches} = Pipeline.batch_stream(pipeline, [])

          # Train one epoch
          {st, cbs} = train_epoch(st, batch_stream, cbs)

          # Compute epoch stats
          epoch_time = System.monotonic_time(:second) - epoch_start
          avg_loss =
            if st.epoch_losses == [] do
              0.0
            else
              Enum.sum(st.epoch_losses) / length(st.epoch_losses)
            end

          st = %{st |
            train_loss: avg_loss,
            epoch_time: epoch_time
          }

          # on_epoch_end (validation, diagnostics, checkpointing, early stopping)
          {result, st, cbs} = Callback.run(cbs, :on_epoch_end, st)

          # Update history
          epoch_entry = %{
            epoch: epoch,
            train_loss: avg_loss,
            val_loss: st.val_loss,
            time_seconds: epoch_time
          }
          st = %{st | history: st.history ++ [epoch_entry]}

          case result do
            :halt -> {:halt, {st, cbs}}
            :cont -> {:cont, {st, cbs}}
          end
        end
      end)

    # on_train_end
    {_, final_state, _} = Callback.run(final_callbacks, :on_train_end, final_state)

    {:ok, final_state}
  end

  # ============================================================================
  # Private — Epoch Training
  # ============================================================================

  defp train_epoch(state, batch_stream, callbacks) do
    accumulation_steps = state.opts[:accumulation_steps] || 1

    if accumulation_steps > 1 do
      # Gradient accumulation compiles a separate JIT program (compute_gradients),
      # which doubles GPU memory for cached programs. On memory-constrained GPUs,
      # this can cause OOM. Log a warning.
      if state.epoch == 1 do
        Logger.info("Gradient accumulation: #{accumulation_steps} steps (effective batch #{(state.opts[:batch_size] || 16) * accumulation_steps})")
      end
      train_epoch_accumulated(state, batch_stream, callbacks, accumulation_steps)
    else
      train_epoch_standard(state, batch_stream, callbacks)
    end
  end

  defp train_epoch_standard(state, batch_stream, callbacks) do
    # NOTE: Prefetcher disabled — EXLA tensors are process-local NIF references
    # and can't be sent across process boundaries. The chunked embedding slicing
    # (174x faster) makes batch creation <1ms, so prefetching is unnecessary.
    # Future: prefetch CPU-only batches, transfer to GPU in training process.
    # Track batch index via TrainingState instead of Stream.with_index (saves ~0.2ms/iter)
    batch_stream
    |> Enum.reduce_while({state, callbacks}, fn batch, {st, cbs} ->
      {new_trainer, metrics} = Imitation.train_step(st.trainer, batch, nil)
      loss = Nx.to_number(metrics.loss)
      batch_idx = st.batch_idx

      check_nan!(loss, batch_idx, st)

      st = %{st |
        trainer: new_trainer,
        step: st.step + 1,
        batch_idx: batch_idx + 1,
        batch_metrics: %{loss: loss},
        epoch_losses: [loss | st.epoch_losses]
      }

      st = Callback.increment_event_count(st, :on_batch_end)
      {result, st, cbs} = Callback.run(cbs, :on_batch_end, st)
      if rem(batch_idx, 100) == 0, do: :erlang.garbage_collect()

      case result do
        :halt_epoch -> {:halt, {st, cbs}}
        :halt -> {:halt, {st, cbs}}
        :cont -> {:cont, {st, cbs}}
      end
    end)
  end

  defp train_epoch_accumulated(state, batch_stream, callbacks, accumulation_steps) do
    alias ExPhil.Training.Imitation.TrainLoop

    init_accum = %{grads: nil, losses: [], count: 0}

    batch_stream
    |> Stream.with_index()
    |> Enum.reduce({state, callbacks, init_accum}, fn {batch, batch_idx}, {st, cbs, accum} ->
      # Compute gradients without applying
      {grads, loss} = TrainLoop.compute_gradients(st.trainer, batch)

      # Accumulate
      new_grads = if accum.grads, do: TrainLoop.add_gradients(accum.grads, grads), else: grads
      accum = %{accum | grads: new_grads, losses: [loss | accum.losses], count: accum.count + 1}

      if accum.count >= accumulation_steps do
        # Average and apply
        avg_grads = TrainLoop.scale_gradients(accum.grads, 1.0 / accumulation_steps)
        avg_loss = Enum.sum(accum.losses) / accumulation_steps
        check_nan!(avg_loss, batch_idx, st)
        new_trainer = TrainLoop.apply_gradients(st.trainer, avg_grads)

        st = %{st |
          trainer: new_trainer,
          step: st.step + 1,
          batch_idx: batch_idx,
          batch_metrics: %{loss: avg_loss},
          epoch_losses: [avg_loss | st.epoch_losses]
        }

        {_result, st, cbs} = Callback.run(cbs, :on_batch_end, st)
        if rem(batch_idx, 100) == 0, do: :erlang.garbage_collect()

        {st, cbs, %{grads: nil, losses: [], count: 0}}
      else
        {st, cbs, accum}
      end
    end)
    |> then(fn {st, cbs, accum} ->
      # Handle remaining partial accumulation
      if accum.count > 0 and accum.grads != nil do
        avg_grads = TrainLoop.scale_gradients(accum.grads, 1.0 / accum.count)
        avg_loss = Enum.sum(accum.losses) / accum.count
        new_trainer = TrainLoop.apply_gradients(st.trainer, avg_grads)
        st = %{st | trainer: new_trainer, step: st.step + 1, epoch_losses: [avg_loss | st.epoch_losses]}
        {st, cbs}
      else
        {st, cbs}
      end
    end)
  end

  defp check_nan!(loss, batch_idx, state) when is_atom(loss) do
    # :nan, :infinity, :neg_infinity
    raise """
    Training diverged: loss is #{loss} at batch #{batch_idx + 1} (epoch #{state.epoch})

    Common fixes:
      - Lower learning rate: --learning-rate 1e-5
      - Enable gradient clipping: --max-grad-norm 1.0
      - Use f32 precision: --precision f32
      - Reduce batch size
    """
  end

  defp check_nan!(loss, _batch_idx, _state) when is_number(loss), do: :ok
end
