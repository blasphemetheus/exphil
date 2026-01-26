defmodule ExPhil.League.Pretraining do
  @moduledoc """
  Imitation pretraining for architecture league competition.

  Trains all architectures on the same dataset to a target loss threshold,
  ensuring fair starting conditions before competition begins.

  ## Training Strategy

  1. **Precompute embeddings once** - Shared by all architectures
  2. **Train each architecture sequentially** - For GPU memory efficiency
  3. **Stop at target loss** - Ensures comparable training levels
  4. **Return trained params** - Ready for league registration

  ## Usage

      # Define architectures to train
      architectures = [
        %{id: :mlp_mewtwo, architecture: :mlp, config: %{hidden_sizes: [256, 256]}},
        %{id: :mamba_mewtwo, architecture: :mamba, config: %{hidden_size: 256, num_layers: 2}},
        %{id: :lstm_mewtwo, architecture: :lstm, config: %{hidden_size: 256, num_layers: 2}}
      ]

      # Train all to target loss
      {:ok, trained} = Pretraining.train_all(
        architectures,
        dataset,
        target_loss: 1.0,
        max_epochs: 50
      )

      # trained is a map: %{:mlp_mewtwo => {model, params, final_loss}, ...}

  """

  alias ExPhil.Training.{Imitation, Output, GPUUtils}
  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings

  require Logger

  @default_opts [
    target_loss: 1.0,
    max_epochs: 50,
    batch_size: 64,
    learning_rate: 1.0e-4,
    patience: 5,
    validation_split: 0.1,
    verbose: true,
    checkpoint_dir: nil
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Train all architectures on shared dataset to target loss.

  ## Parameters

  - `architectures` - List of architecture specs (maps with :id, :architecture, :config)
  - `dataset` - Training dataset (list of %{state: ..., action: ...} maps)

  ## Options

  - `:target_loss` - Stop training when validation loss reaches this (default: 1.0)
  - `:max_epochs` - Maximum epochs per architecture (default: 50)
  - `:batch_size` - Batch size (default: 64)
  - `:learning_rate` - Learning rate (default: 1.0e-4)
  - `:patience` - Early stopping patience (default: 5)
  - `:validation_split` - Fraction for validation (default: 0.1)
  - `:verbose` - Print progress (default: true)
  - `:checkpoint_dir` - Save checkpoints here (optional)

  ## Returns

  `{:ok, %{arch_id => {model, params, final_loss, epochs_trained}}}`

  """
  @spec train_all([map()], [map()], keyword()) :: {:ok, map()} | {:error, term()}
  def train_all(architectures, dataset, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    verbose = Keyword.get(opts, :verbose)

    if verbose do
      Output.banner("Architecture League Pretraining")

      Output.config([
        {"Architectures", length(architectures)},
        {"Dataset size", length(dataset)},
        {"Target loss", opts[:target_loss]},
        {"Max epochs", opts[:max_epochs]},
        {"Batch size", opts[:batch_size]}
      ])
    end

    # Split dataset
    {train_data, val_data} = split_dataset(dataset, opts[:validation_split])

    if verbose do
      Output.puts("Train: #{length(train_data)} samples, Val: #{length(val_data)} samples")
      Output.puts("")
    end

    # Precompute embeddings
    if verbose, do: Output.puts("Precomputing embeddings...")

    embed_config = Embeddings.config([])
    {train_embedded, val_embedded} = precompute_embeddings(train_data, val_data, embed_config)

    if verbose do
      Output.success("Embeddings ready")
      Output.puts("")
    end

    # Train each architecture
    total = length(architectures)

    results =
      architectures
      |> Enum.with_index(1)
      |> Enum.reduce(%{}, fn {arch_spec, idx}, acc ->
        arch_id = arch_spec.id
        architecture = arch_spec.architecture
        config = Map.get(arch_spec, :config, %{})

        if verbose do
          Output.step(idx, total, "Training #{arch_id} (#{architecture})")
          Output.puts(GPUUtils.memory_status_string())
        end

        # Build model for this architecture
        model = build_model(architecture, embed_config, config)

        # Train to target loss
        result =
          train_to_target(
            model,
            train_embedded,
            val_embedded,
            embed_config,
            opts
          )

        case result do
          {:ok, params, final_loss, epochs} ->
            if verbose do
              Output.success("#{arch_id}: loss=#{Float.round(final_loss, 4)} in #{epochs} epochs")
            end

            # Save checkpoint if configured
            if opts[:checkpoint_dir] do
              save_checkpoint(opts[:checkpoint_dir], arch_id, model, params)
            end

            Map.put(acc, arch_id, {model, params, final_loss, epochs})

          {:error, reason} ->
            Logger.warning("[Pretraining] Failed to train #{arch_id}: #{inspect(reason)}")
            acc
        end
      end)

    if verbose do
      Output.puts("")
      Output.success("Pretraining complete: #{map_size(results)}/#{total} architectures trained")
    end

    {:ok, results}
  end

  @doc """
  Train a single architecture to target loss.

  ## Parameters

  - `model` - Axon model
  - `train_data` - Embedded training data
  - `val_data` - Embedded validation data
  - `embed_config` - Embedding configuration
  - `opts` - Training options

  ## Returns

  `{:ok, params, final_loss, epochs_trained}` or `{:error, reason}`
  """
  @spec train_to_target(Axon.t(), map(), map(), map(), keyword()) ::
          {:ok, map(), float(), non_neg_integer()} | {:error, term()}
  def train_to_target(model, train_data, val_data, embed_config, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    target_loss = opts[:target_loss]
    max_epochs = opts[:max_epochs]
    batch_size = opts[:batch_size]
    learning_rate = opts[:learning_rate]
    patience = opts[:patience]
    verbose = opts[:verbose]

    # Initialize trainer
    trainer =
      Imitation.new(
        embed_config: embed_config,
        hidden_sizes: [256, 256],
        learning_rate: learning_rate,
        batch_size: batch_size,
        temporal: is_temporal_model?(model),
        window_size: 30
      )

    # Initialize params using Axon.build
    sample_input = create_sample_input(embed_config, batch_size)
    {init_fn, _predict_fn} = Axon.build(model, mode: :train)
    params = init_fn.(sample_input, Axon.ModelState.empty())

    # Training loop
    train_loop(
      trainer,
      model,
      params,
      train_data,
      val_data,
      target_loss,
      max_epochs,
      patience,
      verbose
    )
  end

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a policy model for a specific architecture type.
  """
  @spec build_model(atom(), map(), map()) :: Axon.t()
  def build_model(architecture, embed_config, config) do
    embed_size = Embeddings.embedding_size(embed_config)

    case architecture do
      :mlp ->
        build_mlp_model(embed_size, config)

      :lstm ->
        build_recurrent_model(embed_size, :lstm, config)

      :gru ->
        build_recurrent_model(embed_size, :gru, config)

      :mamba ->
        build_mamba_model(embed_size, config)

      :attention ->
        build_attention_model(embed_size, config)

      :jamba ->
        build_jamba_model(embed_size, config)

      other ->
        raise ArgumentError, "Unknown architecture: #{inspect(other)}"
    end
  end

  defp build_mlp_model(embed_size, config) do
    hidden_sizes = Map.get(config, :hidden_sizes, [256, 256])
    dropout = Map.get(config, :dropout, 0.1)

    Policy.build(
      embed_size: embed_size,
      hidden_sizes: hidden_sizes,
      dropout: dropout,
      temporal: false
    )
  end

  defp build_recurrent_model(embed_size, cell_type, config) do
    hidden_size = Map.get(config, :hidden_size, 256)
    num_layers = Map.get(config, :num_layers, 2)
    window_size = Map.get(config, :window_size, 30)
    dropout = Map.get(config, :dropout, 0.1)

    Policy.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      temporal: true,
      backbone: cell_type,
      window_size: window_size
    )
  end

  defp build_mamba_model(embed_size, config) do
    hidden_size = Map.get(config, :hidden_size, 256)
    num_layers = Map.get(config, :num_layers, 2)
    state_size = Map.get(config, :state_size, 16)
    window_size = Map.get(config, :window_size, 30)

    Policy.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      state_size: state_size,
      temporal: true,
      backbone: :mamba,
      window_size: window_size
    )
  end

  defp build_attention_model(embed_size, config) do
    hidden_size = Map.get(config, :hidden_size, 256)
    num_layers = Map.get(config, :num_layers, 2)
    num_heads = Map.get(config, :num_heads, 4)
    window_size = Map.get(config, :window_size, 30)
    dropout = Map.get(config, :dropout, 0.1)

    Policy.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      dropout: dropout,
      temporal: true,
      backbone: :attention,
      window_size: window_size
    )
  end

  defp build_jamba_model(embed_size, config) do
    hidden_size = Map.get(config, :hidden_size, 256)
    num_layers = Map.get(config, :num_layers, 4)
    state_size = Map.get(config, :state_size, 16)
    num_heads = Map.get(config, :num_heads, 4)
    attention_every = Map.get(config, :attention_every, 3)
    window_size = Map.get(config, :window_size, 30)

    Policy.build(
      embed_size: embed_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      state_size: state_size,
      num_heads: num_heads,
      attention_every: attention_every,
      temporal: true,
      backbone: :jamba,
      window_size: window_size
    )
  end

  # ============================================================================
  # Training Loop
  # ============================================================================

  defp train_loop(
         trainer,
         model,
         params,
         train_data,
         val_data,
         target_loss,
         max_epochs,
         patience,
         verbose
       ) do
    # Build training functions
    {_loss_fn, grad_fn} = build_training_fns(model, trainer)

    # Optimizer - returns {init_fn, update_fn} tuple
    {optimizer_init, optimizer_update} =
      Polaris.Optimizers.adamw(learning_rate: trainer.config.learning_rate)

    opt_state = optimizer_init.(params)

    # Training state
    state = %{
      params: params,
      opt_state: opt_state,
      best_loss: :infinity,
      best_params: params,
      no_improve_count: 0,
      epoch: 0
    }

    do_train_loop(
      state,
      trainer,
      model,
      grad_fn,
      optimizer_update,
      train_data,
      val_data,
      target_loss,
      max_epochs,
      patience,
      verbose
    )
  end

  defp do_train_loop(
         state,
         _trainer,
         _model,
         _grad_fn,
         _optimizer,
         _train_data,
         _val_data,
         target_loss,
         _max_epochs,
         _patience,
         _verbose
       )
       when state.best_loss <= target_loss do
    # Reached target loss
    {:ok, state.best_params, state.best_loss, state.epoch}
  end

  defp do_train_loop(
         state,
         _trainer,
         _model,
         _grad_fn,
         _optimizer,
         _train_data,
         _val_data,
         _target_loss,
         max_epochs,
         _patience,
         _verbose
       )
       when state.epoch >= max_epochs do
    # Reached max epochs
    {:ok, state.best_params, state.best_loss, state.epoch}
  end

  defp do_train_loop(
         state,
         _trainer,
         _model,
         _grad_fn,
         _optimizer,
         _train_data,
         _val_data,
         _target_loss,
         _max_epochs,
         patience,
         _verbose
       )
       when state.no_improve_count >= patience do
    # Early stopping
    {:ok, state.best_params, state.best_loss, state.epoch}
  end

  defp do_train_loop(
         state,
         trainer,
         model,
         grad_fn,
         optimizer,
         train_data,
         val_data,
         target_loss,
         max_epochs,
         patience,
         verbose
       ) do
    # Run one epoch
    {new_params, new_opt_state, train_loss} =
      run_epoch(
        state.params,
        state.opt_state,
        train_data,
        grad_fn,
        optimizer,
        trainer.config.batch_size
      )

    # Validate
    val_loss = compute_validation_loss(model, new_params, val_data)

    # Update state
    {best_loss, best_params, no_improve} =
      if val_loss < state.best_loss do
        {val_loss, new_params, 0}
      else
        {state.best_loss, state.best_params, state.no_improve_count + 1}
      end

    if verbose do
      IO.write(
        :stderr,
        "\r  Epoch #{state.epoch + 1}: train_loss=#{Float.round(train_loss, 4)}, " <>
          "val_loss=#{Float.round(val_loss, 4)}, best=#{Float.round(best_loss, 4)}"
      )
    end

    new_state = %{
      state
      | params: new_params,
        opt_state: new_opt_state,
        best_loss: best_loss,
        best_params: best_params,
        no_improve_count: no_improve,
        epoch: state.epoch + 1
    }

    do_train_loop(
      new_state,
      trainer,
      model,
      grad_fn,
      optimizer,
      train_data,
      val_data,
      target_loss,
      max_epochs,
      patience,
      verbose
    )
  end

  defp run_epoch(params, opt_state, data, grad_fn, optimizer_update, batch_size) do
    # Create batches
    batches = create_batches(data, batch_size)

    # Process each batch
    {final_params, final_opt_state, total_loss, batch_count} =
      Enum.reduce(batches, {params, opt_state, 0.0, 0}, fn batch, {p, opt, loss_sum, count} ->
        # Compute loss and gradients
        {loss, grads} = grad_fn.(p, batch)

        # Apply updates using optimizer update function
        {updates, new_opt} = optimizer_update.(grads, opt, p)
        new_params = Polaris.Updates.apply_updates(updates, p)

        {new_params, new_opt, loss_sum + Nx.to_number(loss), count + 1}
      end)

    avg_loss = if batch_count > 0, do: total_loss / batch_count, else: 0.0

    {final_params, final_opt_state, avg_loss}
  end

  defp compute_validation_loss(model, params, val_data) do
    batches = create_batches(val_data, 64)

    {total_loss, count} =
      Enum.reduce(batches, {0.0, 0}, fn batch, {sum, c} ->
        output = Axon.predict(model, params, %{"state" => batch.states})
        loss = compute_policy_loss(output, batch.actions)
        {sum + Nx.to_number(loss), c + 1}
      end)

    if count > 0, do: total_loss / count, else: 0.0
  end

  # ============================================================================
  # Data Processing
  # ============================================================================

  defp split_dataset(dataset, split_ratio) do
    shuffled = Enum.shuffle(dataset)
    split_idx = round(length(shuffled) * (1 - split_ratio))

    train = Enum.take(shuffled, split_idx)
    val = Enum.drop(shuffled, split_idx)

    {train, val}
  end

  defp precompute_embeddings(train_data, val_data, embed_config) do
    # Embed training data
    train_embedded =
      Enum.map(train_data, fn sample ->
        %{
          state: embed_sample(sample.state, embed_config),
          action: sample.action
        }
      end)

    # Embed validation data
    val_embedded =
      Enum.map(val_data, fn sample ->
        %{
          state: embed_sample(sample.state, embed_config),
          action: sample.action
        }
      end)

    {train_embedded, val_embedded}
  end

  defp embed_sample(state, _embed_config) do
    # Use the game embedding system
    # For now, simple tensor conversion
    case state do
      %Nx.Tensor{} = t -> t
      list when is_list(list) -> Nx.tensor(list, type: :f32)
      _ -> Nx.tensor([0.0], type: :f32)
    end
  end

  defp create_batches(data, batch_size) do
    data
    |> Enum.shuffle()
    |> Enum.chunk_every(batch_size, batch_size, :discard)
    |> Enum.map(&stack_batch/1)
  end

  defp stack_batch(samples) do
    states = samples |> Enum.map(& &1.state) |> Nx.stack()
    actions = samples |> Enum.map(& &1.action) |> stack_actions()

    %{states: states, actions: actions}
  end

  defp stack_actions(actions) do
    # Actions can be maps or simple values
    case hd(actions) do
      %{} = _action_map ->
        keys = Map.keys(hd(actions))

        Map.new(keys, fn key ->
          values = Enum.map(actions, &Map.get(&1, key))
          {key, Nx.stack(values)}
        end)

      _ ->
        Nx.stack(actions)
    end
  end

  defp create_sample_input(embed_config, batch_size) do
    embed_size = Embeddings.embedding_size(embed_config)
    %{"state" => Nx.broadcast(0.0, {batch_size, embed_size})}
  end

  # ============================================================================
  # Training Functions
  # ============================================================================

  defp build_training_fns(model, _trainer) do
    # Loss function
    loss_fn = fn params, batch ->
      output = Axon.predict(model, params, %{"state" => batch.states})
      compute_policy_loss(output, batch.actions)
    end

    # Gradient function
    grad_fn = fn params, batch ->
      Nx.Defn.value_and_grad(params, fn p ->
        output = Axon.predict(model, p, %{"state" => batch.states})
        compute_policy_loss(output, batch.actions)
      end)
    end

    {loss_fn, grad_fn}
  end

  defp compute_policy_loss(output, target_actions) do
    # Cross-entropy loss for discrete action heads
    # Simplified - actual implementation would handle all action heads

    case output do
      %{"buttons" => buttons_logits} ->
        target =
          case target_actions do
            %{buttons: t} -> t
            _ -> Nx.broadcast(0, Nx.shape(buttons_logits) |> elem(0))
          end

        Axon.Losses.categorical_cross_entropy(
          target,
          Axon.Activations.softmax(buttons_logits),
          reduction: :mean
        )

      _ ->
        # Fallback for different output formats
        Nx.tensor(0.0, type: :f32)
    end
  end

  defp is_temporal_model?(_model) do
    # Check if model expects sequence input
    # For now, default to false
    false
  end

  # ============================================================================
  # Checkpointing
  # ============================================================================

  defp save_checkpoint(dir, arch_id, _model, params) do
    File.mkdir_p!(dir)

    # Save params
    params_path = Path.join(dir, "#{arch_id}_pretrained.bin")

    params_binary =
      params
      |> Enum.map(fn {k, v} -> {k, Nx.backend_transfer(v, Nx.BinaryBackend)} end)
      |> Map.new()

    File.write!(params_path, :erlang.term_to_binary(params_binary))

    Logger.debug("[Pretraining] Saved checkpoint for #{arch_id} to #{params_path}")
  end
end
