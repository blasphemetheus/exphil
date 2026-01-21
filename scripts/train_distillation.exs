#!/usr/bin/env elixir
# Train a small MLP student model using knowledge distillation.
#
# The student learns to match the soft probability distributions generated
# by a larger teacher model (LSTM/Mamba), achieving similar behavior with
# much faster inference.
#
# Usage:
#   mix run scripts/train_distillation.exs \
#     --soft-labels soft_labels.bin \
#     --hidden 64,64 \
#     --epochs 10 \
#     --output distilled_policy.bin
#
# The loss function combines:
#   - Soft loss: KL divergence from teacher distributions (weighted by alpha)
#   - Hard loss: Cross-entropy with original ground truth (weighted by 1-alpha)
#
# Recommended settings:
#   - alpha=0.7: Balance soft and hard labels
#   - hidden=64,64: Good speed/accuracy tradeoff (~2ms inference)
#   - hidden=32,32: Maximum speed (~1ms inference)

Mix.install([])

defmodule DistillationTrainer do
  @moduledoc """
  Train a student MLP using knowledge distillation from soft labels.
  """

  import Nx.Defn
  alias ExPhil.Networks.Policy
  alias ExPhil.Embeddings.Game, as: GameEmbed
  alias ExPhil.Training.Output

  @default_hidden_sizes [64, 64]
  @default_alpha 0.7  # Weight for soft labels vs hard labels
  @default_learning_rate 1.0e-3
  @default_batch_size 128
  @default_epochs 10

  def run(args) do
    opts = parse_args(args)

    Output.banner("Knowledge Distillation Training")
    Output.config([
      {"Soft labels", opts.soft_labels},
      {"Hidden sizes", inspect(opts.hidden_sizes)},
      {"Epochs", opts.epochs},
      {"Batch size", opts.batch_size},
      {"Learning rate", opts.learning_rate},
      {"Alpha (soft weight)", opts.alpha},
      {"Output", opts.output}
    ])

    # Load soft labels
    Output.step(1, 7, "Loading soft labels")
    %{config: config, labels: labels} = load_soft_labels(opts.soft_labels)
    Output.puts("  Loaded #{length(labels)} frames")
    Output.puts("  Teacher: #{config.teacher_path}")
    Output.puts("  Temperature: #{config.temperature}")

    embed_size = config.embed_size

    # Build student model
    Output.step(2, 7, "Building student MLP")
    Output.puts("  Hidden sizes: #{inspect(opts.hidden_sizes)}")

    student_model = Policy.build(
      embed_size: embed_size,
      hidden_sizes: opts.hidden_sizes,
      dropout: opts.dropout
    )

    # Count parameters
    {init_fn, _} = Axon.build(student_model, mode: :inference)
    dummy_input = Nx.broadcast(0.0, {1, embed_size})
    params = init_fn.(dummy_input, Axon.ModelState.empty())
    param_count = count_params(params)
    Output.puts("  Parameters: #{format_number(param_count)}")

    # Prepare dataset
    Output.step(3, 7, "Preparing dataset")
    {train_data, val_data} = prepare_dataset(labels, opts.val_split)
    Output.puts("  Training: #{length(train_data)} frames")
    Output.puts("  Validation: #{length(val_data)} frames")

    # Initialize optimizer
    Output.step(4, 7, "Initializing optimizer")
    optimizer = Polaris.Optimizers.adamw(learning_rate: opts.learning_rate)
    optimizer_state = Polaris.Updates.init(optimizer)

    # Build training functions
    {_, predict_fn} = Axon.build(student_model, mode: :inference)

    # Training loop
    Output.step(5, 7, "Training")
    Output.puts("  Epochs: #{opts.epochs}")
    Output.puts("  Batch size: #{opts.batch_size}")
    Output.puts("  Alpha (soft weight): #{opts.alpha}")
    Output.puts("")

    final_state = train_loop(
      train_data, val_data,
      params, optimizer, optimizer_state,
      predict_fn, opts
    )

    # Export policy
    Output.step(6, 7, "Exporting distilled policy")
    export_policy(student_model, final_state.params, opts.output, %{
      embed_size: embed_size,
      hidden_sizes: opts.hidden_sizes,
      dropout: opts.dropout,
      distilled: true,
      teacher_path: config.teacher_path,
      teacher_config: config.teacher_config
    })

    # Benchmark inference
    Output.step(7, 7, "Benchmarking inference speed")
    benchmark_inference(student_model, final_state.params, embed_size)

    Output.divider()
    Output.section("Complete!")
    Output.puts("Output: #{opts.output}")
    Output.puts("Final validation loss: #{Float.round(final_state.val_loss, 4)}")
    Output.puts("")
    Output.puts("Test the policy:")
    Output.puts("  mix run scripts/eval_model.exs --policy #{opts.output}")
    Output.puts("")
    Output.puts("Play in Dolphin:")
    Output.puts("  mix run scripts/play_dolphin.exs --policy #{opts.output}")
  end

  defp parse_args(args) do
    {parsed, _, _} = OptionParser.parse(args, strict: [
      soft_labels: :string,
      hidden: :string,
      epochs: :integer,
      batch_size: :integer,
      learning_rate: :float,
      alpha: :float,
      dropout: :float,
      val_split: :float,
      output: :string
    ])

    hidden_sizes = case Keyword.get(parsed, :hidden) do
      nil -> @default_hidden_sizes
      str -> str |> String.split(",") |> Enum.map(&String.to_integer/1)
    end

    %{
      soft_labels: Keyword.fetch!(parsed, :soft_labels),
      hidden_sizes: hidden_sizes,
      epochs: Keyword.get(parsed, :epochs, @default_epochs),
      batch_size: Keyword.get(parsed, :batch_size, @default_batch_size),
      learning_rate: Keyword.get(parsed, :learning_rate, @default_learning_rate),
      alpha: Keyword.get(parsed, :alpha, @default_alpha),
      dropout: Keyword.get(parsed, :dropout, 0.1),
      val_split: Keyword.get(parsed, :val_split, 0.1),
      output: Keyword.get(parsed, :output, "distilled_policy.bin")
    }
  end

  defp load_soft_labels(path) do
    path
    |> File.read!()
    |> :erlang.binary_to_term()
  end

  defp prepare_dataset(labels, val_split) do
    # Shuffle and split
    shuffled = Enum.shuffle(labels)
    val_count = round(length(shuffled) * val_split)

    {val_data, train_data} = Enum.split(shuffled, val_count)
    {train_data, val_data}
  end

  defp train_loop(train_data, val_data, params, optimizer, opt_state, predict_fn, opts) do
    # Cache JIT-compiled functions
    apply_updates_fn = Nx.Defn.jit(&Polaris.Updates.apply_updates/2)

    initial_state = %{
      params: params,
      opt_state: opt_state,
      epoch: 0,
      step: 0,
      val_loss: 0.0
    }

    Enum.reduce(1..opts.epochs, initial_state, fn epoch, state ->
      epoch_start = System.monotonic_time(:millisecond)

      # Shuffle training data each epoch
      shuffled = Enum.shuffle(train_data)
      batches = Enum.chunk_every(shuffled, opts.batch_size, opts.batch_size, :discard)
      num_batches = length(batches)

      # Train epoch
      {epoch_state, losses} = Enum.reduce(
        Enum.with_index(batches),
        {state, []},
        fn {batch, batch_idx}, {s, losses} ->
          # Prepare batch tensors
          {states, soft_labels, hard_labels} = prepare_batch(batch)

          # Compute gradients and update
          {loss, grads} = compute_loss_and_grads(
            s.params, predict_fn, states, soft_labels, hard_labels, opts.alpha
          )

          # Apply updates
          {updates, new_opt_state} = Polaris.Updates.update(
            optimizer, grads, s.opt_state, s.params
          )
          new_params = apply_updates_fn.(s.params, updates)

          # Progress
          if rem(batch_idx + 1, max(1, div(num_batches, 10))) == 0 do
            pct = round((batch_idx + 1) / num_batches * 100)
            IO.write(:stderr, "\r  Epoch #{epoch}: #{pct}% | loss: #{Float.round(Nx.to_number(loss), 4)}")
          end

          {%{s |
            params: new_params,
            opt_state: new_opt_state,
            step: s.step + 1
          }, [Nx.to_number(loss) | losses]}
        end
      )

      # Validation
      val_loss = compute_validation_loss(val_data, epoch_state.params, predict_fn, opts)

      epoch_time = System.monotonic_time(:millisecond) - epoch_start
      avg_loss = Enum.sum(losses) / length(losses)

      Output.puts("\r  Epoch #{epoch}: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_loss, 4)} (#{epoch_time}ms)")

      %{epoch_state | epoch: epoch, val_loss: val_loss}
    end)
  end

  defp prepare_batch(batch) do
    # Embed game states
    states = batch
    |> Enum.map(fn frame -> GameEmbed.embed(frame.game_state) end)
    |> Nx.stack()

    # Stack soft labels
    soft_labels = %{
      buttons: batch |> Enum.map(& &1.soft_labels.buttons) |> Nx.stack(),
      main_x: batch |> Enum.map(& &1.soft_labels.main_x) |> Nx.stack(),
      main_y: batch |> Enum.map(& &1.soft_labels.main_y) |> Nx.stack(),
      c_x: batch |> Enum.map(& &1.soft_labels.c_x) |> Nx.stack(),
      c_y: batch |> Enum.map(& &1.soft_labels.c_y) |> Nx.stack(),
      shoulder: batch |> Enum.map(& &1.soft_labels.shoulder) |> Nx.stack()
    }

    # Convert controller states to hard labels
    hard_labels = batch
    |> Enum.map(fn frame ->
      ExPhil.Embeddings.Controller.to_training_targets(frame.controller)
    end)
    |> stack_targets()

    {states, soft_labels, hard_labels}
  end

  defp stack_targets(targets) do
    %{
      buttons: targets |> Enum.map(& &1.buttons) |> Nx.stack(),
      main_x: targets |> Enum.map(& &1.main_x) |> Nx.stack(),
      main_y: targets |> Enum.map(& &1.main_y) |> Nx.stack(),
      c_x: targets |> Enum.map(& &1.c_x) |> Nx.stack(),
      c_y: targets |> Enum.map(& &1.c_y) |> Nx.stack(),
      shoulder: targets |> Enum.map(& &1.shoulder) |> Nx.stack()
    }
  end

  defp compute_loss_and_grads(params, predict_fn, states, soft_labels, hard_labels, alpha) do
    # Copy tensors to avoid EXLA/Defn.Expr mismatch
    states = Nx.backend_copy(states)
    soft_labels = Map.new(soft_labels, fn {k, v} -> {k, Nx.backend_copy(v)} end)
    hard_labels = Map.new(hard_labels, fn {k, v} -> {k, Nx.backend_copy(v)} end)
    params = deep_backend_copy(params)

    loss_fn = fn p ->
      logits = predict_fn.(p, states)
      distillation_loss(logits, soft_labels, hard_labels, alpha)
    end

    Nx.Defn.jit(&Nx.Defn.value_and_grad/1).(loss_fn).(params)
  end

  defn distillation_loss(logits, soft_labels, hard_labels, alpha) do
    {btn_logits, mx_logits, my_logits, cx_logits, cy_logits, sh_logits} = logits

    # Soft loss: KL divergence from teacher distributions
    soft_loss = kl_divergence_buttons(btn_logits, soft_labels.buttons)
      |> Nx.add(kl_divergence_categorical(mx_logits, soft_labels.main_x))
      |> Nx.add(kl_divergence_categorical(my_logits, soft_labels.main_y))
      |> Nx.add(kl_divergence_categorical(cx_logits, soft_labels.c_x))
      |> Nx.add(kl_divergence_categorical(cy_logits, soft_labels.c_y))
      |> Nx.add(kl_divergence_categorical(sh_logits, soft_labels.shoulder))

    # Hard loss: Cross-entropy with ground truth
    hard_loss = button_bce(btn_logits, hard_labels.buttons)
      |> Nx.add(categorical_ce(mx_logits, hard_labels.main_x))
      |> Nx.add(categorical_ce(my_logits, hard_labels.main_y))
      |> Nx.add(categorical_ce(cx_logits, hard_labels.c_x))
      |> Nx.add(categorical_ce(cy_logits, hard_labels.c_y))
      |> Nx.add(categorical_ce(sh_logits, hard_labels.shoulder))

    # Combined loss
    alpha * soft_loss + (1 - alpha) * hard_loss
  end

  defnp kl_divergence_buttons(logits, soft_targets) do
    # KL divergence for Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
    # Where p = soft_targets, q = sigmoid(logits)
    probs = Nx.sigmoid(logits)
    eps = 1.0e-7

    # Clamp to avoid log(0)
    probs = Nx.clip(probs, eps, 1.0 - eps)
    soft_targets = Nx.clip(soft_targets, eps, 1.0 - eps)

    kl = soft_targets * Nx.log(soft_targets / probs) +
         (1 - soft_targets) * Nx.log((1 - soft_targets) / (1 - probs))

    Nx.mean(kl)
  end

  defnp kl_divergence_categorical(logits, soft_targets) do
    # KL divergence for categorical: sum(p * log(p/q))
    # Where p = soft_targets, q = softmax(logits)
    probs = Axon.Activations.softmax(logits, axis: -1)
    eps = 1.0e-7

    probs = Nx.clip(probs, eps, 1.0)
    soft_targets = Nx.clip(soft_targets, eps, 1.0)

    kl = soft_targets * Nx.log(soft_targets / probs)
    Nx.mean(Nx.sum(kl, axes: [-1]))
  end

  defnp button_bce(logits, targets) do
    # Binary cross-entropy with hard labels
    probs = Nx.sigmoid(logits)
    eps = 1.0e-7
    probs = Nx.clip(probs, eps, 1.0 - eps)

    loss = -targets * Nx.log(probs) - (1 - targets) * Nx.log(1 - probs)
    Nx.mean(loss)
  end

  defnp categorical_ce(logits, targets) do
    # Categorical cross-entropy with one-hot hard labels
    # targets should be one-hot encoded
    log_probs = Axon.Activations.log_softmax(logits, axis: -1)
    loss = -Nx.sum(targets * log_probs, axes: [-1])
    Nx.mean(loss)
  end

  defp compute_validation_loss(val_data, params, predict_fn, opts) do
    batches = Enum.chunk_every(val_data, opts.batch_size, opts.batch_size, :discard)

    losses = Enum.map(batches, fn batch ->
      {states, soft_labels, hard_labels} = prepare_batch(batch)

      # Copy for JIT
      states = Nx.backend_copy(states)
      soft_labels = Map.new(soft_labels, fn {k, v} -> {k, Nx.backend_copy(v)} end)
      hard_labels = Map.new(hard_labels, fn {k, v} -> {k, Nx.backend_copy(v)} end)
      params_copy = deep_backend_copy(params)

      logits = predict_fn.(params_copy, states)
      loss = distillation_loss(logits, soft_labels, hard_labels, opts.alpha)
      Nx.to_number(loss)
    end)

    if length(losses) > 0 do
      Enum.sum(losses) / length(losses)
    else
      0.0
    end
  end

  defp export_policy(model, params, path, config) do
    # Convert to binary backend
    params_binary = deep_backend_copy_to_binary(params)

    policy = %{
      model: model,
      params: params_binary,
      config: config
    }

    File.write!(path, :erlang.term_to_binary(policy, [:compressed]))
    Output.puts("  Saved: #{path}")
  end

  defp benchmark_inference(model, params, embed_size) do
    {_, predict_fn} = Axon.build(model, mode: :inference)

    # Warmup
    dummy = Nx.broadcast(0.5, {1, embed_size})
    for _ <- 1..10, do: predict_fn.(params, dummy)

    # Benchmark
    iterations = 100
    start = System.monotonic_time(:microsecond)
    for _ <- 1..iterations, do: predict_fn.(params, dummy)
    elapsed = System.monotonic_time(:microsecond) - start

    avg_ms = elapsed / iterations / 1000
    Output.puts("  Average inference: #{Float.round(avg_ms, 2)} ms")
    Output.puts("  60 FPS ready: #{if avg_ms < 16.67, do: "Yes", else: "No (need <16.67ms)"}")
  end

  defp count_params(params) do
    params
    |> Axon.ModelState.data()
    |> count_params_recursive()
  end

  defp count_params_recursive(%Nx.Tensor{} = t), do: Nx.size(t)
  defp count_params_recursive(map) when is_map(map) do
    map |> Map.values() |> Enum.map(&count_params_recursive/1) |> Enum.sum()
  end
  defp count_params_recursive(_), do: 0

  defp deep_backend_copy(%Nx.Tensor{} = t), do: Nx.backend_copy(t)
  defp deep_backend_copy(%Axon.ModelState{} = state) do
    %{state |
      data: deep_backend_copy(state.data),
      state: deep_backend_copy(state.state)
    }
  end
  defp deep_backend_copy(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_backend_copy(v)} end)
  end
  defp deep_backend_copy(other), do: other

  defp deep_backend_copy_to_binary(%Nx.Tensor{} = t) do
    Nx.backend_copy(t, Nx.BinaryBackend)
  end
  defp deep_backend_copy_to_binary(%Axon.ModelState{} = state) do
    %{state |
      data: deep_backend_copy_to_binary(state.data),
      state: deep_backend_copy_to_binary(state.state)
    }
  end
  defp deep_backend_copy_to_binary(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_backend_copy_to_binary(v)} end)
  end
  defp deep_backend_copy_to_binary(other), do: other

  defp format_number(n) when n >= 1_000_000, do: "#{Float.round(n / 1_000_000, 2)}M"
  defp format_number(n) when n >= 1_000, do: "#{Float.round(n / 1_000, 1)}K"
  defp format_number(n), do: "#{n}"
end

# Run trainer
DistillationTrainer.run(System.argv())
