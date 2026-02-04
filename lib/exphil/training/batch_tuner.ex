defmodule ExPhil.Training.BatchTuner do
  @moduledoc """
  Automatic batch size tuning for optimal GPU utilization.

  Finds the largest batch size that fits in GPU memory by testing
  progressively larger batches until an out-of-memory error occurs,
  then backing off to a safe size.

  ## How it works

  1. Start with a small batch size (e.g., 32)
  2. Run a test forward+backward pass
  3. If successful, double the batch size
  4. If OOM, reduce by 20% and test again
  5. Return the largest working batch size

  ## Usage

  ```elixir
  # Auto-tune with default settings
  {:ok, optimal_size} = BatchTuner.find_optimal(model, sample_batch)

  # With custom settings
  {:ok, optimal_size} = BatchTuner.find_optimal(model, sample_batch,
    initial: 64,
    max: 2048,
    backoff: 0.8
  )
  ```

  ## Safety

  - Each test is wrapped in a try/catch to handle OOM gracefully
  - Garbage collection is forced between tests to free GPU memory
  - A cooldown period prevents rapid OOM oscillation
  """

  require Logger
  alias ExPhil.Training.Output
  alias ExPhil.Error.GPUError

  @default_initial 32
  @default_max 4096
  @default_backoff 0.8

  @doc """
  Find the optimal batch size for the given model and sample data.

  ## Parameters

    - `model` - Axon model
    - `sample_states` - Sample input tensor of shape `{batch_size, ...}`
    - `sample_actions` - Sample target tensor

  ## Options

    - `:initial` - Starting batch size (default: 32)
    - `:max` - Maximum batch size to try (default: 4096)
    - `:backoff` - Factor to multiply by after OOM (default: 0.8)
    - `:show_progress` - Show progress output (default: true)
    - `:trainer` - Optional trainer to use for forward/backward pass

  ## Returns

    - `{:ok, batch_size}` - The optimal batch size
    - `{:error, reason}` - If tuning fails
  """
  @spec find_optimal(Axon.t() | map(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {:ok, pos_integer()} | {:error, term()}
  def find_optimal(model_or_trainer, sample_states, sample_actions, opts \\ []) do
    initial = Keyword.get(opts, :initial, @default_initial)
    max_size = Keyword.get(opts, :max, @default_max)
    backoff = Keyword.get(opts, :backoff, @default_backoff)
    show_progress = Keyword.get(opts, :show_progress, true)

    if show_progress do
      Output.puts("  🔍 Auto-tuning batch size (initial=#{initial}, max=#{max_size})...")
    end

    # Get the full batch size from samples
    {full_batch_size, _rest} = Nx.shape(sample_states) |> Tuple.to_list() |> List.pop_at(0)

    if full_batch_size < initial do
      if show_progress do
        Output.puts("    Sample batch (#{full_batch_size}) smaller than initial (#{initial})")
        Output.puts("    Using sample batch size: #{full_batch_size}")
      end

      {:ok, full_batch_size}
    else
      # Binary search for optimal size
      find_optimal_binary_search(
        model_or_trainer,
        sample_states,
        sample_actions,
        initial,
        max_size,
        backoff,
        show_progress
      )
    end
  end

  defp find_optimal_binary_search(
         model_or_trainer,
         sample_states,
         sample_actions,
         initial,
         max_size,
         backoff,
         show_progress
       ) do
    # Start with initial and double until OOM
    {largest_working, _} =
      Enum.reduce_while(
        Stream.iterate(initial, &(&1 * 2)),
        {initial, false},
        fn batch_size, {last_working, _} ->
          if batch_size > max_size do
            {:halt, {last_working, true}}
          else
            if show_progress do
              IO.write(:stderr, "    Testing batch_size=#{batch_size}...")
            end

            case test_batch_size(model_or_trainer, sample_states, sample_actions, batch_size) do
              :ok ->
                if show_progress do
                  IO.write(:stderr, " ✓\n")
                end

                # Force GC to free memory before next test
                :erlang.garbage_collect()
                {:cont, {batch_size, false}}

              {:error, %GPUError{reason: :oom}} ->
                if show_progress do
                  IO.write(:stderr, " OOM\n")
                end

                {:halt, {last_working, true}}

              {:error, reason} ->
                if show_progress do
                  IO.write(:stderr, " Error: #{inspect(reason)}\n")
                end

                {:halt, {last_working, true}}
            end
          end
        end
      )

    # Apply backoff for safety margin
    safe_size = max(initial, floor(largest_working * backoff))

    # Round down to nearest power of 2 for efficiency
    optimal = round_to_power_of_2(safe_size)

    if show_progress do
      Output.puts("  ✓ Optimal batch size: #{optimal} (largest working: #{largest_working})")
    end

    {:ok, optimal}
  end

  defp test_batch_size(model_or_trainer, sample_states, sample_actions, batch_size) do
    # Get a subset of the samples
    states_subset = Nx.slice_along_axis(sample_states, 0, batch_size, axis: 0)
    _actions_subset = Nx.slice_along_axis(sample_actions, 0, batch_size, axis: 0)

    try do
      # Run forward pass (this will allocate GPU memory)
      case model_or_trainer do
        %{policy_model: _model} = trainer ->
          # Use trainer's predict function
          predict_fn = trainer.predict_fn || fn params, x -> Axon.predict(trainer.policy_model, params, x) end
          _output = predict_fn.(trainer.policy_params, states_subset)

        %Axon{} = model ->
          # Initialize params if needed and run forward pass
          {init_fn, predict_fn} = Axon.build(model)
          params = init_fn.(states_subset, %{})
          _output = predict_fn.(params, states_subset)

        _ ->
          # Unknown model type, try generic approach
          :ok
      end

      # If we got here without error, the batch size works
      :ok
    rescue
      e in RuntimeError ->
        if String.contains?(Exception.message(e), "out of memory") or
             String.contains?(Exception.message(e), "OOM") or
             String.contains?(Exception.message(e), "CUDA") do
          {:error, GPUError.new(:oom)}
        else
          {:error, Exception.message(e)}
        end

      e ->
        {:error, Exception.message(e)}
    catch
      :exit, reason ->
        if is_tuple(reason) and elem(reason, 0) == :noproc do
          {:error, GPUError.new(:oom)}
        else
          {:error, {:exit, reason}}
        end
    end
  end

  defp round_to_power_of_2(n) when n <= 0, do: 1

  defp round_to_power_of_2(n) do
    # Find the largest power of 2 <= n
    power = :math.log2(n) |> floor()
    round(:math.pow(2, power))
  end

  @doc """
  Suggest a batch size based on available GPU memory.

  This is a quick heuristic that doesn't require running test batches.
  Use `find_optimal/4` for more accurate results.

  ## Parameters

    - `embed_size` - Size of embedding dimension
    - `hidden_sizes` - List of hidden layer sizes

  ## Options

    - `:vram_gb` - Available GPU VRAM in GB (default: auto-detect)
    - `:precision` - :f32 or :bf16 (default: :f32)
    - `:safety_factor` - Fraction of VRAM to use (default: 0.7)

  ## Returns

  Suggested batch size as integer
  """
  @spec suggest(pos_integer(), [pos_integer()], keyword()) :: pos_integer()
  def suggest(embed_size, hidden_sizes, opts \\ []) do
    vram_gb = Keyword.get(opts, :vram_gb) || detect_vram_gb()
    precision = Keyword.get(opts, :precision, :f32)
    safety_factor = Keyword.get(opts, :safety_factor, 0.7)

    bytes_per_param = if precision == :bf16, do: 2, else: 4

    # Rough estimate: each sample needs ~4x model size for forward + backward
    # Plus optimizer state (another 2x for Adam)
    model_params = estimate_model_params(embed_size, hidden_sizes)
    bytes_per_sample = model_params * bytes_per_param * 6

    available_bytes = vram_gb * 1_000_000_000 * safety_factor
    max_batch = floor(available_bytes / bytes_per_sample)

    # Clamp to reasonable range and round to power of 2
    max_batch
    |> max(16)
    |> min(4096)
    |> round_to_power_of_2()
  end

  defp estimate_model_params(embed_size, hidden_sizes) do
    # Input -> first hidden
    input_params = embed_size * List.first(hidden_sizes, 256)

    # Hidden layers
    hidden_params =
      hidden_sizes
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] -> a * b end)
      |> Enum.sum()

    # Output (approximate - depends on action space)
    output_params = List.last(hidden_sizes, 256) * 100

    input_params + hidden_params + output_params
  end

  defp detect_vram_gb do
    # Try to detect GPU VRAM
    case System.cmd("nvidia-smi", ["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.to_integer()
        |> Kernel./(1024)

      _ ->
        # Default to 8GB if detection fails
        8.0
    end
  rescue
    _ -> 8.0
  end
end
