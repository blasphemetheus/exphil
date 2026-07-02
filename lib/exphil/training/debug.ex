defmodule ExPhil.Training.Debug do
  @moduledoc """
  JIT debugging utilities using Nx.Defn.Kernel.hook and print_value.

  These tools let you inspect tensor values inside JIT-compiled functions
  without breaking the computation graph.

  ## Usage

      # Print intermediate values during JIT execution
      Debug.print_tensor(tensor, label: "after_forward")

      # Hook for custom inspection
      Debug.hook_gradient_norm(grads, "layer1")

      # Enable debug mode for training
      Debug.enable()  # sets process flag
      Debug.enabled?() # check flag

  ## In Loss Functions

  Add debug prints inside defn functions:

      defn my_loss(logits, targets) do
        loss = Nx.mean(Nx.subtract(logits, targets) |> Nx.pow(2))
        loss = if Debug.enabled?(), do: print_value(loss, label: "loss"), else: loss
        loss
      end

  Note: `print_value` and `hook` are Nx.Defn.Kernel functions — they only
  work inside `defn` or JIT-compiled functions.
  """

  @doc """
  Enable debug tensor printing. Sets a process-level flag.
  """
  def enable do
    Process.put(:exphil_debug_tensors, true)
    :ok
  end

  @doc """
  Disable debug tensor printing.
  """
  def disable do
    Process.put(:exphil_debug_tensors, false)
    :ok
  end

  @doc """
  Check if debug mode is enabled.
  """
  def enabled? do
    Process.get(:exphil_debug_tensors, false) == true or
      System.get_env("EXPHIL_DEBUG_TENSORS") == "1"
  end

  @doc """
  Print a summary of a tensor (shape, type, min, max, mean, has_nan).
  Works outside JIT — for inspecting tensors between steps.

      Debug.inspect_tensor(loss, "train_loss")
  """
  def inspect_tensor(tensor, label \\ "tensor") do
    shape = Nx.shape(tensor)
    type = Nx.type(tensor)
    flat = Nx.flatten(tensor)

    min_val = Nx.reduce_min(flat) |> Nx.to_number()
    max_val = Nx.reduce_max(flat) |> Nx.to_number()
    mean_val = Nx.mean(flat) |> Nx.to_number()

    has_nan = Nx.any(Nx.is_nan(flat)) |> Nx.to_number() > 0
    has_inf = Nx.any(Nx.is_infinity(flat)) |> Nx.to_number() > 0

    status = cond do
      has_nan -> " [NaN!]"
      has_inf -> " [Inf!]"
      true -> ""
    end

    IO.puts("[DEBUG] #{label}: shape=#{inspect(shape)} type=#{inspect(type)} " <>
            "min=#{Float.round(min_val * 1.0, 4)} max=#{Float.round(max_val * 1.0, 4)} " <>
            "mean=#{Float.round(mean_val * 1.0, 4)}#{status}")
  end

  @doc """
  Inspect gradient norms for each parameter group.
  Works outside JIT.

      Debug.inspect_gradients(grads)
  """
  def inspect_gradients(grads) do
    IO.puts("[DEBUG] Gradient norms:")
    flatten_and_inspect(grads, "")
  end

  defp flatten_and_inspect(%Nx.Tensor{} = t, path) do
    norm = t |> Nx.flatten() |> Nx.LinAlg.norm() |> Nx.to_number()
    has_nan = Nx.any(Nx.is_nan(t)) |> Nx.to_number() > 0
    status = if has_nan, do: " [NaN!]", else: ""
    IO.puts("  #{path}: norm=#{Float.round(norm, 6)}#{status}")
  end

  defp flatten_and_inspect(map, path) when is_map(map) and not is_struct(map) do
    Enum.each(map, fn {k, v} ->
      new_path = if path == "", do: to_string(k), else: "#{path}.#{k}"
      flatten_and_inspect(v, new_path)
    end)
  end

  defp flatten_and_inspect(_, _), do: :ok

  @doc """
  Run a training step with full debug output.
  Prints: input shapes, output loss, gradient norms, param update norms.

      Debug.debug_step(trainer, batch)
  """
  def debug_step(trainer, batch) do
    IO.puts("\n[DEBUG] === Training Step Debug ===")

    # Input shapes
    IO.puts("[DEBUG] Input shapes:")
    IO.puts("  states: #{inspect(Nx.shape(batch.states))}")
    IO.puts("  buttons: #{inspect(Nx.shape(batch.actions.buttons))}")

    # Forward + backward
    alias ExPhil.Training.Imitation.TrainLoop
    {grads, loss} = TrainLoop.compute_gradients(trainer, batch)

    IO.puts("[DEBUG] Loss: #{loss}")
    inspect_gradients(grads)

    # Check param values
    IO.puts("[DEBUG] Param norms:")
    alias ExPhil.Training.Imitation.TrainLoop
    params = TrainLoop.get_params_data(trainer.policy_params)
    flatten_and_inspect(params, "")

    IO.puts("[DEBUG] === End Debug ===\n")

    {grads, loss}
  end
end
