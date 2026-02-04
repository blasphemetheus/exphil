defmodule ExPhil.Training.GPUUtils do
  @moduledoc """
  GPU utilities for training - memory tracking, device info, etc.

  Uses nvidia-smi for NVIDIA GPU information since EXLA doesn't expose
  memory APIs directly.

  ## Error Handling

  Returns structured `ExPhil.Error.GPUError` on failure:

      case GPUUtils.get_memory_info() do
        {:ok, info} -> info.used_mb
        {:error, %GPUError{reason: :not_found}} ->
          Logger.warning("No GPU detected, using CPU")
      end

  """

  alias ExPhil.Error.GPUError

  @doc """
  Get GPU memory usage information via nvidia-smi.

  Returns `{:ok, %{used_mb: int, total_mb: int, free_mb: int, utilization: int}}` on success,
  or `{:error, reason}` if GPU info is unavailable.

  ## Examples

      iex> GPUUtils.get_memory_info()
      {:ok, %{used_mb: 2048, total_mb: 8192, free_mb: 6144, utilization: 45}}

      iex> GPUUtils.get_memory_info()
      {:error, %GPUError{reason: :not_found}}
  """
  @spec get_memory_info(non_neg_integer()) :: {:ok, map()} | {:error, GPUError.t()}
  def get_memory_info(device_id \\ 0) do
    # Query nvidia-smi for memory info
    # Format: memory.used, memory.total, memory.free, utilization.gpu
    query = "memory.used,memory.total,memory.free,utilization.gpu"

    case System.cmd(
           "nvidia-smi",
           [
             "--query-gpu=#{query}",
             "--format=csv,noheader,nounits",
             "-i",
             to_string(device_id)
           ],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        parse_nvidia_smi_output(output)

      {_, _} ->
        {:error, GPUError.new(:nvidia_smi_failed)}
    end
  rescue
    ErlangError -> {:error, GPUError.new(:not_found)}
  end

  defp parse_nvidia_smi_output(output) do
    output
    |> String.trim()
    |> String.split(",")
    |> Enum.map(&String.trim/1)
    |> case do
      [used, total, free, util] ->
        {:ok,
         %{
           used_mb: String.to_integer(used),
           total_mb: String.to_integer(total),
           free_mb: String.to_integer(free),
           utilization: parse_utilization(util)
         }}

      _ ->
        {:error, GPUError.new(:nvidia_smi_failed, context: %{details: "parse failed"})}
    end
  rescue
    _ -> {:error, GPUError.new(:nvidia_smi_failed, context: %{details: "parse failed"})}
  end

  defp parse_utilization(util) do
    case Integer.parse(util) do
      {n, _} -> n
      :error -> 0
    end
  end

  @doc """
  Format memory size in human-readable format.

  ## Examples

      iex> GPUUtils.format_mb(2048)
      "2.00 GB"

      iex> GPUUtils.format_mb(512)
      "512 MB"
  """
  @spec format_mb(non_neg_integer()) :: String.t()
  def format_mb(mb) when is_integer(mb) and mb >= 1024 do
    gb = mb / 1024
    :io_lib.format("~.2f GB", [gb]) |> IO.iodata_to_binary()
  end

  def format_mb(mb) when is_integer(mb) do
    "#{mb} MB"
  end

  @doc """
  Get a formatted string of GPU memory usage.

  Returns a string like "GPU: 2.50/8.00 GB (31%) | Util: 45%" or
  "GPU: N/A" if unavailable.
  """
  @spec memory_status_string(non_neg_integer()) :: String.t()
  def memory_status_string(device_id \\ 0) do
    case get_memory_info(device_id) do
      {:ok, %{used_mb: used, total_mb: total, utilization: util}} ->
        pct = round(used / total * 100)
        "GPU: #{format_mb(used)}/#{format_mb(total)} (#{pct}%) | Util: #{util}%"

      {:error, _} ->
        "GPU: N/A (CPU mode)"
    end
  end

  @doc """
  Check if NVIDIA GPU is available (via nvidia-smi).
  """
  @spec gpu_available?() :: boolean()
  def gpu_available? do
    case System.cmd("nvidia-smi", ["-L"], stderr_to_stdout: true) do
      {_, 0} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  @doc """
  Get GPU device name if available.
  """
  @spec device_name(non_neg_integer()) :: {:ok, String.t()} | {:error, GPUError.t()}
  def device_name(device_id \\ 0) do
    case System.cmd(
           "nvidia-smi",
           [
             "--query-gpu=name",
             "--format=csv,noheader",
             "-i",
             to_string(device_id)
           ],
           stderr_to_stdout: true
         ) do
      {name, 0} -> {:ok, String.trim(name)}
      _ -> {:error, GPUError.new(:nvidia_smi_failed)}
    end
  rescue
    _ -> {:error, GPUError.new(:not_found)}
  end

  @doc """
  Convenience alias for `get_memory_info/1` that returns simplified status.

  Returns `{:ok, %{used_mb: int, total_mb: int, utilization: float}}` on success,
  where utilization is a fraction (0.0 to 1.0).

  ## Examples

      iex> GPUUtils.memory_status()
      {:ok, %{used_mb: 4521, total_mb: 24564, utilization: 0.18}}
  """
  @spec memory_status(non_neg_integer()) :: {:ok, map()} | {:error, GPUError.t()}
  def memory_status(device_id \\ 0) do
    case get_memory_info(device_id) do
      {:ok, %{used_mb: used, total_mb: total, utilization: util}} ->
        {:ok,
         %{
           used_mb: used,
           total_mb: total,
           utilization: util / 100
         }}

      error ->
        error
    end
  end

  @doc """
  Check GPU memory and return warning if usage is above threshold.

  Returns `{:warning, message}` if memory usage exceeds threshold,
  `:ok` if memory is fine, or `{:error, reason}` if GPU info unavailable.

  ## Options
  - `:threshold` - Memory usage fraction to warn at (default: 0.85 = 85%)

  ## Examples

      iex> GPUUtils.check_memory_warning()
      :ok

      iex> GPUUtils.check_memory_warning(threshold: 0.8)
      {:warning, "GPU memory usage is high: 7.2/8.0 GB (90%). Consider reducing batch size."}
  """
  @spec check_memory_warning(keyword()) :: :ok | {:warning, String.t()} | {:error, GPUError.t()}
  def check_memory_warning(opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.85)

    case get_memory_info() do
      {:ok, %{used_mb: used, total_mb: total}} ->
        usage_pct = used / total

        if usage_pct >= threshold do
          used_str = format_mb(used)
          total_str = format_mb(total)
          pct_str = round(usage_pct * 100)

          {:warning,
           "GPU memory usage is high: #{used_str}/#{total_str} (#{pct_str}%). " <>
             "Consider reducing batch size or using --gradient-checkpoint."}
        else
          :ok
        end

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Check GPU memory and return warning if free memory is below requirement.

  Useful before training to warn if the model might not fit.

  ## Options
  - `:required_mb` - Minimum free memory in MB

  ## Examples

      iex> GPUUtils.check_free_memory(required_mb: 4000)
      {:warning, "Only 2.5 GB free, may need 4.0 GB. Consider closing other GPU processes."}
  """
  @spec check_free_memory(keyword()) :: :ok | {:warning, String.t()} | {:error, GPUError.t()}
  def check_free_memory(opts \\ []) do
    required_mb = Keyword.get(opts, :required_mb, 0)

    case get_memory_info() do
      {:ok, %{free_mb: free}} ->
        if free < required_mb do
          {:warning,
           "Only #{format_mb(free)} free, may need #{format_mb(required_mb)}. " <>
             "Consider closing other GPU processes or reducing model size."}
        else
          :ok
        end

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Estimate GPU memory requirement for training based on model config.

  Returns estimated memory in MB. This is a rough estimate based on:
  - Parameter count (4 bytes per f32, 2 bytes per bf16)
  - Optimizer state (2x parameters for Adam)
  - Activation memory (depends on batch size and sequence length)
  - Gradient buffers (1x parameters)

  ## Examples

      iex> GPUUtils.estimate_memory_mb(
      ...>   param_count: 5_000_000,
      ...>   batch_size: 64,
      ...>   precision: :bf16
      ...> )
      2400
  """
  @spec estimate_memory_mb(keyword()) :: non_neg_integer()
  def estimate_memory_mb(opts \\ []) do
    param_count = Keyword.get(opts, :param_count, 1_000_000)
    batch_size = Keyword.get(opts, :batch_size, 64)
    precision = Keyword.get(opts, :precision, :bf16)
    temporal = Keyword.get(opts, :temporal, false)
    window_size = Keyword.get(opts, :window_size, 60)

    # Bytes per parameter
    bytes_per_param = if precision == :bf16, do: 2, else: 4

    # Parameters + optimizer state (Adam has 2 momentum tensors)
    param_memory = param_count * bytes_per_param * 3

    # Gradients
    gradient_memory = param_count * bytes_per_param

    # Activation memory (rough estimate)
    # Scales with batch size and sequence length for temporal models
    activation_multiplier = if temporal, do: window_size, else: 1
    activation_memory = batch_size * activation_multiplier * 10_000 * bytes_per_param

    total_bytes = param_memory + gradient_memory + activation_memory

    # Convert to MB and add 20% safety margin
    round(total_bytes / 1_048_576 * 1.2)
  end

  @doc """
  Estimate checkpoint file size based on parameter count.

  Returns estimated size in bytes. Checkpoints store:
  - Model parameters (4 bytes per f32 - saved as BinaryBackend)
  - Config/metadata (small)
  - Optimizer state (2x parameters for Adam)

  ## Examples

      iex> GPUUtils.estimate_checkpoint_size(param_count: 5_000_000)
      60_000_000  # ~60 MB
  """
  @spec estimate_checkpoint_size(keyword()) :: non_neg_integer()
  def estimate_checkpoint_size(opts \\ []) do
    param_count = Keyword.get(opts, :param_count, 1_000_000)
    include_optimizer = Keyword.get(opts, :include_optimizer, true)

    # Parameters are stored as f32 (4 bytes each)
    param_bytes = param_count * 4

    # Optimizer state (Adam has momentum + variance = 2x params)
    optimizer_bytes = if include_optimizer, do: param_count * 4 * 2, else: 0

    # Metadata overhead (~10KB)
    metadata_bytes = 10_000

    # Erlang term format overhead (~10%)
    total = param_bytes + optimizer_bytes + metadata_bytes
    round(total * 1.1)
  end

  @doc """
  Check if estimated checkpoint size exceeds a threshold and return warning.

  ## Options
  - `:param_count` - Number of model parameters
  - `:threshold_mb` - Size threshold in MB (default: 500 MB)

  ## Examples

      iex> GPUUtils.check_checkpoint_size_warning(param_count: 50_000_000)
      {:warning, "Estimated checkpoint size: 572 MB. Large checkpoints slow down saving/loading."}
  """
  @spec check_checkpoint_size_warning(keyword()) :: :ok | {:warning, String.t()}
  def check_checkpoint_size_warning(opts \\ []) do
    threshold_mb = Keyword.get(opts, :threshold_mb, 500)
    estimated_bytes = estimate_checkpoint_size(opts)
    estimated_mb = round(estimated_bytes / 1_048_576)

    if estimated_mb >= threshold_mb do
      {:warning,
       "Estimated checkpoint size: #{estimated_mb} MB. " <>
         "Large checkpoints slow down saving/loading. " <>
         "Consider using smaller hidden sizes or fewer layers."}
    else
      :ok
    end
  end

  @doc """
  Count parameters in a nested map of tensors.

  ## Examples

      iex> params = %{dense1: %{kernel: Nx.iota({512, 256}), bias: Nx.iota({256})}}
      iex> GPUUtils.count_params(params)
      131328
  """
  @spec count_params(map() | Nx.Tensor.t() | any()) :: non_neg_integer()
  def count_params(%Nx.Tensor{} = tensor), do: Nx.size(tensor)
  def count_params(%Axon.ModelState{data: data}), do: count_params(data)

  def count_params(params) when is_map(params) and not is_struct(params) do
    params
    |> Enum.reduce(0, fn {_key, value}, acc ->
      acc + count_params(value)
    end)
  end

  def count_params(_), do: 0

  @doc """
  Get all GPU devices with their info.
  """
  @spec list_devices() :: {:ok, [map()]} | {:error, GPUError.t()}
  def list_devices do
    case System.cmd(
           "nvidia-smi",
           [
             "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"
           ],
           stderr_to_stdout: true
         ) do
      {output, 0} ->
        devices =
          output
          |> String.trim()
          |> String.split("\n")
          |> Enum.map(fn line ->
            case String.split(line, ",") |> Enum.map(&String.trim/1) do
              [idx, name, mem] ->
                %{index: String.to_integer(idx), name: name, memory_mb: String.to_integer(mem)}

              _ ->
                nil
            end
          end)
          |> Enum.reject(&is_nil/1)

        {:ok, devices}

      _ ->
        {:error, GPUError.new(:nvidia_smi_failed)}
    end
  rescue
    _ -> {:error, GPUError.new(:not_found)}
  end
end
