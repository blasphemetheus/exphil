defmodule ExPhil.Training.GPUUtils do
  @moduledoc """
  GPU utilities for training - memory tracking, device info, etc.

  Uses nvidia-smi for NVIDIA GPU information since EXLA doesn't expose
  memory APIs directly.
  """

  @doc """
  Get GPU memory usage information via nvidia-smi.

  Returns `{:ok, %{used_mb: int, total_mb: int, free_mb: int, utilization: int}}` on success,
  or `{:error, reason}` if GPU info is unavailable.

  ## Examples

      iex> GPUUtils.get_memory_info()
      {:ok, %{used_mb: 2048, total_mb: 8192, free_mb: 6144, utilization: 45}}

      iex> GPUUtils.get_memory_info()
      {:error, :nvidia_smi_not_found}
  """
  @spec get_memory_info(non_neg_integer()) :: {:ok, map()} | {:error, atom()}
  def get_memory_info(device_id \\ 0) do
    # Query nvidia-smi for memory info
    # Format: memory.used, memory.total, memory.free, utilization.gpu
    query = "memory.used,memory.total,memory.free,utilization.gpu"

    case System.cmd("nvidia-smi", [
      "--query-gpu=#{query}",
      "--format=csv,noheader,nounits",
      "-i", to_string(device_id)
    ], stderr_to_stdout: true) do
      {output, 0} ->
        parse_nvidia_smi_output(output)
      {_, _} ->
        {:error, :nvidia_smi_failed}
    end
  rescue
    ErlangError -> {:error, :nvidia_smi_not_found}
  end

  defp parse_nvidia_smi_output(output) do
    output
    |> String.trim()
    |> String.split(",")
    |> Enum.map(&String.trim/1)
    |> case do
      [used, total, free, util] ->
        {:ok, %{
          used_mb: String.to_integer(used),
          total_mb: String.to_integer(total),
          free_mb: String.to_integer(free),
          utilization: parse_utilization(util)
        }}
      _ ->
        {:error, :parse_failed}
    end
  rescue
    _ -> {:error, :parse_failed}
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
  @spec device_name(non_neg_integer()) :: {:ok, String.t()} | {:error, atom()}
  def device_name(device_id \\ 0) do
    case System.cmd("nvidia-smi", [
      "--query-gpu=name",
      "--format=csv,noheader",
      "-i", to_string(device_id)
    ], stderr_to_stdout: true) do
      {name, 0} -> {:ok, String.trim(name)}
      _ -> {:error, :nvidia_smi_failed}
    end
  rescue
    _ -> {:error, :nvidia_smi_not_found}
  end

  @doc """
  Convenience alias for `get_memory_info/1` that returns simplified status.

  Returns `{:ok, %{used_mb: int, total_mb: int, utilization: float}}` on success,
  where utilization is a fraction (0.0 to 1.0).

  ## Examples

      iex> GPUUtils.memory_status()
      {:ok, %{used_mb: 4521, total_mb: 24564, utilization: 0.18}}
  """
  @spec memory_status(non_neg_integer()) :: {:ok, map()} | {:error, atom()}
  def memory_status(device_id \\ 0) do
    case get_memory_info(device_id) do
      {:ok, %{used_mb: used, total_mb: total, utilization: util}} ->
        {:ok, %{
          used_mb: used,
          total_mb: total,
          utilization: util / 100
        }}

      error ->
        error
    end
  end

  @doc """
  Get all GPU devices with their info.
  """
  @spec list_devices() :: {:ok, [map()]} | {:error, atom()}
  def list_devices do
    case System.cmd("nvidia-smi", [
      "--query-gpu=index,name,memory.total",
      "--format=csv,noheader,nounits"
    ], stderr_to_stdout: true) do
      {output, 0} ->
        devices = output
        |> String.trim()
        |> String.split("\n")
        |> Enum.map(fn line ->
          case String.split(line, ",") |> Enum.map(&String.trim/1) do
            [idx, name, mem] ->
              %{index: String.to_integer(idx), name: name, memory_mb: String.to_integer(mem)}
            _ -> nil
          end
        end)
        |> Enum.reject(&is_nil/1)

        {:ok, devices}
      _ ->
        {:error, :nvidia_smi_failed}
    end
  rescue
    _ -> {:error, :nvidia_smi_not_found}
  end
end
