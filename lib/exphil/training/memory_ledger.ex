defmodule ExPhil.Training.MemoryLedger do
  @moduledoc """
  Bytes-per-frame RAM scaling model for training runs.

  Each training run (and each `--preflight`) appends one JSONL entry to
  the ledger: pool size, embed/window config, and the process's peak RSS
  (`VmHWM` — the kernel's high-water mark, which survives transient
  spikes no sampler would catch). `predict/2` fits peak-RSS-vs-pool-frames
  over past entries so a run can refuse a pool that won't fit BEFORE
  paying the parse + embed + JIT cost.

  Built after r16 (2026-07-22): the run died at epoch 10 and the OOM
  theory was unfalsifiable because no run had ever recorded its actual
  peak. With two ledger points at different pool sizes the model answers
  "how far can the pool scale on this box" in closed form.

  Ledger entry fields (all optional except `pool_frames` and
  `peak_rss_bytes`): `mode` ("preflight" | "train"), `tag`, `pool_frames`,
  `embed_size`, `window`, `backbone`, `peak_rss_bytes`, `gpu_used_mb`,
  `recorded_at`.
  """

  @default_path "logs/memory_ledger.jsonl"

  # Predicted peak must fit inside headroom with this margin — covers
  # allocator fragmentation and the train-mode peak sitting above a
  # preflight-mode measurement.
  @safety_margin 1.2

  def default_path, do: System.get_env("EXPHIL_MEMORY_LEDGER") || @default_path

  # ============================================================================
  # Measurement
  # ============================================================================

  @doc """
  Current and peak RSS of this OS process, from `/proc/self/status`.

  Returns `%{rss_bytes: n, hwm_bytes: n}`. `hwm_bytes` (VmHWM) is the
  lifetime high-water mark — read it at exit to get the true peak even
  if the spike lasted one allocation.
  """
  def process_memory do
    parse_proc_status(File.read!("/proc/self/status"))
  end

  @doc false
  def parse_proc_status(text) do
    %{
      rss_bytes: proc_kb(text, "VmRSS") * 1024,
      hwm_bytes: proc_kb(text, "VmHWM") * 1024
    }
  end

  @doc """
  System memory headroom, from `/proc/meminfo`.

  Returns `%{available_bytes: n, total_bytes: n, swap_free_bytes: n}`.
  """
  def system_memory do
    parse_meminfo(File.read!("/proc/meminfo"))
  end

  @doc false
  def parse_meminfo(text) do
    %{
      available_bytes: proc_kb(text, "MemAvailable") * 1024,
      total_bytes: proc_kb(text, "MemTotal") * 1024,
      swap_free_bytes: proc_kb(text, "SwapFree") * 1024
    }
  end

  defp proc_kb(text, key) do
    case Regex.run(~r/^#{key}:\s+(\d+)\s+kB/m, text) do
      [_, kb] -> String.to_integer(kb)
      nil -> 0
    end
  end

  @doc "Human-readable byte count (\"12.3 GB\")."
  def format_bytes(nil), do: "?"

  def format_bytes(bytes) when is_number(bytes) do
    cond do
      bytes >= 1_000_000_000 -> "#{Float.round(bytes / 1.0e9, 1)} GB"
      bytes >= 1_000_000 -> "#{Float.round(bytes / 1.0e6, 1)} MB"
      true -> "#{bytes} B"
    end
  end

  # ============================================================================
  # Ledger
  # ============================================================================

  @doc "Append an entry (map) to the ledger; stamps `recorded_at`."
  def append(entry, path \\ default_path()) do
    File.mkdir_p!(Path.dirname(path))

    line =
      entry
      |> Map.put(:recorded_at, DateTime.utc_now() |> DateTime.to_iso8601())
      |> Jason.encode!()

    File.write!(path, line <> "\n", [:append])
  end

  @doc "All ledger entries (atom keys); missing file or bad lines -> skipped."
  def entries(path \\ default_path()) do
    case File.read(path) do
      {:ok, text} ->
        text
        |> String.split("\n", trim: true)
        |> Enum.flat_map(fn line ->
          case Jason.decode(line, keys: :atoms) do
            {:ok, %{pool_frames: f, peak_rss_bytes: b} = e}
            when is_number(f) and is_number(b) ->
              [e]

            _ ->
              []
          end
        end)

      {:error, _} ->
        []
    end
  end

  # ============================================================================
  # Model
  # ============================================================================

  @doc """
  Predict peak RSS for a pool of `pool_frames` frames.

  Options: `:path` (ledger), plus entry filters `:window`, `:embed_size`,
  `:backbone` — an entry matches a filter if the field is equal OR the
  entry predates the field (nil), so early ledgers stay usable.

  Returns `{:ok, %{predicted_bytes, model, points, slope_bytes_per_frame,
  intercept_bytes}}` or `{:error, :no_data}`. Model is `:linear_fit`
  (>= 2 distinct pool sizes) or `:proportional` (1 point, scaled through
  the origin — safe for extrapolating UP, the direction that OOMs).
  """
  def predict(pool_frames, opts \\ []) do
    points =
      opts
      |> Keyword.get(:path, default_path())
      |> entries()
      |> Enum.filter(&matches_filters?(&1, opts))
      |> Enum.map(fn e -> {e.pool_frames, e.peak_rss_bytes} end)

    case fit(points) do
      {:ok, slope, intercept, model} ->
        {:ok,
         %{
           predicted_bytes: round(slope * pool_frames + intercept),
           model: model,
           points: length(points),
           slope_bytes_per_frame: Float.round(slope, 1),
           intercept_bytes: round(intercept)
         }}

      {:error, :no_data} ->
        {:error, :no_data}
    end
  end

  @doc false
  # Least squares over [{frames, bytes}]. One point -> proportional.
  def fit([]), do: {:error, :no_data}

  def fit(points) do
    distinct_x = points |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> length()

    if distinct_x < 2 do
      # All points at one pool size: slope through the origin from the
      # WORST observed peak (safe direction for scaling up).
      {frames, bytes} = Enum.max_by(points, &elem(&1, 1))
      {:ok, bytes / max(frames, 1), 0.0, :proportional}
    else
      n = length(points)
      {xs, ys} = Enum.unzip(points)
      mx = Enum.sum(xs) / n
      my = Enum.sum(ys) / n

      sxy = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end) |> Enum.sum()
      sxx = xs |> Enum.map(fn x -> (x - mx) * (x - mx) end) |> Enum.sum()

      slope = sxy / sxx
      {:ok, slope, my - slope * mx, :linear_fit}
    end
  end

  defp matches_filters?(entry, opts) do
    Enum.all?([:window, :embed_size, :backbone], fn key ->
      case Keyword.get(opts, key) do
        nil -> true
        want -> entry[key] == nil or to_string(entry[key]) == to_string(want)
      end
    end)
  end

  # ============================================================================
  # Headroom check
  # ============================================================================

  @doc """
  Compare the predicted peak for `pool_frames` against this process's
  memory budget (current RSS + MemAvailable — the predicted peak already
  contains what we hold now). Swap is deliberately NOT counted: a
  training run pushed into swap is already lost.

  Returns `{:ok | :warn | :no_data, info}` where info carries
  `predicted_bytes`, `budget_bytes`, `margin` and the model fields.
  A 1.2x safety margin covers fragmentation and preflight-vs-train gap.
  """
  def headroom_check(pool_frames, opts \\ []) do
    case predict(pool_frames, opts) do
      {:error, :no_data} ->
        {:no_data, %{}}

      {:ok, model} ->
        budget = process_memory().rss_bytes + system_memory().available_bytes
        info = Map.merge(model, %{budget_bytes: budget, margin: @safety_margin})

        if model.predicted_bytes * @safety_margin <= budget do
          {:ok, info}
        else
          {:warn, info}
        end
    end
  end
end
