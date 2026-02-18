defmodule ExPhil.Test.Helpers do
  @moduledoc """
  Common test helpers and assertions.

  Usage:
    import ExPhil.Test.Helpers

    # Assert tensor shapes
    assert_tensor_shape(tensor, {32, 64})

    # Assert tensor values are in range
    assert_tensor_in_range(tensor, 0.0, 1.0)

    # Create temporary directory for test
    with_temp_dir(fn dir -> ... end)
  """

  import ExUnit.Assertions

  # ============================================================================
  # Tensor Assertions
  # ============================================================================

  @doc """
  Assert that a tensor has the expected shape.
  """
  def assert_tensor_shape(tensor, expected_shape) do
    actual_shape = Nx.shape(tensor)

    assert actual_shape == expected_shape,
           "Expected tensor shape #{inspect(expected_shape)}, got #{inspect(actual_shape)}"
  end

  @doc """
  Assert that all tensor values are within a range.
  """
  def assert_tensor_in_range(tensor, min_val, max_val) do
    min_actual = Nx.reduce_min(tensor) |> Nx.to_number()
    max_actual = Nx.reduce_max(tensor) |> Nx.to_number()

    assert min_actual >= min_val,
           "Tensor min #{min_actual} is below expected min #{min_val}"

    assert max_actual <= max_val,
           "Tensor max #{max_actual} is above expected max #{max_val}"
  end

  @doc """
  Assert that a tensor is finite (no NaN or Inf).
  """
  def assert_tensor_finite(tensor) do
    # Check for NaN
    has_nan = Nx.is_nan(tensor) |> Nx.any() |> Nx.to_number() == 1
    refute has_nan, "Tensor contains NaN values"

    # Check for Inf
    has_inf = Nx.is_infinity(tensor) |> Nx.any() |> Nx.to_number() == 1
    refute has_inf, "Tensor contains Inf values"
  end

  @doc """
  Assert that two tensors are approximately equal.
  """
  def assert_tensors_close(tensor1, tensor2, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-5)

    assert Nx.shape(tensor1) == Nx.shape(tensor2),
           "Tensor shapes don't match: #{inspect(Nx.shape(tensor1))} vs #{inspect(Nx.shape(tensor2))}"

    diff = Nx.abs(Nx.subtract(tensor1, tensor2))
    threshold = Nx.add(atol, Nx.multiply(rtol, Nx.abs(tensor2)))
    all_close = Nx.all(Nx.less_equal(diff, threshold)) |> Nx.to_number() == 1

    assert all_close, "Tensors are not close within atol=#{atol}, rtol=#{rtol}"
  end

  @doc """
  Assert that tensor dtype matches expected type.
  """
  def assert_tensor_type(tensor, expected_type) do
    actual_type = Nx.type(tensor)

    assert actual_type == expected_type,
           "Expected tensor type #{inspect(expected_type)}, got #{inspect(actual_type)}"
  end

  # ============================================================================
  # File System Helpers
  # ============================================================================

  @doc """
  Execute a function with a temporary directory, cleaning up afterward.

  ## Example

      with_temp_dir(fn dir ->
        path = Path.join(dir, "test.txt")
        File.write!(path, "hello")
        assert File.exists?(path)
      end)
  """
  def with_temp_dir(fun) do
    dir = System.tmp_dir!()
    unique_dir = Path.join(dir, "exphil_test_#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(unique_dir)

    try do
      fun.(unique_dir)
    after
      File.rm_rf!(unique_dir)
    end
  end

  @doc """
  Create a temporary file with content and return its path.
  Caller is responsible for cleanup.
  """
  def temp_file(content, opts \\ []) do
    extension = Keyword.get(opts, :extension, ".tmp")
    dir = System.tmp_dir!()
    filename = "exphil_test_#{:erlang.unique_integer([:positive])}#{extension}"
    path = Path.join(dir, filename)
    File.write!(path, content)
    path
  end

  # ============================================================================
  # Timing Helpers
  # ============================================================================

  @doc """
  Measure execution time of a function in milliseconds.

  Returns `{result, time_ms}`.
  """
  def timed(fun) do
    {time_us, result} = :timer.tc(fun)
    {result, time_us / 1000}
  end

  @doc """
  Assert that a function completes within a time limit.

  ## Example

      assert_completes_within(1000, fn ->
        some_fast_operation()
      end)
  """
  def assert_completes_within(max_ms, fun) do
    {result, time_ms} = timed(fun)

    assert time_ms <= max_ms,
           "Operation took #{Float.round(time_ms, 1)}ms, expected <= #{max_ms}ms"

    result
  end

  # ============================================================================
  # Process Helpers
  # ============================================================================

  @doc """
  Capture all messages sent to the current process within a timeout.
  """
  def capture_messages(timeout_ms \\ 100) do
    capture_messages_loop([], timeout_ms)
  end

  defp capture_messages_loop(acc, timeout) do
    receive do
      msg -> capture_messages_loop([msg | acc], timeout)
    after
      timeout -> Enum.reverse(acc)
    end
  end

  # ============================================================================
  # Async Helpers
  # ============================================================================

  @doc """
  Wait for a condition to become true, with timeout.

  ## Example

      wait_until(fn -> File.exists?(path) end, timeout: 5000)
  """
  def wait_until(condition_fn, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5000)
    interval = Keyword.get(opts, :interval, 50)
    deadline = System.monotonic_time(:millisecond) + timeout

    wait_until_loop(condition_fn, interval, deadline)
  end

  defp wait_until_loop(condition_fn, interval, deadline) do
    if condition_fn.() do
      :ok
    else
      now = System.monotonic_time(:millisecond)

      if now >= deadline do
        flunk("Condition not met within timeout")
      else
        Process.sleep(interval)
        wait_until_loop(condition_fn, interval, deadline)
      end
    end
  end

  # ============================================================================
  # Model Testing Helpers
  # ============================================================================

  @doc """
  Build and initialize an Axon model for testing.

  Returns `{params, predict_fn}`.
  """
  def build_and_init_model(model, input_shape) do
    {init_fn, predict_fn} = Axon.build(model)
    template = Nx.template(input_shape, :f32)
    params = init_fn.(template, Axon.ModelState.empty())
    {params, predict_fn}
  end

  @doc """
  Count the number of parameters in a model.
  """
  def count_params(params) when is_struct(params, Axon.ModelState) do
    count_params(params.data)
  end

  def count_params(params) when is_map(params) do
    params
    |> Enum.map(fn {_key, value} ->
      case value do
        %Nx.Tensor{} = t -> Nx.size(t)
        map when is_map(map) -> count_params(map)
        _ -> 0
      end
    end)
    |> Enum.sum()
  end

  # ============================================================================
  # Flaky Test Helpers
  # ============================================================================

  @doc """
  Retry a test function multiple times until it succeeds.

  Use this for tests that are known to be flaky due to timing,
  network conditions, or other non-deterministic factors.

  ## Options

    - `:retries` - Number of retry attempts (default: 3)
    - `:delay` - Delay in ms between retries (default: 100)
    - `:log` - Whether to log retry attempts (default: false)

  ## Example

      test "flaky network operation" do
        retry(retries: 3, delay: 200) do
          result = NetworkClient.fetch()
          assert result.status == 200
        end
      end

  ## Best Practices

  1. Investigate the root cause first - retries are a workaround
  2. Tag flaky tests with `@tag :flaky` for tracking
  3. Keep retry counts low (2-3 max)
  4. Consider if the test can be made deterministic instead
  """
  defmacro retry(opts \\ [], do: block) do
    quote do
      retries = Keyword.get(unquote(opts), :retries, 3)
      delay = Keyword.get(unquote(opts), :delay, 100)
      log? = Keyword.get(unquote(opts), :log, false)

      ExPhil.Test.Helpers.do_retry(retries, delay, log?, fn ->
        unquote(block)
      end)
    end
  end

  @doc false
  def do_retry(retries, delay, log?, fun, attempt \\ 1) do
    try do
      fun.()
    rescue
      e in [ExUnit.AssertionError] ->
        if attempt < retries do
          if log? do
            IO.puts("[Flaky retry] Attempt #{attempt}/#{retries} failed: #{Exception.message(e)}")
          end

          Process.sleep(delay)
          do_retry(retries, delay, log?, fun, attempt + 1)
        else
          reraise e, __STACKTRACE__
        end
    end
  end

  @doc """
  Run a function and return whether it succeeded (for conditional logic).

  ## Example

      if eventually_succeeds?(fn -> check_resource_available() end, retries: 5) do
        # proceed with test
      else
        # skip or handle gracefully
      end
  """
  def eventually_succeeds?(fun, opts \\ []) do
    retries = Keyword.get(opts, :retries, 3)
    delay = Keyword.get(opts, :delay, 100)

    Enum.reduce_while(1..retries, false, fn attempt, _acc ->
      try do
        fun.()
        {:halt, true}
      rescue
        _ ->
          if attempt < retries, do: Process.sleep(delay)
          {:cont, false}
      end
    end)
  end

  # ============================================================================
  # Benchmark Helpers
  # ============================================================================

  @benchmark_baseline_path "test/fixtures/benchmark_baselines.json"

  @doc """
  Run a benchmark and compare against stored baseline.

  ## Options

    - `:name` - Benchmark name for tracking (required)
    - `:warmup` - Number of warmup iterations (default: 3)
    - `:iterations` - Number of measured iterations (default: 10)
    - `:tolerance` - Allowed percentage regression (default: 0.20 = 20%)

  ## Example

      benchmark(name: "policy_inference", warmup: 5, iterations: 20) do
        predict_fn.(params, input)
      end

  Returns `{:ok, %{mean: ms, median: ms, min: ms, max: ms}}` or raises on regression.
  """
  defmacro benchmark(opts, do: block) do
    quote do
      name = Keyword.fetch!(unquote(opts), :name)
      warmup = Keyword.get(unquote(opts), :warmup, 3)
      iterations = Keyword.get(unquote(opts), :iterations, 10)
      tolerance = Keyword.get(unquote(opts), :tolerance, 0.20)

      ExPhil.Test.Helpers.run_benchmark(name, warmup, iterations, tolerance, fn ->
        unquote(block)
      end)
    end
  end

  @doc false
  def run_benchmark(name, warmup, iterations, tolerance, fun) do
    # Warmup
    for _ <- 1..warmup, do: fun.()

    # Measure
    times =
      for _ <- 1..iterations do
        {time_us, _result} = :timer.tc(fun)
        # Convert to ms
        time_us / 1000
      end

    stats = %{
      mean: Enum.sum(times) / length(times),
      median: median(times),
      min: Enum.min(times),
      max: Enum.max(times),
      stddev: stddev(times)
    }

    # Update baseline FIRST if requested (before regression check,
    # otherwise flunk() prevents the update from ever running)
    if System.get_env("BENCHMARK_UPDATE") do
      update_baseline(name, stats)
    end

    # Check against baseline (skip if we just updated)
    unless System.get_env("BENCHMARK_UPDATE") do
      baselines = load_baselines()
      baseline = Map.get(baselines, name)

      if baseline do
        regression = (stats.mean - baseline["mean"]) / baseline["mean"]

        if regression > tolerance do
          flunk("""
          Performance regression detected for '#{name}'!

          Baseline: #{Float.round(baseline["mean"], 2)}ms (±#{Float.round(baseline["stddev"], 2)}ms)
          Current:  #{Float.round(stats.mean, 2)}ms (±#{Float.round(stats.stddev, 2)}ms)
          Regression: #{Float.round(regression * 100, 1)}% (tolerance: #{Float.round(tolerance * 100, 1)}%)

          Run with BENCHMARK_UPDATE=1 to update baselines if this is expected.
          """)
        end
      end
    end

    {:ok, stats}
  end

  defp median(list) do
    sorted = Enum.sort(list)
    len = length(sorted)
    mid = div(len, 2)

    if rem(len, 2) == 0 do
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
    else
      Enum.at(sorted, mid)
    end
  end

  defp stddev(list) do
    mean = Enum.sum(list) / length(list)
    variance = Enum.sum(Enum.map(list, fn x -> (x - mean) * (x - mean) end)) / length(list)
    :math.sqrt(variance)
  end

  defp load_baselines do
    case File.read(@benchmark_baseline_path) do
      {:ok, content} -> Jason.decode!(content)
      {:error, _} -> %{}
    end
  end

  defp update_baseline(name, stats) do
    baselines = load_baselines()

    updated =
      Map.put(baselines, name, %{
        "mean" => stats.mean,
        "median" => stats.median,
        "min" => stats.min,
        "max" => stats.max,
        "stddev" => stats.stddev,
        "updated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
      })

    File.mkdir_p!(Path.dirname(@benchmark_baseline_path))
    File.write!(@benchmark_baseline_path, Jason.encode!(updated, pretty: true))

    IO.puts("[Benchmark] Updated baseline for '#{name}': #{Float.round(stats.mean, 2)}ms")
  end
end
