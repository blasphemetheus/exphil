defmodule ExPhil.Bridge.BendPort do
  @moduledoc """
  Simple port wrapper for Bend (HVM2) linear scan.

  Bend has no C FFI or CUDA interop, so this uses a simple CLI interface:
  compile the Bend program, run it, parse stdout. This is for learning
  and exploration only — not practical for production use.

  ## Usage

      # Run tiny test case
      ExPhil.Bridge.BendPort.run_test()

      # Benchmark against pure Nx
      ExPhil.Bridge.BendPort.benchmark(batch, seq_len, hidden)

  ## Limitations

  - No binary tensor I/O (Bend has no f32 arrays)
  - Numbers are f24 (24-bit float), not IEEE 754
  - GPU backend (HVM2) is experimental
  - Dramatically slower than CUDA C for real workloads
  """

  require Logger

  @doc """
  Check if Bend is installed and available.
  """
  @spec available?() :: boolean()
  def available? do
    find_bend() != nil
  end

  @doc """
  Run the built-in test case (batch=1, seq_len=4, hidden=2).

  Returns raw output from Bend for manual inspection.
  """
  @spec run_test(keyword()) :: {:ok, String.t()} | {:error, String.t()}
  def run_test(opts \\ []) do
    backend = Keyword.get(opts, :backend, "rust")

    # Use project root (not _build) since native/ isn't copied to _build
    bend_file = Path.join([File.cwd!(), "native", "bend_scan", "linear_scan.bend"])

    bend_exe = find_bend()

    if bend_exe == nil do
      {:error, "bend not found in PATH or ~/.cargo/bin"}
    else
      run_cmd = case backend do
        "rust" -> "run-rs"
        "c" -> "run-c"
        "cuda" -> "run-cu"
        _other -> "run-rs"
      end

      case System.cmd(bend_exe, [run_cmd, bend_file], stderr_to_stdout: true) do
        {output, 0} -> {:ok, String.trim(output)}
        {output, code} -> {:error, "Exit code #{code}: #{String.trim(output)}"}
      end
    end
  end

  @doc """
  Benchmark Bend vs pure Nx for tiny sizes.

  Since Bend can't handle real tensor I/O, this compares:
  - Bend CLI execution time (compile + run)
  - Pure Nx sequential scan

  Only meaningful for understanding Bend's execution model,
  not for performance comparison against CUDA C.
  """
  @spec benchmark(pos_integer(), pos_integer(), pos_integer(), keyword()) :: map()
  def benchmark(batch, seq_len, hidden, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 5)

    # Bend timing (includes compilation overhead)
    bend_times =
      if available?() do
        for _ <- 1..iterations do
          t0 = System.monotonic_time(:microsecond)
          run_test(backend: "rust")
          System.monotonic_time(:microsecond) - t0
        end
      else
        []
      end

    # Nx timing for same small size
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    nx_times =
      for _ <- 1..iterations do
        t0 = System.monotonic_time(:microsecond)

        Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
          a_t = a_vals[[.., t, ..]]
          b_t = b_vals[[.., t, ..]]
          h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
          {h_new, [h_new | acc]}
        end)

        System.monotonic_time(:microsecond) - t0
      end

    bend_sorted = Enum.sort(bend_times)
    nx_sorted = Enum.sort(nx_times)

    %{
      bend_available: available?(),
      bend_median_us: if(bend_sorted != [], do: Enum.at(bend_sorted, div(length(bend_sorted), 2)), else: nil),
      bend_times: bend_times,
      nx_median_us: Enum.at(nx_sorted, div(length(nx_sorted), 2)),
      nx_times: nx_times
    }
  end

  # Check PATH first, then ~/.cargo/bin (where cargo install puts it)
  defp find_bend do
    System.find_executable("bend") ||
      (cargo_bin = Path.join(System.user_home!(), ".cargo/bin/bend")
       if File.exists?(cargo_bin), do: cargo_bin, else: nil)
  end
end
