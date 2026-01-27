#!/usr/bin/env elixir
# Benchmark: Rust NIF Selective Scan vs Nx/XLA implementations
#
# This script compares the Rust NIF CUDA implementation against
# our Nx-based implementations to measure the GPU↔CPU transfer overhead.
#
# Usage:
#   mix run scripts/benchmark_nif_scan.exs
#   mix run scripts/benchmark_nif_scan.exs --iterations 50
#   mix run scripts/benchmark_nif_scan.exs --batch 64 --seq 120

alias ExPhil.Training.Output

# Parse command line args
{opts, _, _} = OptionParser.parse(System.argv(),
  strict: [
    iterations: :integer,
    batch: :integer,
    seq: :integer,
    hidden: :integer,
    state: :integer,
    help: :boolean
  ]
)

if opts[:help] do
  IO.puts("""
  Benchmark Rust NIF Selective Scan

  Options:
    --iterations N   Number of benchmark iterations (default: 20)
    --batch N        Batch size (default: 32)
    --seq N          Sequence length (default: 60)
    --hidden N       Hidden dimension (default: 512)
    --state N        State dimension (default: 16)
    --help           Show this help
  """)
  System.halt(0)
end

iterations = opts[:iterations] || 20
batch = opts[:batch] || 32
seq_len = opts[:seq] || 60
hidden = opts[:hidden] || 512
state_dim = opts[:state] || 16

Output.banner("Selective Scan NIF Benchmark")

Output.config([
  {"Batch size", batch},
  {"Sequence length", seq_len},
  {"Hidden dimension", hidden},
  {"State dimension", state_dim},
  {"Iterations", iterations}
])

# Check NIF availability
nif_available = try do
  ExPhil.Native.SelectiveScan.available?()
rescue
  _ -> false
end

cuda_available = if nif_available do
  try do
    ExPhil.Native.SelectiveScan.cuda_available?()
  rescue
    _ -> false
  end
else
  false
end

IO.puts("")
Output.puts("NIF loaded: #{if nif_available, do: "✓ Yes", else: "✗ No"}")
Output.puts("CUDA available: #{if cuda_available, do: "✓ Yes", else: "✗ No"}")

if nif_available and cuda_available do
  case ExPhil.Native.SelectiveScan.device_info() do
    {:ok, info} -> Output.puts("Device: #{info}")
    {:error, _} -> :ok
  end
end

IO.puts("")

# Generate test data
Output.puts("Generating test tensors...")

key = Nx.Random.key(42)
{x, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{dt, key} = Nx.Random.uniform(key, 0.001, 0.1, shape: {batch, seq_len, hidden}, type: :f32)
{a, key} = Nx.Random.uniform(key, -2.0, -0.5, shape: {hidden, state_dim}, type: :f32)
{b, key} = Nx.Random.normal(key, shape: {batch, seq_len, state_dim}, type: :f32)
{c, _key} = Nx.Random.normal(key, shape: {batch, seq_len, state_dim}, type: :f32)

# Force evaluation
x = Nx.backend_copy(x, Nx.BinaryBackend)
dt = Nx.backend_copy(dt, Nx.BinaryBackend)
a = Nx.backend_copy(a, Nx.BinaryBackend)
b = Nx.backend_copy(b, Nx.BinaryBackend)
c = Nx.backend_copy(c, Nx.BinaryBackend)

tensor_bytes = (batch * seq_len * hidden * 4 * 2 + hidden * state_dim * 4 + batch * seq_len * state_dim * 4 * 2)
Output.puts("Total tensor data: #{Float.round(tensor_bytes / 1_000_000, 2)} MB")
IO.puts("")

# Benchmark function
benchmark = fn name, fun ->
  Output.puts("Benchmarking #{name}...")

  # Warmup
  for _ <- 1..3, do: fun.()

  # Timed runs
  times = for _ <- 1..iterations do
    {time_us, _result} = :timer.tc(fun)
    time_us / 1000  # Convert to ms
  end

  avg = Enum.sum(times) / length(times)
  min = Enum.min(times)
  max = Enum.max(times)

  meets_60fps = avg < 16.67

  IO.puts("  Average: #{Float.round(avg, 2)} ms")
  IO.puts("  Min: #{Float.round(min, 2)} ms")
  IO.puts("  Max: #{Float.round(max, 2)} ms")
  IO.puts("  60 FPS: #{if meets_60fps, do: "✓ YES", else: "✗ NO"}")
  IO.puts("")

  {name, avg}
end

results = []

# Benchmark Rust NIF (if available)
results = if nif_available and cuda_available do
  {name, time} = benchmark.("Rust NIF (CUDA)", fn ->
    ExPhil.Native.SelectiveScan.scan(x, dt, a, b, c)
  end)
  [{name, time} | results]
else
  Output.warning("Skipping Rust NIF benchmark (not available)")
  IO.puts("")
  results
end

# Benchmark Nx/XLA Blelloch (our current best)
results = try do
  alias ExPhil.Networks.Mamba

  {name, time} = benchmark.("Nx/XLA Blelloch", fn ->
    Mamba.selective_scan(x, dt, a, b, c)
  end)
  [{name, time} | results]
rescue
  e ->
    Output.warning("Skipping Nx/XLA Blelloch: #{inspect(e)}")
    IO.puts("")
    results
end

# Benchmark PyTorch Port (if available)
results = try do
  if ExPhil.Bridge.PyTorchPort.available?() do
    {name, time} = benchmark.("PyTorch Port", fn ->
      ExPhil.Bridge.PyTorchPort.selective_scan!(x, dt, a, b, c)
    end)
    [{name, time} | results]
  else
    Output.warning("Skipping PyTorch Port (not available)")
    IO.puts("")
    results
  end
rescue
  _ ->
    Output.warning("Skipping PyTorch Port (not available)")
    IO.puts("")
    results
end

# Summary
IO.puts("=" |> String.duplicate(60))
IO.puts("SUMMARY")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

results = Enum.reverse(results)

if length(results) > 0 do
  fastest = Enum.min_by(results, fn {_, time} -> time end)
  {fastest_name, fastest_time} = fastest

  for {name, time} <- results do
    speedup = if time == fastest_time, do: "", else: " (#{Float.round(time / fastest_time, 1)}x slower)"
    status = if time < 16.67, do: "✓", else: "✗"
    IO.puts("#{status} #{name}: #{Float.round(time, 2)} ms#{speedup}")
  end

  IO.puts("")
  Output.success("Fastest: #{fastest_name} at #{Float.round(fastest_time, 2)} ms")

  if fastest_time < 16.67 do
    Output.success("60 FPS target (16.67ms) achieved!")
  else
    Output.warning("60 FPS target (16.67ms) not met. Gap: #{Float.round(fastest_time - 16.67, 2)} ms")
  end
else
  Output.error("No benchmarks completed successfully")
end
