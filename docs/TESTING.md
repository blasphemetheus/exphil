# ExPhil Testing Guide

This document describes the test harness, how to run tests efficiently, and best practices for writing tests.

## Quick Reference

```bash
# Fast tests only (default - excludes slow/integration)
mix test

# Include slow tests
mix test --include slow

# Include all tests (slow + integration + external)
mix test --include slow --include integration --include external

# Run specific test file
mix test test/exphil/training/config_test.exs

# Run specific test by line number
mix test test/exphil/training/config_test.exs:42

# Run tests matching a pattern
mix test --only describe:"parse_args"

# Run with coverage
mix test --cover

# Run with verbose output
EXUNIT_VERBOSE=1 mix test

# Run with specific seed (reproducibility)
mix test --seed 12345
```

## Test Categories (Tags)

Tests are organized using ExUnit tags. By default, only fast unit tests run.

| Tag | Description | Default |
|-----|-------------|---------|
| `:slow` | Tests taking >1 second | Excluded |
| `:integration` | Tests with external dependencies | Excluded |
| `:external` | Tests needing external files (replays) | Excluded |
| `:gpu` | Tests requiring GPU/CUDA | Excluded |
| (none) | Fast unit tests | Included |

### Running Different Test Categories

```bash
# Only fast unit tests (default)
mix test

# Include slow tests (training, network tests)
mix test --include slow

# Only integration tests
mix test --only integration

# Everything except GPU tests
mix test --include slow --include integration --include external

# Only GPU tests (for CI with GPU)
mix test --only gpu
```

## Test Structure

```
test/
├── test_helper.exs          # ExUnit configuration
├── support/                  # Shared test utilities
│   ├── factories.ex         # Test data factories
│   └── test_helpers.ex      # Common assertions & helpers
├── fixtures/                # Test fixture files
│   └── replays/             # Sample replay files
├── exphil/
│   ├── training/            # Training module tests
│   │   ├── config_test.exs
│   │   ├── imitation_test.exs
│   │   └── ...
│   ├── networks/            # Network tests
│   │   ├── policy_test.exs
│   │   ├── mamba_test.exs
│   │   └── ...
│   ├── embeddings/          # Embedding tests
│   └── ...
└── exphil_bridge/           # Bridge/integration tests
```

## Using Test Factories

The `ExPhil.Test.Factories` module provides builders for test data:

```elixir
defmodule MyTest do
  use ExUnit.Case
  import ExPhil.Test.Factories

  test "example using factories" do
    # Build a training batch
    batch = build_batch(batch_size: 8, embed_size: 64)

    # Build a game state
    game_state = build_game_state(frame: 100)

    # Build a player with custom values
    player = build_player(damage: 50.0, stocks: 2, x: -30.0)

    # Build a PPO rollout
    rollout = build_rollout(num_steps: 32)

    # Build training frames
    frames = build_training_frames(100)
  end
end
```

### Available Factories

| Factory | Description | Options |
|---------|-------------|---------|
| `build_batch/1` | Training batch with state/targets | `:batch_size`, `:embed_size`, `:seq_len` |
| `build_targets/1` | Controller output targets | batch_size |
| `build_rollout/1` | PPO rollout data | `:num_steps`, `:embed_size` |
| `build_game_state/1` | Game state struct | `:frame`, `:stage`, `:players` |
| `build_player/1` | Player struct | `:port`, `:damage`, `:stocks`, `:x`, `:y`, ... |
| `build_controller_state/1` | Controller inputs | `:button_a`, `:main_stick`, ... |
| `build_training_frame/1` | Game state + controller pair | `:game_state`, `:controller` |
| `build_training_frames/2` | List of training frames | count, opts |
| `random_tensor/2` | Random Nx tensor | shape, `:type` |

## Using Test Helpers

The `ExPhil.Test.Helpers` module provides common assertions:

```elixir
defmodule MyTest do
  use ExUnit.Case
  import ExPhil.Test.Helpers

  test "tensor assertions" do
    tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Assert shape
    assert_tensor_shape(tensor, {2, 2})

    # Assert value range
    assert_tensor_in_range(tensor, 0.0, 5.0)

    # Assert no NaN/Inf
    assert_tensor_finite(tensor)

    # Assert tensors are close
    assert_tensors_close(tensor, other_tensor, atol: 1.0e-4)
  end

  test "timing assertions" do
    # Assert operation completes within time limit
    result = assert_completes_within(1000, fn ->
      some_operation()
    end)
  end

  test "temp file helpers" do
    with_temp_dir(fn dir ->
      path = Path.join(dir, "test.axon")
      # ... test with temp files
    end)
    # Directory is automatically cleaned up
  end
end
```

### Available Helpers

| Helper | Description |
|--------|-------------|
| `assert_tensor_shape/2` | Assert tensor has expected shape |
| `assert_tensor_in_range/3` | Assert tensor values in [min, max] |
| `assert_tensor_finite/1` | Assert no NaN or Inf values |
| `assert_tensors_close/3` | Assert tensors approximately equal |
| `assert_tensor_type/2` | Assert tensor dtype |
| `with_temp_dir/1` | Execute with temp directory, auto-cleanup |
| `temp_file/2` | Create temp file with content |
| `timed/1` | Measure execution time |
| `assert_completes_within/2` | Assert function completes in time |
| `wait_until/2` | Wait for condition with timeout |
| `build_and_init_model/2` | Build and initialize Axon model |
| `count_params/1` | Count model parameters |

## Writing Tests

### Tagging Slow Tests

Any test taking more than ~1 second should be tagged:

```elixir
@tag :slow
test "trains for multiple epochs" do
  # ... slow test
end
```

### Tagging Integration Tests

Tests that depend on external services or files:

```elixir
@tag :integration
@tag :external
test "parses real replay file" do
  # ... test with external file
end
```

### Module-Level Tags

Apply tags to all tests in a module:

```elixir
defmodule ExPhil.Training.SlowTest do
  use ExUnit.Case

  @moduletag :slow

  # All tests in this module are tagged :slow
end
```

### Async Tests

Tests that don't share state can run in parallel:

```elixir
defmodule MyTest do
  use ExUnit.Case, async: true

  # These tests run concurrently with other async test modules
end
```

**Note:** Don't use `async: true` for tests that:
- Share global state (ETS tables, agents)
- Use the same ports or files
- Depend on specific process ordering

## Coverage

### Running with Coverage

```bash
# Basic coverage
mix test --cover

# Coverage with HTML report (requires excoveralls)
mix coveralls.html
```

### Coverage Configuration

Coverage is configured in `mix.exs`:

```elixir
def project do
  [
    # ...
    test_coverage: [tool: ExCoveralls],
    preferred_cli_env: [
      coveralls: :test,
      "coveralls.html": :test
    ]
  ]
end
```

## CI Configuration

### GitHub Actions Example

```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: erlef/setup-beam@v1
      with:
        elixir-version: '1.16'
        otp-version: '26'

    - run: mix deps.get
    - run: mix compile --warnings-as-errors

    # Fast tests first (quick feedback)
    - run: mix test

    # Slow tests (can be in separate job)
    - run: mix test --include slow
```

### Parallel CI Jobs

For faster CI, split tests by category:

```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: mix test  # Fast only

  slow-tests:
    runs-on: ubuntu-latest
    steps:
      - run: mix test --include slow --exclude integration

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - run: mix test --only integration

  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - run: mix test --only gpu
```

## Debugging Tests

### Run Single Test

```bash
# By file and line
mix test test/exphil/training/config_test.exs:42

# By pattern
mix test --only describe:"parse_args"
```

### Verbose Output

```bash
# Show test names as they run
mix test --trace

# Show configuration
EXUNIT_VERBOSE=1 mix test
```

### Reproduce Failures

```bash
# Run with same seed as CI failure
mix test --seed 12345

# Run only failed tests from last run
mix test --failed
```

### Debug with IEx

```bash
iex -S mix test test/mytest.exs:42
```

## Best Practices

1. **Keep unit tests fast** - If a test takes >1s, tag it `:slow`
2. **Use factories** - Don't duplicate test data setup
3. **Use descriptive names** - `test "returns error when input is nil"` not `test "error case"`
4. **One assertion per concept** - Multiple assertions are OK if testing one thing
5. **Don't test implementation** - Test behavior, not internal details
6. **Isolate tests** - Each test should be independent
7. **Tag appropriately** - Help others run the right subset of tests

## Mocking with Mox

Mox enables mock-based testing for components with behaviours.

### Setup

Mox is configured in test_helper.exs. Mock definitions live in `test/support/mocks.ex`.

### Usage

```elixir
defmodule MyTest do
  use ExUnit.Case, async: true
  import Mox

  # Verify all expectations are met after each test
  setup :verify_on_exit!

  test "uses mock implementation" do
    # Set expectation
    expect(ExPhil.SomeMock, :function, fn arg ->
      {:ok, arg}
    end)

    # Call code that uses the mocked module
    assert {:ok, "test"} = SomeModule.call("test")
  end

  test "uses stub for default behavior" do
    stub(ExPhil.SomeMock, :function, fn _ -> :default end)

    # Stub is used for all calls
    assert :default = SomeModule.call("anything")
  end
end
```

### Adding New Mocks

1. Define a behaviour in your module:
   ```elixir
   defmodule ExPhil.SomeBehaviour do
     @callback function(arg :: term()) :: {:ok, term()} | {:error, term()}
   end
   ```

2. Add the mock to `test/support/mocks.ex`:
   ```elixir
   Mox.defmock(ExPhil.SomeMock, for: ExPhil.SomeBehaviour)
   ```

3. Configure the application to use the mock in tests (config/test.exs):
   ```elixir
   config :exphil, :some_module, ExPhil.SomeMock
   ```

## Property-Based Testing with StreamData

StreamData enables property-based testing - verifying that code works for
a wide range of randomly generated inputs.

### Generators

Custom generators live in `test/support/generators.ex`:

```elixir
import ExPhil.Test.Generators

# Generate valid game state
game_state <- game_state_gen()

# Generate valid player
player <- player_gen()

# Generate controller state
controller <- controller_state_gen()

# Generate training frames
frames <- training_frames_gen(10, 100)

# Generate random tensor
tensor <- tensor_gen({32, 64})
```

### Writing Property Tests

```elixir
defmodule MyPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties
  import ExPhil.Test.Generators

  # Tag as slow - property tests run many iterations
  @moduletag :slow

  property "embedding produces valid output for any player" do
    check all player <- player_gen(),
              max_runs: 50 do
      embedded = Player.embed(player, 1)

      # Assertions that should hold for ALL valid players
      assert is_struct(embedded, Nx.Tensor)
      assert Nx.all(Nx.is_nan(embedded) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
```

### Best Practices

1. **Tag property tests as `:slow`** - They run many iterations
2. **Use `max_runs: N`** - Control iteration count (default: 100)
3. **Test invariants** - Properties that should hold for ALL inputs
4. **Common properties:**
   - Output shape is consistent
   - No NaN/Inf in outputs
   - Roundtrip serialization
   - Idempotency

## Test Improvements Roadmap

### Completed
- [x] Enhanced test_helper.exs with tag configuration
- [x] Test factories for common data structures
- [x] Test helpers with tensor assertions
- [x] Documentation (this file)
- [x] Add excoveralls for coverage reporting
- [x] Standardize tags across all existing tests
- [x] Add mix aliases for common test commands
- [x] Mox for mock-based testing
- [x] Property-based testing with StreamData
- [x] Profile and tag slow tests for faster feedback
- [x] Mutation testing with Muzak
- [x] Doctest coverage for EarlyStopping
- [x] Flaky test retry mechanism

### High Priority
- [x] Test fixtures for replay files - Small deterministic fixtures for integration tests
- [x] Snapshot testing for embeddings - Save/compare expected tensor outputs
- [x] Benchmark tests - Track inference performance over time
- [ ] CI configuration - GitHub Actions for fast/slow/GPU test splits

### Medium Priority
- [ ] Contract tests for Bridge - Verify Elixir-Python interface contracts
- [ ] More doctest coverage - Training.Data, Embeddings.Primitives, utilities
- [ ] Test coverage gates - Fail CI if coverage drops below threshold

### Lower Priority
- [ ] Visual regression tests - For training plots/reports
- [ ] Load testing - Agent GenServer under heavy message volume
- [ ] Chaos testing - Network failures, resource exhaustion

## Mutation Testing

Mutation testing verifies test quality by introducing small changes (mutations) to
your code and checking if tests catch them. If a mutation survives (tests still pass),
it indicates a gap in test coverage.

### Running Mutation Tests

```bash
# Run with default profile (core modules)
mix test.mutate

# Quick profile (single module)
mix test.mutate.quick

# Run mutation testing on a specific module
mix muzak --only ExPhil.Embeddings.Player

# Run with coverage threshold
mix muzak --min-coverage 80
```

### Configuration

Profiles are defined in `.muzak.exs`:
- `default` - Core modules (embeddings, training config/targets)
- `ci` - More thorough for CI (all embeddings and training)
- `quick` - Single module for rapid iteration

### Interpreting Results

- **Killed mutations**: Tests caught the change (good)
- **Survived mutations**: Tests missed the change (needs improvement)
- **Equivalent mutations**: Change doesn't affect behavior (ignore)

## Doctest Coverage

Doctests serve dual purposes: documentation examples and executable tests.

### Writing Doctests

```elixir
defmodule ExPhil.Example do
  @moduledoc "Example module with doctests"

  @doc """
  Normalizes a value to the range [0, 1].

  ## Examples

      iex> ExPhil.Example.normalize(5, 0, 10)
      0.5

      iex> ExPhil.Example.normalize(0, 0, 100)
      0.0

      iex> ExPhil.Example.normalize(100, 0, 100)
      1.0
  """
  def normalize(value, min, max) do
    (value - min) / (max - min)
  end
end
```

### Running Doctests

```bash
# Doctests run automatically with mix test
mix test

# Run only doctests
mix test --only doctest
```

## Flaky Test Handling

Flaky tests (tests that sometimes pass, sometimes fail) undermine CI reliability.

### Using the Retry Helper

The `ExPhil.Test.Helpers` module provides a `retry` macro for handling flaky tests:

```elixir
defmodule MyTest do
  use ExUnit.Case, async: true
  import ExPhil.Test.Helpers

  @tag :flaky
  test "network operation that occasionally times out" do
    retry retries: 3, delay: 200 do
      result = NetworkClient.fetch()
      assert result.status == 200
    end
  end
end
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `:retries` | 3 | Number of retry attempts |
| `:delay` | 100 | Delay in ms between retries |
| `:log` | false | Log retry attempts for debugging |

### Checking Success

Use `eventually_succeeds?/2` for conditional test logic:

```elixir
test "resource-dependent operation" do
  if eventually_succeeds?(fn -> check_resource_available() end, retries: 5) do
    # proceed with test
    result = use_resource()
    assert result == :ok
  else
    # skip gracefully or use alternative
    IO.puts("Resource unavailable, skipping")
  end
end
```

### Finding Flaky Tests

ExUnit 1.17+ includes `--repeat-until-failure` to help identify flaky tests:

```bash
# Run a test 100 times until it fails
mix test test/my_test.exs:42 --repeat-until-failure 100

# Reproduce a failure with the same seed
mix test test/my_test.exs:42 --seed 123456
```

### Best Practices

1. **Investigate root cause first** - Don't just add retries
2. **Tag flaky tests explicitly** - `@tag :flaky` for tracking
3. **Set reasonable retry limits** - Usually 2-3 retries max
4. **Log retry attempts** - For debugging persistent issues
5. **Fix or remove** - Flaky tests should be temporary

## Test File Naming Conventions

Consistent naming helps locate tests quickly:

| Module | Test File |
|--------|-----------|
| `ExPhil.Training.Config` | `test/exphil/training/config_test.exs` |
| `ExPhil.Networks.Policy` | `test/exphil/networks/policy_test.exs` |
| `ExPhil.Bridge.GameState` | `test/exphil/bridge/game_state_test.exs` |

### Rules

1. Test file mirrors module path: `lib/exphil/foo/bar.ex` → `test/exphil/foo/bar_test.exs`
2. Always use `_test.exs` suffix
3. Integration tests go in `test/integration/`
4. Property tests use `_property_test.exs` suffix

## Benchmark Tests

Benchmark tests detect performance regressions by comparing against stored baselines.

### Running Benchmarks

```bash
# Run benchmark tests
mix test.benchmark

# Update baselines when changes are intentional
mix test.benchmark.update
# Or: BENCHMARK_UPDATE=1 mix test --only benchmark
```

### Writing Benchmark Tests

```elixir
defmodule MyBenchmarkTest do
  use ExUnit.Case
  import ExPhil.Test.Helpers

  @moduletag :benchmark

  test "embedding performance" do
    {:ok, stats} = benchmark name: "embed_player", iterations: 50, warmup: 10 do
      Player.embed(player)
    end

    # Assert reasonable performance
    assert stats.mean < 5, "Embedding too slow: #{stats.mean}ms"
  end
end
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `:name` | required | Unique name for baseline tracking |
| `:iterations` | 10 | Number of measured iterations |
| `:warmup` | 3 | Number of warmup iterations |
| `:tolerance` | 0.20 | Allowed regression (20% by default) |

### Baselines

Baselines are stored in `test/fixtures/benchmark_baselines.json` and tracked in git.

### Expected GPU vs CPU Performance

Benchmark times vary significantly between CPU (BinaryBackend) and GPU (EXLA with CUDA).
The table below shows expected ranges - use these to verify your setup is performing correctly.

| Benchmark | CPU (BinaryBackend) | GPU (EXLA/CUDA) | Notes |
|-----------|---------------------|-----------------|-------|
| `policy_single_frame` | ~250ms | ~1.6ms | **156x speedup** |
| `policy_batch_32` | ~250ms | ~1.9ms | **131x speedup** |
| `mamba_temporal` | timeout (>120s) | ~40ms | CPU can't run Mamba |
| `embed_player` | ~0.2ms | ~1.9ms | GPU overhead > CPU compute |
| `embed_game_state` | ~0.3ms | ~4.5ms | GPU overhead > CPU compute |

**Important Notes:**
- Mamba benchmarks are tagged `:gpu` and excluded from CPU test runs
- First GPU run includes JIT compilation overhead - use warmup iterations
- ONNX INT8 quantized models achieve ~0.55ms inference regardless of GPU

### Updating Benchmark Baselines

When running benchmarks on a new GPU setup or after intentional performance changes:

```bash
# Run benchmarks and update baselines
mix test.benchmark.update

# After GPU testing, commit the updated baselines
git add test/fixtures/benchmark_baselines.json
git commit -m "Update benchmark baselines from GPU run"
```

**When to update baselines:**
- After switching to a new GPU
- After optimizing inference code
- After changing batch sizes or model architecture
- When setting up a new CI GPU runner

**When NOT to update:**
- If benchmarks fail unexpectedly (investigate first)
- If regression tolerance is exceeded without code changes

## GPU Testing

GPU tests verify CUDA/EXLA functionality and catch GPU-specific issues like memory leaks, blocking operations, and performance regressions.

### Running GPU Tests

```bash
# On a machine with GPU
MIX_ENV=test mix test --only gpu

# With verbose output
MIX_ENV=test mix test --only gpu --trace

# Specific GPU test file
MIX_ENV=test mix test test/exphil/benchmarks/training_speed_test.exs
```

**Important:** On RunPod/cloud pods, `MIX_ENV=prod` is the default. Always use `MIX_ENV=test` explicitly.

### Current GPU Tests

Located in `test/exphil/benchmarks/training_speed_test.exs`:

| Test | Purpose | Threshold |
|------|---------|-----------|
| MLP training speed | Baseline comparison | <500ms/batch |
| LSTM training speed | Verify BPTT performance | <15000ms/batch |
| GRU training speed | Verify BPTT performance | <15000ms/batch |
| Mamba training speed | Verify SSM performance | <2000ms/batch |
| Jamba training speed | Verify hybrid architecture | <3000ms/batch |
| Mamba vs LSTM comparison | Mamba should be faster | Mamba < LSTM * 1.5 |
| Tensor accumulation regression | Catch Gotcha #18 | <5000ms total |
| Periodic conversion correctness | Verify progress display fix | Loss within 0.0001 |
| GPU utilization proxy | Detect blocking | <1000ms/batch |

### GPU Test Ideas (Roadmap)

#### Memory Tests

| Test | Description | Priority |
|------|-------------|----------|
| **OOM boundary detection** | Find max batch size before OOM for each backbone | High |
| **Memory leak detection** | Run 1000 batches, assert memory doesn't grow | High |
| **Gradient accumulation memory** | Verify accumulated grads use less memory than large batch | Medium |
| **Checkpoint memory spike** | Measure memory during save/load operations | Medium |
| **Multi-model memory** | Load multiple models, verify total memory fits | Low |

```elixir
# Example: OOM boundary test
@tag :gpu
@tag :slow
test "finds max batch size for MLP before OOM" do
  max_batch = find_max_batch_size(:mlp, starting: 64, max: 4096)
  assert max_batch >= 512, "Expected at least batch_size=512 on this GPU"
  IO.puts("\n  [INFO] Max MLP batch size: #{max_batch}")
end
```

#### Performance Tests

| Test | Description | Priority |
|------|-------------|----------|
| **JIT compilation time** | Measure first-batch compilation overhead | High |
| **Warm vs cold inference** | Compare first vs subsequent batch timing | High |
| **Batch size scaling** | Verify throughput scales with batch size | Medium |
| **Sequence length scaling** | Measure how seq_len affects speed | Medium |
| **bf16 vs f32 speed** | Measure mixed precision speedup | Medium |
| **ONNX vs Axon inference** | Compare exported model speed | Medium |
| **Real-time budget check** | Verify <16.6ms inference for 60fps | High |

```elixir
# Example: Real-time inference check
@tag :gpu
test "inference meets 60fps budget" do
  trainer = Imitation.new(embed_size: 408, backbone: :mamba, window_size: 30)
  batch = generate_batch(1, 408, temporal: true, seq_len: 30)

  # Warmup
  _ = Imitation.predict(trainer, batch)

  # Measure
  {time_us, _} = :timer.tc(fn ->
    for _ <- 1..60, do: Imitation.predict(trainer, batch)
  end)

  avg_ms = time_us / 1000 / 60
  assert avg_ms < 16.6, "Inference too slow for 60fps: #{avg_ms}ms"
end
```

#### Numerical Stability Tests

| Test | Description | Priority |
|------|-------------|----------|
| **No NaN/Inf in training** | Run 100 batches, assert all outputs finite | High |
| **Gradient explosion detection** | Verify grad norms stay bounded | High |
| **bf16 precision loss** | Compare bf16 vs f32 loss values | Medium |
| **Long training stability** | 10000 steps without NaN | Medium |
| **Extreme input handling** | Very large/small input values | Low |

```elixir
# Example: Numerical stability
@tag :gpu
@tag :slow
test "training stays numerically stable for 1000 batches" do
  trainer = Imitation.new(backbone: :mamba)
  batches = generate_training_batches(1000, 32, 408, temporal: true)

  final_trainer = Enum.reduce(batches, trainer, fn batch, t ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    loss = Nx.to_number(metrics.loss)

    assert is_float(loss), "Loss is not a float: #{inspect(loss)}"
    refute is_nan(loss), "NaN loss at step #{new_t.step}"
    refute is_infinite(loss), "Infinite loss at step #{new_t.step}"

    new_t
  end)

  assert final_trainer.step == 1000
end
```

#### Checkpoint Tests

| Test | Description | Priority |
|------|-------------|----------|
| **Save/load roundtrip** | Save, load, verify identical predictions | High |
| **Cross-device load** | Save on GPU, load on CPU (and vice versa) | High |
| **Corrupted checkpoint** | Graceful error on bad checkpoint | Medium |
| **Checkpoint size** | Verify file size is reasonable | Low |
| **Incremental checkpointing** | Verify only changed params saved | Low |

```elixir
# Example: Roundtrip test
@tag :gpu
test "checkpoint roundtrip produces identical predictions" do
  trainer = Imitation.new(backbone: :mamba)
  batch = generate_batch(8, 408, temporal: true)

  # Get predictions before save
  pred_before = Imitation.predict(trainer, batch)

  with_temp_dir(fn dir ->
    path = Path.join(dir, "test.axon")

    # Save and reload
    Imitation.save_checkpoint(trainer, path)
    loaded_trainer = Imitation.load_checkpoint(path)

    # Get predictions after load
    pred_after = Imitation.predict(loaded_trainer, batch)

    # Should be identical
    assert_tensors_close(pred_before, pred_after, atol: 1.0e-6)
  end)
end
```

#### Gradient Tests

| Test | Description | Priority |
|------|-------------|----------|
| **Gradient accumulation correctness** | 4x small batches = 1x large batch | High |
| **Gradient clipping** | Verify clipping activates for large grads | Medium |
| **Frozen layer gradients** | Verify frozen params have zero grad | Medium |
| **Gradient checkpointing** | Verify memory reduction works | Low |

```elixir
# Example: Gradient accumulation
@tag :gpu
test "gradient accumulation matches large batch" do
  # Single large batch
  large_trainer = Imitation.new(batch_size: 128)
  large_batch = generate_batch(128, 408)
  {_, large_grads} = compute_gradients(large_trainer, large_batch)

  # 4 small batches accumulated
  small_trainer = Imitation.new(batch_size: 32)
  small_batches = for _ <- 1..4, do: generate_batch(32, 408)
  accumulated_grads = accumulate_gradients(small_trainer, small_batches)

  # Should be approximately equal (within numerical precision)
  assert_tensors_close(large_grads, accumulated_grads, rtol: 0.01)
end
```

#### Multi-GPU Tests (Future)

| Test | Description | Priority |
|------|-------------|----------|
| **Device placement** | Model on GPU 0, data on GPU 1 | Low |
| **Data parallel training** | Same model, split batches across GPUs | Low |
| **Model parallel** | Split model across GPUs | Low |
| **NCCL communication** | Verify all-reduce works | Low |

#### Backend Comparison Tests

| Test | Description | Priority |
|------|-------------|----------|
| **EXLA vs BinaryBackend** | Verify same results (within precision) | High |
| **GPU vs CPU predictions** | Same input → same output | High |
| **JIT vs eager mode** | Verify JIT doesn't change results | Medium |

```elixir
# Example: Backend comparison
@tag :gpu
test "GPU and CPU produce same predictions" do
  # Build on GPU
  gpu_trainer = Imitation.new(backbone: :mamba)

  # Generate data on CPU, copy to GPU
  batch_cpu = generate_batch(8, 408, temporal: true)
  batch_gpu = deep_copy_to_device(batch_cpu, EXLA.Backend)

  # Predict on GPU
  pred_gpu = Imitation.predict(gpu_trainer, batch_gpu)

  # Move model to CPU and predict
  cpu_trainer = to_cpu(gpu_trainer)
  pred_cpu = Imitation.predict(cpu_trainer, batch_cpu)

  # Should match within floating point tolerance
  assert_tensors_close(
    Nx.backend_copy(pred_gpu, Nx.BinaryBackend),
    pred_cpu,
    atol: 1.0e-4
  )
end
```

#### Stress Tests

| Test | Description | Priority |
|------|-------------|----------|
| **Continuous training** | Train for 1 hour without failure | Low |
| **Rapid model switching** | Load/unload different models quickly | Low |
| **Concurrent inference** | Multiple processes, same GPU | Low |
| **Recovery from OOM** | Hit OOM, recover gracefully | Low |

### Writing GPU Tests

```elixir
defmodule MyGpuTest do
  use ExUnit.Case, async: false  # GPU tests should NOT be async

  @moduletag :gpu
  @moduletag :slow
  @moduletag timeout: 300_000  # 5 min timeout for GPU tests

  setup do
    # Ensure GPU is available
    case System.get_env("CUDA_VISIBLE_DEVICES") do
      nil -> :ok
      "" -> raise "CUDA_VISIBLE_DEVICES is empty - no GPU available"
      _ -> :ok
    end

    # Clear GPU memory before test
    :erlang.garbage_collect()

    :ok
  end

  @tag :gpu
  test "my gpu test" do
    # Test implementation
  end
end
```

### GPU Test Best Practices

1. **Always use `async: false`** - GPU tests should not run concurrently
2. **Set generous timeouts** - JIT compilation can take minutes
3. **Include warmup iterations** - First batch includes JIT time
4. **Clear memory between tests** - Prevent OOM from accumulation
5. **Tag appropriately** - `:gpu`, `:slow`, `:benchmark` as needed
6. **Test on target hardware** - Performance varies by GPU
7. **Document thresholds** - Explain why specific limits were chosen
8. **Handle missing GPU gracefully** - Skip tests if no CUDA

### GPU Debugging Tips

```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor GPU during test
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Verify EXLA sees GPU
iex -S mix
iex> EXLA.Client.default_device_id()

# Run with verbose XLA output
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" mix test --only gpu
```

## Snapshot Testing

Snapshot tests verify that embeddings produce consistent outputs over time.

### Running Snapshot Tests

```bash
# Run snapshot tests
mix test.snapshot

# Update snapshots when changes are intentional
mix test.snapshot.update
# Or: SNAPSHOT_UPDATE=1 mix test --only snapshot
```

### Writing Snapshot Tests

```elixir
defmodule MySnapshotTest do
  use ExUnit.Case
  import ExPhil.Test.Helpers

  @moduletag :snapshot

  test "player embedding output" do
    player = build_player(character: 10, x: 0.0)
    embedding = Player.embed(player)

    assert_snapshot("player_neutral", embedding)
  end
end
```

### How It Works

1. First run creates snapshot files in `test/fixtures/embedding_snapshots/`
2. Subsequent runs compare output against saved snapshots
3. If output differs beyond tolerance (atol=1e-5), test fails
4. Use `SNAPSHOT_UPDATE=1` when changes are intentional

## Replay Fixtures

The `ExPhil.Test.ReplayFixtures` module provides realistic game scenario data.

### Available Fixtures

```elixir
import ExPhil.Test.ReplayFixtures

# Neutral game scenarios
game_state = neutral_game_fixture(:mewtwo_vs_fox)
game_state = neutral_game_fixture(:marth_vs_sheik)
game_state = neutral_game_fixture(:low_tier)

# Edge guard scenarios
game_state = edge_guard_fixture(:fox_recovering_low)
game_state = edge_guard_fixture(:mewtwo_offstage)

# Combo sequences (list of {game_state, controller} tuples)
frames = combo_sequence_fixture(:fox_upthrow_upair)
```

### Converting to Peppi Format

```elixir
# For testing code that expects Peppi.ParsedReplay
game_states = [state1, state2, state3]
parsed_replay = to_parsed_replay(game_states, stage: 32)
```

### Why Fixtures Instead of Real Replays

- **Deterministic**: Same output every time
- **Small**: No large binary files in repo
- **Fast**: No file I/O or parsing
- **Targeted**: Test specific scenarios
