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

### Planned
- [ ] Create test fixtures for replay files
- [ ] CI configuration examples
