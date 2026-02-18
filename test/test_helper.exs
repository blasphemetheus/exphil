# ExPhil Test Configuration
#
# Test Categories (use with --include/--exclude/--only):
#   :slow        - Tests taking >1s (excluded by default)
#   :integration - Tests with external dependencies (excluded by default)
#   :gpu         - Tests requiring GPU/CUDA
#   :external    - Tests needing external files (replays, models)
#   :benchmark   - Performance regression tests (excluded by default)
#   :property    - Property-based tests with StreamData (excluded by default)
#   :nif         - Tests requiring compiled Rust NIF (excluded by default)
#   :cuda        - Tests requiring CUDA device + NIF (excluded by default)
#
# Examples:
#   mix test                           # Fast tests only (default)
#   mix test --include slow            # Include slow tests
#   mix test --include property        # Include property tests
#   mix test --only integration        # Only integration tests
#   mix test --only benchmark          # Only benchmark tests
#   mix test --include slow --include integration --include property  # Everything
#
# Quick aliases (defined in mix.exs):
#   mix test.fast      # Same as: mix test
#   mix test.slow      # Same as: mix test --include slow
#   mix test.all       # Same as: mix test --include slow --include integration
#   mix test.benchmark # Same as: mix test --only benchmark
#   mix test.coverage  # With coverage report

# Build base configuration
base_config = [
  # Exclude slow, integration, benchmark, property and snapshot tests by default for fast feedback
  exclude: [:slow, :integration, :external, :gpu, :benchmark, :snapshot, :property, :nif, :cuda],

  # Timeout for individual tests (2 minutes default, can override with @tag timeout: N)
  timeout: 120_000,

  # Run tests in parallel where possible
  max_cases: System.schedulers_online() * 2,

  # Capture logs during tests (show on failure)
  capture_log: true
]

# Add seed only if explicitly set via environment variable
config =
  case System.get_env("EXUNIT_SEED") do
    nil -> base_config
    val -> Keyword.put(base_config, :seed, String.to_integer(val))
  end

# Configure ExUnit
ExUnit.configure(config)

# Start ExUnit
ExUnit.start()

# Set up Mox for mock-based testing
# Mox allows defining mock modules that implement behaviours
# See test/support/mocks.ex for mock definitions
Application.ensure_all_started(:mox)

# Optionally print test configuration
if System.get_env("EXUNIT_VERBOSE") do
  IO.puts("\n[TestConfig] Excluded tags: #{inspect(ExUnit.configuration()[:exclude])}")
  IO.puts("[TestConfig] Max cases: #{ExUnit.configuration()[:max_cases]}")
end
