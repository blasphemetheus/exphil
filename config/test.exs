import Config

# Use binary backend for tests (faster, no GPU needed)
config :nx, default_backend: Nx.BinaryBackend

# Disable telemetry in tests
config :telemetry, :enabled, false
