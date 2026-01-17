import Config

# Development configuration

# Use binary backend for faster iteration (no GPU)
config :nx, default_backend: Nx.BinaryBackend

# Logging
config :logger, level: :debug
