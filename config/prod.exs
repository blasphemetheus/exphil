import Config

# Production configuration

# Use EXLA backend with GPU
config :nx, default_backend: EXLA.Backend

# Configure EXLA for CUDA
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.9],
  default: [platform: :cuda]

config :exla, default_client: :cuda

# Logging
config :logger, level: :info
