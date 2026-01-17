import Config

# Configure Nx backend (defaults to binary, use EXLA for GPU)
config :nx, default_backend: Nx.BinaryBackend

# For GPU training, uncomment and configure:
# config :nx, default_backend: EXLA.Backend

# EXLA configuration for CUDA
# config :exla, :clients,
#   cuda: [platform: :cuda, memory_fraction: 0.8],
#   default: [platform: :host]

# Configure EXLA to prefer CUDA when available
# config :exla, default_client: :cuda

# Telemetry configuration
config :telemetry, :enabled, true

# Import environment specific config
import_config "#{config_env()}.exs"
