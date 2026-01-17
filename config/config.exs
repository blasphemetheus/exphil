import Config

# Configure Nx backend - use EXLA for multi-core CPU acceleration
config :nx, default_backend: EXLA.Backend

# Use host (CPU) backend with XLA optimization (uses all cores)
config :exla, default_client: :host

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
