import Config

# Production configuration

# Use EXLA backend
config :nx, default_backend: EXLA.Backend

# Configure EXLA - use CUDA if available, fall back to CPU
# Set EXLA_TARGET=cuda to force GPU, EXLA_TARGET=host for CPU
exla_target = System.get_env("EXLA_TARGET", "host")

if exla_target == "cuda" do
  config :exla, :clients,
    cuda: [platform: :cuda, memory_fraction: 0.9],
    default: [platform: :cuda]

  config :exla, default_client: :cuda
else
  # CPU with XLA optimizations
  config :exla, :clients,
    host: [platform: :host],
    default: [platform: :host]

  config :exla, default_client: :host
end

# Logging
config :logger, level: :info
