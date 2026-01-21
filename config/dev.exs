import Config

# Development configuration

# Use EXLA for accelerated training
config :nx, default_backend: EXLA.Backend

# Configure EXLA - auto-detect CUDA if EXLA_TARGET=cuda is set
# This allows GPU presets to work without requiring MIX_ENV=prod
#
# Usage:
#   CPU training:  mix run scripts/train_from_replays.exs --preset quick
#   GPU training:  EXLA_TARGET=cuda mix run scripts/train_from_replays.exs --preset gpu_quick
#
# The 4090 and other NVIDIA GPUs require EXLA_TARGET=cuda to enable CUDA backend.
# Without it, training runs on CPU with XLA optimizations.
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

# Logging - debug level for development
config :logger, level: :debug
