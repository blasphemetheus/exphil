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
  # EXPHIL_GPU_MEMORY_FRACTION: how much VRAM EXLA reserves UP FRONT.
  # 2026-07-31 machine freeze: the 0.75 default (24.6GB of the 5090) while
  # Bradley played Melee starved the DESKTOP of VRAM (nvidia-drm "Failed to
  # allocate NVKMS memory"), spiraled into system memory pressure and a
  # hard reboot. Drill-scale training needs ~2GB — cap those runs low
  # (farm scripts export 0.25) and reserve 0.75 only for big-model work.
  gpu_fraction =
    case Float.parse(System.get_env("EXPHIL_GPU_MEMORY_FRACTION", "0.75")) do
      {f, _} when f > 0.0 and f <= 1.0 -> f
      _ -> 0.75
    end

  config :exla, :clients,
    cuda: [platform: :cuda, memory_fraction: gpu_fraction],
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
