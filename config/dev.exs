import Config

# Development configuration

# Use EXLA for accelerated training (CPU with XLA optimizations)
# Falls back gracefully if EXLA isn't available
config :nx, default_backend: EXLA.Backend

# Logging
config :logger, level: :debug
