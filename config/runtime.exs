import Config

# Runtime configuration - loaded at runtime, not compile time
# This file is for configuration that depends on environment variables

# XLA compilation cache - avoids recompiling operations on restart
# Set XLA_CACHE_PATH environment variable to customize location
if cache_path = System.get_env("XLA_CACHE_PATH") do
  config :exla, :cache_path, cache_path
else
  # Default cache location
  config :exla, :cache_path, Path.join(System.tmp_dir!(), "exphil_xla_cache")
end

# CUDA memory configuration
# Limit GPU memory usage to avoid OOM with other processes
if memory_fraction = System.get_env("EXLA_MEMORY_FRACTION") do
  {fraction, ""} = Float.parse(memory_fraction)
  config :exla, :memory_fraction, fraction
end
