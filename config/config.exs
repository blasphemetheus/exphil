import Config

# Configure timezone database for Central time timestamps
config :elixir, :time_zone_database, Tz.TimeZoneDatabase

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

# Tensor dtype does not pin GPU dot-product arithmetic. V3's strict comparison
# uses highest precision for both sequence and per-frame execution. Opt in at
# launch so existing policies retain their original arithmetic by default.
case System.get_env("EXPHIL_EXLA_PRECISION") do
  nil ->
    :ok

  value when value in ["default", "high", "highest"] ->
    precision = %{"default" => :default, "high" => :high, "highest" => :highest}[value]
    config :nx, :default_defn_options, compiler: EXLA, precision: precision

  value ->
    raise "Invalid EXPHIL_EXLA_PRECISION: #{inspect(value)}"
end
