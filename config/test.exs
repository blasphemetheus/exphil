import Config

# Detect GPU availability for tests
# Uses EXLA (GPU) when available, falls back to BinaryBackend (CPU)
#
# Override with environment variable:
#   EXPHIL_GPU=1 mix test    # Force GPU
#   EXPHIL_GPU=0 mix test    # Force CPU
nx_backend =
  case System.get_env("EXPHIL_GPU") do
    "1" ->
      EXLA.Backend

    "0" ->
      Nx.BinaryBackend

    _ ->
      # Auto-detect: use EXLA if CUDA is available
      if System.get_env("EXLA_TARGET") == "cuda" or
           System.get_env("XLA_TARGET") == "cuda12" do
        EXLA.Backend
      else
        Nx.BinaryBackend
      end
  end

config :nx, default_backend: nx_backend

# Disable telemetry in tests
config :telemetry, :enabled, false
