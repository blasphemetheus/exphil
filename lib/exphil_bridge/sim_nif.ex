defmodule ExPhil.Bridge.SimNif do
  @moduledoc """
  Raw NIF bindings over melee-sim-light's C batch API (`native/exphil_msl`,
  SIM_INTEGRATION.md step 10b). Rows are raw C-struct binaries:
  `MslMatchConfig` 52 B, `MslInput` 32 B, `MslObservation` 980 B,
  `MslTerminal` 16 B — decoded/encoded by `ExPhil.Bridge.SimRows` with the
  layouts in `priv/sim/layouts.json`. Use `ExPhil.Bridge.SimBatch`, not this.
  """

  use Rustler,
    otp_app: :exphil,
    crate: "exphil_msl",
    skip_compilation?:
      Mix.env() == :prod or System.find_executable("cargo") == nil or
        System.get_env("EXPHIL_SKIP_NIF_COMPILE") == "1"

  @doc "Load `libmelee_core.so` and create a batch of `batch_size` envs over `data_root`."
  def open(_lib_path, _data_root, _batch_size), do: :erlang.nif_error(:nif_not_loaded)
  def batch_size(_batch), do: :erlang.nif_error(:nif_not_loaded)
  def match_config_default(_batch), do: :erlang.nif_error(:nif_not_loaded)
  @doc "configs = batch_size × 52 bytes; mask = batch_size bytes or <<>> for all. Returns observation rows."
  def reset(_batch, _configs, _mask), do: :erlang.nif_error(:nif_not_loaded)
  @doc "inputs = batch_size × 32 bytes (raw MslInput). Returns {observation rows, terminal rows}."
  def step(_batch, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def observe(_batch), do: :erlang.nif_error(:nif_not_loaded)
  @doc "inputs = batch_size × 80 bytes (raw MslReplayInput, replay-exact). Returns {observation rows, terminal rows}."
  def step_replay(_batch, _inputs), do: :erlang.nif_error(:nif_not_loaded)
  def save(_batch, _index), do: :erlang.nif_error(:nif_not_loaded)
  def restore(_batch, _index, _state), do: :erlang.nif_error(:nif_not_loaded)
end
