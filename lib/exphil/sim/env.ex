defmodule ExPhil.Sim.Env do
  @moduledoc """
  Backend dispatch for sim environments. A `sim` is either a bare pid (the
  Python worker, `ExPhil.Bridge.SimPort`) or `{ExPhil.Bridge.SimBatch, pid}`
  (the C-API NIF). Drill and Search call only this module, so an experiment
  can switch backends with one option (`--backend port|nif`).

  `start/2` returns the handle: `Env.start(:port, opts)` / `Env.start(:nif, opts)`.
  """

  alias ExPhil.Bridge.{SimBatch, SimPort}

  @type sim :: pid() | {module(), pid()}

  def start(:port, opts) do
    with {:ok, pid} <- SimPort.start_link(opts), do: {:ok, pid}
  end

  def start(:nif, opts) do
    with {:ok, pid} <- SimBatch.start_link(opts), do: {:ok, {SimBatch, pid}}
  end

  def backend(pid) when is_pid(pid), do: :port
  def backend({SimBatch, _}), do: :nif

  def frames(sim), do: call(sim, :frames, [])
  def step(sim, controllers \\ nil), do: call(sim, :step, [controllers])
  def reinit(sim, req), do: call(sim, :reinit, [req])
  def reset(sim, env_ids \\ nil), do: call(sim, :reset, [env_ids])
  def save(sim, env \\ 0, opts \\ []), do: call(sim, :save, [env, opts])
  def upload(sim, blob), do: call(sim, :upload, [blob])
  def restore(sim, env, state, opts \\ []), do: call(sim, :restore, [env, state, opts])
  def observe(sim), do: call(sim, :observe, [])
  def request(sim, req), do: call(sim, :request, [req])
  def stop(sim), do: call(sim, :stop, [])

  defp call(pid, fun, args) when is_pid(pid), do: apply(SimPort, fun, [pid | args])
  defp call({mod, pid}, fun, args), do: apply(mod, fun, [pid | args])
end
