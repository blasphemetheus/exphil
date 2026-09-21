defmodule ExPhil.Bridge.SimBatch do
  @moduledoc """
  GenServer over `ExPhil.Bridge.SimBatch.Core` (the C-API NIF batch) with the
  same call surface as `ExPhil.Bridge.SimPort`, so `ExPhil.Sim.Env` can
  address either backend by pid: `frames/1`, `step/2`, `reinit/2`,
  `reset/2`, `save/3`, `upload/2`, `restore/4`, `observe/1`, `stop/1`.
  """

  use GenServer
  alias ExPhil.Bridge.SimBatch.Core

  @timeout 60_000

  def start_link(opts \\ []) do
    {name_opts, opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, opts, name_opts)
  end

  def frames(pid), do: GenServer.call(pid, :frames, @timeout)
  def step(pid, controllers \\ nil), do: GenServer.call(pid, {:step, controllers}, @timeout)
  def reinit(pid, req), do: GenServer.call(pid, {:reinit, req}, @timeout)
  def reset(pid, env_ids \\ nil), do: GenServer.call(pid, {:reset, env_ids}, @timeout)
  def save(pid, env \\ 0, opts \\ []), do: GenServer.call(pid, {:save, env, opts}, @timeout)
  def upload(pid, blob), do: GenServer.call(pid, {:upload, blob}, @timeout)
  def restore(pid, env, state, opts \\ []), do: GenServer.call(pid, {:restore, env, state, opts}, @timeout)
  def observe(pid), do: GenServer.call(pid, :observe, @timeout)
  def request(_pid, _req), do: {:error, :not_supported_on_nif_backend}
  def stop(pid), do: GenServer.stop(pid, :normal)

  @impl true
  def init(opts) do
    case Core.start(opts) do
      {:ok, _frames, batch} -> {:ok, batch}
      {:error, reason} -> {:stop, {:sim_nif_init_failed, reason}}
    end
  end

  @impl true
  def handle_call(:frames, _from, b), do: {:reply, Core.frames(b), b}

  def handle_call({:step, controllers}, _from, b) do
    case Core.step(b, controllers) do
      {:ok, frames, terminals, b2} -> {:reply, {:ok, frames, terminals}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end

  def handle_call({:reinit, req}, _from, b) do
    case Core.reinit(b, req) do
      {:ok, frames, b2} -> {:reply, {:ok, frames}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end

  def handle_call({:reset, env_ids}, _from, b) do
    case Core.reset(b, env_ids) do
      {:ok, frames, b2} -> {:reply, {:ok, frames}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end

  def handle_call({:save, env, opts}, _from, b) do
    case Core.save(b, env, opts) do
      {:ok, blob, id, b2} -> {:reply, {:ok, blob, id}, b2}
      {:ok, blob, b2} -> {:reply, {:ok, blob}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end

  def handle_call({:upload, blob}, _from, b) do
    {:ok, id, b2} = Core.upload(b, blob)
    {:reply, {:ok, id}, b2}
  end

  def handle_call({:restore, env, state, opts}, _from, b) do
    case Core.restore(b, env, state, opts) do
      {:ok, frames, b2} -> {:reply, {:ok, frames}, b2}
      {:ok, b2} -> {:reply, {:ok, :restored}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end

  def handle_call(:observe, _from, b) do
    case Core.observe(b) do
      {:ok, frames, terminals, b2} -> {:reply, {:ok, frames, terminals}, b2}
      {:error, reason} -> {:reply, {:error, reason}, b}
    end
  end
end
