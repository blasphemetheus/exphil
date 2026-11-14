defmodule ExPhil.Stomped.Seat do
  @moduledoc """
  Hands the one warm policy agent to one match at a time.

  A policy agent is expensive to warm (JIT compilation runs far past the
  arena's 5 s and 10 s step deadlines), so the server warms one at boot and
  matches borrow it. The agent carries recurrent state, so it cannot be
  shared: a second match while one is live gets `:busy`.

  The borrower is monitored and the seat frees itself when it dies. The arena
  reconnects a dropped socket as a new process, and its first message can
  arrive before the old socket's exit is processed, so a `checkout` against a
  dead owner hands the agent over instead of answering `:busy`.
  """

  use GenServer

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, Keyword.fetch!(opts, :agent), Keyword.take(opts, [:name]))
  end

  @doc "Borrow the agent for the calling process."
  @spec checkout(GenServer.server()) :: {:ok, pid()} | :busy
  def checkout(seat), do: GenServer.call(seat, {:checkout, self()})

  @doc "Give the agent back (also automatic when the borrower dies)."
  @spec release(GenServer.server()) :: :ok
  def release(seat), do: GenServer.call(seat, {:release, self()})

  @impl true
  def init(agent), do: {:ok, %{agent: agent, owner: nil, ref: nil}}

  @impl true
  def handle_call({:checkout, pid}, _from, %{owner: owner} = state) do
    cond do
      owner == pid ->
        {:reply, {:ok, state.agent}, state}

      owner == nil or not Process.alive?(owner) ->
        if state.ref, do: Process.demonitor(state.ref, [:flush])
        {:reply, {:ok, state.agent}, %{state | owner: pid, ref: Process.monitor(pid)}}

      true ->
        {:reply, :busy, state}
    end
  end

  def handle_call({:release, pid}, _from, %{owner: pid} = state) do
    Process.demonitor(state.ref, [:flush])
    {:reply, :ok, %{state | owner: nil, ref: nil}}
  end

  def handle_call({:release, _pid}, _from, state), do: {:reply, :ok, state}

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, _reason}, %{ref: ref} = state) do
    {:noreply, %{state | owner: nil, ref: nil}}
  end

  def handle_info(_message, state), do: {:noreply, state}
end
