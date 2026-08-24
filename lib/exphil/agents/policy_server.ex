defmodule ExPhil.Agents.PolicyServer do
  @moduledoc """
  Resident policy server (JIT_WARMUP option 3; design in
  docs/planning/POLICY_SERVER_DESIGN.md): one long-lived beam holds
  warmed Agents and serves them to play/eval sessions over local
  Erlang distribution. The JIT compile happens once per (checkpoint,
  config) per server boot — EXLA's executable cache is per-beam, so
  every later Agent for the same shapes warms in ~0s — and session
  boots collapse to menu time.

  A session checks out an Agent (a plain pid; `Agent`'s API is
  GenServer calls, so remote pids drop into AsyncRunner unchanged),
  and the server monitors the session: a crashed or finished session
  auto-releases its Agent.

  ## Hazards (from the design doc — read it before changing this)

  - Sessions must NEVER init a CUDA client (the second-EXLA-client
    law) — `--policy-server` sessions run `EXLA_CPU_ONLY=1`.
  - The server is a live exphil beam: the NO-MIX law applies while it
    runs, exactly like a training run.
  - The server runs the code it booted with; `status/1` reports its
    git rev so sessions can spot staleness.
  """

  use GenServer
  require Logger

  alias ExPhil.Agents.Agent

  @global_name {:global, __MODULE__}

  # ============================================================================
  # Client API (callable from remote session nodes)
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: @global_name)
  end

  @doc "Locate the server (global registry; sessions connect the node first)."
  def whereis, do: :global.whereis_name(__MODULE__)

  @doc """
  Check out a warmed Agent for `policy_path` with the given Agent opts.
  Returns `{:ok, agent_pid, %{warmup_ms: ms, cached: boolean}}`. The
  caller is monitored; its death releases the Agent automatically.
  """
  def checkout(policy_path, agent_opts \\ [], timeout \\ 120_000) do
    GenServer.call(@global_name, {:checkout, policy_path, agent_opts, self()}, timeout)
  end

  @doc "Release a checked-out Agent explicitly (also automatic on caller death)."
  def release(agent_pid) do
    GenServer.call(@global_name, {:release, agent_pid})
  end

  @doc "Pre-warm a checkpoint before any session asks for it."
  def preload(policy_path, agent_opts \\ [], timeout \\ 120_000) do
    GenServer.call(@global_name, {:preload, policy_path, agent_opts}, timeout)
  end

  @doc "Server status: rev, warmed checkpoints, live checkouts."
  def status do
    GenServer.call(@global_name, :status)
  end

  # ============================================================================
  # Server
  # ============================================================================

  @impl true
  def init(_opts) do
    # Agents are start_link'ed from this process; trap exits so a
    # crashed Agent cleans up its checkout instead of killing the
    # server (and every other session's Agent with it).
    Process.flag(:trap_exit, true)

    rev =
      case System.cmd("git", ["rev-parse", "--short", "HEAD"], stderr_to_stdout: true) do
        {out, 0} -> String.trim(out)
        _ -> "unknown"
      end

    Logger.info("[PolicyServer] up (rev #{rev})")

    {:ok,
     %{
       rev: rev,
       # policy_path => true once its first Agent warmed
       warmed: %{},
       # agent_pid => %{policy: path, session: pid, monitor: ref, key: term}
       checkouts: %{},
       # {policy_path, agent_opts} => [warm idle agents]. The design's
       # per-beam JIT amortization does NOT hold across Agent instances
       # (each Axon.build mints fresh closures = new cache identity —
       # measured 19.2s for the second Agent), so warm REUSE is the
       # real amortizer: released Agents park here and later checkouts
       # with the same key get them back instantly (reset, still warm).
       pool: %{}
     }}
  end

  # Options that change WHAT gets built/JIT'd (pool key); everything
  # else is a session tunable applied to a pooled agent via
  # Agent.reconfigure/2.
  @structural_opts [
    :stateful_step,
    :leace_eraser,
    :steer_vector,
    :steer_alpha,
    :player_registry,
    :uncertainty_log
  ]

  defp split_opts(agent_opts) do
    {structural, tunable} = Keyword.split(agent_opts, @structural_opts)

    # Normalize: drop nil/false entries so "explicitly default" equals
    # "absent" — a session's `stateful_step: false` must key the same
    # as a preload that never mentioned it (first smoke: every session
    # missed the preloaded pool over exactly this). steer_alpha only
    # matters alongside a steer_vector.
    structural =
      structural
      |> Enum.reject(fn
        {:steer_alpha, _} -> Keyword.get(structural, :steer_vector) in [nil, false]
        {_k, v} -> v in [nil, false]
      end)
      |> Enum.sort()

    {structural, tunable}
  end

  @impl true
  def handle_call({:checkout, policy_path, agent_opts, session}, _from, state) do
    {structural, tunable} = split_opts(agent_opts)
    key = {policy_path, structural}

    case pop_pooled(state, key) do
      {agent, state} when agent != nil ->
        case Agent.reconfigure(agent, tunable) do
          :ok ->
            :ok = Agent.reset_buffer(agent)
            ref = Process.monitor(session)

            checkouts =
              Map.put(state.checkouts, agent, %{
                policy: policy_path,
                session: session,
                monitor: ref,
                key: key
              })

            Logger.info(
              "[PolicyServer] checkout #{Path.basename(policy_path)} -> #{inspect(agent)} " <>
                "for #{inspect(session)} (POOLED, warmup 0ms)"
            )

            {:reply, {:ok, agent, %{warmup_ms: 0, cached: true}},
             %{state | checkouts: checkouts}}

          {:error, reason} ->
            # Bad tunables (e.g. untrained delay id): the agent stays
            # pooled; the session gets the error.
            {:reply, {:error, reason},
             %{state | pool: Map.update(state.pool, key, [agent], &[agent | &1])}}
        end

      {nil, state} ->
        case start_and_warm(policy_path, agent_opts, state) do
          {:ok, agent, warmup_ms} ->
            ref = Process.monitor(session)

            checkouts =
              Map.put(state.checkouts, agent, %{
                policy: policy_path,
                session: session,
                monitor: ref,
                key: key
              })

            Logger.info(
              "[PolicyServer] checkout #{Path.basename(policy_path)} -> #{inspect(agent)} " <>
                "for #{inspect(session)} (warmup #{warmup_ms}ms)"
            )

            {:reply, {:ok, agent, %{warmup_ms: warmup_ms, cached: false}},
             %{state | checkouts: checkouts, warmed: Map.put(state.warmed, policy_path, true)}}

          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
  end

  def handle_call({:preload, policy_path, agent_opts}, _from, state) do
    {structural, _tunable} = split_opts(agent_opts)
    key = {policy_path, structural}

    case start_and_warm(policy_path, agent_opts, state) do
      {:ok, agent, warmup_ms} ->
        # Park the warm Agent in the pool — Agent instances don't share
        # JIT identity, so the WARM PROCESS is the asset, not a cache.
        Logger.info("[PolicyServer] preloaded #{Path.basename(policy_path)} (#{warmup_ms}ms)")

        {:reply, {:ok, warmup_ms},
         %{
           state
           | warmed: Map.put(state.warmed, policy_path, true),
             pool: Map.update(state.pool, key, [agent], &[agent | &1])
         }}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:release, agent_pid}, _from, state) do
    {:reply, :ok, do_release(agent_pid, state)}
  end

  def handle_call(:status, _from, state) do
    {:reply,
     %{
       rev: state.rev,
       warmed: Map.keys(state.warmed),
       checkouts:
         Map.new(state.checkouts, fn {agent, %{policy: p, session: s}} ->
           {inspect(agent), %{policy: p, session: inspect(s)}}
         end)
     }, state}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Enum.find(state.checkouts, fn {_a, %{monitor: r}} -> r == ref end) do
      {agent, %{policy: policy}} ->
        Logger.info(
          "[PolicyServer] session for #{Path.basename(policy)} down (#{inspect(reason)}) — " <>
            "releasing #{inspect(agent)}"
        )

        {:noreply, do_release(agent, state)}

      nil ->
        {:noreply, state}
    end
  end

  def handle_info({:EXIT, pid, reason}, state) do
    if Map.has_key?(state.checkouts, pid) do
      Logger.warning("[PolicyServer] checked-out Agent #{inspect(pid)} exited: #{inspect(reason)}")
      {:noreply, do_release(pid, state)}
    else
      {:noreply, state}
    end
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ============================================================================
  # Internals
  # ============================================================================

  defp start_and_warm(policy_path, agent_opts, _state) do
    opts = Keyword.merge(agent_opts, policy_path: policy_path)

    with {:ok, agent} <- Agent.start_link(opts),
         {:ok, warmup_ms} <- Agent.warmup(agent) do
      {:ok, agent, warmup_ms}
    else
      {:error, reason} -> {:error, reason}
      other -> {:error, other}
    end
  end

  @pool_cap 4

  defp do_release(agent_pid, state) do
    case Map.pop(state.checkouts, agent_pid) do
      {nil, _} ->
        state

      {%{monitor: ref, key: key}, checkouts} ->
        Process.demonitor(ref, [:flush])
        state = %{state | checkouts: checkouts}

        cond do
          not Process.alive?(agent_pid) ->
            state

          length(Map.get(state.pool, key, [])) >= @pool_cap ->
            GenServer.stop(agent_pid, :normal)
            state

          true ->
            # Warm reuse: park it for the next same-key checkout.
            %{state | pool: Map.update(state.pool, key, [agent_pid], &[agent_pid | &1])}
        end
    end
  end

  # Pop a live pooled agent for the key (skipping any that died idle).
  defp pop_pooled(state, key) do
    {agent, rest} =
      state.pool
      |> Map.get(key, [])
      |> Enum.split_while(fn a -> not Process.alive?(a) end)
      |> then(fn {_dead, alive} ->
        case alive do
          [a | rest] -> {a, rest}
          [] -> {nil, []}
        end
      end)

    {agent, %{state | pool: Map.put(state.pool, key, rest)}}
  end
end
