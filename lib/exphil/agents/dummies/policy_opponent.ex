defmodule ExPhil.Agents.Dummies.PolicyOpponent do
  @moduledoc """
  A second POLICY on the opponent port — the checkpoint ladder's (task
  #19) policy-vs-policy engine, riding the existing external-dummy hook.

  Not a scripted dummy: `new/1` starts a full `ExPhil.Agents.Agent` for
  the given checkpoint, and `step_game/3` (the rich dummy contract —
  full game state, not just the two player structs) swaps the players
  map so the second policy sees ITSELF as port 1, then returns the
  agent's ACTION for the runner to convert with its own
  `action_to_input/2` (exact same conversion path as the port-1 policy).

  Inference runs INLINE in the frame loop (the port-1 agent infers in a
  separate process). Headless games pace to the loop (GOTCHAS #69), so a
  ladder match simply runs sub-real-time; both policies still see every
  frame. Async-split for port 2 is a later optimization if ladder
  throughput matters.
  """

  alias ExPhil.Agents.Agent

  defstruct [:agent]

  @doc """
  Start the opponent agent. `opts` mirror the port-1 agent flags the
  ladder cares about: `:policy_path` (required), `:deterministic`,
  `:temperature`, `:press_threshold`, `:release_threshold`.
  """
  def new(opts) do
    {:ok, agent} =
      Agent.start_link(
        policy_path: Keyword.fetch!(opts, :policy_path),
        deterministic: Keyword.get(opts, :deterministic, false),
        temperature: Keyword.get(opts, :temperature, 1.0),
        press_threshold: Keyword.get(opts, :press_threshold),
        release_threshold: Keyword.get(opts, :release_threshold)
      )

    # CRITICAL: warm the JIT now, not on the first frame. The play script
    # warms P1 only; without this, port 2's first inference compiles XLA
    # INSIDE the frame loop — the loop starves (~0.1 fps observed,
    # 2026-07-20 ladder smoke) until the bridge watchdog kills the match.
    case Agent.warmup(agent) do
      {:ok, _ms} -> :ok
      other -> raise "PolicyOpponent warmup failed: #{inspect(other)}"
    end

    %__MODULE__{agent: agent}
  end

  @doc """
  Rich dummy contract: full game state + the port this policy controls.
  Returns `{controller_state | nil, state}` — a fully-decoded
  `%ControllerState{}` (via the agent's own decode pipeline, hysteresis
  and all), which is the ONLY shape `action_to_input/2` accepts. nil on
  inference error (runner skips the send that frame).
  """
  def step_game(game_state, self_port, %__MODULE__{agent: agent} = s) do
    case Agent.get_controller(agent, swap_perspective(game_state, self_port)) do
      {:ok, controller} -> {controller, s}
      {:error, _} -> {nil, s}
    end
  end

  # The policy was trained seeing itself as players[1]; mirror the map so
  # port `self_port` lands there (same normalization the drill applies to
  # port-flipped fixtures).
  defp swap_perspective(game_state, 1), do: game_state

  defp swap_perspective(game_state, self_port) do
    other = if self_port == 1, do: 2, else: 1
    players = game_state.players

    %{game_state | players: %{1 => players[self_port], 2 => players[other]}}
  end
end
