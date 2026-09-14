defmodule ExPhil.Eval.ScenarioHistory do
  @moduledoc """
  Reconstructs committed decision history from a recorded demonstration.

  If the suite sends recorded input `t + input_offset` at frame `t`, a
  decision delayed by `response_delay` must have committed the recorded
  input `t + input_offset + response_delay`. This lookup warms the agent's
  decision history; it must not change the input physically replayed now.
  """

  def committed_input(inputs, frame, input_offset, response_delay)
      when is_integer(response_delay) and response_delay >= 0 do
    target_frame = frame + input_offset + response_delay

    case Map.fetch(inputs, target_frame) do
      {:ok, {player_input, _opponent_input}} -> player_input
      :error -> raise ArgumentError, "missing recorded decision input at frame #{target_frame}"
    end
  end

  def prepare_handoff(state, reset \\ &ExPhil.Agents.Agent.reset_buffer/1)

  def prepare_handoff(%{driver: :policy, prefix_history: "cold", agent: agent} = state, reset) do
    :ok = reset.(agent)
    state
  end

  def prepare_handoff(state, _reset), do: state
end
