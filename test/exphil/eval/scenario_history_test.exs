defmodule ExPhil.Eval.ScenarioHistoryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.ScenarioHistory

  test "cold handoff clears the real agent buffer and controller queue" do
    {:ok, agent} = ExPhil.Agents.Agent.start_link([])
    on_exit(fn -> if Process.alive?(agent), do: GenServer.stop(agent) end)

    :sys.replace_state(agent, fn state ->
      %{
        state
        | frame_buffer: :queue.from_list([:old_frame]),
          controller_queue: [:old_command],
          last_controller: :old_command,
          last_action: :cached_action,
          step_frame_buffer: [:old_frame]
      }
    end)

    ScenarioHistory.prepare_handoff(%{driver: :policy, prefix_history: "cold", agent: agent})
    state = :sys.get_state(agent)
    assert :queue.is_empty(state.frame_buffer)
    assert state.controller_queue == []
    assert state.last_controller == nil
    assert state.last_action == nil
    assert state.step_frame_buffer == []
  end

  test "cold handoff resets agent history but preserves physical inputs and pending delivery" do
    state = %{
      driver: :policy,
      prefix_history: "cold",
      agent: :test_agent,
      inputs: %{100 => {:player, :opponent}},
      pending: [:committed],
      response_delay: 2
    }

    reset = fn agent ->
      send(self(), {:reset, agent})
      :ok
    end

    assert ScenarioHistory.prepare_handoff(state, reset) == state
    assert_received {:reset, :test_agent}

    for mode <- ["applied", "committed"] do
      warm = %{state | prefix_history: mode}
      assert ScenarioHistory.prepare_handoff(warm, reset) == warm
      refute_received {:reset, _}
    end

    teacher = %{state | driver: :teacher}
    assert ScenarioHistory.prepare_handoff(teacher, reset) == teacher
    refute_received {:reset, _}
  end

  test "warm-up uses the future committed input without changing replayed inputs" do
    inputs = Map.new(90..110, &{&1, {&1, -&1}})
    assert ScenarioHistory.committed_input(inputs, 100, 1, 4) == 105
    assert inputs[101] == {101, -101}
    assert ScenarioHistory.committed_input(inputs, 99, 1, 4) == 104
  end

  test "zero delay preserves current causal input and offsets compose" do
    inputs = Map.new(90..110, &{&1, {&1, -&1}})
    assert ScenarioHistory.committed_input(inputs, 100, 1, 0) == 101
    assert ScenarioHistory.committed_input(inputs, 100, 0, 4) == 104
  end

  test "missing future evidence fails rather than fabricating neutral history" do
    assert_raise ArgumentError, ~r/missing recorded decision input at frame 105/, fn ->
      ScenarioHistory.committed_input(%{101 => {:present, :opponent}}, 100, 1, 4)
    end
  end

  test "history before handoff consists of already committed pending inputs" do
    inputs = Map.new(90..110, &{&1, {&1, -&1}})

    pending =
      for frame <- 96..99,
          do: ScenarioHistory.committed_input(inputs, frame, 1, 4)

    assert pending == [101, 102, 103, 104]
    assert ScenarioHistory.committed_input(inputs, 100, 1, 4) == 105
  end
end
