defmodule ExPhil.Bridge.InputPollRetryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.MeleePort

  test "internal timeout retries wait for the existing commitment instead of flushing again" do
    parent = self()

    console =
      spawn_link(fn ->
        for attempt <- 1..3 do
          receive do
            {:"$gen_call", from, {:step, flush?}} ->
              send(parent, {:flushed, flush?})
              reply = if attempt == 3, do: {:error, :enet_disconnected}, else: nil
              GenServer.reply(from, reply)
          end
        end
      end)

    state = %MeleePort.State{running: true, console: console}

    assert {:reply, {:game_ended, "dolphin_disconnected"}, %{running: false}} =
             MeleePort.handle_call({:step, [poll: false]}, nil, state)

    assert_receive {:flushed, true}
    assert_receive {:flushed, false}
    assert_receive {:flushed, false}
    refute_receive {:flushed, _}
  end
end
