defmodule ExPhil.Bridge.SimPortTest do
  @moduledoc """
  End-to-end worker test (SIM_INTEGRATION.md step 2). Needs the sim main
  clone built, so it is tagged `:external` and excluded by default:

      mix test test/exphil_bridge/sim_port_test.exs --include external
  """
  use ExUnit.Case, async: false

  alias ExPhil.Bridge.{ControllerState, GameState, Player, SimPort}

  @moduletag :external

  setup do
    {:ok, sim} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox"}, %{character: "fox", costume: 1}], length: 16)
    on_exit(fn -> if Process.alive?(sim), do: SimPort.stop(sim) end)
    %{sim: sim}
  end

  test "init returns a mapped Fox ditto on FD at Slippi frame -124 (one tick before the first recorded frame)", %{sim: sim} do
    {:ok, [%GameState{} = gs]} = SimPort.frames(sim)
    assert gs.frame == -124 and gs.stage == 32
    assert %Player{character: 1, stock: 4, action: 322} = gs.players[1]
    assert gs.players[1].x < 0 and gs.players[2].x > 0
  end

  test "stepping past the buffer window keeps counting frames", %{sim: sim} do
    y = %ControllerState{
      main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
      button_a: false, button_b: false, button_x: false, button_y: true, button_z: false,
      button_l: false, button_r: false, button_d_up: false
    }

    frames =
      for t <- 1..40 do
        {:ok, [gs], [_term]} = SimPort.step(sim, [[y, nil]])
        gs.frame
      end

    assert frames == Enum.to_list(-123..-84)
  end

  test "binary step and JSON step produce identical mapped frames", %{sim: sim} do
    {:ok, [a], [ta]} = SimPort.step(sim, [[nil, nil]])
    {:ok, json_sim} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox"}, %{character: "fox", costume: 1}], length: 16, binary: false)
    {:ok, [b], [tb]} = SimPort.step(json_sim, [[nil, nil]])
    assert a == b and ta == tb
    SimPort.stop(json_sim)
  end

  test "save/restore returns to the saved frame", %{sim: sim} do
    for _ <- 1..5, do: {:ok, _, _} = SimPort.step(sim)
    {:ok, [before]} = SimPort.frames(sim)
    {:ok, blob} = SimPort.save(sim, 0)
    for _ <- 1..5, do: {:ok, _, _} = SimPort.step(sim)
    {:ok, [restored]} = SimPort.restore(sim, 0, blob)
    assert restored.frame == before.frame
    assert restored.players[1] == before.players[1]
  end
end
