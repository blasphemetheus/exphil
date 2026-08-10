defmodule ExPhil.InspectTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Bridge.GameState
  alias ExPhil.Bridge.Player
  alias ExPhil.Inspect

  @window 4

  # Stub loaded-policy map (Activations.load_heads shape) with a fixed
  # head output: B strongly pressed, main_x peaked at bucket 3
  defp stub_loaded do
    %{
      kind: :heads,
      params: %{},
      window: @window,
      config: %{temporal: true, use_prev_action: false},
      predict_fn: fn _params, win ->
        {n, _w, _e} = Nx.shape(win)

        {
          Nx.tensor([[-4.0, 4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0]]) |> Nx.tile([n, 1]),
          Nx.iota({n, 17}, axis: 1) |> Nx.equal(3) |> Nx.multiply(8.0) |> Nx.as_type(:f32),
          Nx.broadcast(0.0, {n, 17}),
          Nx.broadcast(0.0, {n, 17}),
          Nx.broadcast(0.0, {n, 17}),
          Nx.broadcast(0.0, {n, 5})
        }
      end
    }
  end

  defp player(overrides) do
    struct(
      %Player{
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        action: 14,
        action_frame: 1,
        facing: 1,
        on_ground: true,
        jumps_left: 2,
        shield_strength: 60.0,
        hitstun_frames_left: 0
      },
      overrides
    )
  end

  defp frames(n) do
    for i <- 0..(n - 1) do
      %{
        game_state: %GameState{
          frame: i,
          stage: 32,
          players: %{1 => player(x: i * 2.0), 2 => player(x: 50.0)}
        },
        controller: ControllerState.neutral()
      }
    end
  end

  defp session do
    Inspect.from_frames(stub_loaded(), frames(10), player_port: 1)
  end

  test "moment before the window has situations but no policy" do
    m = Inspect.moment(session(), 0)
    assert m.policy == nil
    assert :neutral in m.situations
    assert m.players.own.x == 0.0
  end

  test "moment after the window exposes per-head distributions" do
    m = Inspect.moment(session(), 6)

    assert m.policy.buttons.b > 0.95
    assert m.policy.buttons.a < 0.05
    assert m.policy.pressed == [:b]

    assert m.policy.main_x.argmax == 3
    assert_in_delta Enum.sum(m.policy.main_x.probs), 1.0, 1.0e-5

    # uniform logits -> max entropy for main_y
    assert m.policy.main_y.entropy > m.policy.main_x.entropy
    assert m.recorded.buttons == []
  end

  test "moment is JSON-encodable" do
    assert {:ok, _} = Jason.encode(Inspect.moment(session(), 6))
  end

  test "out-of-range index raises" do
    assert_raise ArgumentError, fn -> Inspect.moment(session(), 99) end
  end

  test "counterfactual patches the whole window and returns both policies" do
    cf = Inspect.counterfactual(session(), 6, fn p -> %{p | x: p.x + 25.0} end)

    assert cf.baseline_policy != nil
    assert cf.policy != nil
    # stub policy ignores input, so distributions match — the contract
    # here is shape + presence; behavioral deltas are integration-tested
    assert cf.policy.main_x.argmax == cf.baseline_policy.main_x.argmax
  end

  test "situation labels line up with the frame index" do
    # x = 2i: by index 9 the fox is at 18, still center-stage neutral
    m = Inspect.moment(session(), 9)
    assert :onstage_center in m.situations
  end
end
