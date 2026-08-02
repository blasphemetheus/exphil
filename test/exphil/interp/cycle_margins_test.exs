defmodule ExPhil.Interp.CycleMarginsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Constants
  alias ExPhil.Interp.CycleMargins

  @ground_reflect Constants.reflector_ground() |> Enum.at(0)
  @jumpsquat Constants.jumpsquat()
  @aerial_jump Constants.aerial_jump()
  # An action id in no shine family (:other)
  @stand 14

  defp ctrl(overrides \\ []) do
    struct!(
      %ControllerState{
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0,
        button_a: false,
        button_b: false,
        button_x: false,
        button_y: false,
        button_z: false,
        button_l: false,
        button_r: false,
        button_d_up: false
      },
      overrides
    )
  end

  defp frame(n, action, ctrl_overrides) do
    %{
      game_state: %{frame: n, players: %{1 => %{action: action}}},
      controller: ctrl(ctrl_overrides)
    }
  end

  describe "event_of/3" do
    test "X edge in grounded reflector is a jc_event" do
      player = %{action: @ground_reflect}
      assert CycleMargins.event_of(player, ctrl(button_x: true), ctrl()) == :jc_event
    end

    test "held X is not an event (edges, not holds)" do
      player = %{action: @ground_reflect}
      held = ctrl(button_x: true)
      assert CycleMargins.event_of(player, held, held) == nil
    end

    test "B edge in jumpsquat or aerial jump is an aerial_shine_event" do
      for action <- [@jumpsquat, @aerial_jump] do
        assert CycleMargins.event_of(%{action: action}, ctrl(button_b: true), ctrl()) ==
                 :aerial_shine_event
      end
    end

    test "B edge in other grounded families is a ground_shine_event" do
      assert CycleMargins.event_of(%{action: @stand}, ctrl(button_b: true), ctrl()) ==
               :ground_shine_event
    end
  end

  describe "events/2" do
    test "classifies by the PRE-edge frame's family, consecutive frames only" do
      # Training frames pair state_t with the controller whose effect
      # state_t already shows — the decision state is t-1, so the family
      # is read there (the fixture's X edge lands on a jumpsquat frame,
      # its B edge on an air_reflect frame).
      frames = [
        frame(10, @ground_reflect, []),
        frame(11, @jumpsquat, button_x: true),
        # splice: frame number jumps — the X edge here is a synthesis artifact
        frame(50, @ground_reflect, button_x: true),
        frame(51, @jumpsquat, button_x: false),
        frame(52, @jumpsquat, button_b: true)
      ]

      assert CycleMargins.events(frames) == [{1, :jc_event}, {4, :aerial_shine_event}]
    end
  end

  describe "prepare/4 + margins/4" do
    test "gathers full windows and reduces margins to signed stats" do
      window = 4
      embed = 8
      emb = Nx.broadcast(0.0, {12, embed})

      events = [{1, :jc_event}, {5, :jc_event}, {9, :aerial_shine_event}]
      assert {stacked, kept} = CycleMargins.prepare(emb, events, window)
      # t=1 has no full window at window 4
      assert kept == [{5, :jc_event}, {9, :aerial_shine_event}]
      assert Nx.shape(stacked) == {2, window, embed}

      # Fake heads: X logit (col 2) positive, B logit (col 1) negative
      predict_fn = fn _params, batch ->
        {n, _, _} = Nx.shape(batch)
        buttons = Nx.tensor([[0.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) |> Nx.tile([n, 1])
        {buttons}
      end

      stats = CycleMargins.margins(predict_fn, nil, {stacked, kept})
      assert stats["jc_n"] == 1
      assert stats["jc_p10"] == 3.0
      assert stats["jc_flip"] == 0.0
      assert stats["aerial_n"] == 1
      assert stats["aerial_p10"] == -2.0
      assert stats["aerial_flip"] == 1.0
      # crit = min p10 over jc + aerial
      assert stats["crit_p10_min"] == -2.0
    end

    test "prepare returns nil with no usable events and caps event count" do
      emb = Nx.broadcast(0.0, {100, 4})
      assert CycleMargins.prepare(emb, [{0, :jc_event}], 16) == nil

      many = for t <- 15..99, do: {t, :jc_event}
      {_stacked, kept} = CycleMargins.prepare(emb, many, 16, max_events: 10)
      assert length(kept) <= 10
    end
  end
end
