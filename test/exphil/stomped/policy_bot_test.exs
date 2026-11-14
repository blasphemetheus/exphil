defmodule ExPhil.Stomped.PolicyBotTest do
  use ExUnit.Case, async: true

  alias ExPhil.Eval.HarnessRung
  alias ExPhil.Stomped.{Match, Pad, Seat}
  alias ExPhil.Stomped.Bots.Policy
  alias ExPhil.Test.StompedStream, as: Stream

  describe "Seat" do
    setup do
      agent = spawn(fn -> Process.sleep(:infinity) end)
      seat = start_supervised!({Seat, agent: agent})
      %{seat: seat, agent: agent}
    end

    test "one borrower at a time", %{seat: seat, agent: agent} do
      assert {:ok, ^agent} = Seat.checkout(seat)
      # The same process asking again keeps its seat.
      assert {:ok, ^agent} = Seat.checkout(seat)

      assert Task.await(Task.async(fn -> Seat.checkout(seat) end)) == :busy

      assert :ok = Seat.release(seat)
      assert {:ok, ^agent} = Task.await(Task.async(fn -> Seat.checkout(seat) end))
    end

    test "a dead borrower's seat is handed over at once", %{seat: seat, agent: agent} do
      owner = spawn(fn -> Seat.checkout(seat) end)
      ref = Process.monitor(owner)
      assert_receive {:DOWN, ^ref, :process, ^owner, _}

      # No wait for the seat to process the exit: a reconnecting socket's first
      # message can beat it.
      assert {:ok, ^agent} = Seat.checkout(seat)
    end
  end

  describe "hold-ahead" do
    test "each pad is repeated for the following frame" do
      match = Match.new(bot: {ExPhil.Stomped.Bots.Walker, []}, hold_ahead: 1)
      {_, match} = Match.handle_text(Stream.step("game", %{"frame" => -124, "port" => 0}), match)
      {_, match} = Match.handle_binary(Stream.header(), match)
      {replies, match} = Match.handle_binary(Stream.frame(0), match)

      assert [{:ok, first}, {:ok, second}] = for({:binary, b} <- replies, do: Pad.decode(b))
      assert first.frame == 2 and second.frame == 3
      assert Map.delete(first, :frame) == Map.delete(second, :frame)

      # The next frame's own pad overwrites the cover it was given.
      {replies, _} = Match.handle_binary(Stream.frame(1), match)

      assert [{:ok, %{frame: 3}}, {:ok, %{frame: 4}}] =
               for({:binary, b} <- replies, do: Pad.decode(b))
    end
  end

  describe "what landed" do
    defp pad(overrides), do: Map.merge(Pad.from_controller(neutral()), Map.new(overrides))

    defp neutral do
      %ExPhil.Bridge.ControllerState{
        main_stick: %{x: 0.5, y: 0.5},
        c_stick: %{x: 0.5, y: 0.5},
        l_shoulder: 0.0,
        r_shoulder: 0.0
      }
    end

    test "a neutral pad cannot be told from a late one" do
      assert Policy.neutral?(pad([]))
      # Inside the dead zone the game reads centre.
      assert Policy.neutral?(pad(stick_x: 128 + 22))
      refute Policy.neutral?(pad(stick_x: 128 + 23))
      refute Policy.neutral?(pad(buttons: 0x0100))
    end

    test "landed: the echo is what we sent" do
      sent = pad(buttons: 0x0100, stick_x: 128 + 80)
      assert Policy.landed?(sent, sent)
      refute Policy.landed?(sent, pad([]))
      refute Policy.landed?(sent, pad(buttons: 0x0200, stick_x: 128 + 80))
    end

    test "landed: a dead-zone axis echoes as centre" do
      sent = pad(buttons: 0x0100, stick_x: 128 + 10, stick_y: 128 - 80)
      assert Policy.landed?(sent, pad(buttons: 0x0100, stick_x: 128, stick_y: 128 - 80))
    end

    test "landed: a corner echoes clamped to the stick's circle" do
      sent = pad(buttons: 0x0100, stick_x: 128 + 80, stick_y: 128 + 80)
      # (80, 80) has radius 113; the game scales it back to radius 80.
      assert Policy.landed?(sent, pad(buttons: 0x0100, stick_x: 128 + 56, stick_y: 128 + 56))
      refute Policy.landed?(sent, pad(buttons: 0x0100, stick_x: 128 + 80, stick_y: 128))
    end
  end

  describe "Pad.from_controller" do
    test "inverts the game's stick and trigger scaling" do
      controller = %ExPhil.Bridge.ControllerState{
        main_stick: %{x: 1.0, y: 0.0},
        c_stick: %{x: 0.75, y: 0.5},
        l_shoulder: 1.0,
        r_shoulder: 0.5,
        button_a: true,
        button_l: true,
        button_d_up: true
      }

      assert Pad.from_controller(controller) == %{
               buttons: 0x0100 + 0x0040 + 0x0008,
               stick_x: 208,
               stick_y: 48,
               c_x: 168,
               c_y: 128,
               l: 140,
               r: 70
             }
    end
  end

  describe "the :stomped harness" do
    test "lead_extra N plays at reaction delay 1 + N" do
      assert HarnessRung.reaction_delay(:stomped, 0) == 1
      assert HarnessRung.reaction_delay(:stomped, 2) == 3
      assert HarnessRung.knob(:stomped, HarnessRung.latency(:training, 3)) == {:ok, 2}
    end

    test "reaction delay 0 is unreachable on the arena" do
      assert {:error, {:unreachable, 2}} =
               HarnessRung.knob(:stomped, HarnessRung.latency(:training, 0))
    end
  end
end
