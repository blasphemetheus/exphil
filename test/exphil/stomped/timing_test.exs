defmodule ExPhil.Stomped.TimingTest do
  @moduledoc """
  The one timing definition, pinned against a recorded arena game.

  Training (`--label-delay k`, causal convention) pairs the state at frame F
  with the raw controller of frame F + 1 + k. The arena asks a bot that has
  seen frame F for the pad of frame `F + lead`, with
  `lead = 1 + inputDelayFrames + lead_extra`. The two agree when
  `k = inputDelayFrames + lead_extra`: on the arena (`inputDelayFrames` 1) a
  checkpoint trained at k = 6 is served with `lead_extra: 5`, and its pads
  target F + 7.

  The deterministic bot below has the recording as an oracle: at frame F it
  returns the pad the recorded controller shows at F + 1 + k. If the lead
  arithmetic is right, every pad is addressed to the frame it was copied
  from, and the echo the arena plays back at that frame scores as landed.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Data.Peppi
  alias ExPhil.Eval.HarnessRung
  alias ExPhil.Stomped.{Decoder, Events, Match, Pad, Replay}
  alias ExPhil.Stomped.Bots.Policy
  alias ExPhil.Test.StompedStream, as: Stream

  @moduletag :nif

  @fixture Path.expand("../../fixtures/replays/stomped_stadium_Clm3suEY.slp", __DIR__)
  @input_delay 1
  @label_delay 6
  @lead_extra @label_delay - @input_delay

  defmodule Oracle do
    @moduledoc false
    @behaviour ExPhil.Stomped.Bot

    @impl true
    def init(opts), do: Map.new(opts) |> Map.put(:seen, [])

    @impl true
    def pick(_step, state), do: {%{"character" => 2, "costume" => 0}, state}

    @impl true
    def handle_frame(frame, _events, ctx, state) do
      # The pad for the target frame is whatever the recording shows the
      # game polled there; nothing is sent for targets past the recording.
      pad = Map.get(state.recorded, ctx.target, %{})
      {pad, %{state | seen: [{frame, ctx} | state.seen]}}
    end
  end

  # The controllers the live decoder reports, by frame, from the same bytes
  # the socket would see (the parity test proves these equal the recording).
  defp decoded_controllers(messages, sizes, decoder) do
    {out, _} =
      Enum.reduce(messages, {%{}, decoder}, fn message, {acc, decoder} ->
        {:ok, events, _} = Events.parse(message, sizes)

        case Events.frames(events) do
          {[{frame, frame_events}], []} ->
            {:ok, decoded, decoder} = Decoder.frame(decoder, frame, frame_events)
            {Map.put(acc, frame, decoded.controller), decoder}

          {[], _} ->
            {acc, decoder}
        end
      end)

    out
  end

  setup_all do
    {:ok, [header | messages]} = Replay.read_messages(@fixture)
    {:ok, header_events, sizes} = Events.parse(header, %{})
    {:ok, replay} = Peppi.parse(@fixture)

    %{
      header: header,
      messages: messages,
      sizes: sizes,
      header_events: header_events,
      replay: replay
    }
  end

  test "lead_extra 5 on the arena is reaction delay 6, the HarnessRung way" do
    assert HarnessRung.reaction_delay(:stomped, @lead_extra) == @label_delay
    assert HarnessRung.deploy_knob(:stomped, %{label_delay: @label_delay}) == {:ok, @lead_extra}
    assert Match.lead_for(0, @input_delay) == 1 + @input_delay
  end

  for port <- [0, 1] do
    test "port #{port}: a pad decided at frame F targets F + 1 + k, and its echo lands",
         %{header: header, messages: messages, sizes: sizes, header_events: header_events} do
      port = unquote(port)

      echoes =
        decoded_controllers(
          messages,
          sizes,
          port |> Decoder.new() |> Decoder.game_start(header_events)
        )

      recorded =
        Map.new(echoes, fn {frame, controller} -> {frame, Pad.from_controller(controller)} end)

      match = Match.new(bot: {Oracle, recorded: recorded}, lead_extra: @lead_extra)

      {_, match} =
        Match.handle_text(Stream.step("game", %{"frame" => -124, "port" => port}), match)

      assert match.input_delay == @input_delay
      assert match.lead == 1 + @input_delay + @lead_extra
      assert Match.reaction_delay(match) == @label_delay

      {_, match} = Match.handle_binary(header, match)

      {pads, match} =
        Enum.flat_map_reduce(messages, match, fn message, match ->
          {replies, match} = Match.handle_binary(message, match)

          pads =
            for {:binary, bytes} <- replies do
              {:ok, pad} = Pad.decode(bytes)
              pad
            end

          {pads, match}
        end)

      seen = Enum.reverse(match.bot_state.seen)
      assert length(seen) == map_size(echoes)

      # Every frame's ctx carries the one definition.
      for {frame, ctx} <- seen do
        assert ctx.target == frame + 1 + @label_delay
        assert ctx.lead == 1 + @input_delay + @lead_extra
        assert ctx.input_delay == @input_delay
        assert ctx.reaction_delay == @label_delay
        assert ctx.live?
      end

      # One pad per frame, addressed to F + 1 + k; it is the recorded pad of
      # that frame, so the controller training would label the state at F
      # with (k = label delay) is the one the arena polls for the pad.
      assert length(pads) == length(seen)

      for {{frame, _}, pad} <- Enum.zip(seen, pads) do
        target = frame + 1 + @label_delay
        assert pad.frame == target

        case recorded[target] do
          nil -> assert Policy.neutral?(Map.delete(pad, :frame))
          want -> assert Map.delete(pad, :frame) == want
        end
      end

      # The echo at the target frame is the pad the bot sent for it: the
      # policy bot's landed/late scoring reads this match as all landed.
      scored =
        for pad <- pads, echo = echoes[pad.frame], echo != nil do
          sent = Map.delete(pad, :frame)

          if Policy.neutral?(sent),
            do: :neutral,
            else: Policy.landed?(sent, Pad.from_controller(echo))
        end

      assert Enum.all?(scored, &(&1 in [true, :neutral]))
      assert Enum.count(scored, &(&1 == true)) > 0

      summary = Match.summary(match)
      assert summary.lead == 1 + @input_delay + @lead_extra
      assert summary.reaction_delay == @label_delay
      assert summary.lead_extra == @lead_extra
      assert summary.input_delay == @input_delay
      assert summary.frames == length(seen)
      assert summary.bot == nil
    end

    test "port #{port}: training's --label-delay k pairs state F with the controller at F + 1 + k",
         %{replay: replay, messages: messages, sizes: sizes, header_events: header_events} do
      port = unquote(port)

      echoes =
        decoded_controllers(
          messages,
          sizes,
          port |> Decoder.new() |> Decoder.game_start(header_events)
        )

      pairs =
        Peppi.to_training_frames(replay,
          player_port: port + 1,
          opponent_port: 2 - port,
          frame_delay: @label_delay,
          remap_ports: true
        )

      assert length(pairs) == map_size(echoes) - 1 - @label_delay

      for pair <- pairs do
        frame = pair.game_state.frame
        assert pair.frame_delay == @label_delay

        assert Pad.from_controller(pair.controller) ==
                 Pad.from_controller(echoes[frame + 1 + @label_delay])
      end
    end
  end

  test "the policy bot counts live frames whose lead disagrees with the agent's delay" do
    state = Policy.init(seat: self(), reaction_delay: @label_delay)
    ctx = %{port: 0, lead: 7, input_delay: 1, reaction_delay: 6, target: 7, live?: true}

    refute Policy.delay_mismatch?(state, ctx)
    assert Policy.delay_mismatch?(state, %{ctx | lead: 8, reaction_delay: 7})
    # A bot not told the agent's delay has nothing to compare against.
    refute Policy.delay_mismatch?(Policy.init(seat: self()), %{ctx | reaction_delay: 7})
  end
end
