defmodule ExPhil.Stomped.MatchTest do
  use ExUnit.Case, async: true

  alias ExPhil.Stomped.{Events, Match, Pad}
  alias ExPhil.Stomped.Bots.Walker
  alias ExPhil.Test.StompedStream, as: Stream

  defmodule Recorder do
    @moduledoc false
    @behaviour ExPhil.Stomped.Bot

    @impl true
    def init(opts), do: %{test: Keyword.fetch!(opts, :test), picks: 0}

    @impl true
    def pick(_step, state), do: {%{character: 10, costume: 1}, %{state | picks: state.picks + 1}}

    @impl true
    def ready(step, state) do
      send(state.test, {:ready, step["players"]})
      state
    end

    @impl true
    def handle_events(events, state) do
      send(state.test, {:events, Enum.map(events, &elem(&1, 0))})
      state
    end

    @impl true
    def handle_frame(frame, _events, ctx, state) do
      send(state.test, {:frame, frame, ctx})
      {%{buttons: [:b], stick_y: 0}, state}
    end

    @impl true
    def over(step, state), do: send(state.test, {:over, step["winner"]})
  end

  defp recorder(opts \\ []) do
    Match.new([bot: {Recorder, test: self()}] ++ opts)
  end

  defp pads(replies) do
    for {:binary, bytes} <- replies do
      {:ok, pad} = Pad.decode(bytes)
      pad
    end
  end

  # Step replies, without the round-trip pings that ride along with them.
  defp texts(replies) do
    for {:text, json} <- replies,
        reply = Jason.decode!(json),
        reply != %{"step" => "ping"},
        do: reply
  end

  defp pings(replies) do
    for {:text, json} <- replies, Jason.decode!(json) == %{"step" => "ping"}, do: :ping
  end

  describe "Pad" do
    test "encodes the documented layout" do
      bytes = Pad.encode(-123, %{buttons: [:a, :z], stick_x: 188, l: 255})

      assert bytes == <<-123::little-signed-32, 0x0110::little-16, 188, 128, 128, 128, 255, 0>>
      assert byte_size(bytes) == 12
    end

    test "neutral is no buttons, centred sticks, released triggers" do
      assert Pad.neutral(7) == <<7::little-32, 0, 0, 128, 128, 128, 128, 0, 0>>
    end

    test "clamps out-of-range values instead of raising" do
      assert {:ok, %{stick_x: 255, stick_y: 0}} =
               Pad.decode(Pad.encode(0, %{stick_x: 300, stick_y: -4}))
    end
  end

  describe "Events" do
    test "reads the size table from the header and frames by bookend" do
      {:ok, header_events, sizes} = Events.parse(Stream.header(), %{})
      assert [{0x35, _}, {0x36, <<0::32>>}] = header_events
      assert sizes[0x38] == 16

      {:ok, events, ^sizes} = Events.parse(Stream.frame(5) <> Stream.frame(6), sizes)
      assert {[{5, first}, {6, _second}], []} = Events.frames(events)
      assert Enum.map(first, &elem(&1, 0)) == [0x3A, 0x37, 0x37, 0x38, 0x38, 0x3C]
    end

    test "stops at a command with no declared size" do
      {:ok, _, sizes} = Events.parse(Stream.header(), %{})

      assert {:partial, [{0x3A, _}], ^sizes, {:unknown_command, 0x99}} =
               Events.parse(<<0x3A, 0::64, 0x99, 1, 2>>, sizes)
    end

    test "stops at a payload cut short" do
      {:ok, _, sizes} = Events.parse(Stream.header(), %{})
      assert {:partial, [], ^sizes, {:truncated_payload, 0x3C}} = Events.parse(<<0x3C, 1>>, sizes)
    end
  end

  describe "steps" do
    test "character: replies with the bot's pick" do
      {replies, match} = Match.handle_text(Stream.step("character"), recorder())

      assert texts(replies) == [%{"step" => "character", "character" => 10, "costume" => 1}]
      assert match.port == 1
      assert match.match_id == "Xk3p9QaB"
    end

    test "character: a resent step repeats the same pick without asking the bot again" do
      {_, match} = Match.handle_text(Stream.step("character"), recorder())

      {replies, match} =
        Match.handle_text(Stream.step("character", %{"error" => "bad costume"}), match)

      assert [%{"character" => 10}] = texts(replies)
      assert match.bot_state.picks == 1
    end

    test "ready: tells the bot the matchup once and replies" do
      players = [%{"character" => 9, "costume" => 0}, %{"character" => 10, "costume" => 1}]

      {replies, match} =
        Match.handle_text(Stream.step("ready", %{"players" => players}), recorder())

      assert texts(replies) == [%{"step" => "ready"}]
      assert_received {:ready, ^players}

      {_, _} = Match.handle_text(Stream.step("ready", %{"players" => players}), match)
      refute_received {:ready, _}
    end

    test "game: covers the frames that are already open with neutral pads" do
      {replies, match} = Match.handle_text(Stream.step("game", %{"frame" => -124}), recorder())

      assert Enum.map(pads(replies), & &1.frame) == [-123, -122]
      assert Enum.all?(pads(replies), &(&1.buttons == 0 and &1.stick_x == 128))
      assert texts(replies) == []
      assert match.lead == 2
    end

    test "game: lead_extra aims further ahead" do
      {replies, match} =
        Match.handle_text(Stream.step("game", %{"frame" => -124}), recorder(lead_extra: 2))

      assert Enum.map(pads(replies), & &1.frame) == [-123, -122, -121, -120]
      assert match.lead == 4
    end

    test "over: records the result and tells the bot" do
      over = Stream.step("over", %{"winner" => 1, "endReason" => "stocks", "missed" => [0, 6]})
      {[], match} = Match.handle_text(over, recorder())

      assert_received {:over, 1}

      assert %{"winner" => 1, "endReason" => "stocks", "missed" => [0, 6]} =
               Match.summary(match).result
    end

    test "unreadable text is ignored" do
      assert {[], %Match{}} = Match.handle_text("not json", recorder())
      assert {[], %Match{}} = Match.handle_text(~s({"no":"step"}), recorder())
    end
  end

  describe "the game stream" do
    setup do
      {_, match} = Match.handle_text(Stream.step("game", %{"frame" => -124}), recorder())
      %{match: match}
    end

    test "the header reaches the bot and asks for no pad", %{match: match} do
      {replies, match} = Match.handle_binary(Stream.header(), match)

      assert replies == []
      assert_received {:events, [0x35, 0x36]}
      assert map_size(match.sizes) > 0
    end

    test "one pad per frame, for frame + 1 + inputDelayFrames", %{match: match} do
      {_, match} = Match.handle_binary(Stream.header(), match)
      {replies, match} = Match.handle_binary(Stream.frame(-123), match)

      assert [%{frame: -121, buttons: 0x0200, stick_y: 0}] = pads(replies)

      assert_received {:frame, -123,
                       %{
                         port: 1,
                         lead: 2,
                         input_delay: 1,
                         reaction_delay: 1,
                         target: -121,
                         live?: true
                       }}

      assert Match.summary(match).frames == 1
    end

    test "frame arrival spacing is measured between consecutive frames", %{match: match} do
      {_, match} = Match.handle_binary(Stream.header(), match)
      assert Match.summary(match).arrival_ms == nil

      {_, match} = Match.handle_binary(Stream.frame(-123), match)
      assert Match.summary(match).arrival_ms == nil

      Process.sleep(40)
      {_, match} = Match.handle_binary(Stream.frame(-122), match)
      {_, match} = Match.handle_binary(Stream.frame(-121), match)

      assert %{n: 2, median: median, p99: p99, max: max, stalls: 1} =
               Match.summary(match).arrival_ms

      assert max >= 40 and p99 == max and median <= max
    end

    test "frames before the header are dropped", %{match: match} do
      assert {[], _} = Match.handle_binary(Stream.frame(10), match)
      refute_received {:frame, _, _}
    end

    test "Game End reaches the bot and asks for no pad", %{match: match} do
      {_, match} = Match.handle_binary(Stream.header(), match)
      {replies, _} = Match.handle_binary(Stream.game_end(), match)

      assert replies == []
      assert_received {:events, [0x39]}
    end
  end

  describe "round trip" do
    test "lead_for: the arena's own figure with no network, a frame more per 16.7 ms" do
      # 11 ms budget at the floor: 2 ms compute + 3 ms margin leaves about 6 ms of network.
      assert Match.lead_for(0, 1) == 2
      assert Match.lead_for(5, 1) == 2
      assert Match.lead_for(7, 1) == 3
      assert Match.lead_for(25, 1) == 4
      assert Match.lead_for(80, 1) == 7
    end

    test "a ping rides with the pick, and its answer sets the lead when auto" do
      {replies, match} = Match.handle_text(Stream.step("character"), recorder(lead_extra: :auto))
      assert pings(replies) == [:ping]
      assert match.lead == 2

      Process.sleep(40)
      {replies, match} = Match.handle_text(Stream.step("character"), match)

      # The answer is the same step again. It is not a new question: replying
      # a second time can land after the step moved on ("wrong step").
      assert texts(replies) == []
      assert [rtt] = match.rtt_all
      assert rtt >= 40
      assert match.lead == Match.lead_for(rtt, 1)
      assert match.lead >= 4
      assert %{n: 1, min: ^rtt} = Match.summary(match).rtt_ms
    end

    test "a pinned lead is measured but not changed" do
      {_, match} = Match.handle_text(Stream.step("character"), recorder(lead_extra: 1))
      Process.sleep(40)
      {_, match} = Match.handle_text(Stream.step("character"), match)

      assert [_rtt] = match.rtt_all
      assert match.lead == 3
    end

    test "a frame of the next step is not taken for the ping's answer" do
      {_, match} = Match.handle_text(Stream.step("character"), recorder(lead_extra: :auto))
      {_, match} = Match.handle_text(Stream.step("ready", %{"players" => []}), match)

      assert match.rtt_all == []
      assert match.ping_at == nil
    end

    test "pings are at most one a second" do
      {first, match} = Match.handle_text(Stream.step("character"), recorder())
      {_, match} = Match.handle_text(Stream.step("character"), match)
      {second, _} = Match.handle_text(Stream.step("ready", %{"players" => []}), match)

      assert pings(first) == [:ping]
      assert pings(second) == []
    end

    test "in game: a ping every two seconds, and its answer does not resend the opening pads" do
      {_, match} = Match.handle_text(Stream.step("game", %{"frame" => -124}), recorder())
      {_, match} = Match.handle_binary(Stream.header(), match)

      {sent, match} =
        Enum.reduce(-123..-80, {[], match}, fn frame, {sent, match} ->
          {replies, match} = Match.handle_binary(Stream.frame(frame), match)
          {sent ++ Enum.map(pings(replies), fn _ -> frame end), match}
        end)

      # First ping half a second in.
      assert sent == [-94]

      {replies, match} = Match.handle_text(Stream.step("game", %{"frame" => -80}), match)
      assert replies == []
      assert [_rtt] = match.rtt_all
    end
  end

  describe "reconnect" do
    test "joining mid-game asks for a replay and drops frames until it arrives" do
      {replies, match} = Match.handle_text(Stream.step("game", %{"frame" => 500}), recorder())

      assert Enum.map(pads(replies), & &1.frame) == [501, 502]
      assert texts(replies) == [%{"step" => "replay"}]

      assert {[], match} = Match.handle_binary(Stream.frame(501), match)
      refute_received {:frame, _, _}

      replay = Stream.header() <> Stream.frame(500) <> Stream.frame(501) <> Stream.frame(502)
      {replies, match} = Match.handle_binary(replay, match)

      # The bot sees every frame; only the newest earns a pad.
      assert_received {:frame, 500, %{live?: false}}
      assert_received {:frame, 501, %{live?: false}}
      assert_received {:frame, 502, %{live?: true}}
      assert [%{frame: 504}] = pads(replies)

      {replies, _} = Match.handle_binary(Stream.frame(503), match)
      assert [%{frame: 505}] = pads(replies)
    end

    test "a reconnect that kept its parser does not ask for a replay" do
      {_, match} = Match.handle_text(Stream.step("game", %{"frame" => -124}), recorder())
      {_, match} = Match.handle_binary(Stream.header(), match)

      {replies, _} = Match.handle_text(Stream.step("game", %{"frame" => 300}), match)
      assert texts(replies) == []
    end
  end

  describe "Walker" do
    defp walker_pad(xs, port) do
      match = Match.new(bot: {Walker, []})

      {_, match} =
        Match.handle_text(Stream.step("game", %{"frame" => -124, "port" => port}), match)

      {_, match} = Match.handle_binary(Stream.header(), match)
      {replies, _} = Match.handle_binary(Stream.frame(0, xs), match)
      hd(pads(replies))
    end

    test "picks its configured character" do
      match = Match.new(bot: {Walker, character: 25, costume: 2})
      {replies, _} = Match.handle_text(Stream.step("character"), match)
      assert texts(replies) == [%{"step" => "character", "character" => 25, "costume" => 2}]
    end

    test "walks toward the opponent from either side" do
      assert %{stick_x: 188, buttons: 0} = walker_pad(%{0 => -40.0, 1 => 40.0}, 0)
      assert %{stick_x: 68, buttons: 0} = walker_pad(%{0 => -40.0, 1 => 40.0}, 1)
    end

    test "stops walking in reach" do
      assert %{stick_x: 128} = walker_pad(%{0 => -4.0, 1 => 4.0}, 0)
    end
  end
end
