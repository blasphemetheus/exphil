defmodule ExPhil.Stomped.DecoderParityTest do
  @moduledoc """
  The decoder's acceptance gate: for real arena games, every frame decoded
  from the live message stream equals the frame training builds from the
  stored replay (Peppi NIF + `to_training_frames`), from both seats.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Data.Peppi
  alias ExPhil.Stomped.{Decoder, Events, Pad, Replay}

  @moduletag :nif

  @fixtures Path.wildcard(Path.expand("../../fixtures/replays/stomped_*.slp", __DIR__))

  # Decode a whole game from its messages, as the socket would see it.
  defp decode(path, port) do
    {:ok, [header | messages]} = Replay.read_messages(path)
    {:ok, header_events, sizes} = Events.parse(header, %{})
    decoder = port |> Decoder.new() |> Decoder.game_start(header_events)

    {decoded, _decoder} =
      Enum.reduce(messages, {[], decoder}, fn message, {acc, decoder} ->
        {:ok, events, _} = Events.parse(message, sizes)

        case Events.frames(events) do
          {[{frame, frame_events}], []} ->
            {:ok, out, decoder} = Decoder.frame(decoder, frame, frame_events)
            {[out | acc], decoder}

          # Game End
          {[], _events} ->
            {acc, decoder}
        end
      end)

    Enum.reverse(decoded)
  end

  test "there are arena fixtures to test against" do
    assert length(@fixtures) >= 5
  end

  for path <- @fixtures do
    name = Path.basename(path, ".slp")

    describe name do
      @describetag path: path

      test "the file cuts into a header, one message per frame, and Game End", %{path: path} do
        {:ok, bytes} = File.read(path)
        {:ok, raw} = Replay.raw(bytes)
        {:ok, [header | rest] = messages} = Replay.messages(raw)

        assert IO.iodata_to_binary(messages) == raw
        assert Events.header?(header)
        {:ok, _, sizes} = Events.parse(header, %{})

        {frames, [game_end]} = Enum.split(rest, -1)
        assert <<0x39, _::binary>> = game_end

        numbers =
          for message <- frames do
            {:ok, events, _} = Events.parse(message, sizes)
            assert {[{frame, _}], []} = Events.frames(events)
            assert {0x3A, _} = hd(events)
            frame
          end

        # No rollbacks in an arena stream: -123, -122, ... without a gap.
        assert numbers == Enum.to_list(-123..(-123 + length(numbers) - 1))
      end

      for port <- [0, 1] do
        test "port #{port}: every frame equals the training frame", %{path: path} do
          port = unquote(port)
          decoded = decode(path, port)

          {:ok, replay} = Peppi.parse(path)

          expected =
            replay
            |> Peppi.to_training_frames(
              player_port: port + 1,
              opponent_port: 2 - port,
              remap_ports: true
            )
            # The one allowed difference: training stamps `own_port: 1` only when
            # it had to move the subject (a subject already on port 1 keeps nil,
            # which every consumer reads as 1). The decoder always stamps it.
            |> Map.new(&{&1.game_state.frame, %{&1.game_state | own_port: 1}})

          assert length(decoded) == length(replay.frames)
          # Training pairs each state with a later input, so the last frames have no pair.
          assert map_size(expected) >= length(decoded) - 2

          mismatches =
            for %{frame: frame, game_state: state} <- decoded,
                want = expected[frame],
                want != nil and want != state do
              {frame, diff(want, state)}
            end

          assert mismatches == [],
                 "#{length(mismatches)} frames differ; first: #{inspect(List.first(mismatches), pretty: true)}"

          compared = Enum.count(decoded, &Map.has_key?(expected, &1.frame))
          assert compared == map_size(expected)
        end

        test "port #{port}: the echoed controller equals the recorded one", %{path: path} do
          port = unquote(port)
          {:ok, replay} = Peppi.parse(path)
          recorded = Map.new(replay.frames, &{&1.frame_number, &1.players[port + 1].controller})

          for %{frame: frame, controller: got} <- decode(path, port) do
            want = recorded[frame]

            assert got.main_stick == %{x: want.main_stick_x, y: want.main_stick_y}
            assert got.c_stick == %{x: want.c_stick_x, y: want.c_stick_y}
            assert got.l_shoulder == want.l_trigger
            assert got.r_shoulder == want.r_trigger

            for button <- ~w(a b x y z l r d_up)a do
              assert Map.fetch!(got, :"button_#{button}") == Map.fetch!(want, :"button_#{button}")
            end
          end
        end

        # The pad we would send for a recorded input is the raw byte the game
        # recorded, wherever the game's own processing is invertible: inside
        # the unit circle and outside the dead zone.
        test "port #{port}: stick bytes invert the game's processing", %{path: path} do
          port = unquote(port)
          {:ok, replay} = Peppi.parse(path)

          checks =
            for frame <- replay.frames,
                c = frame.players[port + 1].controller,
                p = c.processed,
                {raw_x, raw_y, x, y, unit_x, unit_y} <- [
                  {p.raw_main_x, p.raw_main_y, p.main_x, p.main_y, c.main_stick_x, c.main_stick_y}
                ],
                raw_x != nil and raw_y != nil,
                raw_x * raw_x + raw_y * raw_y <= 80 * 80,
                {raw, processed, unit} <- [{raw_x, x, unit_x}, {raw_y, y, unit_y}],
                processed != 0.0 do
              {Pad.stick_byte(unit), 128 + raw}
            end

          # A bot may never tilt a stick (one of these seats does not), so an
          # empty set is not a failure here; the count is printed for the record.
          wrong = Enum.reject(checks, fn {ours, recorded} -> ours == recorded end)

          assert wrong == [],
                 "#{length(wrong)}/#{length(checks)} axes differ, e.g. #{inspect(Enum.take(wrong, 5))}"

          IO.puts("  #{Path.basename(path)} port #{port}: #{length(checks)} stick axes checked")
        end
      end
    end
  end

  defp diff(want, got) do
    top =
      for key <- Map.keys(Map.from_struct(want)),
          key != :players,
          Map.get(want, key) != Map.get(got, key),
          do: {key, want: Map.get(want, key), got: Map.get(got, key)}

    players =
      for slot <- [1, 2],
          w = want.players[slot],
          g = got.players[slot],
          w != g,
          key <- Map.keys(Map.from_struct(w || %ExPhil.Bridge.Player{})),
          Map.get(w || %{}, key) != Map.get(g || %{}, key),
          do: {{:player, slot, key}, want: Map.get(w || %{}, key), got: Map.get(g || %{}, key)}

    top ++ players
  end
end
