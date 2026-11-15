defmodule ExPhil.Stomped.ProbeReplayTest do
  @moduledoc """
  The failing test for the arena: given the replay of a match played by
  `ExPhil.Stomped.Bots.Probe`, every frame's pre-frame controller must be
  the pad stamped for that frame (the spec: "the pad for frame F is what the
  game polls at frame F"). Run it with the replay's path:

      STOMPED_PROBE_REPLAY=path/to/match.slp mix test \
        test/exphil/stomped/probe_replay_test.exs --include external

  On 2026-11-15 arena replays failed it at 33-46 % of frames, in stretches
  recurring every 16.7 s (1000 frames), the beat of 59.94 Hz against 60 Hz.
  """
  use ExUnit.Case, async: true

  @moduletag :external
  @moduletag :nif

  @replay System.get_env("STOMPED_PROBE_REPLAY")

  if @replay do
    alias ExPhil.Stomped.{Decoder, Events, Replay}
    alias ExPhil.Stomped.Bots.Probe

    test "every pad is applied at the frame it was stamped for" do
      {:ok, [header | messages]} = Replay.read_messages(@replay)
      {:ok, header_events, sizes} = Events.parse(header, %{})

      offsets =
        for port <- [0, 1] do
          decoder = port |> Decoder.new() |> Decoder.game_start(header_events)

          {out, _} =
            Enum.reduce(messages, {[], decoder}, fn message, {acc, decoder} ->
              {:ok, events, _} = Events.parse(message, sizes)

              case Events.frames(events) do
                {[{frame, frame_events}], []} ->
                  {:ok, decoded, decoder} = Decoder.frame(decoder, frame, frame_events)
                  {[{frame, Probe.offset(decoded.controller, frame)} | acc], decoder}

                _ ->
                  {acc, decoder}
              end
            end)

          Enum.reverse(out)
        end
        |> Enum.max_by(fn offsets -> Enum.count(offsets, fn {_, o} -> is_integer(o) end) end)

      scored = Enum.filter(offsets, fn {_, o} -> is_integer(o) end)
      assert length(scored) > 600, "not a probe replay: too few frames echo a probe code"

      # Ordinary play also lands stick bytes in the code range, but spread over
      # all 50 offsets; a probe's sit on one or two.
      top_two =
        scored
        |> Enum.frequencies_by(&elem(&1, 1))
        |> Map.values()
        |> Enum.sort(:desc)
        |> Enum.take(2)
        |> Enum.sum()

      assert top_two >= 0.8 * length(scored),
             "not a probe replay: the decoded offsets are spread over many values"

      late = Enum.filter(scored, fn {_, o} -> o > 0 end)

      assert late == [],
             "#{length(late)} of #{length(scored)} frames (#{Float.round(100 * length(late) / length(scored), 1)}%) " <>
               "applied the pad stamped for an earlier frame; first: #{inspect(Enum.take(late, 5))}"
    end
  else
    @tag :skip
    test "needs STOMPED_PROBE_REPLAY" do
      flunk("set STOMPED_PROBE_REPLAY to a probe match's .slp")
    end
  end
end
