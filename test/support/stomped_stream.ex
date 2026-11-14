defmodule ExPhil.Test.StompedStream do
  @moduledoc """
  Synthetic stomped.dev traffic for tests: a minimal `.slp` event stream and
  the arena's JSON step frames.

  The payload sizes here are small on purpose. Real streams declare their own
  sizes in the Event Payloads event, and the code under test must honour the
  declared table rather than assume the real ones.
  """

  @sizes [{0x36, 4}, {0x3A, 8}, {0x37, 8}, {0x38, 16}, {0x3C, 8}, {0x39, 2}]

  @doc "The first binary message: Event Payloads then Game Start."
  def header do
    table = for {cmd, size} <- @sizes, into: <<>>, do: <<cmd, size::big-16>>
    <<0x35, byte_size(table) + 1>> <> table <> <<0x36, 0::32>>
  end

  @doc "One frame's message. `xs` maps port → x position."
  def frame(number, xs \\ %{0 => -40.0, 1 => 40.0}) do
    ports = xs |> Map.keys() |> Enum.sort()

    pre = for port <- ports, into: <<>>, do: <<0x37, number::big-signed-32, port, 0::24>>

    post =
      for port <- ports, into: <<>> do
        <<0x38, number::big-signed-32, port, 0, 2, 14::16, xs[port]::big-float-32, 0::24>>
      end

    <<0x3A, number::big-signed-32, 0::32>> <>
      pre <> post <> <<0x3C, number::big-signed-32, number::big-signed-32>>
  end

  @doc "The last binary message."
  def game_end, do: <<0x39, 2, 0>>

  @doc "A step frame as the arena sends it (JSON text)."
  def step(name, extra \\ %{}) do
    %{
      "step" => name,
      "version" => 1,
      "sentAt" => 1_761_740_000_123,
      "matchId" => "Xk3p9QaB",
      "mode" => "live",
      "port" => 1,
      "opponent" => %{"botId" => "b7Qm2ZkA", "name" => "rusher"},
      "stage" => 31,
      "inputDelayFrames" => 1
    }
    |> Map.merge(extra)
    |> Jason.encode!()
  end
end
