defmodule ExPhil.Stomped.Socket do
  @moduledoc """
  The WebSocket handler for one stomped.dev match.

  All protocol logic is in `ExPhil.Stomped.Match`; this module moves frames
  between the socket and that state machine and logs the match summary when
  the socket closes.
  """

  @behaviour WebSock

  require Logger

  alias ExPhil.Stomped.Match

  @impl true
  def init(opts) do
    {:ok, Match.new(opts)}
  end

  @impl true
  def handle_in({data, opcode: :text}, match) do
    data |> Match.handle_text(match) |> push()
  end

  def handle_in({data, opcode: :binary}, match) do
    data |> Match.handle_binary(match) |> push()
  end

  @impl true
  def handle_info(_message, match), do: {:ok, match}

  @impl true
  def terminate(_reason, %Match{} = match) do
    # The deploy check opens and closes a socket without sending a step.
    if match.step != nil do
      summary = Match.summary(match)

      Logger.info(
        "[stomped #{summary.match_id}] closed: port #{inspect(summary.port)}, " <>
          "#{summary.frames} frames, lead #{summary.lead}, " <>
          "compute mean #{Float.round(summary.compute_ms_mean * 1.0, 2)} ms / " <>
          "max #{Float.round(summary.compute_ms_max * 1.0, 2)} ms, " <>
          "round trip #{rtt(summary.rtt_ms)}, " <>
          "result #{inspect(summary.result)}"
      )
    end

    :ok
  end

  def terminate(_reason, _state), do: :ok

  defp rtt(nil), do: "not measured"

  defp rtt(%{n: n, min: min, median: median, max: max}),
    do: "#{min}/#{median}/#{max} ms min/median/max over #{n} pings"

  defp push({[], match}), do: {:ok, match}
  defp push({replies, match}), do: {:push, replies, match}
end
