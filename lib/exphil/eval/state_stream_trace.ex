defmodule ExPhil.Eval.StateStreamTrace do
  @moduledoc """
  Emits the LIVE half of a state-stream pair (task #8, GOTCHAS #81).

  A "pair" is one game recorded twice: the `.slp` Dolphin writes (what Peppi
  parses, the space policies train in) and a per-frame log of what the
  libmelee bridge reported as that same game ran (what a policy actually
  receives). Diffing a pair is how the parsed<->live `action_frame` mapping in
  `ExPhil.Data.ActionFrameConvention` was derived.

  The emitter lives at the bridge's decode boundary
  (`ExPhil.Bridge.MeleePort`), so ANY live script — drills, demos, policy
  rollouts, the multishine recorder — can produce a pair without its own
  tracing code. That matters because the table needs coverage of ordinary
  play (Mewtwo especially), not just the Fox multishine loop it came from.

  ## Recording a pair

      EXPHIL_STATE_TRACE=1 mix run scripts/<any live script>.exs ... \\
        > mewtwo_pair.live-trace.log 2>&1

  Then pair that log with the `.slp` Dolphin wrote for the SAME run (the
  newest file in your Slippi replay dir) and diff them:

      mix run scripts/diff_state_streams.exs \\
        --slp mewtwo_pair.slp --trace mewtwo_pair.live-trace.log

  Options are environment variables so no script needs a new flag:

    * `EXPHIL_STATE_TRACE=1` — enable
    * `EXPHIL_STATE_TRACE_PORT=N` — port to trace (default 1)

  ## Format

  One line per frame, mixed into stdout (build noise is fine —
  `StateStreamDiff.parse_trace/1` keeps only `[trace]` lines):

      [trace] f123 act=24 af=1 gnd=true y=0.0 vy=0.0

  `line/2` is the single definition of that format, shared with the parser's
  regex by a round-trip test, so emitter and parser cannot drift apart.

  ## Frame numbering

  This emitter reports the TRUE in-game frame, so a pair recorded with it
  aligns against the replay at offset 0. The two committed fixtures predate
  it and used a recorder-local 0-based counter (offset -123). Both work:
  `StateStreamDiff` anchors alignment on the first jumpsquat entry rather
  than assuming any particular numbering.
  """

  alias ExPhil.Bridge.GameState

  @enabled_var "EXPHIL_STATE_TRACE"
  @port_var "EXPHIL_STATE_TRACE_PORT"

  @doc """
  Whether tracing is switched on for this run.
  """
  @spec enabled?() :: boolean()
  def enabled?, do: System.get_env(@enabled_var) == "1"

  @doc """
  Port being traced (default 1).
  """
  @spec port() :: pos_integer()
  def port do
    case System.get_env(@port_var) do
      nil ->
        1

      raw ->
        case Integer.parse(raw) do
          {n, _} when n > 0 -> n
          _ -> 1
        end
    end
  end

  @doc """
  The canonical trace line for one frame of one player.

  Kept in lockstep with `ExPhil.Eval.StateStreamDiff.parse_trace/1` by a
  round-trip test — change one and the other must follow.

  ## Examples

      iex> player = %ExPhil.Bridge.Player{action: 24, action_frame: 1, on_ground: true, y: 0.0, speed_y_self: 0.0}
      iex> ExPhil.Eval.StateStreamTrace.line(7, player)
      "[trace] f7 act=24 af=1 gnd=true y=0.0 vy=0.0"

  """
  @spec line(integer(), map()) :: String.t()
  def line(frame, player) do
    "[trace] f#{frame} act=#{trunc_or(player.action)} af=#{trunc_or(player.action_frame)} " <>
      "gnd=#{player.on_ground == true} y=#{round2(player.y)} vy=#{round3(player.speed_y_self)}"
  end

  @doc """
  Emit a trace line for `game_state` when tracing is enabled.

  A no-op otherwise, so the call is safe to leave on the per-frame path.
  Returns the state unchanged so it can sit inline in a pipeline.
  """
  @spec maybe_emit(GameState.t() | nil) :: GameState.t() | nil
  def maybe_emit(state)

  def maybe_emit(%GameState{} = state) do
    if enabled?(), do: emit(state, port())
    state
  end

  def maybe_emit(other), do: other

  defp emit(%GameState{frame: frame, players: players}, port)
       when is_integer(frame) and is_map(players) do
    case players[port] do
      nil -> :ok
      player -> IO.puts(line(frame, player))
    end
  end

  defp emit(_state, _port), do: :ok

  defp trunc_or(nil), do: nil
  defp trunc_or(v) when is_number(v), do: trunc(v)
  defp trunc_or(v), do: v

  defp round2(nil), do: 0.0
  defp round2(v) when is_number(v), do: Float.round(v * 1.0, 2)

  defp round3(nil), do: 0.0
  defp round3(v) when is_number(v), do: Float.round(v * 1.0, 3)
end
