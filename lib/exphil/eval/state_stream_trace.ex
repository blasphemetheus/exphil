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
  Port being traced (default 1). First port when several are configured.
  """
  @spec port() :: pos_integer()
  def port, do: ports() |> hd()

  @doc """
  Ports being traced, from `EXPHIL_STATE_TRACE_PORT` (default `[1]`).

  Accepts a comma-separated list (`"1,2"`). Tracing two ports in ONE recording
  is the only way to compare them under identical run timing — which is what
  separates a per-character effect from a per-run one, since both ports then
  share the same sampling phase by construction.
  """
  @spec ports() :: [pos_integer(), ...]
  def ports do
    case System.get_env(@port_var) do
      nil ->
        [1]

      raw ->
        parsed =
          raw
          |> String.split(",", trim: true)
          |> Enum.flat_map(fn part ->
            case Integer.parse(String.trim(part)) do
              {n, _} when n > 0 -> [n]
              _ -> []
            end
          end)
          |> Enum.uniq()

        if parsed == [], do: [1], else: parsed
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
  def line(frame, player), do: line(frame, player, nil)

  @doc """
  As `line/2`, tagging the line with `p=<port>`.

  The tag is emitted ONLY when several ports are traced, so a single-port
  trace stays byte-identical to the three vendored pairs in
  `test/fixtures/statestream/` (whose format is pinned by round-trip tests).
  `parse_trace/2` treats an untagged line as belonging to whichever port is
  asked for, which is what keeps those legacy traces readable.
  """
  @spec line(integer(), map(), pos_integer() | nil) :: String.t()
  def line(frame, player, port) do
    tag = if port, do: " p=#{port}", else: ""

    "[trace]#{tag} f#{frame} act=#{trunc_or(player.action)} af=#{trunc_or(player.action_frame)} " <>
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
    if enabled?() and in_game?(state) do
      case ports() do
        [single] -> emit(state, single, nil)
        many -> Enum.each(many, &emit(state, &1, &1))
      end
    end

    state
  end

  def maybe_emit(other), do: other

  # Menu frames must NOT be traced. The frame counter resets between games and
  # idles during menus, so a trace spanning menus contains DUPLICATE frame
  # numbers — and a duplicate silently pairs the wrong live frame to a parsed
  # one. Observed cost: a Mewtwo pair scored 98.0% action agreement (looks
  # nearly right, is unusable) purely from post-game frames after the counter
  # reset; trimming to the in-game run took it to exactly 100%.
  #
  # libmelee Menu values: IN_GAME = 2, SUDDEN_DEATH = 3.
  @in_game_menu_states [2, 3]

  defp in_game?(%GameState{menu_state: menu}) when is_integer(menu),
    do: menu in @in_game_menu_states

  # Unknown/absent menu_state: emit rather than silently record nothing —
  # a noisy trace is recoverable, a missing one is not.
  defp in_game?(_state), do: true

  defp emit(%GameState{frame: frame, players: players}, port, tag)
       when is_integer(frame) and is_map(players) do
    case players[port] do
      nil -> :ok
      player -> IO.puts(line(frame, player, tag))
    end
  end

  defp emit(_state, _port, _tag), do: :ok

  defp trunc_or(nil), do: nil
  defp trunc_or(v) when is_number(v), do: trunc(v)
  defp trunc_or(v), do: v

  defp round2(nil), do: 0.0
  defp round2(v) when is_number(v), do: Float.round(v * 1.0, 2)

  defp round3(nil), do: 0.0
  defp round3(v) when is_number(v), do: Float.round(v * 1.0, 3)
end
