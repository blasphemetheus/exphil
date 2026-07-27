defmodule ExPhil.Eval.StateStreamTraceTest do
  use ExUnit.Case, async: false

  alias ExPhil.Bridge.GameState
  alias ExPhil.Bridge.Player, as: PlayerState
  alias ExPhil.Eval.StateStreamDiff
  alias ExPhil.Eval.StateStreamTrace

  doctest ExPhil.Eval.StateStreamTrace

  @enabled_var "EXPHIL_STATE_TRACE"
  @port_var "EXPHIL_STATE_TRACE_PORT"

  setup do
    # async: false — these mutate process env.
    prev = {System.get_env(@enabled_var), System.get_env(@port_var)}

    on_exit(fn ->
      {e, p} = prev
      if e, do: System.put_env(@enabled_var, e), else: System.delete_env(@enabled_var)
      if p, do: System.put_env(@port_var, p), else: System.delete_env(@port_var)
    end)

    System.delete_env(@enabled_var)
    System.delete_env(@port_var)
    :ok
  end

  defp player(opts \\ []) do
    %PlayerState{
      action: Keyword.get(opts, :action, 24),
      action_frame: Keyword.get(opts, :af, 1),
      on_ground: Keyword.get(opts, :gnd, true),
      y: Keyword.get(opts, :y, 0.0),
      speed_y_self: Keyword.get(opts, :vy, 0.0)
    }
  end

  describe "line/2 matches the committed fixture format exactly" do
    test "reproduces a real line from fox_ms_frame1" do
      # Taken verbatim from test/fixtures/statestream/fox_ms_frame1.live-trace.log.
      # The emitter was refactored onto this shared formatter; if the format
      # drifts, previously recorded pairs stop parsing.
      assert StateStreamTrace.line(0, player(action: 322, af: -1, gnd: false, y: 10.0, vy: 0.0)) ==
               "[trace] f0 act=322 af=-1 gnd=false y=10.0 vy=0.0"

      assert StateStreamTrace.line(5, player(action: 323, af: 1, gnd: false, y: 10.04, vy: 0.0)) ==
               "[trace] f5 act=323 af=1 gnd=false y=10.04 vy=0.0"
    end

    test "every line in a committed fixture round-trips through the formatter" do
      {:ok, rows} =
        StateStreamDiff.parse_trace("test/fixtures/statestream/fox_ms_frame1.live-trace.log")

      # Re-emit each parsed row and re-parse it; values must survive.
      for row <- Enum.take(rows, 50) do
        reemitted =
          StateStreamTrace.line(row.f, %PlayerState{
            action: row.act,
            action_frame: row.af,
            on_ground: row.gnd,
            y: row.y,
            speed_y_self: row.vy
          })

        {:ok, [back]} = parse_one(reemitted)
        assert back == row
      end
    end
  end

  describe "emitter and parser cannot drift" do
    test "a freshly emitted line parses back to the same values" do
      emitted =
        StateStreamTrace.line(-123, player(action: 24, af: 3, gnd: true, y: -1.25, vy: 2.5))

      {:ok, [row]} = parse_one(emitted)

      assert row.f == -123
      assert row.act == 24
      assert row.af == 3
      assert row.gnd == true
      assert row.y == -1.25
      assert row.vy == 2.5
    end

    test "float action ids are emitted as integers so the parser accepts them" do
      # The old inline emitter used inspect/1, which would render a float
      # action as "322.0" — unmatched by the parser's act=(-?\\d+) and thus
      # silently dropped. Normalizing here removes that trap.
      emitted = StateStreamTrace.line(1, player(action: 322.0, af: 2.0))
      assert emitted =~ "act=322 "
      assert emitted =~ "af=2 "
      assert {:ok, [_]} = parse_one(emitted)
    end

    test "nil velocity and nil y degrade to 0.0 rather than crashing a live run" do
      emitted = StateStreamTrace.line(1, player(y: nil, vy: nil))
      assert emitted =~ "y=0.0"
      assert emitted =~ "vy=0.0"
    end
  end

  describe "enablement" do
    test "disabled by default" do
      refute StateStreamTrace.enabled?()
    end

    test "enabled only by exactly \"1\"" do
      System.put_env(@enabled_var, "1")
      assert StateStreamTrace.enabled?()

      System.put_env(@enabled_var, "true")
      refute StateStreamTrace.enabled?()
    end

    test "port defaults to 1 and tolerates junk" do
      assert StateStreamTrace.port() == 1

      System.put_env(@port_var, "2")
      assert StateStreamTrace.port() == 2

      System.put_env(@port_var, "garbage")
      assert StateStreamTrace.port() == 1
    end

    test "ports/0 parses a comma-separated list" do
      assert StateStreamTrace.ports() == [1]

      System.put_env(@port_var, "1,2")
      assert StateStreamTrace.ports() == [1, 2]

      System.put_env(@port_var, " 2 , 1 ")
      assert StateStreamTrace.ports() == [2, 1]

      System.put_env(@port_var, "2,2")
      assert StateStreamTrace.ports() == [2]

      System.put_env(@port_var, "garbage,0,-1")
      assert StateStreamTrace.ports() == [1]
    end
  end

  describe "multi-port tracing" do
    test "single port stays byte-identical to the vendored format" do
      # The three committed pairs have no p= tag. If a tag leaked into the
      # single-port path they would stop matching the pinned format.
      refute StateStreamTrace.line(7, player()) =~ "p="
      assert StateStreamTrace.line(7, player(), nil) == StateStreamTrace.line(7, player())
    end

    test "several ports emit one tagged line each per frame" do
      System.put_env(@enabled_var, "1")
      System.put_env(@port_var, "1,2")

      state = %GameState{
        frame: 5,
        players: %{1 => player(action: 24), 2 => player(action: 365)}
      }

      out = ExUnit.CaptureIO.capture_io(fn -> StateStreamTrace.maybe_emit(state) end)
      lines = out |> String.split("\n", trim: true)

      assert length(lines) == 2
      assert Enum.any?(lines, &(&1 =~ "p=1" and &1 =~ "act=24"))
      assert Enum.any?(lines, &(&1 =~ "p=2" and &1 =~ "act=365"))
    end

    test "a tagged trace round-trips per port through the parser" do
      System.put_env(@enabled_var, "1")
      System.put_env(@port_var, "1,2")

      state = %GameState{
        frame: 5,
        players: %{1 => player(action: 24, af: 2), 2 => player(action: 365, af: 3)}
      }

      out = ExUnit.CaptureIO.capture_io(fn -> StateStreamTrace.maybe_emit(state) end)
      path = Path.join(System.tmp_dir!(), "dual_#{System.unique_integer([:positive])}.log")
      File.write!(path, out)

      {:ok, p1} = StateStreamDiff.parse_trace(path, port: 1)
      {:ok, p2} = StateStreamDiff.parse_trace(path, port: 2)
      File.rm(path)

      assert [%{act: 24, af: 2}] = p1
      assert [%{act: 365, af: 3}] = p2
    end

    test "untagged legacy lines are returned for whichever port is asked" do
      # Vendored pairs predate the tag; filtering must not silently drop them.
      {:ok, rows} =
        StateStreamDiff.parse_trace(
          "test/fixtures/statestream/fox_ms_frame1.live-trace.log",
          port: 2
        )

      assert length(rows) == 300
    end
  end

  describe "maybe_emit/1" do
    test "is a silent pass-through when disabled" do
      state = %GameState{frame: 10, players: %{1 => player()}}

      out =
        ExUnit.CaptureIO.capture_io(fn -> assert StateStreamTrace.maybe_emit(state) == state end)

      assert out == ""
    end

    test "emits the traced port when enabled" do
      System.put_env(@enabled_var, "1")
      state = %GameState{frame: 10, players: %{1 => player(action: 24, af: 1)}}

      out = ExUnit.CaptureIO.capture_io(fn -> StateStreamTrace.maybe_emit(state) end)

      assert out =~ "[trace] f10 act=24 af=1"
    end

    test "honours EXPHIL_STATE_TRACE_PORT" do
      System.put_env(@enabled_var, "1")
      System.put_env(@port_var, "2")

      state = %GameState{
        frame: 3,
        players: %{1 => player(action: 24), 2 => player(action: 365)}
      }

      out = ExUnit.CaptureIO.capture_io(fn -> StateStreamTrace.maybe_emit(state) end)

      assert out =~ "act=365"
      refute out =~ "act=24"
    end

    test "stays quiet on a menu frame with no such port" do
      System.put_env(@enabled_var, "1")
      state = %GameState{frame: 1, players: %{}}

      out = ExUnit.CaptureIO.capture_io(fn -> StateStreamTrace.maybe_emit(state) end)

      assert out == ""
    end

    test "passes non-GameState values through untouched" do
      assert StateStreamTrace.maybe_emit(nil) == nil
    end
  end

  defp parse_one(line) do
    path = Path.join(System.tmp_dir!(), "trace_#{System.unique_integer([:positive])}.log")
    File.write!(path, line <> "\n")
    result = StateStreamDiff.parse_trace(path)
    File.rm(path)
    result
  end
end
