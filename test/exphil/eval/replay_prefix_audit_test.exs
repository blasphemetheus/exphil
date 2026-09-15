defmodule ExPhil.Eval.ReplayPrefixAuditTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.ReplayPrefixAudit, as: Audit

  test "waits for an incomplete recording and fails closed on missing or ambiguous files" do
    source = "test/fixtures/replays/fox_multishine_closed_d1.slp"
    directory = Path.join(System.tmp_dir!(), "prefix-audit-#{System.unique_integer([:positive])}")
    File.mkdir!(directory)
    on_exit(fn -> File.rm_rf!(directory) end)
    assert %{valid: false} = Audit.verify_directory(source, directory, -39, -38, [1, 2], 0)
    recording = Path.join(directory, "test.slp")
    File.write!(recording, "incomplete")

    writer =
      Task.async(fn ->
        Process.sleep(100)
        File.cp!(source, recording)
      end)

    assert %{valid: true} = Audit.verify_directory(source, directory, -39, -38, [1, 2])
    Task.await(writer)
    File.cp!(source, Path.join(directory, "second.slp"))

    assert %{valid: false, error: error} =
             Audit.verify_directory(source, directory, -39, -38, [1, 2])

    assert error =~ "ambiguous"
  end

  defp frame(n, x \\ 0.0, input \\ 0.0) do
    %{frame_number: n, players: %{1 => %{x: x, action_frame: 1.25, controller: %{main_x: input}}}}
  end

  defmodule SampleController do
    defstruct axis: 0.0
  end

  test "compares nested structs returned by the NIF" do
    f = put_in(frame(0), [:players, 1, :controller], %SampleController{})
    assert %{valid: true} = Audit.compare([f], [f], 0, 0, [1])
    changed = put_in(f, [:players, 1, :controller], %SampleController{axis: -0.0})
    assert %{valid: false} = Audit.compare([f], [changed], 0, 0, [1])
  end

  test "mixed audit exempts only byte-port raw sticks, and still checks later input changes" do
    base = %{
      frame_number: 0,
      players:
        Map.new([1, 2], fn port ->
          {port, %{x: 0.0, controller: %{processed: %{main_x: 0.65, raw_main_x: 70}}}}
        end)
    }

    changed = put_in(base, [:players, 1, :controller, :processed, :raw_main_x], 52)
    assert %{valid: false} = Audit.compare([base], [changed], 0, 0)
    assert %{valid: true} = Audit.compare_mixed([base], [changed], 0, 0, [2])
    changed = put_in(changed, [:players, 2, :controller, :processed, :raw_main_x], 52)
    assert %{valid: false} = Audit.compare_mixed([base], [changed], 0, 0, [2])
    next = %{base | frame_number: 1}
    next_changed = put_in(next, [:players, 1, :controller, :processed, :main_x], 0.5)
    assert %{valid: false} = Audit.compare_mixed([base, next], [base, next_changed], 0, 1, [2])
  end

  test "compares the whole requested interval" do
    frames = [frame(-39), frame(-38)]
    assert %{valid: true, ports: [%{compared: 2}]} = Audit.compare(frames, frames, -39, -38, [1])
  end

  test "detects differences hidden by the old rounding and truncation" do
    a = frame(0)
    b = frame(0, 0.00001, 0.000001)

    assert %{valid: false, ports: [%{input: %{frame: 0}, state: %{frame: 0}}]} =
             Audit.compare([a], [b], 0, 0, [1])

    b = put_in(a, [:players, 1, :action_frame], 1.5)
    assert %{valid: false} = Audit.compare([a], [b], 0, 0, [1])
  end

  test "missing frames and missing players fail even when both are absent" do
    for {a, b, port} <- [{[], [], 1}, {[frame(0)], [], 1}, {[frame(0)], [frame(0)], 2}] do
      assert %{valid: false, ports: [%{missing: %{frame: 0}, compared: 0}]} =
               Audit.compare(a, b, 0, 0, [port])
    end
  end

  test "distinguishes signed zero in recorded inputs" do
    assert %{valid: false, ports: [%{input: %{frame: 0}}]} =
             Audit.compare([frame(0)], [frame(0, 0.0, -0.0)], 0, 0, [1])
  end

  test "rejects duplicate frames instead of silently choosing one" do
    assert_raise ArgumentError, ~r/duplicate frame/, fn ->
      Audit.compare([frame(0), frame(0)], [frame(0)], 0, 0, [1])
    end
  end

  test "does not compare intentional changes after the handoff boundary" do
    assert %{valid: true} =
             Audit.compare([frame(0), frame(1)], [frame(0), frame(1, 2.0)], 0, 0, [1])
  end
end
