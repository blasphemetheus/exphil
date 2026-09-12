defmodule ExPhil.Bridge.LatencyProbeTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.LatencyProbe, as: P

  defp gs(frame, stick_x \\ 0.5) do
    %{frame: frame, players: %{1 => %{controller_state: %{main_stick: %{x: stick_x, y: 0.5}}}}}
  end

  # Drive the probe through a synthetic countdown where the game reports an
  # input `latency` frames after it was sent.
  defp run(probe, latency, frames \\ -123..-1) do
    Enum.reduce_while(frames, {probe, %{}, nil}, fn f, {p, sent, _} ->
      seen_x = Map.get(sent, f - latency, 0.5)
      {what, p} = P.step(p, gs(f, seen_x), 1)

      sent =
        case what do
          {:send, %{main_stick: %{x: x}}} when x > 0.9 -> Map.put(sent, f, x)
          _ -> sent
        end

      case what do
        {:done, r} -> {:halt, {p, sent, r}}
        _ -> {:cont, {p, sent, nil}}
      end
    end)
  end

  test "measures the frame gap between the marker send and its report" do
    for l <- [1, 2, 3, 5] do
      {p, _, r} = run(P.new(), l)
      assert r == {:ok, l}
      assert P.latency(p) == l
      assert P.done?(p)
    end
  end

  test "expected latency: ok when equal, mismatch otherwise, and describe says which way" do
    {_, _, r} = run(P.new(expected: 3), 3)
    assert r == {:ok, 3}

    {p, _, r} = run(P.new(expected: 3), 2)
    assert r == {:mismatch, 2}
    assert P.describe(p) =~ "FASTER"

    {p, _, r} = run(P.new(expected: 3), 5)
    assert r == {:mismatch, 5}
    assert P.describe(p) =~ "slower"
  end

  test "sends the marker once, then neutral while waiting, then passes forever" do
    {p, sent, _} = run(P.new(), 2)
    assert map_size(sent) == 1
    assert [{sent_frame, 0.95}] = Map.to_list(sent)
    assert sent_frame == -110
    assert P.step(p, gs(0), 1) == {:pass, p}
  end

  test "gives up after max_wait frames without a report" do
    {_, _, r} = run(P.new(max_wait: 5), 50)
    assert r == :unmeasured
  end

  test "a game already past the countdown cannot be probed" do
    {what, p} = P.step(P.new(), gs(10), 1)
    assert what == {:done, :unmeasured}
    assert P.done?(p)
  end

  test "frames before first_frame and non-integer frames pass" do
    assert {:pass, _} = P.step(P.new(), gs(-120), 1)
    assert {:pass, _} = P.step(P.new(), %{frame: nil, players: %{}}, 1)
  end
end
