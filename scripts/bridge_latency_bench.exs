# Step-latency benchmark: native libmelee_ex bridge vs the legacy Python
# melee_bridge.
#
# Measures the per-frame cost a bot actually pays: step (receive a frame)
# + send_controller (write inputs). Runs a headless FD game vs a stand
# dummy at realtime pace so both bridges face the same 60 Hz frame supply,
# then reports percentiles over the in-game frames.
#
# Usage: devenv shell -- mix run scripts/bridge_latency_bench.exs [--frames N]

alias ExPhil.Bridge.MeleePort

defmodule Bench do
  @dolphin Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless")
  @iso "/home/blewf/isos/melee.iso"

  def config(slippi_port) do
    %{
      dolphin_path: @dolphin,
      iso_path: @iso,
      headless: true,
      character: :fox,
      stage: :final_destination,
      dummy_mode: "stand",
      dummy_character: :fox,
      slippi_port: slippi_port,
      console_timeout: 0.1,
      # Unthrottled + blocking input: the game paces to OUR loop, so
      # per-frame time measures bridge overhead (plus emulator frame
      # compute, identical for both) instead of a 60 Hz wait.
      emulation_speed: 0.0
    }
  end

  def input, do: %{main_stick: %{x: 0.6, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, buttons: %{a: true}}

  # --- native -----------------------------------------------------------

  def run_native(frames) do
    {:ok, bridge} = MeleePort.start_link([])
    {:ok, _} = MeleePort.init_console(bridge, config(51_520), 120_000)
    samples = native_loop(bridge, frames, [])
    MeleePort.stop(bridge)
    samples
  end

  defp native_loop(_bridge, 0, acc), do: acc

  defp native_loop(bridge, n, acc) do
    t0 = System.monotonic_time(:microsecond)
    result = MeleePort.step(bridge, [], 30_000)

    case result do
      {:ok, _gs} ->
        t_step = System.monotonic_time(:microsecond)
        :ok = MeleePort.send_controller(bridge, input())
        t1 = System.monotonic_time(:microsecond)
        native_loop(bridge, n - 1, [{t1 - t0, t1 - t_step} | acc])

      {:menu, _} ->
        native_loop(bridge, n, acc)

      :no_frame ->
        native_loop(bridge, n, acc)

      other ->
        IO.puts("[bench] native ended: #{inspect(other)}")
        acc
    end
  end

  # --- python -----------------------------------------------------------

  def run_python(frames) do
    python = Path.join([File.cwd!(), ".venv", "bin", "python3"])
    script = Path.join([File.cwd!(), "priv", "python", "melee_bridge.py"])

    port =
      Port.open({:spawn_executable, python}, [
        :binary,
        :exit_status,
        :use_stdio,
        {:args, ["-u", script]},
        {:cd, File.cwd!()}
      ])

    cfg = config(51_521) |> Map.put(:dummy_mode, "stand")
    %{"ok" => true} = rpc(port, %{cmd: "init", config: cfg}, 120_000)

    samples = python_loop(port, frames, [])
    rpc(port, %{cmd: "stop"}, 30_000)
    Port.close(port)
    samples
  end

  defp python_loop(_port, 0, acc), do: acc

  defp python_loop(port, n, acc) do
    t0 = System.monotonic_time(:microsecond)
    resp = rpc(port, %{cmd: "step", auto_menu: true}, 60_000)

    case resp do
      %{"ok" => true, "no_frame" => true} ->
        python_loop(port, n, acc)

      %{"ok" => true, "is_menu" => false} ->
        t_step = System.monotonic_time(:microsecond)
        %{"ok" => true} = rpc(port, %{cmd: "send_controller", input: input()}, 30_000)
        t1 = System.monotonic_time(:microsecond)
        python_loop(port, n - 1, [{t1 - t0, t1 - t_step} | acc])

      %{"ok" => true} ->
        python_loop(port, n, acc)

      other ->
        IO.puts("[bench] python ended: #{inspect(other)}")
        acc
    end
  end

  defp rpc(port, request, timeout) do
    Port.command(port, Jason.encode!(request) <> "\n")
    recv_line(port, Process.get({:buf, port}, ""), timeout)
  end

  defp recv_line(port, buffer, timeout) do
    case String.split(buffer, "\n", parts: 2) do
      [line, rest] when line != "" ->
        Process.put({:buf, port}, rest)

        case Jason.decode(line) do
          {:ok, json} -> json
          {:error, _} -> recv_line(port, rest, timeout)
        end

      _ ->
        receive do
          {^port, {:data, data}} -> recv_line(port, buffer <> data, timeout)
          {^port, {:exit_status, n}} -> raise "python bridge exited: #{n}"
        after
          timeout -> raise "python bridge timeout"
        end
    end
  end

  # --- stats ------------------------------------------------------------

  def report(label, samples) do
    totals = samples |> Enum.map(&elem(&1, 0)) |> Enum.sort()
    sends = samples |> Enum.map(&elem(&1, 1)) |> Enum.sort()
    n = length(totals)

    if n == 0 do
      IO.puts("[bench] #{label}: no samples")
      []
    else
      stats = fn sorted ->
        pct = fn p -> Enum.at(sorted, min(length(sorted) - 1, trunc(length(sorted) * p / 100))) end
        mean = Enum.sum(sorted) / length(sorted)

        "mean=#{Float.round(mean, 1)}µs p50=#{pct.(50)}µs p90=#{pct.(90)}µs p99=#{pct.(99)}µs"
      end

      IO.puts("[bench] #{label} step+send (n=#{n}): #{stats.(totals)}")
      IO.puts("[bench] #{label} send only:        #{stats.(sends)}")

      throughput = 1_000_000 / (Enum.sum(totals) / n)
      IO.puts("[bench] #{label} throughput: #{Float.round(throughput, 1)} frames/sec")

      totals
    end
  end
end

frames =
  case System.argv() do
    ["--frames", n | _] -> String.to_integer(n)
    _ -> 1500
  end

IO.puts("[bench] measuring step+send_controller over #{frames} in-game frames per bridge")

IO.puts("[bench] === native ===")
native = Bench.report("native", Bench.run_native(frames))

IO.puts("[bench] === python ===")
python = Bench.report("python", Bench.run_python(frames))

if native != [] and python != [] do
  med = fn s -> Enum.at(s, div(length(s), 2)) end
  IO.puts("[bench] median speedup: #{Float.round(med.(python) / max(med.(native), 1), 2)}x")

  # Frame budget context: one Melee frame is 16667µs.
  IO.puts(
    "[bench] p99 as share of a 16.67ms frame — native #{Float.round(Enum.at(native, trunc(length(native) * 0.99)) / 166.67, 2)}%, " <>
      "python #{Float.round(Enum.at(python, trunc(length(python) * 0.99)) / 166.67, 2)}%"
  )
end
