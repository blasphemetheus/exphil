# Differential parity harness: Python melee_bridge vs native libmelee_ex.
#
# Runs TWO headless Dolphin sessions (fox vs stand-dummy fox on FD — FD is
# the determinism-safe stage, GOTCHAS) driving the SAME frame-keyed input
# schedule through:
#   A. the native bridge (ExPhil.Bridge.MeleePort -> libmelee_ex)
#   B. the legacy Python bridge (priv/python/melee_bridge.py over a Port)
# then diffs the recorded gamestate streams frame-by-frame.
#
# Melee on FD with identical inputs is deterministic, so any field
# mismatch is a bridge-parity defect (parse, controller encoding, or
# timing), not game noise.
#
# Usage:
#   devenv shell -- mix run scripts/parity_harness.exs [--frames N]
#
# Exit: prints a per-field mismatch report and first divergence; exits
# nonzero on divergence beyond tolerance.

require Logger
alias ExPhil.Bridge.MeleePort

defmodule Parity do
  @dolphin Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless")
  @iso "/home/blewf/isos/melee.iso"
  @native_port 51_490
  @python_port 51_491
  @eps 1.0e-4

  # ---------------------------------------------------------------------------
  # Deterministic input schedule (period 240, function of in-game frame).
  # Covers: stick angles, taps, jumps, B moves, c-stick, crouch, analog +
  # digital shield. No START (would pause).
  # ---------------------------------------------------------------------------

  def input_for(frame) do
    t = Integer.mod(frame, 240)

    cond do
      t < 30 -> neutral()
      t < 46 -> %{neutral() | main_stick: %{x: 1.0, y: 0.5}}
      t < 60 -> if t in 50..52, do: buttons(%{y: true}), else: neutral()
      t < 90 -> %{neutral() | shoulder: 0.7}
      t < 120 -> if rem(t, 6) < 3, do: %{buttons(%{b: true}) | main_stick: %{x: 0.5, y: 0.0}}, else: neutral()
      t < 150 -> %{neutral() | c_stick: %{x: 0.9, y: 0.5}}
      t < 180 -> %{neutral() | main_stick: %{x: 0.3, y: 0.5}}
      t < 200 -> %{neutral() | main_stick: %{x: 0.5, y: 0.0}}
      t < 210 -> buttons(%{a: true})
      t < 225 -> buttons(%{r: true})
      true -> neutral()
    end
  end

  defp neutral,
    do: %{main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, shoulder: 0.0, buttons: %{}}

  defp buttons(map), do: %{neutral() | buttons: map}

  # ---------------------------------------------------------------------------
  # Snapshots: one flat map per (frame, player) from either bridge.
  # ---------------------------------------------------------------------------

  def snapshot_native(gs) do
    for {port, p} <- gs.players, p != nil, into: %{} do
      cs = p.controller_state || %ExPhil.Bridge.ControllerState{}

      {port,
       %{
         x: p.x,
         y: p.y,
         percent: p.percent,
         stock: p.stock,
         facing: p.facing,
         action: p.action,
         action_frame: p.action_frame,
         jumps_left: p.jumps_left,
         on_ground: p.on_ground,
         shield: p.shield_strength,
         hitstun: p.hitstun_frames_left,
         speed_air_x: p.speed_air_x_self,
         speed_y: p.speed_y_self,
         speed_gx: p.speed_ground_x_self,
         main_x: cs.main_stick && cs.main_stick.x,
         main_y: cs.main_stick && cs.main_stick.y,
         c_x: cs.c_stick && cs.c_stick.x,
         l_shoulder: cs.l_shoulder,
         btn_a: cs.button_a,
         btn_b: cs.button_b,
         btn_y: cs.button_y,
         btn_r: cs.button_r
       }}
    end
  end

  def snapshot_python(gs_json) do
    for {port, p} <- gs_json["players"] || %{}, p != nil, into: %{} do
      cs = p["controller_state"] || %{}
      main = cs["main_stick"] || %{}
      c = cs["c_stick"] || %{}

      {String.to_integer(port),
       %{
         x: p["x"],
         y: p["y"],
         percent: p["percent"],
         stock: p["stock"],
         facing: p["facing"],
         action: p["action"],
         action_frame: p["action_frame"],
         jumps_left: p["jumps_left"],
         on_ground: p["on_ground"],
         shield: p["shield_strength"],
         hitstun: p["hitstun_frames_left"],
         speed_air_x: p["speed_air_x_self"],
         speed_y: p["speed_y_self"],
         speed_gx: p["speed_ground_x_self"],
         main_x: main["x"],
         main_y: main["y"],
         c_x: c["x"],
         l_shoulder: cs["l_shoulder"],
         btn_a: cs["button_a"],
         btn_b: cs["button_b"],
         btn_y: cs["button_y"],
         btn_r: cs["button_r"]
       }}
    end
  end

  # ---------------------------------------------------------------------------
  # Session A: native bridge
  # ---------------------------------------------------------------------------

  def run_native(max_frames) do
    {:ok, bridge} = MeleePort.start_link([])

    {:ok, _} =
      MeleePort.init_console(
        bridge,
        %{
          dolphin_path: @dolphin,
          iso_path: @iso,
          headless: true,
          character: :fox,
          stage: :final_destination,
          dummy_mode: "stand",
          dummy_character: :fox,
          slippi_port: @native_port,
          console_timeout: 0.1,
          emulation_speed: 0.0
        },
        120_000
      )

    frames = native_loop(bridge, max_frames, %{})
    MeleePort.stop(bridge)
    frames
  end

  defp native_loop(bridge, max_frames, acc) do
    if map_size(acc) >= max_frames do
      acc
    else
      case MeleePort.step(bridge, [], 30_000) do
        {:ok, gs} ->
          :ok = MeleePort.send_controller(bridge, input_for(gs.frame))
          native_loop(bridge, max_frames, Map.put(acc, gs.frame, snapshot_native(gs)))

        {:menu, _} -> native_loop(bridge, max_frames, acc)
        :no_frame -> native_loop(bridge, max_frames, acc)
        {:postgame, _} -> acc
        other ->
          IO.puts("[parity] native ended early: #{inspect(other)}")
          acc
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Session B: legacy Python bridge over a raw Port (line JSON protocol)
  # ---------------------------------------------------------------------------

  def run_python(max_frames) do
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

    init = %{
      cmd: "init",
      config: %{
        dolphin_path: @dolphin,
        iso_path: @iso,
        headless: true,
        character: "fox",
        stage: "final_destination",
        dummy_mode: "stand",
        dummy_character: "fox",
        slippi_port: @python_port,
        console_timeout: 0.1,
        emulation_speed: 0.0
      }
    }

    case rpc(port, init, 120_000) do
      %{"ok" => true} -> :ok
      other -> raise "python init failed: #{inspect(other)}"
    end

    frames = python_loop(port, max_frames, %{})
    rpc(port, %{cmd: "stop"}, 30_000)
    Port.close(port)
    frames
  rescue
    e ->
      IO.puts("[parity] python session error: #{Exception.message(e)}")
      %{}
  end

  defp python_loop(port, max_frames, acc) do
    if map_size(acc) >= max_frames do
      acc
    else
      case rpc(port, %{cmd: "step", auto_menu: true}, 60_000) do
        %{"ok" => true, "no_frame" => true} ->
          python_loop(port, max_frames, acc)

        %{"ok" => true, "is_postgame" => true} ->
          acc

        %{"ok" => true, "is_menu" => false, "game_state" => gs} ->
          frame = gs["frame"]
          %{"ok" => true} = rpc(port, %{cmd: "send_controller", input: input_for(frame)}, 30_000)
          python_loop(port, max_frames, Map.put(acc, frame, snapshot_python(gs)))

        %{"ok" => true} ->
          python_loop(port, max_frames, acc)

        other ->
          IO.puts("[parity] python ended early: #{inspect(other)}")
          acc
      end
    end
  end

  defp rpc(port, request, timeout) do
    Port.command(port, Jason.encode!(request) <> "\n")
    recv_line(port, "", timeout)
  end

  defp recv_line(port, buffer, timeout) do
    case String.split(buffer, "\n", parts: 2) do
      [line, rest] when line != "" ->
        Process.put({:pybuf, port}, rest)

        case Jason.decode(line) do
          {:ok, json} -> json
          {:error, _} -> recv_line(port, rest, timeout)
        end

      _ ->
        buffered = Process.get({:pybuf, port}, "")

        if buffered != "" and buffer == "" do
          Process.put({:pybuf, port}, "")
          recv_line(port, buffered, timeout)
        else
          receive do
            {^port, {:data, data}} -> recv_line(port, buffer <> data, timeout)
            {^port, {:exit_status, n}} -> raise "python bridge exited: #{n}"
          after
            timeout -> raise "python bridge timeout"
          end
        end
    end
  end

  # ---------------------------------------------------------------------------
  # Diff
  # ---------------------------------------------------------------------------

  def diff(native, python) do
    common = MapSet.intersection(MapSet.new(Map.keys(native)), MapSet.new(Map.keys(python)))
    frames = Enum.sort(common)

    IO.puts(
      "[parity] native frames=#{map_size(native)} python frames=#{map_size(python)} " <>
        "common=#{length(frames)} (#{List.first(frames)}..#{List.last(frames)})"
    )

    {mismatch_counts, first_divergence, max_float_diff} =
      Enum.reduce(frames, {%{}, nil, 0.0}, fn frame, {counts, first, maxd} ->
        Enum.reduce([1, 2], {counts, first, maxd}, fn port, {counts, first, maxd} ->
          a = native[frame][port]
          b = python[frame][port]

          cond do
            a == nil or b == nil ->
              {counts, first, maxd}

            true ->
              Enum.reduce(Map.keys(a), {counts, first, maxd}, fn field, {counts, first, maxd} ->
                va = a[field]
                vb = b[field]

                {match?, d} = compare(va, vb)
                maxd = if is_number(d), do: max(maxd, d), else: maxd

                if match? do
                  {counts, first, maxd}
                else
                  counts = Map.update(counts, field, 1, &(&1 + 1))

                  first =
                    first ||
                      {frame, port, field, va, vb}

                  {counts, first, maxd}
                end
              end)
          end
        end)
      end)

    IO.puts("[parity] max float diff observed: #{max_float_diff}")

    if map_size(mismatch_counts) == 0 do
      IO.puts("[parity] PARITY OK: zero field mismatches across #{length(frames)} frames x 2 players")
      :ok
    else
      IO.puts("[parity] MISMATCHES by field: #{inspect(Enum.sort_by(mismatch_counts, &(-elem(&1, 1))))}")
      {frame, port, field, va, vb} = first_divergence
      IO.puts("[parity] first divergence: frame=#{frame} port=#{port} #{field}: native=#{inspect(va)} python=#{inspect(vb)}")
      :divergence
    end
  end

  defp compare(a, b) when is_number(a) and is_number(b) do
    d = abs(a - b)
    {d <= @eps, d}
  end

  defp compare(a, b), do: {a == b, nil}
end

max_frames =
  case System.argv() do
    ["--frames", n | _] -> String.to_integer(n)
    _ -> 1800
  end

IO.puts("[parity] target #{max_frames} in-game frames per bridge")

IO.puts("[parity] === session A: native bridge ===")
native = Parity.run_native(max_frames)

IO.puts("[parity] === session B: python bridge ===")
python = Parity.run_python(max_frames)

case Parity.diff(native, python) do
  :ok -> IO.puts("[parity] DONE: bridges are frame-identical")
  :divergence -> System.halt(1)
end
