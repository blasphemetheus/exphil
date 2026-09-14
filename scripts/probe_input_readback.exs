alias ExPhil.Bridge.{ControllerState, MeleePort}
alias ExPhil.Data.Peppi

{opts, _, []} = OptionParser.parse(System.argv(), strict: [out: :string])
out = Keyword.fetch!(opts, :out)
File.mkdir!(out)
neutral = ControllerState.neutral() |> ControllerState.to_input()
axes = Enum.map(0..16, &(&1 / 16)) ++ Enum.map([-24, -23, -22, 22, 23, 24], &(&1 / 160 + 0.5))

sticks =
  for horizontal <- axes, vertical <- axes do
    %{neutral | main_stick: %{x: horizontal, y: vertical}, c_stick: %{x: vertical, y: horizontal}}
  end

triggers =
  for shoulder <- [0.0, 0.25, 0.5, 0.75, 1.0], left <- [false, true], right <- [false, true] do
    %{neutral | shoulder: shoulder, buttons: %{neutral.buttons | l: left, r: right}}
  end

commands = sticks ++ triggers
{:ok, bridge} = MeleePort.start_link()

config = %{
  dolphin_path: Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
  iso_path: Path.expand("~/isos/melee.iso"),
  controller_port: 1,
  opponent_port: 2,
  character: :fox,
  stage: :final_destination,
  online_delay: 0,
  dummy_mode: "external",
  dummy_character: "fox",
  dummy_cpu_level: 0,
  no_audio: true,
  headless: true,
  emulation_speed: 0.0,
  replay_dir: Path.expand(out),
  slippi_port: 51960
}

try do
  case MeleePort.init_console(bridge, config, 180_000) do
    {:ok, _} -> :ok
    :ok -> :ok
    error -> raise "console init: #{inspect(error)}"
  end

  {_remaining, trace} =
    Enum.reduce_while(1..20_000, {commands, []}, fn _, {remaining, trace} ->
      case MeleePort.step(bridge, auto_menu: true) do
        {:ok, gs} when gs.frame >= 60 ->
          case remaining do
            [] ->
              {:halt, {[], Enum.reverse(trace)}}

            [input | rest] ->
              :ok = MeleePort.send_controller(bridge, input)
              :ok = MeleePort.send_controller(bridge, Map.put(neutral, :port, 2))
              {:cont, {rest, [%{frame: gs.frame, sent: input, issued: input} | trace]}}
          end

        {:ok, _} ->
          {:cont, {remaining, trace}}

        {:menu, _} ->
          {:cont, {remaining, trace}}

        other ->
          raise "unexpected probe step: #{inspect(other)}"
      end
    end)

  if length(trace) != length(commands), do: raise("incomplete probe")
  MeleePort.stop(bridge)
  Process.sleep(1000)
  [path] = Path.wildcard(Path.join(out, "*.slp"))
  {:ok, replay} = Peppi.parse(path)
  frames = Map.new(replay.frames, &{&1.frame_number, Map.from_struct(&1.players[1].controller)})
  rows = Enum.map(trace, &Map.put(&1, :recorded, Map.fetch!(frames, &1.frame + 1)))

  File.write!(
    Path.join(out, "readback.json"),
    Jason.encode!(%{replay: path, rows: rows}, pretty: true),
    [:exclusive]
  )

  IO.puts("Recorded #{length(rows)} input samples")
after
  if Process.alive?(bridge) do
    MeleePort.stop(bridge)
    GenServer.stop(bridge, :normal, 5000)
  end
end
