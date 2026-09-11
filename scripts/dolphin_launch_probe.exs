# Launch Dolphin exactly the way MeleePort does (Melee.Dolphin.launch/1)
# and dump whatever the process says for a few seconds — the fastest way
# to see WHY a headless gate dies with {:enet_disconnected, :timeout}
# (2026-09-10: every gate failed from 11:27 on with no Dolphin output in
# the run logs).
#
#   mix run scripts/dolphin_launch_probe.exs [--dolphin DIR] [--iso ISO] [--seconds 8]

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [dolphin: :string, iso: :string, seconds: :integer, windowed: :boolean])

dolphin = opts[:dolphin] || Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless")
iso = opts[:iso] || Path.expand("~/isos/melee.iso")
secs = opts[:seconds] || 8

launch_opts = [
  path: dolphin,
  iso_path: iso,
  slippi_port: 51442,
  headless: not (opts[:windowed] || false),
  blocking_input: true,
  online_delay: 0,
  emulation_speed: 0.0,
  save_replays: false,
  controller_ports: [{1, :standard}, {2, :standard}]
]

IO.puts("prepare_home:")

case Melee.Dolphin.prepare_home(launch_opts) do
  {:ok, prep} ->
    IO.puts("  exe:  #{prep.exe}")
    IO.puts("  home: #{prep.home} (temp? #{prep.temp_home?})")
    IO.puts("  args: #{Enum.join(prep.args, " ")}")

  {:error, reason} ->
    IO.puts("  ERROR: #{inspect(reason)}")
    System.halt(1)
end

IO.puts("launch:")

case Melee.Dolphin.launch(launch_opts) do
  {:ok, dolphin} ->
    IO.inspect(Map.drop(dolphin, [:port]), label: "  dolphin")
    port = dolphin.port
    deadline = System.monotonic_time(:millisecond) + secs * 1000

    loop = fn loop ->
      remaining = deadline - System.monotonic_time(:millisecond)

      if remaining > 0 do
        receive do
          {^port, {:data, data}} ->
            IO.write("  [dolphin] " <> String.replace(data, "\n", "\n  [dolphin] ") <> "\n")
            loop.(loop)

          {^port, {:exit_status, s}} ->
            IO.puts("  [dolphin] EXITED status #{s}")
        after
          remaining -> IO.puts("  (still running after #{secs}s — no exit)")
        end
      end
    end

    loop.(loop)
    IO.puts("procs: #{:os.cmd(~c"pgrep -a dolphin-emu") |> to_string() |> String.trim()}")
    Melee.Dolphin.stop(dolphin)
    IO.puts("stopped")

  {:error, reason} ->
    IO.puts("  ERROR: #{inspect(reason)}")
end
