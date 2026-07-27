# Was that opponent actually the CPU you asked for?
#
#   mix run scripts/check_replay_ports.exs <replay.slp> [more.slp ...] \
#     [--expect-cpu PORT] [--expect-level N]
#
# Slippi records each port's TYPE in the game-start block (0 HUMAN, 1 CPU,
# 2 DEMO, 3 empty). Peppi's PlayerMeta does not expose it and libmelee never
# logged the achieved level, so a dummy that silently came up HUMAN was
# invisible — it just looked like a very passive CPU.
#
# That is not hypothetical. On 2026-07-26, five of six recordings made with
# `--dummy cpu --dummy-cpu-level 9` produced a HUMAN port 2: it sat in Wait
# 76% of frames, never once jumped, and drifted onto the ledge. "Level 9 CPUs
# don't ledge plank" is what gave it away. Cause was an autostart race at
# character select (GOTCHAS #57); this script is how you check the fix held.
#
# Exits 1 if an --expect-* assertion fails, so a recording loop can gate on it.

alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(), strict: [expect_cpu: :integer, expect_level: :integer])

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/check_replay_ports.exs <replay.slp> ...")
  System.halt(2)
end

# py-slippi reads the start block; the project venv already has it.
python =
  [Path.join([File.cwd!(), ".venv", "bin", "python3"]), System.find_executable("python3")]
  |> Enum.find(&(&1 && File.exists?(&1)))

if is_nil(python) do
  Output.error("No python found (.venv/bin/python3 or python3) — py-slippi reads the start block")
  System.halt(2)
end

script = """
import json, sys
from slippi import Game
out = []
for p in sys.argv[1:]:
    try:
        g = Game(p)
        ports = []
        for i, pl in enumerate(g.start.players):
            if pl is None:
                continue
            ports.append({
                "port": i + 1,
                "type": int(pl.type) if pl.type is not None else None,
                "character": str(pl.character),
                "cpu_level": getattr(pl, "cpu_level", None),
            })
        out.append({"path": p, "ports": ports})
    except Exception as e:
        out.append({"path": p, "error": str(e)})
print(json.dumps(out))
"""

{json, 0} = System.cmd(python, ["-c", script | paths], stderr_to_stdout: false)
reports = Jason.decode!(json)

type_name = fn
  0 -> "HUMAN"
  1 -> "CPU"
  2 -> "DEMO"
  3 -> "empty"
  other -> "type=#{inspect(other)}"
end

failures =
  Enum.reduce(reports, 0, fn r, acc ->
    Output.puts("")
    Output.puts(Path.basename(r["path"]))

    case r["error"] do
      nil ->
        Enum.each(r["ports"], fn p ->
          lvl = if p["cpu_level"] in [nil, 0], do: "", else: " level=#{p["cpu_level"]}"
          Output.puts("  port #{p["port"]}: #{type_name.(p["type"])}#{lvl}  #{p["character"]}")
        end)

        want_port = opts[:expect_cpu]

        if want_port do
          port = Enum.find(r["ports"], &(&1["port"] == want_port))

          cond do
            is_nil(port) ->
              Output.error("  expected a CPU on port #{want_port}, but that port is absent")
              acc + 1

            port["type"] != 1 ->
              Output.error(
                "  port #{want_port} is #{type_name.(port["type"])}, NOT a CPU — the dummy " <>
                  "never finished character-select setup (GOTCHAS #57)"
              )

              acc + 1

            opts[:expect_level] && port["cpu_level"] &&
                port["cpu_level"] != opts[:expect_level] ->
              Output.error(
                "  port #{want_port} is a CPU but level #{port["cpu_level"]}, " <>
                  "expected #{opts[:expect_level]} — Melee defaults a fresh CPU to 1, " <>
                  "so this is the autostart race firing before the slider drag finished"
              )

              acc + 1

            true ->
              Output.success("  port #{want_port} is a CPU as requested")
              acc
          end
        else
          acc
        end

      err ->
        # Truncated replays are the usual cause: Slippi only finalizes on game
        # end, so a killed run leaves a file peppi/py-slippi both reject.
        Output.error("  UNREADABLE: #{err}")
        Output.error("  (a run killed mid-game leaves a truncated .slp — use --seconds)")
        acc + 1
    end
  end)

Output.puts("")

if failures > 0 do
  Output.error("#{failures} check(s) failed")
  System.halt(1)
else
  Output.success("all checks passed")
end
