# Plain Elixir launcher. Each Mix/Dolphin process exits before the next starts.
[policy | only] = System.argv()  # optional case ids to run (one foreground call each)
base = Path.join(Path.dirname(__ENV__.file), "live") |> Path.expand()
registry = String.replace_suffix(policy, "_policy.bin", "_players.json")

cases = [
  {"fd_fox_p1_anonymous", "final_destination", "fox", 1, false, false},
  {"bf_marth_p2_anonymous", "battlefield", "marth", 2, false, false},
  {"fod_falco_p1_style", "fountain_of_dreams", "falco", 1, true, false},
  {"ps_samus_p2_graphical", "pokemon_stadium", "samus", 2, true, true}
]

for {id, stage, opponent, port, styled, graphical} <- cases, only == [] or id in only do
  dir = Path.join(base, id)
  File.mkdir_p!(dir)

  args = [
    "run",
    "scripts/play_dolphin.exs",
    "--policy",
    Path.expand(policy),
    "--character",
    "fox",
    "--stage",
    stage,
    "--reaction-delay",
    "0",
    "--live-af",
    "--stateful-step",
    "--temperature",
    "1.0",
    "--port",
    to_string(port),
    "--opponent-port",
    to_string(3 - port),
    "--dummy",
    "cpu",
    "--dummy-character",
    opponent,
    "--dummy-cpu-level",
    "6",
    "--blocking-input",
    "--seconds",
    "30",
    "--dolphin",
    # DOLPHIN.md: --headless needs the headless build (eval_live_protocol.sh
    # default); the netplay AppImage wrapper is the graphical one.
    if(graphical,
      do: Path.join(System.user_home!(), ".config/Slippi Launcher/netplay-beta-nixos"),
      else: Path.join(System.user_home!(), ".local/share/slippi/exi-ai/dolphin-emu-headless")
    ),
    "--iso",
    Path.join(System.user_home!(), "isos/melee.iso"),
    "--replay-dir",
    Path.join(dir, "replays"),
    "--session-report",
    Path.join(dir, "session.json")
  ]

  args =
    args ++
      if(styled,
        do: ["--style-tag", "TITP", "--player-registry", Path.expand(registry)],
        else: []
      )

  args =
    args ++
      if(graphical,
        do: ["--emulation-speed", "1"],
        else: ["--headless", "--no-audio", "--emulation-speed", "0"]
      )

  File.write!(Path.join(dir, "args.json"), JSON.encode!(args))
  if File.exists?(Path.join(dir, "session.json")), do: raise("session already exists: #{id}")
  IO.puts("Starting #{id}")

  {_, status} =
    System.cmd("mix", args,
      into: File.stream!(Path.join(dir, "play.log")),
      stderr_to_stdout: true,
      env: [
        {"EXLA_TARGET", "cuda"},
        {"EXPHIL_GPU_MEMORY_FRACTION", "0.15"},
        {"EXPHIL_EXLA_PRECISION", "highest"}
      ]
    )

  IO.puts("Finished #{id}: exit #{status}")
  if status != 0, do: System.halt(status)
end
