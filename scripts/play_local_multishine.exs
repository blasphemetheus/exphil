# Local GUI demonstration, using the synchronous state[t] -> input[t+1] path.
# Run from the repository: devenv shell -- elixir scripts/play_local_multishine.exs
{opts, rest, invalid} =
  OptionParser.parse(System.argv(),
    strict: [
      stage: :string,
      opponent: :string,
      mode: :string,
      human_port: :integer,
      dolphin: :string,
      iso: :string,
      policy: :string,
      seconds: :integer,
      replay_dir: :string,
      help: :boolean
    ]
  )

unless rest == [] and invalid == [],
  do: raise("Unrecognized arguments: #{inspect(rest ++ invalid)}")

if opts[:help] do
  IO.puts("""
  Local stationary multishine Fox, zero added delay.
  --stage final_destination|battlefield|dreamland|yoshis_story|fountain_of_dreams|pokemon_stadium
  --mode human|stand|cpu (default human)
  --opponent fox|falco|marth|peach|samus (CPU/stand only; humans choose on CSS)
  --human-port 2 (physical GameCube adapter port)
  --seconds N (optional bounded session; otherwise play until you quit)
  --policy PATH --dolphin PATH --iso PATH --replay-dir NEW_DIR
  Finish the game or LRAS before stopping the terminal so Slippi finalizes the replay.
  """)

  System.halt(0)
end

stage = opts[:stage] || "final_destination"
opponent = opts[:opponent] || "fox"
mode = opts[:mode] || "human"

unless stage in ~w(final_destination battlefield dreamland yoshis_story fountain_of_dreams pokemon_stadium),
  do: raise("Unsupported demo stage: #{stage}")

unless opponent in ~w(fox falco marth peach samus),
  do: raise("Unsupported demo opponent: #{opponent}")

unless mode in ~w(human stand cpu), do: raise("Mode must be human, stand, or cpu")

unless (opts[:human_port] || 2) == 2,
  do: raise("This demo uses Fox on port 1 and the human on port 2")

if opts[:seconds] && opts[:seconds] <= 0, do: raise("seconds must be positive")

File.cd!(Path.expand("..", __DIR__))
policy = opts[:policy] || "eval_runs/0915_local_zero/corrected_bootstrap/candidate.bin"

dolphin =
  opts[:dolphin] ||
    Path.expand(
      "~/.config/Slippi Launcher/netplay-beta-nixos/Slippi_Netplay_Mainline-x86_64.AppImage"
    )

iso = opts[:iso] || Path.expand("~/isos/melee.iso")
for path <- [policy, dolphin, iso], do: File.stat!(path)
stamp = Calendar.strftime(DateTime.utc_now(), "%Y%m%dT%H%M%S")
directory = opts[:replay_dir] || "eval_runs/local_zero_demo_#{stamp}"
File.mkdir_p!(Path.dirname(directory))
File.mkdir!(directory)

args = [
  "run",
  "scripts/play_dolphin.exs",
  "--policy",
  policy,
  "--dolphin",
  dolphin,
  "--iso",
  iso,
  "--replay-dir",
  directory,
  "--session-report",
  Path.join(directory, "session.json"),
  "--character",
  "fox",
  "--stage",
  stage,
  "--reaction-delay",
  "0",
  "--live-af",
  "--deterministic",
  "--port",
  "1",
  "--opponent-port",
  "2",
  "--blocking-input",
  "--frozen-stadium",
  "--postgame-delay",
  "15"
]

args =
  args ++
    if(mode == "human",
      do: ["--human-port", "2"],
      else: [
        "--dummy",
        mode,
        "--dummy-character",
        opponent,
        "--dummy-cpu-level",
        if(mode == "cpu", do: "1", else: "0")
      ]
    )

args = args ++ if(opts[:seconds], do: ["--seconds", to_string(opts[:seconds])], else: [])
sha = :crypto.hash(:sha256, File.read!(policy)) |> Base.encode16(case: :lower)

File.write!(
  Path.join(directory, "launch.json"),
  JSON.encode!(%{
    args: args,
    policy_sha256: sha,
    stage: stage,
    mode: mode,
    reaction_delay: 0,
    opponent: if(mode == "human", do: "chosen_by_human", else: opponent)
  })
)

IO.puts("Replays and launch details: #{Path.expand(directory)}")

IO.puts(
  "Fox is on port 1. Connect your controller to adapter port 2. Wait for JIT warmup before starting."
)

{_, status} =
  System.cmd("mix", args,
    into: IO.stream(),
    stderr_to_stdout: true,
    env: [{"EXLA_TARGET", "cuda"}, {"EXPHIL_GPU_MEMORY_FRACTION", "0.15"}]
  )

System.halt(status)
