# Local stationary multishine Fox

This launcher uses a learned reaction-delay-0 policy and synchronous input
delivery: the decision from frame `t` is applied on frame `t+1`. Fox stays in
place and attempts to resume multishining after interruptions. It does not
chase the opponent. The initial 60-match automated qualification is complete; see the
[experiment plan](../planning/LOCAL_MULTISHINE_DEMO.md) for current evidence.

“Zero added delay” means ordinary local play with no extra bot reaction queue
or configured online delay. It does not remove Melee's normal input processing
or display latency. The measured one-frame figure is the bot's state-to-input
handoff, not a measurement of your controller-to-screen latency.

After a hit, Fox resumes where it lands. If knocked offstage, it may lose the
stock and resume after respawning. Steering back from offstage is outside
the agreed demo scope.

## Play locally

Graphical startup is verified on this machine with the corrected zero-delay
checkpoint: a 30-second Final Destination/Fox standing run rendered at 59.94 fps,
measured one-frame input latency, and scored a 200-shine chain. Three human games
also completed, including rematches. Those human games preceded the trigger fix;
the corrected trigger delivery has since been verified against a CPU.
The runner preserves Mainline's memory-card configuration, matching the established
play recipe; disabling those EXI slots caused the earlier black screen.

The Mainline trigger encoding is also corrected: earlier graphical sessions
accidentally held analog shield. Fresh standing and CPU runs now record zero
trigger pressure and zero shield frames. The CPU check resumed after six hit
episodes; one additional episode was interrupted again. Use a fresh launch to
pick up this controller fix; the checkpoint itself has not changed.

Both the 30-match CPU pass and 30-match standing-opponent pass are complete.
All standing cells passed with no deaths and chains of 96–200; CPU cells had
chains of 31–198 and four stock losses. All 60 runs measured one-frame bot
input delivery. This is one 30-second run per matchup and mode, not a guarantee
of flawless play. Fountain of Dreams' descending platforms can interrupt
the fast chain while Fox continues slower ground/aerial shines. Offstage stock
losses remain possible. These limits matter when describing the demonstration.

From `/home/blewf/git/exphil`:

```sh
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15 \
  mix run scripts/play_dolphin.exs \
  --policy eval_runs/0915_local_zero/corrected_bootstrap/candidate.bin \
  --character fox --stage battlefield \
  --reaction-delay 0 --live-af --deterministic \
  --port 1 --opponent-port 2 --human-port 2 \
  --blocking-input --frozen-stadium --postgame-delay 15 \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --replay-dir "eval_runs/local_recording_$(date +%Y%m%d_%H%M%S)"
```

This is the direct `mix run` command. The optional convenience wrapper is
`devenv shell -- elixir scripts/play_local_multishine.exs --stage battlefield`.

Connect your GameCube controller to **adapter port 2**. Fox uses port 1.
Wait for the terminal's JIT warmup message, then select your character and
start the match. The controller adapter is already visible to this computer.

Stage names are `final_destination`, `battlefield`, `dreamland`,
`yoshis_story`, `fountain_of_dreams`, and `pokemon_stadium`. Stadium is frozen.
The confirmed opponent set is Fox, Falco, Marth, Peach, and Samus; in human
mode you choose your character on the character-select screen.

The launcher's defaults point to the local Slippi Mainline AppImage and
`~/isos/melee.iso`. Use `--help` for path overrides. Do not run training or
another evaluation alongside the demo; they compete for the GPU and can
slow local play.

For an unattended graphical demonstration:

```sh
devenv shell -- elixir scripts/play_local_multishine.exs \
  --stage battlefield --mode cpu --opponent marth --seconds 60
```

## Record the demonstration

1. Focus the monitor containing Dolphin.
2. Press **Super+Shift+V** to start the existing screen recorder.
3. Show an uninterrupted chain, hit Fox once, then leave it alone long enough
   to show whether it resumes. Repeat a few times. Keep failed recoveries in
   the evidence when assessing consistency.
4. Press **Super+Shift+V** again to stop and name the video. Your desktop's
   recorder saves MP4 files under `~/Videos/Recordings/`.
5. End the match normally or quit it with L+R+A+Start before stopping the
   terminal. This lets Slippi finalize the `.slp` replay. At the result/menu
   screen, stop the terminal with Ctrl+C (Elixir may then ask for `a` to abort).

The existing desktop recording shortcut captures the focused monitor. Its
current `wf-recorder` command does not enable audio. This guide reflects the
installed Hyprland binding and recording script on this machine.

The direct command saves replays under `eval_runs/local_recording_*`.
Keep the terminal's measured-latency message with your recording notes.

The convenience wrapper prints a new `eval_runs/local_zero_demo_*` directory. It contains
the replay and `launch.json`, including exact arguments and the policy's
SHA-256 hash. A clean bounded session also writes `session.json` with measured
latency. A terminal abort may leave no session report, so retain the terminal's
“Latency measured 1 frame … aligned” message for a manual recording.

A video demonstrates visible behavior; the replay, checkpoint hash, and live
latency measurement make the result inspectable. The automated stage/opponent
results belong alongside the clip when making a consistency claim.
