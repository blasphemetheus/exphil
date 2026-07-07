# Record a Fox multishine .slp fixture by scripting inputs through the
# libmelee bridge (Tier-2 of the overfit-replication harness — see
# docs/planning/HANDOFF.md and test/support/replication_check.ex).
#
# Slippi Dolphin auto-saves the game to SlippiReplayDir; after the scripted
# run this copies the newest .slp to test/fixtures/replays/fox_multishine.slp.
#
# A human (or at least a plugged-in controller) is needed on the opponent
# port to get past the character select screen — pick any character.
#
# Usage:
#   mix run scripts/record_multishine.exs \
#     --dolphin ~/.local/share/slippi/netplay --iso ~/isos/melee.iso \
#     --seconds 20 [--mode multishine|simple]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.MeleePort
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      dolphin: :string,
      iso: :string,
      seconds: :integer,
      mode: :string,
      stage: :string,
      port: :integer,
      slippi_dir: :string
    ]
  )

dolphin = opts[:dolphin] || raise "--dolphin required (folder containing Slippi AppImage)"
iso = opts[:iso] || raise "--iso required"
seconds = opts[:seconds] || 20
mode = String.to_atom(opts[:mode] || "multishine")
stage = String.to_atom(opts[:stage] || "final_destination")
player_port = opts[:port] || 1
slippi_dir = opts[:slippi_dir] || Path.expand("~/Slippi")
fixture_path = "test/fixtures/replays/fox_multishine.slp"

defmodule Multishine do
  @moduledoc false

  # All patterns hold the main stick DOWN (y = 0.0 in libmelee's 0..1 space).
  # Shine = down + B; jump-cancel = X during shine; aerial shine on the first
  # airborne frame after 3 frames of jumpsquat. Open-loop period of 12 frames —
  # phase alignment with the game doesn't matter for a periodic fixture, and
  # the replication checker only needs shine events recurring at a stable
  # period (strictness :periodic).
  @period_multishine 12
  @period_simple 15

  def input(:multishine, frame) do
    case rem(frame, @period_multishine) do
      0 -> controller(b: true)
      3 -> controller(x: true)
      7 -> controller(b: true)
      _ -> controller([])
    end
  end

  # Fallback: plain repeated ground shine (shine, release, wait). Less cool,
  # equally periodic.
  def input(:simple, frame) do
    case rem(frame, @period_simple) do
      0 -> controller(b: true)
      1 -> controller(b: true)
      _ -> controller([])
    end
  end

  # Neutral everything except: main stick held down, plus requested buttons.
  defp controller(buttons) do
    %{
      main_stick: %{x: 0.5, y: 0.0},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{
        a: false,
        b: Keyword.get(buttons, :b, false),
        x: Keyword.get(buttons, :x, false),
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      }
    }
  end

  def neutral do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
    }
  end

  # Hold full left: walk/fall off the stage and don't recover. Used to burn
  # Fox's stocks after recording so the game ENDS — Slippi only finalizes the
  # .slp on game end; a mid-game quit leaves a truncated file peppi rejects
  # ("failed to fill whole buffer").
  def hold_left do
    %{
      main_stick: %{x: 0.0, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
    }
  end
end

defmodule RecordLoop do
  @moduledoc false
  alias ExPhil.Bridge.MeleePort

  def run(bridge, mode, target_frames, frame \\ 0, in_game_frames \\ 0) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, _game_state} when in_game_frames >= target_frames ->
        # Enough footage recorded
        {:ok, in_game_frames}

      {:ok, _game_state} ->
        if in_game_frames == 0 do
          IO.puts("[record] In game — starting #{mode} input loop")
        end

        input = Multishine.input(mode, in_game_frames)
        MeleePort.send_controller(bridge, input)

        if rem(in_game_frames, 300) == 299 do
          IO.puts("[record] #{div(in_game_frames + 1, 60)}s recorded...")
        end

        run(bridge, mode, target_frames, frame + 1, in_game_frames + 1)

      {:menu, _game_state} ->
        run(bridge, mode, target_frames, frame + 1, in_game_frames)

      {:postgame, _game_state} ->
        # Game ended before we hit the target (e.g. someone died a lot) —
        # what's recorded is recorded.
        {:ok, in_game_frames}

      {:game_ended, reason} ->
        {:ok, {:ended, reason, in_game_frames}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # After the footage is captured: SD until the game ends (max ~2 min safety),
  # so Slippi finalizes the replay file.
  def sd_until_game_end(bridge, frames_left \\ 7200) do
    if frames_left <= 0 do
      {:error, :sd_timeout}
    else
      case MeleePort.step(bridge, auto_menu: false) do
        {:ok, _game_state} ->
          MeleePort.send_controller(bridge, Multishine.hold_left())
          sd_until_game_end(bridge, frames_left - 1)

        {:postgame, _} -> {:ok, :game_ended}
        {:menu, _} -> {:ok, :game_ended}
        {:game_ended, _} -> {:ok, :game_ended}
        {:error, reason} -> {:error, reason}
      end
    end
  end
end

# ============================================================================

Output.banner("Fox Multishine Recorder")

Output.config([
  {"Mode", mode},
  {"Duration", "#{seconds}s (#{seconds * 60} frames)"},
  {"Stage", stage},
  {"Slippi dir", slippi_dir},
  {"Fixture out", fixture_path}
])

started_at = System.os_time(:second)

{:ok, bridge} = MeleePort.start_link([])

Output.puts("Launching Dolphin (pick any character on the opponent port)...")

{:ok, _} =
  MeleePort.init_console(bridge, %{
    dolphin_path: dolphin,
    iso_path: iso,
    controller_port: player_port,
    opponent_port: if(player_port == 1, do: 2, else: 1),
    character: :fox,
    stage: stage
  })

result = RecordLoop.run(bridge, mode, seconds * 60)

Output.puts("Recording loop done: #{inspect(result)}")
Output.puts("SD-ing Fox to end the game (Slippi finalizes the .slp on game end)...")
sd_result = RecordLoop.sd_until_game_end(bridge)
Output.puts("Game end: #{inspect(sd_result)}")
# Let Slippi write the game-end payload
Process.sleep(3_000)

# Send neutral, give Slippi a beat to flush frames, then stop the console.
# unlink first: MeleePort's teardown races the Python process's own exit and
# must not take the script down before the fixture copy below.
Process.unlink(bridge)
MeleePort.send_controller(bridge, Multishine.neutral())
Process.sleep(1_000)
try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> :ok
end

Process.sleep(2_000)

# Find the .slp created during this run and copy it to the fixture path.
newest_slp =
  slippi_dir
  |> Path.join("**/*.slp")
  |> Path.wildcard()
  |> Enum.map(&{&1, File.stat!(&1).mtime |> :calendar.datetime_to_gregorian_seconds()})
  |> Enum.filter(fn {_, mtime} ->
    mtime >= :calendar.datetime_to_gregorian_seconds(:calendar.system_time_to_universal_time(started_at, :second))
  end)
  |> Enum.max_by(fn {_, mtime} -> mtime end, fn -> nil end)

case newest_slp do
  {path, _} ->
    File.mkdir_p!(Path.dirname(fixture_path))
    File.cp!(path, fixture_path)
    size_kb = div(File.stat!(fixture_path).size, 1024)
    Output.success("Fixture saved: #{fixture_path} (#{size_kb} KB, from #{path})")

  nil ->
    Output.error(
      "No new .slp found in #{slippi_dir} — check SlippiSaveReplays in Dolphin.ini " <>
        "and copy the replay manually"
    )
end
