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
      slippi_dir: :string,
      out: :string
    ]
  )

dolphin = opts[:dolphin] || raise "--dolphin required (folder containing Slippi AppImage)"
iso = opts[:iso] || raise "--iso required"
seconds = opts[:seconds] || 20
mode = String.to_atom(opts[:mode] || "multishine")
stage = String.to_atom(opts[:stage] || "final_destination")
player_port = opts[:port] || 1
slippi_dir = opts[:slippi_dir] || Path.expand("~/Slippi")
fixture_path = opts[:out] || "test/fixtures/replays/fox_multishine.slp"

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

  # CLOSED-LOOP multishine: inputs are a PURE FUNCTION OF FOX'S STATE, not
  # the wall clock. This is what makes the recording fully learnable — every
  # frame's correct input is recoverable from the game state, so a policy
  # memorizing it acquires transferable "in state X do Y" rules and the
  # Tier-2 gate ambiguity disappears.
  #
  # Fox reflector action states occupy 360..369 (observed 365/366 in parsed
  # recordings); KNEE_BEND (jumpsquat) = 24. Shine is jump-cancelable from
  # action_frame 4 onward while grounded.
  @reflector_range 360..369
  @jumpsquat 24

  def input(:closed_loop, _frame, player) do
    in_reflector = player.action in @reflector_range
    af = max(player.action_frame || 0, 0)

    # Melee registers button EDGES, not levels — a B held continuously (e.g.
    # from the entry platform onward) never triggers anything. Alternating on
    # action_frame parity guarantees a fresh edge within 2 frames while
    # remaining a pure function of observable state (af ticks every frame).
    b_edge = rem(trunc(af), 2) == 0

    cond do
      # Grounded reflector, too early to JC: hold B (keep it out).
      # Window shifted 1 early vs the true JC frame (4): the bridge applies
      # inputs one frame late, so pressing at af 3 lands at af 4.
      in_reflector and player.on_ground and af <= 2 ->
        controller(b: true)

      # Grounded reflector, JC window (af 3..4, arrives 4..5): X pressed and
      # B RELEASED — the release is what makes the jumpsquat branch's B a
      # fresh edge. X false in the af<=2 branch → crisp jump edge too.
      in_reflector and player.on_ground and af <= 4 ->
        controller(x: true)

      # af 6..8: JC should have taken; keep the reflector out while the
      # state transition lands.
      in_reflector and player.on_ground and af <= 8 ->
        controller(b: true)

      # Grounded reflector, af 5..8: transition guard with X RELEASED — this
      # gap guarantees the af>8 branch below always opens with a fresh X edge.
      in_reflector and player.on_ground and af <= 8 ->
        controller(b: true)

      # Grounded reflector with LARGE af: only reachable by landing from an
      # aerial shine (af carries over from the air; ground shines get JC'd
      # by af 5). X held here — the air branch keeps X false, so the landing
      # transition itself is the edge → instant JC out of the landed shine.
      # This is the frame the real multishine loop pivots on.
      in_reflector and player.on_ground ->
        controller(b: true, x: true)

      # Aerial shine startup (365, "stun" frames): B released, stick NEUTRAL
      # — deliberately, so the 366 branch below constitutes a FRESH downward
      # flick. Fastfall requires the stick to cross into down while falling;
      # a stick held down since the ground never triggers it (measured: the
      # un-fastfallen aerial reflector falls at ~0.027/frame² — 10x weaker
      # than Fox's real gravity — turning the canonical 5 airborne frames
      # into 17).
      player.action == 365 ->
        %{
          main_stick: %{x: 0.5, y: 0.5},
          c_stick: %{x: 0.5, y: 0.5},
          shoulder: 0.0,
          buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
        }

      # Aerial reflector out (366+): slam the stick DOWN — fresh flick →
      # FASTFALL → land within a couple of frames, reflector persisting into
      # the grounded state where the JC branch takes over.
      in_reflector ->
        controller([])

      # Special fall (helpless, post-reflector-release): no inputs — pressing
      # B here did nothing but destroy the next shine's edge.
      player.action == 85 ->
        controller([])

      # Jumpsquat: B released frames 1-2, PRESSED when af>=2 is observed.
      # Latency calibration (take-5 trace): observations are 1 frame stale
      # and inputs apply 1 frame later, so the af2 observation lands B on
      # frame 4 — the jumpsquat-exit frame. A frame-perfect down-B there is
      # the actual multishine: the shine comes out GROUNDED, zero airborne
      # frames ("one frame of airborne Fox" is the documented FAILURE mode).
      # This edge is real because the JC window above releases B. If the
      # timing slips a frame, the airborne branch below still delivers an
      # aerial shine + fastfall recovery instead of an empty hop.
      player.action == @jumpsquat ->
        # take-6 trace: af>=2 landed the edge on jumpsquat frame 3 (consumed
        # no-op) — live af leads the recording by one. af>=3 puts it on the
        # exit frame.
        controller(b: af >= 3)

      # Airborne without reflector (just left the ground from the JC):
      # aerial shine IMMEDIATELY — B was released during jumpsquat, so this
      # is always a fresh edge; no alternation delay.
      not player.on_ground ->
        controller(b: true)

      # Grounded and anything else (wait/crouch/landing): start a shine,
      # edge-safe.
      true ->
        controller(b: b_edge)
    end
  end

  # JC-UPSMASH out of shield, on repeat: shield → X (jump out of shield) →
  # up+A during jumpsquat (the upsmash CANCELS the jump — same
  # act-during-jumpsquat mechanic the multishine's timing depends on) →
  # upsmash finishes → shield again. Visually unambiguous validation of the
  # bridge hitting 3-frame cancel windows: an upsmash with NO jump means the
  # window was hit; a jump means it was missed.
  @shield_states 178..181
  # AttackHi4 (upsmash) start/charge/release
  @upsmash_states [62, 63, 64]

  def input(:jc_upsmash, _frame, player) do
    af = max(player.action_frame || 0, 0)

    cond do
      # Jumpsquat: up + A → jump-cancel upsmash. A was false in the shield
      # branch, so this is always a fresh edge.
      player.action == @jumpsquat ->
        %{
          main_stick: %{x: 0.5, y: 1.0},
          c_stick: %{x: 0.5, y: 0.5},
          shoulder: 0.0,
          buttons: %{a: true, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
        }

      # Upsmash in progress: release everything, let it play out.
      player.action in @upsmash_states ->
        neutral()

      # Shielding: jump out of shield (X edge-alternated on af parity).
      player.action in @shield_states ->
        %{
          main_stick: %{x: 0.5, y: 0.5},
          c_stick: %{x: 0.5, y: 0.5},
          shoulder: 0.8,
          buttons: %{a: false, b: false, x: rem(trunc(af), 2) == 0, y: false, z: false, l: false, r: false, d_up: false}
        }

      # Grounded and actionable: raise shield.
      player.on_ground ->
        %{
          main_stick: %{x: 0.5, y: 0.5},
          c_stick: %{x: 0.5, y: 0.5},
          shoulder: 0.8,
          buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
        }

      # Airborne (missed cancel → we jumped): wait to land.
      true ->
        neutral()
    end
  end

  # CAPABILITY SUITE: choreographed routine exercising every validated
  # mechanic in sequence —
  #   visible shield (~20f) → JC upsmash → lightshield blip (1f) →
  #   5× multishine (shine-JC loop) → visible shield → repeat.
  # Macro-sequencing uses explicit phase state (script memory); each phase's
  # inputs remain state-reactive. Returns {input, new_state}.
  def capability(player, st) do
    af = max(player.action_frame || 0, 0)
    st = Map.merge(%{phase: :shield, t: 20, reps: 0, prev_action: nil}, st)
    entered_jumpsquat = player.action == @jumpsquat and st.prev_action != @jumpsquat
    st2 = %{st | prev_action: player.action}

    {input, st3} =
      case st2.phase do
        :shield ->
          if st2.t > 0 do
            {shield_input(false), %{st2 | t: st2.t - 1}}
          else
            {shield_input(false), %{st2 | phase: :upsmash}}
          end

        :upsmash ->
          cond do
            player.action == @jumpsquat ->
              {upsmash_input(), st2}

            player.action in @upsmash_states ->
              # riding the upsmash; move on once it ends
              {neutral(), %{st2 | phase: :upsmash_end}}

            player.action in @shield_states ->
              {shield_jump_input(af), st2}

            player.on_ground ->
              {shield_input(false), st2}

            true ->
              {neutral(), st2}
          end

        :upsmash_end ->
          if player.action in @upsmash_states or player.action == @jumpsquat do
            {neutral(), st2}
          else
            # upsmash done → 1-frame lightshield blip
            {shield_input(true), %{st2 | phase: :multishine, reps: 5}}
          end

        :multishine ->
          st2 = if entered_jumpsquat, do: %{st2 | reps: st2.reps - 1}, else: st2

          cond do
            # Shield still up/dropping from the blip: stick-down now would
            # spotdodge (down + shoulder). Wait it out at neutral.
            player.action in @shield_states ->
              {neutral(), st2}

            # Done with reps but the perfect multishine never leaves
            # reflector/jumpsquat — release everything and let the reflector
            # drop before moving on (the loop is otherwise unexitable).
            st2.reps <= 0 and
                (player.action in @reflector_range or player.action == @jumpsquat) ->
              {neutral(), st2}

            st2.reps <= 0 and player.on_ground and
                player.action not in @upsmash_states ->
              {shield_input(false), %{st2 | phase: :shield, t: 20}}

            true ->
              {input(:closed_loop, 0, player), st2}
          end
      end

    {input, st3}
  end

  # Full shield = DIGITAL L press (unambiguous — the analog shoulder has
  # thresholds: below ~0.3 nothing, mid-range LIGHT shield, and only the top
  # of the range/digital press gives full shield). Light shield = analog.
  defp shield_input(light) do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: if(light, do: 0.35, else: 0.0),
      buttons: %{
        a: false,
        b: false,
        x: false,
        y: false,
        z: false,
        l: not light,
        r: false,
        d_up: false
      }
    }
  end

  defp shield_jump_input(af) do
    %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{
        a: false,
        b: false,
        x: rem(trunc(af), 2) == 0,
        y: false,
        z: false,
        l: true,
        r: false,
        d_up: false
      }
    }
  end

  defp upsmash_input do
    %{
      main_stick: %{x: 0.5, y: 1.0},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{a: true, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
    }
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

  def run(bridge, mode, target_frames, frame \\ 0, in_game_frames \\ 0, mstate \\ %{}) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:ok, _game_state} when in_game_frames >= target_frames ->
        # Enough footage recorded
        {:ok, in_game_frames}

      {:ok, game_state} ->
        if in_game_frames == 0 do
          IO.puts("[record] In game — starting #{mode} input loop")
        end

        {input, mstate} =
          case {mode, game_state.players[1]} do
            {m, nil} when m in [:closed_loop, :jc_upsmash, :capability] ->
              if rem(in_game_frames, 60) == 0, do: IO.puts("[#{mode}] P1 is nil!")
              {Multishine.neutral(), mstate}

            {:capability, player} ->
              {input, mstate} = Multishine.capability(player, mstate)

              if rem(in_game_frames, 60) == 0 do
                IO.puts(
                  "[capability] f#{in_game_frames} phase=#{mstate[:phase]} reps=#{mstate[:reps]} " <>
                    "action=#{inspect(player.action)}"
                )
              end

              {input, mstate}

            {m, player} when m in [:closed_loop, :jc_upsmash] ->
              if rem(in_game_frames, 60) == 0 do
                IO.puts(
                  "[#{mode}] f#{in_game_frames} action=#{inspect(player.action)} " <>
                    "af=#{inspect(player.action_frame)} ground=#{inspect(player.on_ground)}"
                )
              end

              {Multishine.input(m, in_game_frames, player), mstate}

            {m, _} ->
              {Multishine.input(m, in_game_frames), mstate}
          end

        MeleePort.send_controller(bridge, input)

        if rem(in_game_frames, 300) == 299 do
          IO.puts("[record] #{div(in_game_frames + 1, 60)}s recorded...")
        end

        run(bridge, mode, target_frames, frame + 1, in_game_frames + 1, mstate)

      {:menu, _game_state} ->
        run(bridge, mode, target_frames, frame + 1, in_game_frames, mstate)

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
