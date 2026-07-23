# Human drill recorder (drills design 2026-07-23, feature B).
#
# Puts YOU into a marked moment from a real game (UnclePunch-style
# savestate practice) and records every attempt you save as training
# data in the exact format `train.exs --mix-frames` eats.
#
# One windowed netplay-build Dolphin session, two phases:
#
#   Phase 1 (automated): prefix-replays BOTH ports of the source .slp to
#   each requested handoff frame (same mechanics as scenario_suite.exs)
#   and presses Shift+F<slot> via wtype at each -> savestate slots 1..8.
#
#   Swap: the script prompts you to open Dolphin's Controllers dialog
#   and switch Port 1 from the pipe pad to your GC adapter.
#
#   Phase 2 (recording): press F<slot> to load a situation and play it.
#   Port 2 is a ghost — it keeps replaying the source game's recorded
#   inputs, re-aligned automatically after every load (savestates
#   restore the frame counter). In-game controls:
#     taunt (D-pad up)      = end attempt, SAVE it
#     L+R (full) + taunt    = end attempt, DISCARD it
#     load a state mid-attempt = DISCARD
#   The match timer rewinds with every load, so a session only ends if
#   you play a long time without loading (or close Dolphin / Ctrl-C —
#   the .frames file is rewritten after every save, so both are safe).
#
# Usage:
#   mix run scripts/human_drill.exs --slp ~/Slippi/Game_X.slp \
#     --frames 4180,7250 [options]
#
# Options:
#   --slp PATH           Source replay (required)
#   --frames F1,F2,...   Handoff frames, ascending, max 8 (required;
#                        get them from scan_bookmarks.exs / a manifest)
#   --dolphin PATH       Netplay build dir (default ~/.config/Slippi Launcher/netplay)
#   --iso PATH           Melee ISO (default ~/isos/melee.iso)
#   --tag NAME           player_tag / expert name (default human_blewf)
#   --out PATH           .frames output (default drills/human/<slp>_<ts>.frames)
#   --port N             Your port in the SOURCE replay (default 1)
#   --input-offset N     Recorded-input frame offset (default 1, see SCENARIOS.md)
#   --drift-tolerance F  Warn threshold at handoffs (default 3.0)
#   --pipe-shim          Analog->digital trigger conversion for the prefix
#                        (default OFF: the netplay build honors analog
#                        triggers over pipe; GOTCHA #66 is ExiAI-only)
#   --min-attempt N      Auto-discard attempts shorter than N frames (default 30)
#   --no-wtype           Don't press hotkeys; prompts you to hit
#                        Shift+F<slot> yourself at each handoff
#   --quiet
#
# UNTESTED as of 2026-07-23 (written while r16 held mix; needs the
# hardware loop) — test plan in docs/planning/DRILLS_DESIGN_2026-07-23.md.

require Logger

alias ExPhil.Bridge.MeleePort
alias ExPhil.Data.Peppi
alias ExPhil.Eval.ScenarioScan
alias ExPhil.Training.Output

defmodule HumanDrill do
  @moduledoc false

  @neutral %{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
  }

  @menu_step_limit 12_000
  # Frames dropped from the start of every attempt (stale post-load frames).
  @post_load_skip 3

  def neutral, do: @neutral

  # ==========================================================================
  # Source replay preparation — same conversion as scenario_suite.exs
  # (duplicated for now; extract to ExPhil.Drills.PrefixReplay once
  # scenario_suite.exs is editable again — the r16 fan-out still calls it).
  # ==========================================================================

  @trigger_digital_threshold 0.31

  def prepare_replay(slp, opts \\ []) do
    shim = Keyword.get(opts, :pipe_shim, false)
    {:ok, replay} = Peppi.parse(Path.expand(slp))

    Enum.reduce(replay.frames, %{inputs: %{}, ref: %{}}, fn f, acc ->
      p1 = f.players[1]
      p2 = f.players[2]

      if p1 && p2 do
        %{
          acc
          | inputs:
              Map.put(
                acc.inputs,
                f.frame_number,
                {rec_input(p1.controller, shim), rec_input(p2.controller, shim)}
              ),
            ref:
              Map.put(acc.ref, f.frame_number, %{
                p1: ScenarioScan.player_summary(p1),
                p2: ScenarioScan.player_summary(p2)
              })
        }
      else
        acc
      end
    end)
  end

  defp rec_input(c, pipe_shim) do
    %{
      main_stick: %{x: c.main_stick_x, y: c.main_stick_y},
      c_stick: %{x: c.c_stick_x, y: c.c_stick_y},
      shoulder: max(c.l_trigger, c.r_trigger),
      buttons: %{
        a: c.button_a,
        b: c.button_b,
        x: c.button_x,
        y: c.button_y,
        z: c.button_z,
        l: c.button_l or (pipe_shim and c.l_trigger > @trigger_digital_threshold),
        r: c.button_r or (pipe_shim and c.r_trigger > @trigger_digital_threshold),
        d_up: c.button_d_up
      }
    }
  end

  # ==========================================================================
  # Savestate hotkeys (Wayland). Dolphin defaults: Save Slot k = Shift+Fk,
  # Load Slot k = Fk. wtype needs the Dolphin window FOCUSED.
  # ==========================================================================

  def save_state!(slot, use_wtype) do
    if use_wtype do
      case System.cmd("wtype", ["-M", "shift", "-P", "F#{slot}", "-p", "F#{slot}", "-m", "shift"],
             stderr_to_stdout: true
           ) do
        {_, 0} ->
          Output.success("savestate -> slot #{slot} (Shift+F#{slot})")

        {out, _} ->
          Output.warning("wtype failed (#{String.trim(out)}) — press Shift+F#{slot} NOW")
      end
    else
      Output.puts(">>> press Shift+F#{slot} NOW to save slot #{slot} <<<")
    end
  end

  # ==========================================================================
  # Episode sink: rewrite the .frames payload after every saved attempt.
  # ==========================================================================

  def write_frames!(out_path, tag, frame_lists) do
    payload = %{
      expert: tag,
      exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      frame_lists: frame_lists
    }

    File.mkdir_p!(Path.dirname(out_path))
    File.write!(out_path, :erlang.term_to_binary(payload))
  end

  def write_sidecar!(out_path, attempts_meta) do
    File.write!(
      out_path <> ".json",
      Jason.encode!(%{"attempts" => Enum.reverse(attempts_meta)}, pretty: true)
    )
  end

  # ==========================================================================
  # Main loop — phase state machine.
  #
  # :menu -> :prefix -> :handover -> :record
  #
  # :record sub-state (st.attempt): nil (waiting for a load) or
  # %{slot:, started_at:, frames: [newest-first]}.
  # ==========================================================================

  def run(bridge, prep, targets, opts) do
    st = %{
      phase: :menu,
      menu_steps: 0,
      # remaining savestate targets: [{handoff_frame, slot}]
      targets: targets,
      all_targets: targets,
      inputs: prep.inputs,
      ref: prep.ref,
      offset: opts[:input_offset],
      tolerance: opts[:drift_tolerance],
      use_wtype: opts[:use_wtype],
      min_attempt: opts[:min_attempt],
      out: opts[:out],
      tag: opts[:tag],
      last_frame: nil,
      attempt: nil,
      prev_cs: nil,
      saved_lists: [],
      discard_lists: [],
      attempts_meta: [],
      counts: %{saved: 0, discarded: 0}
    }

    loop(bridge, st)
  end

  defp loop(bridge, st) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:menu, _gs} ->
        handle_out_of_game(bridge, st)

      {:postgame, _gs} ->
        handle_out_of_game(bridge, st)

      {:ok, gs} ->
        st = if st.phase == :menu, do: %{st | phase: :prefix}, else: st

        case st.phase do
          :prefix -> prefix_frame(bridge, gs, st)
          :handover -> handover_frame(bridge, gs, st)
          :record -> record_frame(bridge, gs, st)
        end

      {:game_ended, reason} ->
        if st.phase == :record do
          finish(st, "game ended (#{reason})")
        else
          %{error: "dolphin ended during #{st.phase}: #{reason}"}
        end

      {:error, reason} ->
        %{error: "step error: #{inspect(reason)}"}
    end
  end

  defp handle_out_of_game(bridge, st) do
    case st.phase do
      :menu ->
        if st.menu_steps > @menu_step_limit do
          %{error: "stuck in menus after #{st.menu_steps} steps"}
        else
          loop(bridge, %{st | menu_steps: st.menu_steps + 1})
        end

      :record ->
        finish(st, "match ended")

      _ ->
        %{error: "game ended during #{st.phase} (divergence? timer?)"}
    end
  end

  # ---- phase 1: prefix ------------------------------------------------------

  defp prefix_frame(bridge, gs, st) do
    f = gs.frame

    case st.targets do
      [{handoff, slot} | rest] when f >= handoff ->
        report_drift(gs, st.ref[f], st.tolerance, slot)
        save_state!(slot, st.use_wtype)

        if rest == [] do
          # Last slot saved -> hand over to the human.
          MeleePort.send_controller(bridge, neutral())
          announce_handover(st)
          loop(bridge, %{st | phase: :handover, targets: [], last_frame: f})
        else
          # More targets later in the same game: keep replaying the prefix.
          send_recorded_both(bridge, st, f)
          loop(bridge, %{st | targets: rest})
        end

      _ ->
        send_recorded_both(bridge, st, f)
        loop(bridge, st)
    end
  end

  defp send_recorded_both(bridge, st, f) do
    case st.inputs[f + st.offset] do
      {p1, p2} ->
        MeleePort.send_controller(bridge, p1)
        MeleePort.send_controller(bridge, Map.put(p2, :port, 2))

      nil ->
        MeleePort.send_controller(bridge, neutral())
        MeleePort.send_controller(bridge, Map.put(neutral(), :port, 2))
    end
  end

  defp announce_handover(st) do
    slots =
      st.all_targets
      |> Enum.map(fn {handoff, slot} -> "  F#{slot} -> frame #{handoff}" end)
      |> Enum.join("\n")

    Output.puts("")
    Output.puts("=====================================================================")
    Output.puts("ALL SAVESTATES CREATED. Now:")
    Output.puts("  1. Dolphin menu: Controllers -> Port 1 -> your GC adapter")
    Output.puts("  2. Press F<slot> to load a situation:")
    Output.puts(slots)
    Output.puts("  3. Play the situation. End every attempt with:")
    Output.puts("       taunt (D-pad up)        SAVE the attempt")
    Output.puts("       hold L+R fully + taunt  DISCARD the attempt")
    Output.puts("       load any state          DISCARD (implicit)")
    Output.puts("  P2 is a ghost of the source game and re-syncs on every load.")
    Output.puts("=====================================================================")
    Output.puts("")
  end

  # ---- swap window: human is reconfiguring Port 1 ---------------------------

  defp handover_frame(bridge, gs, st) do
    # Ghost P2 keeps playing; P1 idles until the first savestate load.
    send_ghost(bridge, st, gs.frame)

    if regression?(st, gs.frame) do
      record_frame(bridge, gs, %{st | phase: :record})
    else
      loop(bridge, %{st | last_frame: gs.frame})
    end
  end

  # ---- phase 2: record ------------------------------------------------------

  defp record_frame(bridge, gs, st) do
    f = gs.frame
    send_ghost(bridge, st, f)

    cs = gs.players[1] && gs.players[1].controller_state

    st =
      cond do
        regression?(st, f) ->
          st
          |> end_attempt(:discard, "state load")
          |> start_attempt(f)

        st.attempt == nil ->
          st

        true ->
          capture(st, gs, cs)
      end

    st =
      case {st.attempt, marker(cs)} do
        {nil, _} -> st
        {_, :save} -> end_attempt(st, :save, "taunt")
        {_, :discard} -> end_attempt(st, :discard, "L+R+taunt")
        {_, nil} -> st
      end

    loop(bridge, %{st | last_frame: f, prev_cs: cs})
  end

  defp send_ghost(bridge, st, f) do
    p2 =
      case st.inputs[f + st.offset] do
        {_p1, p2} -> p2
        nil -> neutral()
      end

    MeleePort.send_controller(bridge, Map.put(p2, :port, 2))
  end

  defp regression?(%{last_frame: last}, f), do: is_integer(last) and f < last - 5

  defp start_attempt(st, f) do
    slot =
      st.all_targets
      |> Enum.min_by(fn {handoff, _} -> abs(handoff - f) end, fn -> {nil, nil} end)
      |> elem(1)

    Output.puts("-- attempt started (slot #{inspect(slot)}, frame #{f})")
    %{st | attempt: %{slot: slot, started_at: f, frames: []}, prev_cs: nil}
  end

  defp capture(st, gs, cs) do
    if cs != nil and gs.players[1] != nil and gs.players[2] != nil do
      frame = %{
        game_state: gs,
        controller: cs,
        prev_controller: st.prev_cs || cs,
        player_tag: st.tag
      }

      %{st | attempt: %{st.attempt | frames: [frame | st.attempt.frames]}}
    else
      st
    end
  end

  defp marker(nil), do: nil

  defp marker(cs) do
    cond do
      cs.button_d_up and cs.button_l and cs.button_r -> :discard
      cs.button_d_up -> :save
      true -> nil
    end
  end

  defp end_attempt(%{attempt: nil} = st, _, _), do: st

  defp end_attempt(st, verdict, why) do
    # Strip trailing marker frames (taunt/chord held while ending) and the
    # stale post-load frames at the start.
    frames =
      st.attempt.frames
      |> Enum.drop_while(fn %{controller: c} -> c.button_d_up end)
      |> Enum.reverse()
      |> Enum.drop(@post_load_skip)

    keep? = verdict == :save and length(frames) >= st.min_attempt

    # Preference pairs (flywheel B4): save/discard verdicts on the SAME
    # slot are labeled preferences. Discards are kept in a parallel
    # .discards.frames file (hard negatives / future DPO-style pairs);
    # sidecar entries carry list indices into their respective files so
    # pairs are recoverable by slot.
    discard? = not keep? and length(frames) >= st.min_attempt

    meta = %{
      "slot" => st.attempt.slot,
      "started_at" => st.attempt.started_at,
      "frames" => length(frames),
      "verdict" => to_string(if keep?, do: :save, else: :discard),
      "why" => why,
      "list_index" =>
        cond do
          keep? -> length(st.saved_lists)
          discard? -> length(st.discard_lists)
          true -> nil
        end,
      "list_file" =>
        cond do
          keep? -> "saved"
          discard? -> "discards"
          true -> nil
        end
    }

    st = %{st | attempt: nil, attempts_meta: [meta | st.attempts_meta]}

    cond do
      keep? ->
        saved = st.saved_lists ++ [frames]
        write_frames!(st.out, st.tag, saved)
        write_sidecar!(st.out, st.attempts_meta)
        counts = %{st.counts | saved: st.counts.saved + 1}
        Output.success("   SAVED #{length(frames)} frames (#{counts.saved} total) -> #{st.out}")
        %{st | saved_lists: saved, counts: counts}

      discard? ->
        discards = st.discard_lists ++ [frames]
        write_frames!(st.out <> ".discards.frames", st.tag, discards)
        write_sidecar!(st.out, st.attempts_meta)
        counts = %{st.counts | discarded: st.counts.discarded + 1}
        Output.puts("   discarded (#{why}) — kept as negative")
        %{st | discard_lists: discards, counts: counts}

      true ->
        counts = %{st.counts | discarded: st.counts.discarded + 1}

        reason =
          if verdict == :save,
            do: "too short (< #{st.min_attempt} frames)",
            else: why

        Output.puts("   discarded (#{reason}, too short to keep)")
        %{st | counts: counts}
    end
  end

  defp report_drift(_gs, nil, _tol, slot),
    do: Output.warning("slot #{slot}: no reference frame for drift check")

  defp report_drift(gs, ref, tol, slot) do
    live1 = ScenarioScan.player_summary(gs.players[1])
    live2 = ScenarioScan.player_summary(gs.players[2])
    d1x = live1.x - ref.p1.x
    d2x = live2.x - ref.p2.x

    if abs(d1x) > tol or abs(d2x) > tol do
      Output.warning(
        "slot #{slot} DRIFTED (p1 dx=#{Float.round(d1x, 2)}, p2 dx=#{Float.round(d2x, 2)}) " <>
          "— situation is approximate"
      )
    else
      Output.puts("slot #{slot} drift ok (p1 dx=#{Float.round(d1x, 2)}, p2 dx=#{Float.round(d2x, 2)})")
    end
  end

  defp finish(st, why) do
    st = end_attempt(st, :discard, "session end")
    if st.saved_lists != [], do: write_frames!(st.out, st.tag, st.saved_lists)
    write_sidecar!(st.out, st.attempts_meta)

    %{
      saved: st.counts.saved,
      discarded: st.counts.discarded,
      out: st.out,
      ended: why
    }
  end
end

# ============================================================================
# CLI
# ============================================================================

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      slp: :string,
      frames: :string,
      dolphin: :string,
      iso: :string,
      tag: :string,
      out: :string,
      port: :integer,
      input_offset: :integer,
      drift_tolerance: :float,
      pipe_shim: :boolean,
      min_attempt: :integer,
      no_wtype: :boolean,
      slippi_port: :integer,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:slp] && opts[:frames] do
  Output.error("--slp and --frames are required")
  System.halt(1)
end

handoffs =
  opts[:frames]
  |> String.split(",", trim: true)
  |> Enum.map(&String.to_integer/1)
  |> Enum.sort()

if length(handoffs) > 8 do
  Output.error("Max 8 handoffs per session (savestate slots F1-F8)")
  System.halt(1)
end

targets = Enum.with_index(handoffs, 1)

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
slp_base = Path.basename(opts[:slp], ".slp")

run_opts = [
  input_offset: opts[:input_offset] || 1,
  drift_tolerance: opts[:drift_tolerance] || 3.0,
  use_wtype: not (opts[:no_wtype] || false),
  min_attempt: opts[:min_attempt] || 30,
  out: opts[:out] || "drills/human/#{slp_base}_#{ts}.frames",
  tag: opts[:tag] || "human_blewf"
]

Output.banner("Human Drill Recorder")

Output.config([
  {"Source", opts[:slp]},
  {"Handoffs", inspect(handoffs)},
  {"Out", run_opts[:out]},
  {"Tag", run_opts[:tag]},
  {"wtype", run_opts[:use_wtype]}
])

Output.puts("Parsing source replay...")
prep = HumanDrill.prepare_replay(opts[:slp], pipe_shim: opts[:pipe_shim] || false)

{:ok, bridge} = MeleePort.start_link()

config = %{
  dolphin_path: Path.expand(opts[:dolphin] || "~/.config/Slippi Launcher/netplay"),
  iso_path: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  controller_port: 1,
  opponent_port: 2,
  character: :mewtwo,
  stage: :final_destination,
  online_delay: 0,
  dummy_mode: "external",
  dummy_character: "fox",
  dummy_cpu_level: 0,
  no_audio: false,
  headless: false,
  # Realtime: the human plays phase 2 in this same session, and
  # emulation_speed is boot-time config. The prefix costs its realtime
  # length once per session.
  emulation_speed: 1.0,
  slippi_port: opts[:slippi_port] || 51490
}

Output.puts("Booting windowed Dolphin (netplay build)...")

result =
  case MeleePort.init_console(bridge, config, 180_000) do
    {:ok, _} -> HumanDrill.run(bridge, prep, targets, run_opts)
    :ok -> HumanDrill.run(bridge, prep, targets, run_opts)
    {:error, reason} -> %{error: "init failed: #{inspect(reason)}"}
  end

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> :ok
end

case result do
  %{error: e} ->
    Output.error(e)
    System.halt(1)

  %{saved: saved, discarded: discarded, out: out, ended: why} ->
    Output.puts("")
    Output.success("Session over (#{why}): #{saved} attempt(s) saved, #{discarded} discarded")
    if saved > 0 do
      Output.puts("Mix into training with:  --mix-frames #{out}")
    end
end
