# Hit-confirm drill episode driver (DRILL_HITCONFIRM.md Route B, build step 2).
#
# Puts the policy IN a hit-confirm state over and over: scripted setup drives
# the port-2 dummy to the drill cell's position/percent band, a scripted
# opener (up-throw) executes on the bot's port, and control HANDS OFF to the
# policy at the throw (the drill_table_mine anchor — bot action enters
# 219..222) for a 240 f continuation window. Score continuation, not the
# opener. Every game's replay is banked; episodes.jsonl records handoff
# frames for the offline scorer (scripts/drill_score.exs), which recomputes
# the cell statistics with the exact mining detector.
#
# Drill 1 (pre-registered): uthrow / victim 0-19% / mid-stage / FD.
# Reference cell (eval_runs/0901_drill_table/RESULTS.md): n=39, mean 3.9
# hits, 87% >=3 hits, 27.0 mean dmg.
#
#   mix run scripts/drill_episode.exs \
#     --policy checkpoints/fox_gen_v1.3_ARrefit_policy.bin \
#     --episodes 20 --out eval_runs/0902_drill_smoke
#
# Options:
#   --policy PATH          Bot policy (required; port 1)
#   --drill NAME|PATH      Drill manifest (default uthrow_low_mid ->
#                          drills/uthrow_low_mid.json): cell, bands, pinned
#                          references, training notes. Snapshotted into the
#                          bank as drill.json for the scorer/slicer.
#   --episodes N           Scored episodes to collect (default 20)
#   --window N             Continuation frames after handoff (manifest, 240)
#   --pct-lo/--pct-hi      Override: single percent band (else manifest
#                          bands + BAND-LADDERING: after a scored episode,
#                          if the victim's risen percent lands in another
#                          band the next episode chains on the same stock)
#   --temperature T        Scalar temperature (default 0.5)
#   --buttons-temperature  Buttons head temperature (default 0.5)
#   --dummy-character C    Port-2 victim (default fox)
#   --no-warm-context      Skip query-and-discard during scripted phases.
#                          Default ON: the agent is queried (output discarded)
#                          through setup/opener so its temporal buffer holds
#                          real history at handoff. Caveat: the prev-action
#                          channel then carries the agent's DISCARDED choices,
#                          not the scripted presses (07-08 lesson violated by
#                          one remove) — acceptable for state-visitation,
#                          revisit if continuation looks input-confused.
#   --out DIR              Bank dir (default eval_runs/drill_<ts>)
#   --dolphin PATH         (default ~/.local/share/slippi/exi-ai/dolphin-emu-headless)
#   --iso PATH             (default ~/isos/melee.iso)
#   --slippi-port N        Base port (default 51700, +1 per game)
#   --quiet
#
# Episode lifecycle (one per dummy stock, ~4 per game):
#   :position -> :opener(grab->throw) -> :continuation(policy) -> :reset
#   (scripted dummy edge-suicide resets percent via the fresh stock; no
#   pummel accumulation needed for the 0-19 band). Game ends when the dummy
#   loses its last stock; the finalized .slp is banked and a new console
#   launches until the episode target is met (:drain burns leftover stocks).

require Logger

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Training.Output

defmodule DrillEpisode do
  @moduledoc false

  @neutral %{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
  }

  # Internal action-state ids (drill_table_mine conventions, GOTCHA notes
  # 09-01): captured 223..232, thrown 239..243, throws 219..222 (uthrow 221),
  # rebirth 12/13. A "hit" on the victim = hitstun OR thrown OR captured.
  @captured 223..232
  @thrown 239..243
  @uthrow 221
  @rebirth [12, 13]
  # Death-fly / dead states (internal ids 0..10): percent reads stale until
  # the respawn, so :position must wait these out rather than judging bands.
  @dead 0..10

  @menu_step_limit 12_000
  @grab_distance 7.0
  @dummy_home 0.0
  @position_budget 1_200
  @opener_capture_wait 20
  @opener_throw_wait 40
  @whiff_cooldown 40
  @max_grab_attempts 6
  @reset_budget 1_800

  def hitstun?(pl) do
    pl != nil and
      ((pl.hitstun_frames_left || 0) > 0 or pl.action in @thrown or pl.action in @captured)
  end

  # Band-laddering: bands are half-open [lo, hi+1) so 19.5% still lands in
  # 0-19. An episode's band is fixed by the victim's percent at capture;
  # after a scored episode the victim's risen percent usually lands in the
  # NEXT band, so episodes chain on the same stock (free percent
  # accumulation, matching how expert conversions ladder percent) until the
  # percent exits the top band — only then does the suicide reset fire.
  def band_of(pct, bands), do: Enum.find(bands, fn {lo, hi} -> pct >= lo and pct < hi + 1 end)
  def band_key(nil), do: nil
  def band_key({lo, hi}), do: "#{lo}-#{hi}"

  # -- one console/game -------------------------------------------------------

  # One console runs the whole session: the bridge's auto_menu restarts
  # games in place (dummy stocks exhausted -> new game, frame counter back
  # to -123), so game boundaries are detected by the frame counter going
  # backward and episodes are tagged with a per-console game index. The
  # replay dir accumulates one timestamped .slp per game; sorted filenames
  # map 1:1 to game indexes.
  def run_console(agent, opts) do
    run_dir = Path.join(opts[:out], "games")
    File.mkdir_p!(run_dir)

    {:ok, bridge} = MeleePort.start_link()

    config = %{
      dolphin_path: opts[:dolphin],
      iso_path: opts[:iso],
      controller_port: 1,
      opponent_port: 2,
      character: :fox,
      stage: opts[:stage],
      online_delay: 0,
      dummy_mode: "external",
      dummy_character: opts[:dummy_character],
      dummy_cpu_level: 0,
      no_audio: true,
      headless: true,
      emulation_speed: 0.0,
      replay_dir: run_dir,
      slippi_port: opts[:slippi_port]
    }

    Agent.reset_buffer(agent)

    st = %{
      phase: :position,
      phase_start: nil,
      menu_steps: 0,
      agent: agent,
      opts: opts,
      grab_attempts: 0,
      cooldown: 0,
      press_frames: 0,
      ep: nil,
      episodes: [],
      remaining: opts[:remaining],
      game_idx: 0,
      last_frame: nil
    }

    result =
      case MeleePort.init_console(bridge, config, 180_000) do
        {:ok, _} -> loop(bridge, st)
        :ok -> loop(bridge, st)
        {:error, reason} -> %{error: "init failed: #{inspect(reason)}", episodes: []}
      end

    try do
      MeleePort.stop(bridge)
    catch
      :exit, _ -> :ok
    end

    if Process.alive?(bridge), do: GenServer.stop(bridge, :normal, 5_000)

    slps = run_dir |> Path.join("*.slp") |> Path.wildcard() |> Enum.sort()
    Map.merge(result, %{run_dir: run_dir, slps: slps})
  rescue
    e -> %{error: Exception.message(e), episodes: [], run_dir: nil, slps: []}
  end

  defp loop(bridge, st) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:menu, _} ->
        if st.menu_steps > @menu_step_limit do
          %{error: "stuck in menus", episodes: st.episodes}
        else
          loop(bridge, %{st | menu_steps: st.menu_steps + 1})
        end

      {:postgame, _} ->
        %{episodes: st.episodes}

      {:game_ended, _reason} ->
        %{episodes: st.episodes}

      {:ok, gs} ->
        # menu_steps guards a single menu VISIT, not the whole session —
        # without this reset the limit is cumulative and trips after ~97
        # auto_menu game transitions (found at 388/500, 09-02).
        frame_step(bridge, gs, %{st | menu_steps: 0})

      {:error, reason} ->
        %{error: "step error: #{inspect(reason)}", episodes: st.episodes}
    end
  end

  defp frame_step(bridge, gs, st) do
    bot = gs.players[1]
    vic = gs.players[2]

    cond do
      bot == nil or vic == nil ->
        send_both(bridge, @neutral, @neutral)
        loop(bridge, st)

      st.last_frame != nil and gs.frame < st.last_frame ->
        # Frame counter went backward: auto_menu started a new game.
        Output.puts("  -- game boundary (game #{st.game_idx + 2} starting)")
        Agent.reset_buffer(st.agent)

        cond do
          st.remaining <= 0 ->
            %{episodes: st.episodes}

          st.episodes == [] and st.game_idx >= 2 ->
            %{error: "3 games with zero episodes — check setup/opener logic", episodes: []}

          true ->
            st = %{
              st
              | game_idx: st.game_idx + 1,
                last_frame: gs.frame,
                phase: :position,
                phase_start: gs.frame,
                ep: nil,
                grab_attempts: 0,
                cooldown: 0
            }

            send_both(bridge, @neutral, @neutral)
            loop(bridge, st)
        end

      true ->
        st = %{st | last_frame: gs.frame}
        st = if st.phase_start == nil, do: %{st | phase_start: gs.frame}, else: st
        debug_frame(gs, bot, vic, st)
        warm_context(bridge, gs, st)
        do_phase(st.phase, bridge, gs, bot, vic, st)
    end
  end

  # --debug: per-step driver view vs the console's applied pad (P1).
  defp debug_frame(gs, bot, vic, st) do
    if st.opts[:debug] do
      cs = bot.controller_state

      pad =
        if cs,
          do:
            "pad=#{Float.round(cs.main_stick.x, 2)},#{Float.round(cs.main_stick.y, 2)}" <>
              " z=#{cs.button_z}",
          else: "pad=nil"

      IO.puts(
        "DBG f#{gs.frame} #{st.phase} p1.act=#{trunc(bot.action || -1)} " <>
          "p2.act=#{trunc(vic.action || -1)} #{pad} dx=#{Float.round((vic.x || 0.0) - (bot.x || 0.0), 1)}"
      )
    end
  end

  # Query-and-discard during scripted phases so the temporal buffer holds
  # real history at handoff (see --no-warm-context in the header).
  defp warm_context(_bridge, gs, %{phase: p, opts: opts, agent: agent})
       when p in [:position, :opener] do
    if opts[:warm_context], do: Agent.get_controller(agent, gs, player_port: 1)
    :ok
  end

  defp warm_context(_bridge, _gs, _st), do: :ok

  # -- :position — script both ports into the drill cell ----------------------

  defp do_phase(:position, bridge, gs, bot, vic, st) do
    elapsed = gs.frame - st.phase_start
    band? = band_of(vic.percent || 0.0, st.opts[:bands]) != nil

    cond do
      st.cooldown > 0 ->
        send_both(bridge, @neutral, @neutral)
        loop(bridge, %{st | cooldown: st.cooldown - 1})

      vic.action in @dead ->
        # Victim is mid-death-fly (stale percent until respawn) — just wait.
        send_both(bridge, @neutral, @neutral)
        loop(bridge, st)

      not band? and not (vic.action in @rebirth) ->
        # A stray hit pushed the victim out of the band — reset via suicide.
        to_phase(bridge, st, :reset, note: "percent out of band")

      elapsed > @position_budget ->
        to_phase(bridge, st, :reset, note: "position timeout")

      ready_to_grab?(bot, vic) ->
        to_phase(bridge, %{st | press_frames: 2}, :opener)

      true ->
        send_both(bridge, approach_input(bot, vic), home_input(vic))
        loop(bridge, st)
    end
  end

  # -- :opener — grab, verify capture, throw up, hand off ----------------------

  defp do_phase(:opener, bridge, gs, bot, vic, st) do
    elapsed = gs.frame - st.phase_start

    cond do
      # HANDOFF: the drill_table_mine anchor — bot enters the throw action.
      bot.action == @uthrow ->
        pct0 = st.ep[:vic_pct0] || vic.percent || 0.0

        ep = %{
          handoff: gs.frame,
          vic_pct0: pct0,
          band: band_key(band_of(pct0, st.opts[:bands])),
          vic_stock0: vic.stock,
          hits: 1,
          prev_hs: true,
          last_pct: vic.percent || 0.0,
          died: false
        }

        cont_st = %{st | ep: ep, grab_attempts: 0, press_frames: 0}
        to_phase(bridge, cont_st, :continuation)

      vic.action in @captured ->
        # Captured — record pre-throw percent once, then PULSE up to throw.
        # A held direction never triggers a throw (the 07-08 press-EDGES
        # lesson: the game wants a fresh cardinal input in GRAB_WAIT, and
        # an up that arrives a frame early, during CatchPull, is ignored
        # forever if simply held). 2 frames up / 2 neutral guarantees an
        # edge lands in an actionable frame.
        ep = st.ep || %{vic_pct0: vic.percent || 0.0}

        if elapsed > @opener_capture_wait + @opener_throw_wait do
          to_phase(bridge, st, :reset, note: "throw never came out")
        else
          input = if rem(elapsed, 4) < 2, do: stick(0.5, 1.0), else: @neutral
          send_both(bridge, input, @neutral)
          loop(bridge, %{st | ep: ep})
        end

      st.press_frames > 0 ->
        send_both(bridge, z_press(), @neutral)
        loop(bridge, %{st | press_frames: st.press_frames - 1})

      elapsed > @opener_capture_wait ->
        # Whiffed. Cool down (let the grab animation end), retry from setup.
        if st.grab_attempts + 1 >= @max_grab_attempts do
          to_phase(bridge, st, :reset, note: "grab whiffed x#{@max_grab_attempts}")
        else
          st = %{st | grab_attempts: st.grab_attempts + 1, cooldown: @whiff_cooldown, ep: nil}
          to_phase(bridge, st, :position)
        end

      true ->
        send_both(bridge, @neutral, @neutral)
        loop(bridge, st)
    end
  end

  # -- :continuation — the policy plays from the hit-confirm state -------------

  defp do_phase(:continuation, bridge, gs, _bot, vic, st) do
    ep = st.ep
    hs = hitstun?(vic)
    hits = if hs and not ep.prev_hs, do: ep.hits + 1, else: ep.hits
    died = ep.died or (vic.stock || 0) < (ep.vic_stock0 || 0)
    last_pct = if died, do: ep.last_pct, else: vic.percent || ep.last_pct
    ep = %{ep | hits: hits, prev_hs: hs, died: died, last_pct: last_pct}

    if gs.frame - ep.handoff >= st.opts[:window] do
      st = score_episode(gs, %{st | ep: ep})

      cond do
        st.remaining <= 0 ->
          to_phase(bridge, st, :drain)

        ep.died ->
          to_phase(bridge, st, :position)

        band_of(vic.percent || 0.0, st.opts[:bands]) != nil ->
          # Ladder: the risen percent still lands in a drill band — chain
          # the next episode on this stock instead of suiciding.
          to_phase(bridge, st, :position)

        true ->
          to_phase(bridge, st, :reset)
      end
    else
      case Agent.get_controller(st.agent, gs, player_port: 1) do
        {:ok, c} -> MeleePort.send_controller(bridge, controller_to_input(c))
        {:error, _} -> MeleePort.send_controller(bridge, @neutral)
      end

      MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      loop(bridge, %{st | ep: ep})
    end
  end

  # -- :reset / :drain — scripted dummy edge-suicide ---------------------------

  defp do_phase(phase, bridge, gs, _bot, vic, st) when phase in [:reset, :drain] do
    stock0 = st.ep[:reset_stock0] || vic.stock

    cond do
      (vic.stock || 0) < (stock0 || 0) or vic.action in @rebirth ->
        next = if phase == :drain, do: :drain_wait, else: :position
        to_phase(bridge, %{st | ep: nil}, next)

      gs.frame - st.phase_start > @reset_budget ->
        %{error: "reset stuck (dummy won't die, action=#{trunc(vic.action || -1)})",
          episodes: st.episodes}

      true ->
        # Melee will not WALK a character off a ledge (walks stop at the
        # brink), and a HELD full deflection never dashes (dash needs a
        # fresh smash input — the press-edges lesson again). So: pulse the
        # stick to re-trigger dash (runs DO carry off edges), and at the
        # brink jump + drift as the guaranteed exit.
        elapsed = gs.frame - st.phase_start
        dir = if (vic.x || 0.0) >= 0.0, do: 1.0, else: 0.0
        drift = stick(dir, 0.5)

        input =
          cond do
            not (vic.on_ground || false) ->
              # Airborne (incl. hanging release): drift toward the blast zone.
              drift

            abs(vic.x || 0.0) > 80.0 ->
              # At the brink: X-tap jump + drift.
              if rem(elapsed, 20) < 2, do: jump_press(dir), else: drift

            true ->
              # Pulse full deflection so each fresh edge dashes edgeward.
              if rem(elapsed, 8) < 6, do: drift, else: @neutral
          end

        send_both(bridge, @neutral, input)
        loop(bridge, %{st | ep: Map.put(st.ep || %{}, :reset_stock0, stock0)})
    end
  end

  # Target met: burn remaining dummy stocks until the game ends naturally
  # (a finalized .slp needs the game to end).
  defp do_phase(:drain_wait, bridge, _gs, _bot, vic, st) do
    if vic.action in @rebirth or not hitstun?(vic) do
      to_phase(bridge, st, :drain)
    else
      send_both(bridge, @neutral, @neutral)
      loop(bridge, st)
    end
  end

  # -- helpers -----------------------------------------------------------------

  defp to_phase(bridge, st, phase, opts \\ []) do
    if note = opts[:note], do: Output.puts("    [#{phase}] #{note}")
    send_both(bridge, @neutral, @neutral)
    loop(bridge, %{st | phase: phase, phase_start: nil})
  end

  defp score_episode(gs, st) do
    ep = st.ep
    dmg = max(ep.last_pct - ep.vic_pct0, 0.0)

    row = %{
      game: st.game_idx,
      handoff: ep.handoff,
      end_frame: gs.frame,
      vic_pct0: Float.round(ep.vic_pct0 * 1.0, 1),
      band: ep.band,
      hits: ep.hits,
      dmg: Float.round(dmg, 1),
      died: ep.died
    }

    seq = length(st.episodes) + 1

    ref_s =
      case st.opts[:references][ep.band] do
        %{"hits" => h, "dmg" => d} -> "ref #{h}/#{d}"
        _ -> "no ref"
      end

    Output.puts(
      "  ep #{st.opts[:episodes] - st.remaining + 1} [#{row.band}%]: hits=#{row.hits} dmg=#{row.dmg}" <>
        if(row.died, do: " STOCK", else: "") <>
        "  (#{ref_s})  [game ep #{seq}, handoff f#{row.handoff}]"
    )

    %{st | episodes: st.episodes ++ [row], ep: nil, remaining: st.remaining - 1}
  end

  defp ready_to_grab?(bot, vic) do
    dx = (vic.x || 0.0) - (bot.x || 0.0)

    abs(dx) <= @grab_distance and bot.on_ground and vic.on_ground and
      not vic.invulnerable and facing_toward?(bot, dx) and
      not hitstun?(vic)
  end

  defp facing_toward?(_bot, dx) when abs(dx) < 1.0, do: true
  defp facing_toward?(bot, dx), do: (bot.facing || 1) * dx > 0

  defp approach_input(bot, vic) do
    dx = (vic.x || 0.0) - (bot.x || 0.0)

    cond do
      abs(dx) <= @grab_distance -> @neutral
      dx > 0 -> stick(0.65, 0.5)
      true -> stick(0.35, 0.5)
    end
  end

  defp home_input(vic) do
    dx = @dummy_home - (vic.x || 0.0)

    cond do
      abs(dx) < 8.0 -> @neutral
      dx > 0 -> stick(0.65, 0.5)
      true -> stick(0.35, 0.5)
    end
  end

  defp stick(x, y), do: %{@neutral | main_stick: %{x: x, y: y}}
  defp z_press, do: %{@neutral | buttons: %{@neutral.buttons | z: true}}
  defp jump_press(dir), do: %{stick(dir, 0.5) | buttons: %{@neutral.buttons | x: true}}

  defp send_both(bridge, p1, p2) do
    MeleePort.send_controller(bridge, p1)
    MeleePort.send_controller(bridge, Map.put(p2, :port, 2))
  end

  defp controller_to_input(cs) do
    %{
      main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
      c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
      shoulder: cs.l_shoulder + cs.r_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end
end

# ============================================================================
# CLI
# ============================================================================

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      drill: :string,
      episodes: :integer,
      window: :integer,
      pct_lo: :integer,
      pct_hi: :integer,
      temperature: :float,
      buttons_temperature: :float,
      dummy_character: :string,
      no_warm_context: :boolean,
      debug: :boolean,
      out: :string,
      dolphin: :string,
      iso: :string,
      slippi_port: :integer,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:policy] do
  Output.error("--policy is required")
  System.halt(1)
end

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
out_dir = Path.expand(opts[:out] || "eval_runs/drill_#{ts}")
File.mkdir_p!(out_dir)

episodes_target = opts[:episodes] || 20

temperature = %{
  buttons: opts[:buttons_temperature] || 0.5,
  main: opts[:temperature] || 0.5,
  c: opts[:temperature] || 0.5,
  shoulder: opts[:temperature] || 0.5
}

# Drill manifest: cell definition, bands, pinned references, training notes.
# (drills/<name>.json — the registry the scorer and retrain slicer read too.)
drill_arg = opts[:drill] || "uthrow_low_mid"
drill_path = if File.exists?(drill_arg), do: drill_arg, else: "drills/#{drill_arg}.json"
drill = drill_path |> File.read!() |> Jason.decode!()

bands =
  if opts[:pct_lo] || opts[:pct_hi] do
    [{opts[:pct_lo] || 0, opts[:pct_hi] || 19}]
  else
    for [lo, hi] <- drill["cell"]["bands"], do: {lo, hi}
  end

references = drill["references"] || %{}

# Snapshot the manifest into the bank — provenance for the scorer/slicer.
File.cp!(drill_path, Path.join(out_dir, "drill.json"))

run_opts = [
  out: out_dir,
  episodes: episodes_target,
  window: opts[:window] || drill["window"] || 240,
  bands: bands,
  references: references,
  stage: String.to_atom(drill["cell"]["stage"] || "final_destination"),
  dummy_character: opts[:dummy_character] || drill["cell"]["dummy_character"] || "fox",
  warm_context: !(opts[:no_warm_context] || false),
  debug: opts[:debug] || false,
  dolphin: Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
  iso: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  slippi_port: opts[:slippi_port] || 51_700
]

Output.banner("Hit-confirm drill — #{drill["name"]}")

Output.config([
  {"Policy", opts[:policy]},
  {"Drill", drill_path},
  {"Episodes", episodes_target},
  {"Window", run_opts[:window]},
  {"Bands", Enum.map_join(bands, ", ", fn {lo, hi} -> "#{lo}-#{hi}%" end)},
  {"Dummy", run_opts[:dummy_character]},
  {"Warm context", run_opts[:warm_context]},
  {"Out", out_dir}
])

{:ok, agent} = Agent.start_link(policy_path: opts[:policy], temperature: temperature)

case Agent.warmup(agent) do
  {:ok, ms} -> Output.success("Agent warmed up (#{ms}ms)")
  {:error, r} -> Output.warning("Agent warmup failed: #{inspect(r)}")
end

cfg = Agent.get_config(agent)

Output.puts(
  "  agent: use_prev_action=#{inspect(cfg[:use_prev_action])} temporal=#{inspect(cfg[:temporal])}"
)

jsonl = Path.join(out_dir, "episodes.jsonl")

t0 = System.monotonic_time(:millisecond)
result = DrillEpisode.run_console(agent, Keyword.put(run_opts, :remaining, episodes_target))
wall = Float.round((System.monotonic_time(:millisecond) - t0) / 1000, 1)

all = result[:episodes] || []
slps = result[:slps] || []

# Map each episode's per-console game index to its finalized .slp (sorted
# timestamped filenames are chronological, matching the boundary counter).
rows =
  Enum.map(all, fn ep ->
    slp =
      case Enum.at(slps, ep.game) do
        nil -> nil
        p -> Path.relative_to(p, out_dir)
      end

    Map.put(ep, :slp, slp)
  end)

File.write!(jsonl, Enum.map_join(rows, "", &(Jason.encode!(&1) <> "\n")), [:append])

unmapped = Enum.count(rows, &(&1.slp == nil))
if unmapped > 0, do: Output.warning("#{unmapped} episodes have no mapped replay (game count #{length(slps)})")

if result[:error] do
  Output.warning("console: #{result[:error]} (#{length(all)} episodes kept, #{wall}s)")
else
  Output.success("console: #{length(all)} episodes across #{length(slps)} games, #{wall}s")
end

if all != [] do
  Output.puts("")
  Output.puts("== smoke summary (live counter; authoritative = drill_score.exs on the bank)")

  all
  |> Enum.group_by(& &1.band)
  |> Enum.sort()
  |> Enum.each(fn {band, eps} ->
    n = length(eps)
    mh = Float.round(Enum.sum(Enum.map(eps, & &1.hits)) / n, 1)
    md = Float.round(Enum.sum(Enum.map(eps, & &1.dmg)) / n, 1)
    deep = round(Enum.count(eps, &(&1.hits >= 3)) / n * 100)

    ref_s =
      case references[band] do
        %{"hits" => h, "deep3" => d3, "dmg" => d, "n" => rn} ->
          "expert n=#{rn}: #{h} / #{d3}% / #{d}"

        _ ->
          "no reference mined"
      end

    Output.puts("   #{band}%: n=#{n}  hits #{mh}  >=3 #{deep}%  dmg #{md}   (#{ref_s})")
  end)

  Output.success("bank -> #{out_dir} (#{jsonl})")
else
  Output.error("no episodes collected")
end

GenServer.stop(agent)

case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} ->
    pids = String.split(out, "\n", trim: true)
    Enum.each(pids, fn pid -> System.cmd("kill", [pid]) end)
    if pids != [], do: Output.warning("Killed #{length(pids)} orphaned Dolphin(s)")

  _ ->
    :ok
end
