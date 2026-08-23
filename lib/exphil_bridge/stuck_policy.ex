defmodule ExPhil.Bridge.StuckPolicy do
  @moduledoc """
  Pure verdict for a MENU STUCK report: alarm, or suppress as a
  legitimate hold (MEMORY_WATCH_PROGRAM application #2).

  The 2026-08-22b diagnosis slice attached RAM ground truth to the
  report (`ram_scene`, `ram_traffic_delta`); this module turns that
  evidence into policy. The constraint that shapes every clause: the
  08-22 bot14 report — settled online CSS, healthy traffic, blind
  fallback NOT finished — was a REAL bug and must still alarm.

  ## Data definitions (HtDP)

      Report  = %{ram_scene: SceneEvidence, ram_traffic_delta: Traffic, ...}
      SceneEvidence = :no_watcher                ; RAM says nothing — legacy alarm path
                    | {:settled, scene}          ; Melee.MemoryMap.scene_view/1
                    | {:leaving, from, to}
                    | :unknown                   ; word unreadable this instant
      Traffic = nil | non_neg_integer()          ; datagram delta over ~250ms
      Verdict = :alarm | {:suppress, Reason}
      Reason  = :transition_in_flight            ; scene change already committed
              | :online_wait                     ; post-pick online hold (opponent/code entry)

  ## The table

  Suppression requires POSITIVE evidence on two axes — the core is
  alive (traffic > 0) AND the scene is a known legitimate hold:

    * `{:leaving, _, _}` — a scene change is committed; the "stall" is
      a load screen. Suppress.
    * settled `:slippi_online_css` with the blind fallback DONE — the
      bot picked and pressed START; the hold is Slippi waiting for the
      opponent (Direct code entry shares this scene word). Suppress.
      With the fallback NOT done, the same scene is the bot14 wedge
      class: alarm.

  Everything else alarms: no watcher, unreadable traffic, zero traffic
  (core wedged — the loudest case), unknown scenes (matchmaking minors
  are not in the taxonomy yet — never suppress on noise; extend the
  table when the scene-word change log captures them), and every other
  settled scene (offline menus always have local feedback, so zero
  progress there is genuinely stuck).

  A suppressed verdict is designed to be RE-ARMED by the caller
  (reset the helper's `stuck_reported`/`stalled_frames`): the detector
  then re-evaluates every stuck window, so a legitimate hold that
  degrades into a core wedge alarms one window later.
  """

  @type reason :: :transition_in_flight | :online_wait
  @type verdict :: :alarm | {:suppress, reason()}

  @doc """
  Verdict for a stuck report. `blind_done?` is MeleePort's
  "the blind CSS fallback completed this cycle" flag — the discriminant
  between an online-CSS wait and the bot14 wedge class.
  """
  @spec verdict(%{optional(atom()) => term()}, boolean()) :: verdict()
  def verdict(report, blind_done?) do
    scene = Map.get(report, :ram_scene, :no_watcher)
    traffic = Map.get(report, :ram_traffic_delta)

    cond do
      # No RAM evidence, or a core that stopped sending datagrams:
      # always loud. Zero traffic during a "hold" is the worst case,
      # not a hold.
      scene == :no_watcher -> :alarm
      not is_integer(traffic) or traffic <= 0 -> :alarm
      match?({:leaving, _, _}, scene) -> {:suppress, :transition_in_flight}
      scene == {:settled, :slippi_online_css} and blind_done? -> {:suppress, :online_wait}
      true -> :alarm
    end
  end
end
