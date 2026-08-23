defmodule ExPhil.Bridge.BlindCss do
  @moduledoc """
  Pure decision core for the online-CSS blind pick fallback — the
  frame-timed open-loop sequence from GOTCHA #101, now SEMI-CLOSED by
  the RAM scene word when a memory watcher is present.

  The netplay-beta build streams no live online-CSS state, so the pick
  itself (the A press) stays open-loop: a character pick does not
  change the scene, and nothing observable confirms it. What IS
  observable is the START press — it commits a scene change, and the
  RAM scene controller shows it twice (`pending_major` flips, then
  `major` lands). Two closure points follow:

    * during the START-pulse phase, the scene moving = selection
      landed -> hand back immediately instead of pulsing blind;
    * after the full pulse window, the scene still settled at the
      online CSS = the pick never took -> retry the A press (bounded)
      instead of handing back a deselected character.

  With no watcher signal every path degrades to the 2026-08-22
  open-loop timings — the change is strictly additive.

  ## Data definitions (HtDP)

      Observation = :unknown | u32          ; raw :menu_state word, or no signal
      Progress    = :at_css                 ; settled at :slippi_online_css
                  | :departing              ; scene change committed (pending != major)
                  | :elsewhere              ; settled at a KNOWN non-CSS scene
                  | :unknown                ; no signal, or settled at an unmapped scene
      Selection   = :unknown                ; no watcher / address unreadable
                  | :none                   ; RAM says coin in hand (0x21)
                  | {:character, ext_id}    ; RAM says locked in
      Action      = :steer                  ; helper drives (cursor -> portrait)
                  | :press_a                ; the ONE pick press (A toggles!)
                  | :release_a              ; settle after the press
                  | {:pulse_start, on?}     ; START pulse duty cycle
                  | {:retry_a, retries'}    ; pick didn't land; caller resets n to @a_press_at
                  | :handback               ; done — helper owns the scene from here

  `Selection` (2026-08-22c) reads the `css_pN_selected` RAM word — the
  static-region selected-character array (0x8043208C stride 8,
  external id, 0x21 = none) that replaced the stream's dead
  `coin_down`. Verified live at the OFFLINE CSS; whether the online
  CSS drives the same array is OWED (next Direct session — the skip
  log line below is the probe). Where it reads `{:character, _}` the
  press phase SKIPS the A press, which makes the toggle-off
  impossible and turns the bounded retry into a safe START re-pulse.
  `:unknown` and `:none` keep the validated open-loop press.

  An unmapped settled scene classifies as `:unknown`, NOT `:elsewhere`:
  the online scene minors are not fully derived yet (program doc), and
  acting on an unrecognized word would close the loop on noise.

  Phase timings are the live-validated 2026-08-22 values: helper
  steers 480 frames (cursor parks), A held 3, released ~2s, START
  pulsed ~5s at 3-of-60 duty.
  """

  alias Melee.MemoryMap

  # Live-validated phase boundaries (frames at the online CSS).
  @a_press_at 480
  @a_release_at 483
  @pulse_at 600
  @handback_at 900
  @max_retries 2
  # Post-warmup re-steer: the loading animation orbits the cursor
  # slightly off-portrait, so the press replays the last stretch of
  # steering before firing.
  @resteer_frames 120

  @doc "Frame index where the A-press phase begins (retry reset point)."
  def a_press_at, do: @a_press_at

  @doc """
  What the fallback does at the ONLINE CSS while JIT warmup is still
  running (2026-08-22c overlap: the steer phase runs concurrently with
  warmup instead of after it — same 480 frames of helper exposure,
  moved ~8s earlier).

      WarmupAction = :steer    ; n < a_press_at — helper parks the cursor
                   | :animate  ; parked; play the loading animation

  ONLINE ONLY: at the local CSS the helper has real feedback and could
  fully confirm mid-JIT, breaking the warmup interlock — local keeps
  the pure animation.
  """
  @spec warmup_step(non_neg_integer()) :: :steer | :animate
  def warmup_step(n) when n < @a_press_at, do: :steer
  def warmup_step(_n), do: :animate

  @doc """
  Counter adjustment when warmup completes: a fully-steered counter
  rewinds to a #{@resteer_frames}-frame re-steer window (the animation
  drifted the cursor); a partial steer keeps its progress.
  """
  @spec ready_resteer_reset(non_neg_integer()) :: non_neg_integer()
  def ready_resteer_reset(n) when n >= @a_press_at, do: @a_press_at - @resteer_frames
  def ready_resteer_reset(n), do: n

  @doc """
  Read the scene word off a memory watcher, totally: `nil` watcher,
  dead watcher, or a not-yet-observed address all yield `:unknown`.
  """
  @spec observe(pid() | nil) :: :unknown | non_neg_integer()
  def observe(nil), do: :unknown

  def observe(watcher) do
    case Melee.MemoryWatcher.get(watcher, :menu_state) do
      {:ok, word} -> word
      :unknown -> :unknown
    end
  catch
    # A dead watcher must not take the menu loop down with it.
    :exit, _ -> :unknown
  end

  @doc """
  Read the local player's CSS selection off a memory watcher, totally:
  `nil`/dead watcher, unobserved address, or an implausible word (an
  id must fit a byte) all yield `:unknown`. `port` is the CSS array
  slot — the online CSS has a single local cursor, slot 1.
  """
  @spec observe_selected(pid() | nil, 1..4) :: :unknown | :none | {:character, byte()}
  def observe_selected(nil, _port), do: :unknown

  def observe_selected(watcher, port) do
    case Melee.MemoryWatcher.get(watcher, :"css_p#{port}_selected") do
      {:ok, word} when word <= 0xFF -> MemoryMap.css_selected(word)
      {:ok, _implausible} -> :unknown
      :unknown -> :unknown
    end
  catch
    :exit, _ -> :unknown
  end

  @doc """
  Observation -> Progress. One clause per SceneView class
  (`Melee.MemoryMap.scene_view/1`).
  """
  @spec classify(:unknown | non_neg_integer()) ::
          :at_css | :departing | :elsewhere | :unknown
  def classify(:unknown), do: :unknown

  # All-zero words appear as ~17ms transients during online scene
  # churn (observed twice, bot14 capture 2026-08-22). A bot genuinely
  # at press-start can never be inside the blind-CSS arm, so zero is
  # load noise, never evidence — without this clause it classified
  # :elsewhere and could trigger a premature handback mid-pulse.
  def classify(0), do: :unknown

  def classify(word) when is_integer(word) do
    case MemoryMap.scene_view(word) do
      {:settled, :slippi_online_css} -> :at_css
      {:leaving, _from, _to} -> :departing
      {:settled, {:unknown, _}} -> :unknown
      {:settled, _known} -> :elsewhere
    end
  end

  @doc """
  The decision table: (frames at CSS, Progress, retries, Selection) ->
  Action.

  Scene evidence is consulted only where it can mean something — the
  pulse phase (early confirm) and the window end (retry-or-give-up).
  Selection evidence is consulted only at the press window: a
  RAM-confirmed `{:character, _}` skips the A press (A toggles — a
  press there would DESELECT); `:none`/`:unknown` keep the validated
  open-loop press. Everything else is timed regardless.
  """
  @spec step(
          non_neg_integer(),
          :at_css | :departing | :elsewhere | :unknown,
          non_neg_integer(),
          :unknown | :none | {:character, byte()}
        ) ::
          :steer
          | :press_a
          | :release_a
          | {:pulse_start, boolean()}
          | {:retry_a, non_neg_integer()}
          | :handback
  def step(n, progress, retries \\ 0, selection \\ :unknown)

  def step(n, _progress, _retries, _selection) when n < @a_press_at, do: :steer

  # Press window, already locked in per RAM: do NOT press (A toggles).
  def step(n, _progress, _retries, {:character, _id}) when n < @a_release_at, do: :release_a

  def step(n, _progress, _retries, _selection) when n < @a_release_at, do: :press_a
  def step(n, _progress, _retries, _selection) when n < @pulse_at, do: :release_a

  # START pulses: confirmed departure ends them early.
  def step(n, progress, _retries, _selection)
      when n < @handback_at and progress in [:departing, :elsewhere],
      do: :handback

  def step(n, _progress, _retries, _selection) when n < @handback_at,
    do: {:pulse_start, rem(n, 60) < 3}

  # Window over, still settled at the CSS, but RAM confirms the pick:
  # hand back NOW instead of burning retry windows (each is ~7s of
  # steer-skip + pulses). Measured 2026-08-23 live: every retry cycle
  # ran with "SKIPPING A press (already locked in)" — pure wait. The
  # handback is safe because the RAM menu merge gives the helper real
  # coin_down/selection at the online CSS, so it presses START itself
  # if the scene truly never left.
  def step(_n, :at_css, _retries, {:character, _id}), do: :handback

  # Window over, still settled at the CSS, pick unconfirmed: the press
  # may never have landed — bounded retry (legacy open-loop path).
  def step(_n, :at_css, retries, _selection) when retries < @max_retries,
    do: {:retry_a, retries + 1}

  def step(_n, _progress, _retries, _selection), do: :handback
end
