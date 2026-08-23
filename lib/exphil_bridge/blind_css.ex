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
                  | :handback               ; done — helper owns the scene from here
      Phase       = see `t:phase/0` — the event-driven machine
                    (2026-08-23) that replaced the global frame counter;
                    re-presses are internal phase transitions now.

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

  Phase budgets are the live-validated 2026-08-22 values (steer 480,
  press 3, settle 117, pulse 300) — since 2026-08-23 they are
  WORST-CASE FALLBACKS: hover/selection/scene readbacks end each phase
  as soon as the evidence lands, and a silent watcher reproduces the
  legacy timeline exactly.
  """

  alias Melee.MemoryMap

  # Phase budgets — the live-validated 2026-08-22 open-loop timings,
  # now WORST-CASE FALLBACKS: each phase exits early on evidence
  # (hover match, selection word, scene word) and only runs out its
  # budget when the readback is silent, which reproduces the legacy
  # timeline exactly (test-pinned below).
  @steer_budget 480
  @press_frames 3
  # 480 steer + 3 press + 1 press-exit release + 116 = 600, the legacy
  # pulse-start frame.
  @confirm_budget 116
  # Legacy pulse window 600..900.
  @pulse_budget 300
  # With the pick RAM-confirmed, two pulse periods are enough — the
  # code-entry departure is unobservable anyway (2026-08-23 sessions:
  # the scene advanced within 1-2 pulses every time).
  @pulse_early_exit 120
  @max_retries 2
  # Post-warmup re-steer: the loading animation orbits the cursor
  # slightly off-portrait, so the press replays the last stretch of
  # steering before firing.
  @resteer_frames 120

  @typedoc """
  The fallback's phase: what it is doing, and how many frames it has
  been doing it.

      Phase = {:steer, n}    ; helper walks the cursor to the portrait
            | {:press, n}    ; the ONE A press (3 frames held)
            | {:confirm, n}  ; released; waiting for the selection word
            | {:pulse, n}    ; START pulses at 3-of-60 duty
  """
  @type phase :: {:steer | :press | :confirm | :pulse, non_neg_integer()}

  @doc "A fresh fallback phase (steering from frame zero)."
  @spec new() :: phase()
  def new, do: {:steer, 0}

  @doc """
  What the fallback does at the ONLINE CSS while JIT warmup is still
  running (2026-08-22c overlap: the steer phase runs concurrently with
  warmup instead of after it). The phase machine may complete steering
  early (hover match) during warmup; it then parks in `:animate` until
  ready — the press never fires mid-JIT.

  ONLINE ONLY: at the local CSS the helper has real feedback and could
  fully confirm mid-JIT, breaking the warmup interlock — local keeps
  the pure animation.
  """
  @spec warmup_step(phase()) :: :steer | :animate
  def warmup_step({:steer, _n}), do: :steer
  def warmup_step(_phase), do: :animate

  @doc """
  Phase adjustment when warmup completes: grant at least a
  #{@resteer_frames}-frame re-steer window (the animation drifted the
  cursor). With hover evidence the re-steer exits as soon as the hand
  reads the target again — the window is a ceiling, not a wait.
  """
  @spec ready_resteer_reset(phase()) :: phase()
  def ready_resteer_reset({:steer, n}), do: {:steer, min(n, @steer_budget - @resteer_frames)}
  def ready_resteer_reset(_past_steer), do: {:steer, @steer_budget - @resteer_frames}

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
  Normalize a selection reading against the TARGET character's
  game-external id. The selected word's scene-ENTRY value can be
  garbage that decodes as a plausible character (26 observed live at
  both online and probe CSS entries; the g5 smoke pulsed START at an
  unpicked CSS because {:character, 26} looked locked). Only the
  target counts as locked; any OTHER character value is noise —
  never "locked" (don't skip the press) and never a whiff (don't
  burn a bounded re-press on it). `:none` passes through as the
  positive whiff signal.
  """
  @spec normalize_selection(
          :unknown | :none | {:character, byte()},
          byte() | nil
        ) :: :unknown | :none | {:character, byte()}
  def normalize_selection({:character, ext}, target_ext) when ext == target_ext,
    do: {:character, ext}

  def normalize_selection({:character, _other}, _target_ext), do: :unknown
  def normalize_selection(other, _target_ext), do: other

  @doc """
  Does the hover byte read the wanted portrait? Totally: `nil`/dead
  watcher, unobserved address, or any non-matching value read `false`
  — early steer-exit requires POSITIVE evidence, so a dead or garbage
  hover byte just means the steer runs its legacy budget. `target_css`
  is the CSS-grid id (fox = 0x0A). Hover-byte validity at the ONLINE
  CSS is still unproven (it is verified at the local CSS; the
  press-point log carries the raw read as the validation trail).
  """
  @spec observe_hover(pid() | nil, byte()) :: boolean()
  def observe_hover(nil, _target_css), do: false

  def observe_hover(watcher, target_css) do
    case Melee.MemoryWatcher.get(watcher, :css_p1_character) do
      {:ok, word} -> Bitwise.bsr(word, 24) == target_css
      :unknown -> false
    end
  catch
    :exit, _ -> false
  end

  @doc """
  The EVENT-DRIVEN step (2026-08-23): `(Phase, Progress, Selection,
  hover_match?, retries) -> {Action, Phase', retries'}`.

  Each phase exits on evidence and falls back to the legacy budget:

    * `:steer` — ends when the hover byte reads the target (the hand
      IS on the portrait), when the selection word already reads
      locked (rematch: skip straight to the pulses), or at 480 frames.
    * `:press` — 3 frames of A; skipped entirely when RAM says locked
      (A toggles — a press there would DESELECT).
    * `:confirm` — ends the instant the selection word flips to
      `{:character, _}` (measured: a few frames), or at the legacy
      117-frame settle; a positive `:none` at budget end = the press
      whiffed -> re-press (bounded), replacing whole retry windows.
    * `:pulse` — 3-of-60 START duty; ends on scene departure, after
      #{@pulse_early_exit} frames with the pick RAM-confirmed (the
      code-entry departure is unobservable; the helper takes over,
      safe under the RAM menu merge), or at the 300-frame budget with
      the legacy retry-or-handback.

  With every observation `:unknown`/`false` the machine reproduces the
  legacy 480/483/600/900 timeline action-for-action (test-pinned).
  """
  @spec step(
          phase(),
          :at_css | :departing | :elsewhere | :unknown,
          :unknown | :none | {:character, byte()},
          boolean(),
          non_neg_integer()
        ) ::
          {:steer | :press_a | :release_a | {:pulse_start, boolean()} | :handback, phase(),
           non_neg_integer()}
  def step(phase, progress, selection, hover_match?, retries)

  def step({:steer, n}, _progress, selection, hover_match?, r) do
    cond do
      # Rematch fast path: pick already locked — nothing to press,
      # straight to the START pulses.
      match?({:character, _}, selection) -> {:steer, {:pulse, 0}, r}
      hover_match? -> {:steer, {:press, 0}, r}
      n + 1 >= @steer_budget -> {:steer, {:press, 0}, r}
      true -> {:steer, {:steer, n + 1}, r}
    end
  end

  def step({:press, n}, _progress, selection, _hover?, r) do
    cond do
      # Locked per RAM: do NOT press (A toggles).
      match?({:character, _}, selection) -> {:release_a, {:pulse, 0}, r}
      n < @press_frames -> {:press_a, {:press, n + 1}, r}
      true -> {:release_a, {:confirm, 0}, r}
    end
  end

  def step({:confirm, n}, _progress, selection, _hover?, r) do
    cond do
      match?({:character, _}, selection) -> {:release_a, {:pulse, 0}, r}
      n + 1 < @confirm_budget -> {:release_a, {:confirm, n + 1}, r}
      # Budget out with a POSITIVE whiff read: re-press immediately —
      # this replaces the legacy full-window retries.
      selection == :none and r < @max_retries -> {:release_a, {:press, 0}, r + 1}
      true -> {:release_a, {:pulse, 0}, r}
    end
  end

  def step({:pulse, n}, progress, selection, _hover?, r) do
    cond do
      # Confirmed departure: the scene is moving — hand back now.
      progress in [:departing, :elsewhere] ->
        {:handback, {:pulse, n}, r}

      # Pick RAM-confirmed and two pulse periods sent: hand back early
      # (measured 2026-08-23: the scene advanced within 1-2 pulses on
      # every cycle; the RAM menu merge lets the helper press START
      # itself if the CSS truly never left).
      match?({:character, _}, selection) and n >= @pulse_early_exit ->
        {:handback, {:pulse, n}, r}

      n < @pulse_budget ->
        {{:pulse_start, rem(n, 60) < 3}, {:pulse, n + 1}, r}

      # Budget out, still settled at the CSS, pick unconfirmed: the
      # press may never have landed — bounded re-press (legacy).
      progress == :at_css and not match?({:character, _}, selection) and r < @max_retries ->
        {:release_a, {:press, 0}, r + 1}

      true ->
        {:handback, {:pulse, n}, r}
    end
  end
end
