defmodule ExPhil.Interp.StyleFingerprint do
  @moduledoc """
  Per-game behavioral fingerprint of one player — the feature vector for
  style clustering and player re-identification (STYLE_IDENTITY.md).

  Two feature families:

  1. **Option habits** (via `ExPhil.Options.events/2`): rates per minute of
     movement/defense options (dashdance, wavedash, rolls, ...) and MIX
     proportions inside forced-choice situations (tech options, ledge
     options, throw directions). Humans are habitually biased in these
     — tech-option mix is a classic exploitable habit.
  2. **Controller micro-mechanics** (from `controller_state`): which jump
     button (X vs Y — near-biometric), c-stick reliance, light-shield
     usage, per-button press rates, stick-position occupancy, input
     rhythm. These survive even when in-game decisions vary by matchup.

  All features are per-minute rates or [0,1] proportions, so games of
  different lengths compare directly. `vector/1` emits a fixed-order
  list for distance computations; `distance/2` is the matcher metric.

  Identity scope: WITHIN-corpus pseudonymous tags only (design doc §4).
  The canonical sanity check: does `--style-tag INFP` conditioning move
  the bot's fingerprint toward Michael's?
  """

  alias ExPhil.Options

  # Aerial attack action states N/F/B/U/D. 66 = ATTACK_AIR_F is pinned by
  # lib/exphil/agents/mewtwo_fair_expert.ex (in-repo ground truth); the
  # family is contiguous 65..69. (The 0x61-0x6B range in
  # embeddings/player/action.ex's table is the CATEGORY doc, not the ids.)
  @aerials %{65 => :nair, 66 => :fair, 67 => :bair, 68 => :uair, 69 => :dair}

  @rate_options [
    :dashdance,
    :wavedash,
    :waveland,
    :spotdodge,
    :roll_forward,
    :roll_backward,
    :grab,
    :throw,
    :double_jump,
    :airdodge,
    :getup_attack
  ]

  @buttons [:a, :b, :x, :y, :z, :l, :r, :d_up]

  @doc """
  Compute the fingerprint for `port` over one game's `game_states`.

  `controllers` is the subject's per-frame controller stream, index-aligned
  with `game_states`. Pass it explicitly when working from training frames
  (`Peppi.to_training_frames` carries `.controller` SEPARATELY and the
  game_state players may lack `controller_state` — reading it implicitly
  there would silently zero the micro features). When nil, falls back to
  `players[port].controller_state`.

  Returns a flat map of `feature_name => float`. Games shorter than
  ~10 seconds give noisy fingerprints — filter upstream.
  """
  @spec fingerprint([map()], pos_integer(), [map()] | nil) :: %{atom() => float()}
  def fingerprint(game_states, port, controllers \\ nil) do
    minutes = max(length(game_states) / 3600, 1.0e-6)
    events = Options.events(game_states, port)
    freq = Options.frequencies(events)

    controllers =
      controllers ||
        Enum.map(game_states, fn gs ->
          case gs.players[port] do
            nil -> nil
            p -> Map.get(p, :controller_state)
          end
        end)

    %{}
    |> Map.merge(option_rates(freq, minutes))
    |> Map.merge(mix(freq, :tech, tech_in_place: :tech_in_place, tech_roll: :tech_roll, missed_tech: :missed_tech))
    |> Map.merge(ledge_mix(freq))
    |> Map.merge(throw_mix(events))
    |> Map.merge(aerial_features(game_states, port, controllers, minutes))
    |> Map.merge(controller_features(controllers, minutes))
  end

  @doc """
  Fixed-order feature vector (missing keys -> 0.0). Pair with `keys/0`.
  """
  @spec vector(%{atom() => float()}) :: [float()]
  def vector(fp), do: Enum.map(keys(), &Map.get(fp, &1, 0.0))

  @doc "The canonical feature order for `vector/1`."
  @spec keys() :: [atom()]
  def keys do
    Enum.map(@rate_options, &:"#{&1}_per_min") ++
      [:tech_in_place_mix, :tech_roll_mix, :missed_tech_mix] ++
      [:ledge_getup_mix, :ledge_attack_mix, :ledge_roll_mix, :ledge_jump_mix] ++
      [:throw_forward_mix, :throw_back_mix, :throw_up_mix, :throw_down_mix] ++
      [:aerial_per_min, :nair_mix, :fair_mix, :bair_mix, :uair_mix, :dair_mix, :cstick_aerial_frac] ++
      Enum.map(@buttons, &:"press_#{&1}_per_min") ++
      [:jump_x_ratio, :cstick_active_frac, :lightshield_frac, :press_interval_mean, :press_interval_cv] ++
      Enum.map(0..8, &:"stick_cell_#{&1}")
  end

  @doc """
  Normalized L2 distance between two fingerprints, computed on
  z-scoreable feature scales: rates are log1p-compressed, proportions
  used raw. Symmetric; 0.0 = identical.
  """
  @spec distance(%{atom() => float()}, %{atom() => float()}) :: float()
  def distance(fp_a, fp_b) do
    a = fp_a |> vector() |> Enum.map(&compress/1)
    b = fp_b |> vector() |> Enum.map(&compress/1)

    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> (x - y) * (x - y) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp compress(v) when v > 1.0, do: :math.log(1.0 + v)
  defp compress(v), do: v

  # -- option habits ---------------------------------------------------------

  defp option_rates(freq, minutes) do
    Map.new(@rate_options, fn opt ->
      {:"#{opt}_per_min", Map.get(freq, opt, 0) / minutes}
    end)
  end

  defp mix(freq, _prefix, pairs) do
    counts = Enum.map(pairs, fn {out, opt} -> {:"#{out}_mix", Map.get(freq, opt, 0)} end)
    total = counts |> Enum.map(&elem(&1, 1)) |> Enum.sum()

    if total == 0 do
      Map.new(counts, fn {k, _} -> {k, 0.0} end)
    else
      Map.new(counts, fn {k, c} -> {k, c / total} end)
    end
  end

  defp ledge_mix(freq) do
    mix(freq, :ledge,
      ledge_getup: :ledge_getup,
      ledge_attack: :ledge_attack,
      ledge_roll: :ledge_roll,
      ledge_jump: :ledge_jump
    )
  end

  defp throw_mix(events) do
    dirs =
      events
      |> Enum.filter(&(&1.option == :throw))
      |> Enum.map(& &1.meta[:direction])
      |> Enum.frequencies()

    total = dirs |> Map.values() |> Enum.sum()

    Map.new([:forward, :back, :up, :down], fn d ->
      {:"throw_#{d}_mix", if(total == 0, do: 0.0, else: Map.get(dirs, d, 0) / total)}
    end)
  end

  # -- aerials ---------------------------------------------------------------

  defp aerial_features(game_states, port, controllers, minutes) do
    ctrl_array = :array.from_list(controllers)

    entries =
      game_states
      |> Enum.map(& &1.players[port])
      |> Enum.with_index()
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.flat_map(fn
        [{%{action: pa}, _}, {%{action: a}, i}] when pa != a ->
          case @aerials[a] do
            nil -> []
            kind -> [{kind, :array.get(i, ctrl_array)}]
          end

        _ ->
          []
      end)

    total = length(entries)
    kinds = entries |> Enum.map(&elem(&1, 0)) |> Enum.frequencies()

    cstick_frac =
      if total == 0 do
        0.0
      else
        entries
        |> Enum.count(fn {_k, ctrl} -> cstick_active?(ctrl) end)
        |> Kernel./(total)
      end

    Map.new(Map.keys(@aerials) |> Enum.map(&@aerials[&1]), fn kind ->
      {:"#{kind}_mix", if(total == 0, do: 0.0, else: Map.get(kinds, kind, 0) / total)}
    end)
    |> Map.put(:aerial_per_min, total / minutes)
    |> Map.put(:cstick_aerial_frac, cstick_frac)
  end

  # -- controller micro ------------------------------------------------------

  defp controller_features(controllers, minutes) do
    controllers = Enum.reject(controllers, &is_nil/1)

    if controllers == [] do
      # No controller data in this replay — emit zeros so vectors stay
      # comparable (distance treats absence as "no signal", not NaN).
      Map.new(
        Enum.map(@buttons, &:"press_#{&1}_per_min") ++
          [:jump_x_ratio, :cstick_active_frac, :lightshield_frac, :press_interval_mean, :press_interval_cv] ++
          Enum.map(0..8, &:"stick_cell_#{&1}"),
        &{&1, 0.0}
      )
    else
      presses = press_edges(controllers)
      total_frames = length(controllers)

      x_presses = presses[:x] || []
      y_presses = presses[:y] || []
      xy = length(x_presses) + length(y_presses)

      all_press_frames = presses |> Map.values() |> List.flatten() |> Enum.sort()
      intervals = all_press_frames |> Enum.chunk_every(2, 1, :discard) |> Enum.map(fn [a, b] -> b - a end)

      {imean, icv} =
        case intervals do
          [] ->
            {0.0, 0.0}

          _ ->
            m = Enum.sum(intervals) / length(intervals)
            var = Enum.sum(Enum.map(intervals, fn i -> (i - m) * (i - m) end)) / length(intervals)
            {m, if(m > 0, do: :math.sqrt(var) / m, else: 0.0)}
        end

      Map.new(@buttons, fn b ->
        {:"press_#{b}_per_min", length(presses[b] || []) / minutes}
      end)
      |> Map.put(:jump_x_ratio, if(xy == 0, do: 0.5, else: length(x_presses) / xy))
      |> Map.put(:cstick_active_frac, frac(controllers, &cstick_active?/1))
      |> Map.put(:lightshield_frac, lightshield_frac(controllers))
      |> Map.put(:press_interval_mean, imean)
      |> Map.put(:press_interval_cv, icv)
      |> Map.merge(stick_occupancy(controllers, total_frames))
    end
  end

  # Rising-edge frames per button: %{button => [frame_index]}
  defp press_edges(controllers) do
    controllers
    |> Enum.with_index()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce(%{}, fn [{prev, _}, {cur, i}], acc ->
      Enum.reduce(@buttons, acc, fn b, acc ->
        key = :"button_#{b}"

        if Map.get(cur, key) == true and Map.get(prev, key) != true do
          Map.update(acc, b, [i], &[i | &1])
        else
          acc
        end
      end)
    end)
    |> Map.new(fn {b, frames} -> {b, Enum.reverse(frames)} end)
  end

  defp cstick_active?(%{c_stick: %{x: x, y: y}}),
    do: abs(x - 0.5) > 0.25 or abs(y - 0.5) > 0.25

  defp cstick_active?(_), do: false

  # Light shield: analog trigger partially depressed WITHOUT the digital
  # click — a deliberate, habitual technique. Fraction of shoulder-active
  # frames that are light (not full).
  defp lightshield_frac(controllers) do
    active =
      Enum.filter(controllers, fn c ->
        (Map.get(c, :l_shoulder) || 0.0) > 0.05 or (Map.get(c, :r_shoulder) || 0.0) > 0.05
      end)

    case active do
      [] ->
        0.0

      _ ->
        light =
          Enum.count(active, fn c ->
            max(Map.get(c, :l_shoulder) || 0.0, Map.get(c, :r_shoulder) || 0.0) < 0.9 and
              Map.get(c, :button_l) != true and Map.get(c, :button_r) != true
          end)

        light / length(active)
    end
  end

  # 3x3 occupancy of the main stick (thirds of the unit square) — a coarse
  # "where does this hand rest" heatmap.
  defp stick_occupancy(controllers, total_frames) do
    cells =
      controllers
      |> Enum.map(fn c ->
        %{x: x, y: y} = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
        col = min(trunc(x * 3), 2)
        row = min(trunc(y * 3), 2)
        row * 3 + col
      end)
      |> Enum.frequencies()

    Map.new(0..8, fn cell ->
      {:"stick_cell_#{cell}", Map.get(cells, cell, 0) / max(total_frames, 1)}
    end)
  end

  defp frac(list, fun), do: Enum.count(list, fun) / max(length(list), 1)
end
