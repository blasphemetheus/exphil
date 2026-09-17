defmodule ExPhil.Interp.StyleTiming do
  @moduledoc """
  STYLE_IDENTITY.md fingerprint v2: frame-timing habits that need the
  animation stream, not just event counts. These are MOTOR habits (when
  you press L before landing, how steep your wavedash is, which jumpsquat
  frame you airdodge on, how fast you act out of shield) — closer to a
  biometric than option choices, which vary by matchup.

  Input: the subject's per-frame player maps (`game_state.players[port]`)
  and the index-aligned controller stream. Every feature is 0.0 when the
  situation never occurs in the game, so vectors stay comparable.

  Action ids: `ExPhil.Interp.ActionNames` (KNEE_BEND 24, JUMPING_F/B 25/26,
  LANDING_SPECIAL 43, *_LANDING 70..74, DAMAGE_* 75..91, SHIELD 178..182
  with SHIELD_STUN 181, AIRDODGE 236).
  """

  @knee_bend 24
  @jumps [25, 26]
  @landing_special 43
  @aerial_landings 70..74
  @damage 75..91
  @shield 178..182
  @shield_stun 181
  @airdodge 236
  # jumps + aerial jumps + falls: the states a jump arc passes through
  @airborne [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 65, 66, 67, 68, 69, 236]
  @wavedash_window 10

  @lcancel_window 7
  @shoulder_press 0.30
  @stick_dead 0.20

  @keys [
    :lcancel_attempt_frac,
    :lcancel_press_offset_mean,
    :lcancel_press_offset_cv,
    :wavedash_angle_mean,
    :wavedash_angle_cv,
    :wavedash_jumpsquat_frame_mean,
    :di_active_frac,
    :di_perp_mean,
    :oos_latency_mean,
    :oos_jump_frac,
    :short_hop_frac
  ]

  @doc "Feature names in canonical order."
  @spec keys() :: [atom()]
  def keys, do: @keys

  @doc "Character-invariant subset (motor timing; travels with the person)."
  @spec invariant_keys() :: [atom()]
  def invariant_keys,
    do: [:lcancel_press_offset_mean, :lcancel_press_offset_cv, :wavedash_angle_mean, :wavedash_angle_cv, :wavedash_jumpsquat_frame_mean, :oos_latency_mean]

  @spec features([map() | nil], [map() | nil]) :: %{atom() => float()}
  def features(players, controllers) do
    p = :array.from_list(players)
    c = :array.from_list(controllers)
    n = length(players)
    act = fn i -> case :array.get(i, p) do %{action: a} -> a; _ -> nil end end
    ctrl = fn i -> :array.get(i, c) end

    transitions =
      for i <- 1..(n - 1)//1, act.(i) != act.(i - 1), do: {i, act.(i - 1), act.(i)}

    zeros = Map.new(@keys, &{&1, 0.0})

    zeros
    |> Map.merge(lcancel(transitions, ctrl))
    |> Map.merge(wavedash(transitions, ctrl))
    |> Map.merge(di(transitions, p, ctrl))
    |> Map.merge(oos(transitions, act, n))
    |> Map.merge(short_hop(transitions, p))
  end

  # -- L-cancel: press edge in the window before an aerial landing ---------

  defp lcancel(transitions, ctrl) do
    landings = for {i, _, a} <- transitions, a in @aerial_landings, do: i

    offsets =
      Enum.flat_map(landings, fn i ->
        window = for k <- 1..@lcancel_window, i - k >= 1, do: i - k
        # nearest press edge before touchdown, in frames-before-landing
        case Enum.find(window, fn j -> shoulder_edge?(ctrl.(j - 1), ctrl.(j)) end) do
          nil -> []
          j -> [i - j]
        end
      end)

    if landings == [] do
      %{}
    else
      %{
        lcancel_attempt_frac: length(offsets) / length(landings),
        lcancel_press_offset_mean: mean(offsets),
        lcancel_press_offset_cv: cv(offsets)
      }
    end
  end

  defp shoulder_edge?(prev, cur) when is_map(prev) and is_map(cur) do
    digital = fn m -> Map.get(m, :button_l) == true or Map.get(m, :button_r) == true or Map.get(m, :button_z) == true end
    analog = fn m -> shoulder(m) >= @shoulder_press end
    (digital.(cur) or analog.(cur)) and not (digital.(prev) or analog.(prev))
  end

  defp shoulder_edge?(_, _), do: false

  # -- Wavedash: KNEE_BEND -> JUMPING_F/B (1-2 f airborne) -> AIRDODGE -> LANDING_SPECIAL
  # (measured on real games 09-17: the airdodge always comes out of the first
  # JUMPING frame, never directly from KNEE_BEND).

  defp wavedash(transitions, ctrl) do
    tr = List.to_tuple(transitions)
    last = tuple_size(tr) - 1

    hits =
      Enum.flat_map(0..last//1, fn k ->
        case elem(tr, k) do
          {i, from, @airdodge} when from in @jumps and k >= 1 ->
            with {kb_start, @knee_bend} when i - kb_start <= @wavedash_window <- knee_bend_before(tr, k),
                 true <- landing_special_within?(tr, k, last, i, 6),
                 {mx, my} <- stick(ctrl.(i)) do
              dx = abs(mx - 0.5) * 2
              dy = (my - 0.5) * 2
              if :math.sqrt(dx * dx + dy * dy) < @stick_dead, do: [], else: [{angle_below(dx, dy), i - kb_start}]
            else
              _ -> []
            end

          _ ->
            []
        end
      end)

    case hits do
      [] ->
        %{}

      _ ->
        angles = Enum.map(hits, &elem(&1, 0))
        %{
          wavedash_angle_mean: mean(angles),
          wavedash_angle_cv: cv(angles),
          wavedash_jumpsquat_frame_mean: mean(Enum.map(hits, &elem(&1, 1)))
        }
    end
  end

  # degrees below horizontal (0 = flat, 90 = straight down); up-angles negative
  defp angle_below(dx, dy), do: :math.atan2(-dy, dx) * 180 / :math.pi()

  # {start_frame, :knee_bend} of the jumpsquat that transition k came out of:
  # k-1 must be the JUMPING entry and k-2 the KNEE_BEND entry.
  defp knee_bend_before(tr, k) when k >= 2 do
    case {elem(tr, k - 1), elem(tr, k - 2)} do
      {{_, @knee_bend, j}, {s, _, @knee_bend}} when j in @jumps -> {s, @knee_bend}
      _ -> nil
    end
  end

  defp knee_bend_before(_, _), do: nil

  defp landing_special_within?(tr, k, last, i, frames) do
    Enum.any?((k + 1)..min(k + 2, last)//1, fn m ->
      match?({j, _, @landing_special} when j - i <= frames, elem(tr, m))
    end)
  end

  # -- DI: stick vs knockback on entering a damage state --------------------

  defp di(transitions, players, ctrl) do
    entries = for {i, from, to} <- transitions, to in @damage and from not in @damage, do: i

    samples =
      Enum.flat_map(entries, fn i ->
        # Knockback direction from the position delta over the two frames after
        # entry (no self-influence in hitstun); replays before the velocity
        # spec carry speed_*_attack = 0.
        with {mx, my} <- stick(ctrl.(i)),
             %{x: x0, y: y0} <- :array.get(i, players),
             %{x: x2, y: y2} <- :array.get(min(i + 2, :array.size(players) - 1), players),
             {kx, ky} <- {x2 - x0, y2 - y0} do
          sx = (mx - 0.5) * 2
          sy = (my - 0.5) * 2
          smag = :math.sqrt(sx * sx + sy * sy)
          kmag = :math.sqrt(kx * kx + ky * ky)
          perp = if smag >= @stick_dead and kmag > 1.0e-6, do: abs(sx * ky - sy * kx) / (smag * kmag), else: nil
          [{smag >= @stick_dead, perp}]
        else
          _ -> []
        end
      end)

    case samples do
      [] ->
        %{}

      _ ->
        perps = for {_, p} <- samples, p != nil, do: p
        %{
          di_active_frac: Enum.count(samples, &elem(&1, 0)) / length(samples),
          di_perp_mean: if(perps == [], do: 0.0, else: mean(perps))
        }
    end
  end

  # -- Out of shield: frames from leaving shieldstun to the first non-shield action

  defp oos(transitions, act, n) do
    exits = for {i, @shield_stun, to} <- transitions, to in @shield, do: i

    latencies =
      Enum.flat_map(exits, fn i ->
        case Enum.find(i..min(i + 120, n - 1)//1, fn j -> act.(j) not in @shield end) do
          nil -> []
          j -> [{j - i, act.(j)}]
        end
      end)

    case latencies do
      [] ->
        %{}

      _ ->
        %{
          oos_latency_mean: mean(Enum.map(latencies, &elem(&1, 0))),
          oos_jump_frac: Enum.count(latencies, fn {_, a} -> a == @knee_bend end) / length(latencies)
        }
    end
  end

  # -- Short hop vs full hop from the first airborne frame's vertical speed --

  # Apex height above the takeoff point, relative to the game's highest jump
  # (replays before the velocity spec carry speed_y_self = 0).
  defp short_hop(transitions, players) do
    n = :array.size(players)

    heights =
      for {i, @knee_bend, to} <- transitions, to in @jumps, i >= 1 do
        y0 = :array.get(i - 1, players).y

        apex =
          Enum.reduce_while(i..min(i + 40, n - 1)//1, y0, fn j, best ->
            p = :array.get(j, players)
            if j > i and (p.on_ground == true or p.action not in @airborne), do: {:halt, best}, else: {:cont, max(best, p.y)}
          end)

        apex - y0
      end
      |> Enum.filter(&(&1 > 0.5))

    case heights do
      [] ->
        %{}

      _ ->
        hmax = Enum.max(heights)
        %{short_hop_frac: Enum.count(heights, &(&1 < 0.6 * hmax)) / length(heights)}
    end
  end

  # Controller rows come in two shapes: ExPhil.Bridge.ControllerState
  # (main_stick: %{x, y}, l_shoulder/r_shoulder) from training frames, and
  # the flat main_stick_x / l_trigger form used by tests and the Peppi
  # controller struct. Read both; the wavedash/DI detectors were silently
  # dead on real data (0 % firing) when they only knew the flat form.
  defp stick(%{main_stick: %{x: x, y: y}}), do: {x, y}
  defp stick(%{main_stick_x: x, main_stick_y: y}), do: {x, y}
  defp stick(_), do: nil

  defp shoulder(m) when is_map(m) do
    max(
      Map.get(m, :l_shoulder) || Map.get(m, :l_trigger) || 0.0,
      Map.get(m, :r_shoulder) || Map.get(m, :r_trigger) || 0.0
    )
  end

  defp shoulder(_), do: 0.0

  defp mean([]), do: 0.0
  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp cv(xs) do
    m = mean(xs)
    if m == 0.0 or xs == [], do: 0.0, else: :math.sqrt(Enum.sum(Enum.map(xs, &((&1 - m) * (&1 - m)))) / length(xs)) / abs(m)
  end
end
