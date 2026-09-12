defmodule ExPhil.Eval.ScenarioScore do
  @moduledoc """
  Pure per-type scoring for the scenario evaluation suite (task #18).

  Given the handoff snapshot and the observed response window, each scorer
  returns `%{score: 0..1, pass: boolean, details: map}`. No I/O, no game —
  everything here is unit-testable with synthetic sequences.

  ## Frame shape

  The handoff is `%{p1: player, p2: player}` and the window is a list of
  `%{frame:, p1:, p2:}` — players in the `ExPhil.Eval.ScenarioScan.player_summary/1`
  shape (`x`, `y`, `action`, `facing`, `on_ground`, `stock`). The window
  starts at the first frame AFTER handoff (index 0 = handoff+1).

  ## Pass criteria (per type)

    * `:opponent_behind` — turned to face (or up/down-smashed) the opponent's
      handoff side within 60f
    * `:tech_chase` — within 25 units of P2 at its first actionable frame
      AND an attack/grab attempt within 30f after
    * `:edgeguard` — moved >= 10 units toward the opponent's side within 60f
      (P2 stock lost in the window is a score bonus, not required)
    * `:getup` — acted from knockdown (tech or getup option) within 40f of
      first becoming actionable (down-wait)
    * `:idle_deadlock` — left Wait within 300f AND moved >= 5 units toward P2

  Scores are graded (faster/cleaner responses score closer to 1.0) so
  distributional runs can rank checkpoints even at equal pass rates.
  """

  @idle 14
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  # Normals 44..69 (jab..dair), grabs, char-specific specials
  @attacks MapSet.new(Enum.to_list(44..69) ++ [212, 214] ++ Enum.to_list(341..380))
  # Up/down smash hit behind without turning
  @rear_hitting [63, 64]
  @down_wait [184, 192]
  @getup_options MapSet.new([186, 187, 188, 189, 194, 195, 196, 197])
  @techs [199, 200, 201]

  @default_window 300

  @doc "Response window length (frames) for a scenario type."
  # Multishine re-entry (closed-loop correction validation, 2026-09-12):
  # 120 frames is ~13 cycles; a driver that has not re-entered by then failed.
  def window(:multishine_reentry), do: 120
  def window(_type), do: @default_window

  @doc """
  Score a response window. See module doc for shapes and criteria.

  Options: `:response_frames` overrides the per-type response deadline.
  """
  @spec score(atom(), map(), [map()], keyword()) ::
          %{score: float(), pass: boolean(), details: map()}
  def score(type, handoff, window, opts \\ [])

  # :multishine_reentry — did port 1 get BACK INTO the shine cycle after the
  # handoff (a chain break mined from a rollout)? A cycle = a second shine-
  # family entry within 12 frames of the previous one. Pass = a cycle within
  # `:response_frames` (default 60). Score grades speed and how many cycles
  # followed; details carry empty hops (jumpsquat -> airborne with no shine
  # within 8 frames) so a driver that jumps without shining is visible.
  def score(:multishine_reentry, _handoff, window, opts) do
    limit = Keyword.get(opts, :response_frames, 60)
    fams = Enum.map(window, fn o -> ExPhil.Eval.ShineChain.family(o.p1.action) end)
    n = length(fams)
    shine? = fn f -> f in [:ground_reflect, :air_reflect] end

    entries =
      fams
      |> Enum.with_index()
      |> Enum.filter(fn {f, i} -> shine?.(f) and not shine?.(if i > 0, do: Enum.at(fams, i - 1), else: :other) end)
      |> Enum.map(&elem(&1, 1))

    pairs = Enum.chunk_every(entries, 2, 1, :discard)
    reentry = Enum.find_value(pairs, fn [a, b] -> if b - a <= 12, do: b, else: nil end)
    cycles = Enum.count(pairs, fn [a, b] -> b - a <= 12 end)

    empty_hops =
      fams
      |> Enum.with_index()
      |> Enum.count(fn {f, i} ->
        next = Enum.at(fams, i + 1)
        after_ = if i + 1 < n, do: Enum.slice(fams, i + 1, 8), else: []
        f == :jumpsquat and next != :jumpsquat and next != nil and not Enum.any?(after_, shine?)
      end)

    # The TIGHT cycle is what matters (2026-09-12 first run: a reactive
    # teacher re-shines in 7-12f but settles into shine -> 18f reflector-
    # open float -> land -> wait, a 43f loop). ShineChain v3 chains only
    # through air gaps <= 8f, so max_chain separates the 9f loop from the
    # float loop; it is the pass criterion, re-entry speed is the tiebreak.
    actions = Enum.map(window, & &1.p1.action)
    detailed = ExPhil.Eval.ShineChain.chains_detailed(actions)
    max_chain = detailed |> Enum.map(& &1.length) |> Enum.max(fn -> 0 end)
    ended_by = detailed |> Enum.map(& &1.ended_by) |> Enum.frequencies()

    pass = reentry != nil and reentry <= limit and max_chain >= 5
    speed = if reentry, do: max(0.0, 1.0 - reentry / limit), else: 0.0
    score = min(1.0, 0.3 * speed + 0.7 * min(max_chain / 12, 1.0))

    %{
      score: Float.round(score * 1.0, 3),
      pass: pass,
      details: %{
        reentry_frame: reentry,
        max_chain: max_chain,
        ended_by: ended_by,
        shine_entries: length(entries),
        cycles: cycles,
        empty_hops: empty_hops
      }
    }
  end

  def score(:opponent_behind, handoff, window, opts) do
    limit = Keyword.get(opts, :response_frames, 60)
    side = sign(handoff.p2.x - handoff.p1.x)

    resp =
      Enum.find_index(window, fn f ->
        f.p1.facing == side or f.p1.action in @rear_hitting
      end)

    pass = resp != nil and resp <= limit

    score =
      cond do
        pass -> 1.0 - 0.5 * resp / limit
        resp != nil -> 0.25
        true -> 0.0
      end

    result(score, pass, %{frames_to_response: resp, opponent_side: side})
  end

  def score(:tech_chase, _handoff, window, opts) do
    prox_limit = Keyword.get(opts, :proximity, 25.0)
    attack_limit = Keyword.get(opts, :response_frames, 30)

    act_i =
      Enum.find_index(window, fn f ->
        not MapSet.member?(@lifecycle, f.p2.action) and
          not MapSet.member?(@hitstun, f.p2.action)
      end)

    {dist, attack_j} =
      case act_i do
        nil ->
          # P2 never became actionable in the window (e.g. stock lost while
          # down): judge by closest approach + any attack attempt.
          d = window |> Enum.map(&dist/1) |> Enum.min(fn -> :infinity end)
          j = Enum.find_index(window, &attack?(&1.p1.action))
          {d, if(j, do: 0, else: nil)}

        i ->
          d = dist(Enum.at(window, i))

          j =
            window
            |> Enum.drop(i)
            |> Enum.find_index(&attack?(&1.p1.action))

          {d, j}
      end

    prox = dist != :infinity and dist <= prox_limit
    attacked = attack_j != nil and attack_j <= attack_limit
    pass = prox and attacked
    score = if(prox, do: 0.5, else: 0.0) + if(attacked, do: 0.5, else: 0.0)

    result(score, pass, %{
      p2_actionable_at: act_i,
      distance_at_actionable: round1(dist),
      frames_to_attack: attack_j
    })
  end

  def score(:edgeguard, handoff, window, opts) do
    limit = Keyword.get(opts, :response_frames, 60)
    needed = Keyword.get(opts, :move_units, 10.0)

    side = if handoff.p2.x == 0, do: sign(handoff.p1.x) * -1, else: sign(handoff.p2.x)
    x0 = handoff.p1.x

    disp =
      window
      |> Enum.take(limit)
      |> Enum.map(fn f -> (f.p1.x - x0) * side end)
      |> Enum.max(fn -> 0.0 end)

    stock_lost =
      case List.last(window) do
        nil -> false
        last -> last.p2.stock < handoff.p2.stock
      end

    pass = disp >= needed
    move_score = 0.7 * min(max(disp, 0.0) / needed, 1.0)
    score = min(1.0, move_score + if(stock_lost, do: 0.3, else: 0.0))

    result(score, pass, %{
      max_toward_disp: round1(disp),
      p2_stock_lost: stock_lost,
      ledge_side: side
    })
  end

  def score(:getup, _handoff, window, opts) do
    limit = Keyword.get(opts, :response_frames, 40)
    p1_actions = Enum.map(window, & &1.p1.action)

    tech_i = Enum.find_index(p1_actions, &(&1 in @techs))
    wait_i = Enum.find_index(p1_actions, &(&1 in @down_wait))
    getup_i = Enum.find_index(p1_actions, &MapSet.member?(@getup_options, &1))

    down_wait_runs =
      p1_actions
      |> Enum.chunk_by(&(&1 in @down_wait))
      |> Enum.count(&(hd(&1) in @down_wait))

    {resp, kind} =
      cond do
        tech_i != nil and (wait_i == nil or tech_i < wait_i) ->
          # Teched before ever reaching down-wait: instant action.
          {0, :tech}

        getup_i != nil and (wait_i == nil or getup_i < wait_i) ->
          # Acted on the first actionable frame — bound transitions straight
          # into a getup option with zero down-wait frames. Best case.
          {0, :getup}

        wait_i != nil ->
          j =
            p1_actions
            |> Enum.drop(wait_i)
            |> Enum.find_index(fn a -> MapSet.member?(@getup_options, a) or a in @techs end)

          {j, :getup}

        true ->
          {nil, :never_actionable}
      end

    pass = resp != nil and resp <= limit

    score =
      cond do
        pass -> max(0.5, 1.0 - resp / (2 * limit))
        resp != nil -> 0.25
        true -> 0.0
      end

    result(score, pass, %{
      frames_to_act: resp,
      response_kind: kind,
      down_wait_runs: down_wait_runs
    })
  end

  def score(:idle_deadlock, handoff, window, opts) do
    limit = Keyword.get(opts, :response_frames, 300)
    needed = Keyword.get(opts, :move_units, 5.0)

    side = sign(handoff.p2.x - handoff.p1.x)
    x0 = handoff.p1.x

    left_i = Enum.find_index(window, fn f -> f.p1.action != @idle end)

    disp =
      window
      |> Enum.map(fn f -> (f.p1.x - x0) * side end)
      |> Enum.max(fn -> 0.0 end)

    left_ok = left_i != nil and left_i <= limit
    moved = disp >= needed
    pass = left_ok and moved
    score = if(left_ok, do: 0.5, else: 0.0) + if(moved, do: 0.5, else: 0.0)

    result(score, pass, %{
      frames_to_leave_idle: left_i,
      max_toward_disp: round1(disp),
      opponent_side: side
    })
  end

  # ==========================================================================
  # Helpers
  # ==========================================================================

  defp result(score, pass, details) do
    %{score: Float.round(score * 1.0, 3), pass: pass, details: details}
  end

  defp attack?(action), do: MapSet.member?(@attacks, action)

  defp dist(%{p1: p1, p2: p2}) do
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    :math.sqrt(dx * dx + dy * dy)
  end

  defp round1(:infinity), do: :infinity
  defp round1(x), do: Float.round(x * 1.0, 1)

  defp sign(x) when x < 0, do: -1
  defp sign(_), do: 1
end
