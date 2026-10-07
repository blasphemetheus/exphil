defmodule ExPhil.Eval.DecisionMap do
  @moduledoc """
  What does the player decide offstage, by state — the recovery-decision
  onset hazard by height (2026-10-07, INPUT_COHERENCE doc "10-07 15:30").

  The double jump turned out to be a *height decision* the bot never
  conditions: the expert's per-frame jump hazard offstage rises 15 % → 44 %
  → 60 % as it falls from the ledge band to y −40..−60, every arm's is flat,
  so the bot drifts through the band where the jump is spent and dies with
  it in hand. `ExPhil.Eval.SilenceMap` asks the same question of one
  decision (letting go); this module asks it of every recovery decision.

  For every frame `t` where the player is **free-falling offstage** —
  airborne, past the edge or below the stage, actionable (no hitstun, not
  helpless, not already in a special, not on the ledge) — with a successor
  in the same game, it classifies what starts at `t + 1`:

    * **jump** — the double jump is spent (`jumps_left` drops) or X/Y is
      pressed with a jump in hand;
    * **special_up / special_side / special_down / special_neutral** — a
      special move begins (`action >= 341`, character-agnostic), by the
      main-stick zone it was started with (Fox: up = Firefox, side =
      Illusion, down = shine, neutral = laser);
    * **airdodge**, **aerial** (an aerial attack starts);
    * otherwise nothing (hold / drift).

  Buckets: `height` band × jumps left (`y>0`, `0..-20`, `-20..-40`,
  `-40..-60`, `<-60`; `j0` / `j1+`). Same frame shape as PlayStats /
  SilenceMap, so the fidelity rollout and the expert-replay pass feed it.
  `summarize/1` turns counts into per-frame hazards, `compare/3` puts model
  and expert side by side with a binomial z, and `slope/2` reports the
  height conditioning of one decision as the ratio of its hazard in the
  `-40..-60` band to the `0..-20` band (expert jump ≈ 4; a flat head ≈ 1).
  """

  @decisions ~w(jump special_up special_side special_down special_neutral airdodge aerial)
  @cliff 252..263
  @helpless 35
  @airdodge 236
  @aerials 65..69
  @first_special 341
  @first_actionable 14

  @type game :: [%{own: map(), opp: map(), controller: map()}]

  @doc "Raw counts for one game. `opts`: `:edge` (stage half-width)."
  @spec from_game(game(), keyword()) :: map()
  def from_game(frames, opts \\ [])
  def from_game([], _opts), do: empty()

  def from_game(frames, opts) do
    edge = Keyword.get(opts, :edge, 85.5656967163)

    frames
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce(empty(), fn [f, g], acc ->
      if eligible?(f.own, edge) and countable?(g.own) do
        bump(acc, bucket(f.own), decision(f.own, g.own, g.controller))
      else
        acc
      end
    end)
  end

  def decisions, do: @decisions

  def empty, do: %{}

  @spec merge(map(), map()) :: map()
  def merge(a, b), do: Map.merge(a, b, fn _k, x, y -> Map.merge(x, y, fn _c, p, q -> p + q end) end)

  @doc "Per-bucket onset hazards (per eligible frame) for every decision."
  @spec summarize(map()) :: map()
  def summarize(counts) do
    Map.new(counts, fn {name, c} ->
      n = c["frames"] || 0
      {name, Map.new(@decisions, fn d -> {String.to_atom(d), rate(c[d], n)} end) |> Map.put(:frames, n)}
    end)
  end

  @doc """
  Model vs expert per bucket for one decision (default `:jump`): hazards,
  ratio, binomial SE of the model hazard and `z`; buckets with fewer than
  `min_n` model frames are dropped; sorted by |z|, worst first.
  """
  @spec compare(map(), map(), keyword()) :: [map()]
  def compare(model, expert, opts \\ []) do
    decision = Keyword.get(opts, :decision, :jump)
    min_n = Keyword.get(opts, :min_n, 100)

    model
    |> Enum.flat_map(fn {k, m} ->
      e = expert[k]
      n = m.frames
      p = m[decision]
      q = e && e[decision]

      if e == nil or n < min_n or p == nil or q == nil do
        []
      else
        se = :math.sqrt(max(p * (1 - p), 1.0e-9) / n)
        ratio = if q > 0, do: Float.round(p / q, 2), else: if(p > 0, do: :infinity, else: 1.0)
        [%{bucket: k, n: n, expert_n: e.frames, model: p, expert: q, ratio: ratio, z: Float.round((p - q) / se, 1)}]
      end
    end)
    |> Enum.sort_by(&abs(&1.z), :desc)
  end

  @doc """
  Height conditioning of one decision with a jump in hand: hazard in the
  `-40..-60` band over the `0..-20` band. `nil` when either band is empty.
  """
  @spec slope(map(), atom()) :: float() | nil
  def slope(summary, decision \\ :jump) do
    lo = get_in(summary, ["-40..-60:j1+", decision])
    hi = get_in(summary, ["0..-20:j1+", decision])
    if is_number(lo) and is_number(hi) and hi > 0, do: Float.round(lo / hi, 2), else: nil
  end

  @doc "Height band × jumps-left bucket for one player state."
  @spec bucket(map()) :: String.t()
  def bucket(p) do
    y = p.y || 0.0
    jumps = if (p.jumps_left || 0) > 0, do: "j1+", else: "j0"

    band =
      cond do
        y > 0 -> "y>0"
        y > -20 -> "0..-20"
        y > -40 -> "-20..-40"
        y > -60 -> "-40..-60"
        true -> "<-60"
      end

    "#{band}:#{jumps}"
  end

  @doc "Free-falling offstage and able to decide."
  @spec eligible?(map() | nil, number()) :: boolean()
  def eligible?(nil, _edge), do: false

  def eligible?(p, edge) do
    act = p.action || 0

    countable?(p) and p.on_ground != true and
      (abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0) and
      (p.hitstun_frames_left || 0) == 0 and act != @helpless and act < @first_special and
      act not in @cliff
  end

  @doc "What begins at `t + 1`, given the states at `t` and `t + 1` and the controller at `t + 1`."
  @spec decision(map(), map(), map()) :: String.t() | nil
  def decision(p, q, controller) do
    a0 = p.action || 0
    a1 = q.action || 0
    jumps0 = p.jumps_left || 0
    jumps1 = q.jumps_left || 0
    pressed_jump? = truthy(Map.get(controller, :button_x)) or truthy(Map.get(controller, :button_y))

    cond do
      a1 >= @first_special and a0 < @first_special -> "special_#{zone(controller)}"
      a1 == @airdodge and a0 != @airdodge -> "airdodge"
      a1 in @aerials and a0 not in @aerials -> "aerial"
      jumps0 > 0 and (jumps1 < jumps0 or pressed_jump?) -> "jump"
      true -> nil
    end
  end

  # -- internals -------------------------------------------------------------

  defp bump(acc, bucket, decision) do
    c = Map.get(acc, bucket, Map.new(["frames" | @decisions], &{&1, 0}))
    c = Map.update!(c, "frames", &(&1 + 1))
    c = if decision, do: Map.update!(c, decision, &(&1 + 1)), else: c
    Map.put(acc, bucket, c)
  end

  defp countable?(nil), do: false
  defp countable?(p), do: (p.action || 0) >= @first_actionable

  defp truthy(v), do: v == true or (is_number(v) and v > 0)

  defp zone(c) do
    ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
    dx = (ms[:x] || 0.5) - 0.5
    dy = (ms[:y] || 0.5) - 0.5

    cond do
      dy >= 0.33 -> "up"
      dy <= -0.33 -> "down"
      abs(dx) >= 0.33 -> "side"
      true -> "neutral"
    end
  end

  defp rate(_num, 0), do: nil
  defp rate(nil, _), do: nil
  defp rate(num, den), do: Float.round(num / den, 4)
end
