defmodule ExPhil.Sim.Search do
  @moduledoc """
  Search-as-teacher v0 (SIM_INTEGRATION.md step 7): random shooting over
  input sequences from a drill start.

  From a pool entry (`ExPhil.Sim.Drill`), restore the savestate, roll `n`
  candidate input sequences of `horizon` frames (each a random program of
  macro-actions: hold one controller for a random duration, then another),
  score each rollout with the drill scorers, and keep the best. The defender
  is `:idle` (pure sim, thousands of fps) or an Agent pid (the frozen prior;
  each candidate then costs policy inference per frame).

  Ranking: converted (true-two-hit or string hit) > first-fair contact >
  damage dealt > connected aerials. The winner's frames + controllers are the
  labels for BC/DAgger (step 8); the summary is the oracle's conversion rate
  to compare with the prior's baseline on the same pool.

  Deterministic given `:seed`: the same pool + seed reproduces the same
  candidates.
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.{ControllerState, SimPort}
  alias ExPhil.Eval.{AerialChain, FairConversion, ScenarioScan}
  alias ExPhil.Sim.Drill

  @neutral Drill.neutral()

  # Macro-action vocabulary for Fox: sticks in the sim's [0, 1] convention.
  @macros [
    {:neutral, @neutral},
    {:dash_r, %{@neutral | main_stick: %{x: 1.0, y: 0.5}}},
    {:dash_l, %{@neutral | main_stick: %{x: 0.0, y: 0.5}}},
    {:walk_r, %{@neutral | main_stick: %{x: 0.7, y: 0.5}}},
    {:walk_l, %{@neutral | main_stick: %{x: 0.3, y: 0.5}}},
    {:jump, %{@neutral | button_y: true}},
    {:jump_r, %{@neutral | button_y: true, main_stick: %{x: 0.8, y: 0.5}}},
    {:jump_l, %{@neutral | button_y: true, main_stick: %{x: 0.2, y: 0.5}}},
    {:fair_r, %{@neutral | c_stick: %{x: 1.0, y: 0.5}}},
    {:fair_l, %{@neutral | c_stick: %{x: 0.0, y: 0.5}}},
    {:nair, %{@neutral | button_a: true}},
    {:uair, %{@neutral | c_stick: %{x: 0.5, y: 1.0}}},
    {:dair, %{@neutral | c_stick: %{x: 0.5, y: 0.0}}},
    {:shine, %{@neutral | button_b: true, main_stick: %{x: 0.5, y: 0.0}}},
    {:lcancel, %{@neutral | l_shoulder: 1.0, button_l: true}},
    {:grab, %{@neutral | button_z: true}},
    {:utilt, %{@neutral | button_a: true, main_stick: %{x: 0.5, y: 0.7}}},
    {:dtilt, %{@neutral | button_a: true, main_stick: %{x: 0.5, y: 0.3}}}
  ]

  def macros, do: @macros

  @doc """
  Sample one candidate program: a list of `horizon` controllers built from
  macro-actions held for 1..`max_hold` frames each. Button macros are held
  for at most 2 frames (a press), movement for longer.
  """
  def random_program(horizon, max_hold \\ 12) do
    Stream.unfold(0, fn
      t when t >= horizon -> nil
      t ->
        {name, ctrl} = Enum.random(@macros)
        hold = if name in [:jump, :jump_r, :jump_l, :fair_r, :fair_l, :nair, :uair, :dair, :shine, :lcancel, :grab, :utilt, :dtilt], do: :rand.uniform(2), else: :rand.uniform(max_hold)
        hold = min(hold, horizon - t)
        {{name, ctrl, hold}, t + hold}
    end)
    |> Enum.flat_map(fn {name, ctrl, hold} -> List.duplicate({name, ctrl}, hold) end)
  end

  @doc """
  Evaluate one program from a pool entry. Returns the rollout result map
  (as `Drill.rollout/5`) plus `:program` and `:score`.
  """
  def evaluate(sim, entry, program, defender, opts \\ []) do
    horizon = length(program)
    {:ok, [gs0]} = SimPort.restore(sim, 0, entry.blob)

    if is_pid(defender) do
      Agent.reset_buffer(defender)
      Enum.each(entry.history, fn {gs, _c1, c2} -> :ok = Agent.observe(defender, %{gs | own_port: 2}, c2, player_port: 2) end)
    end

    {states, ctrls, _} =
      Enum.reduce_while(program, {[gs0], [], gs0}, fn {_name, c1}, {acc, cs, gs} ->
        c2 = defender_controller(defender, gs)

        case SimPort.step(sim, [[c1, c2]]) do
          {:ok, [next], [term]} ->
            if term["done"] == 1, do: {:halt, {[next | acc], [c1 | cs], next}}, else: {:cont, {[next | acc], [c1 | cs], next}}

          {:error, reason} ->
            raise "sim step failed: #{inspect(reason)}"
        end
      end)

    states = Enum.reverse(states)
    ctrls = Enum.reverse(ctrls)
    frames = Enum.map(states, fn s -> %{frame: s.frame, p1: ScenarioScan.player_summary(s.players[1]), p2: ScenarioScan.player_summary(s.players[2])} end)
    trials = FairConversion.trials(frames, window: horizon)
    chain = AerialChain.summary(frames)
    damage = (List.last(states).players[2].percent - gs0.players[2].percent) * 1.0
    converted = Enum.any?(trials, &(&1.outcome in [:true_two_hit, :string_hit]))
    contact = trials != []
    p1_end = List.last(states).players[1]
    # penalize dying / leaving the stage: a "combo" that ends in an SD is not a label
    alive = p1_end.stock == gs0.players[1].stock and abs(p1_end.x) < 90.0

    score =
      (if converted, do: 1000.0, else: 0.0) + (if contact, do: 100.0, else: 0.0) + damage + 10.0 * chain.mean_connected_aerials - (if alive, do: 0.0, else: 500.0)

    %{program: Enum.map(program, &elem(&1, 0)), controllers: ctrls, states: states, frames: frames, converted?: converted, contact?: contact, damage: damage, chain: chain, alive?: alive, score: score, fair: FairConversion.summary(frames, window: horizon)}
  end

  defp defender_controller(:idle, _gs), do: @neutral

  defp defender_controller(agent, gs) do
    case Agent.get_controller(agent, %{gs | own_port: 2}, player_port: 2) do
      {:ok, c} -> c
      _ -> @neutral
    end
  end

  @doc """
  Random shooting for one entry: `n` programs, best by score. Options:
  `:horizon` (90), `:seed`, `:max_hold` (12). Returns `%{best, tried, converted_any?, contact_any?}`.
  """
  def shoot(sim, entry, defender, opts \\ []) do
    n = Keyword.get(opts, :n, 64)
    horizon = Keyword.get(opts, :horizon, 90)
    max_hold = Keyword.get(opts, :max_hold, 12)
    if seed = Keyword.get(opts, :seed), do: :rand.seed(:exsss, {seed, entry.id, 3})

    results = for _ <- 1..n, do: evaluate(sim, entry, random_program(horizon, max_hold), defender, opts)
    best = Enum.max_by(results, & &1.score)

    %{
      best: best,
      tried: n,
      converted_any?: Enum.any?(results, & &1.converted?),
      contact_any?: Enum.any?(results, & &1.contact?),
      n_converted: Enum.count(results, & &1.converted?),
      n_contact: Enum.count(results, & &1.contact?)
    }
  end

  @doc "Controller -> compact JSON for the label file."
  def controller_json(%ControllerState{} = c) do
    %{mx: c.main_stick.x, my: c.main_stick.y, cx: c.c_stick.x, cy: c.c_stick.y, a: c.button_a, b: c.button_b, x: c.button_x, y: c.button_y, z: c.button_z, l: c.button_l, r: c.button_r, sh: max(c.l_shoulder || 0.0, c.r_shoulder || 0.0)}
  end
end
