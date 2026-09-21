defmodule ExPhil.Sim.Drill do
  @moduledoc """
  Curriculum environment v0 over melee-sim-light (SIM_INTEGRATION.md step 6).

  A **drill** is `(start distribution, scorer, horizon)`:

    * **start pool** — savestates reached by playing a random walk from the
      spawn (random `start_percent`s from `configure_match`, then W frames of
      random dash / jump / neutral inputs for both players after the
      countdown), filtered to on-stage, alive, not-in-hitstun states. Each
      entry keeps the sim blob plus the last `warm` mapped frames so an agent
      can be warmed with real history (`Agent.observe/4`) before the drill.
    * **rollout** — restore the blob, warm the agents, run `horizon` frames
      with the attacker on port 1 and the defender on port 2 (a policy, or
      `:idle`), collecting slim frames (`ScenarioScan.player_summary/1`).
    * **scorers** — `ExPhil.Eval.FairConversion` (first-fair trials and their
      outcome) and `ExPhil.Eval.AerialChain` (connected aerials per opening),
      both unchanged from the replay path, so sim numbers and Dolphin numbers
      are the same instrument.

  The frozen prior's conversion rate on a fixed pool is the baseline every
  teacher (search, DAgger, PPO) must beat; the same pool is the search
  oracle's input (step 7).
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Sim.Env
  alias ExPhil.Eval.{AerialChain, FairConversion, Opening, ScenarioScan}

  @neutral %ControllerState{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    l_shoulder: 0.0,
    r_shoulder: 0.0,
    button_a: false,
    button_b: false,
    button_x: false,
    button_y: false,
    button_z: false,
    button_l: false,
    button_r: false,
    button_d_up: false
  }

  @players [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]

  def neutral, do: @neutral

  @doc """
  Build a start pool of `n` entries. Options: `:seed` (default 1), `:warm`
  (frames of history kept per entry, default 30), `:max_distance` (start
  separation cap, default 60.0), `:walk` (random-walk
  frames after frame 0, range `{min, max}`, default `{45, 150}`),
  `:percent` (max start percent, default 60), `:stage` (default FD).

  Returns `[%{id, blob, frame, summary, history}]`; `history` is the list of
  the last `warm` `GameState`s with the P1/P2 controllers used
  (`[{gs, c1, c2}]`), oldest first.
  """
  def build_pool(sim, n, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    warm = Keyword.get(opts, :warm, 30)
    {wmin, wmax} = Keyword.get(opts, :walk, {45, 150})
    max_pct = Keyword.get(opts, :percent, 60)
    max_dist = Keyword.get(opts, :max_distance, 60.0)
    stage = Keyword.get(opts, :stage, "final_destination")
    :rand.seed(:exsss, {seed, 7, 11})

    Stream.iterate(0, &(&1 + 1))
    |> Stream.map(fn i ->
      p1 = %{character: "fox", costume: 1, start_percent: :rand.uniform(max_pct + 1) - 1}
      p2 = %{character: "fox", costume: 0, start_percent: :rand.uniform(max_pct + 1) - 1}
      {:ok, _} = Env.reinit(sim, %{stage: stage, players: [p1, p2], length: 256, seed: seed * 100_000 + i})
      walk = wmin + :rand.uniform(wmax - wmin + 1) - 1
      run_walk(sim, walk, warm, i, max_dist)
    end)
    |> Stream.reject(&is_nil/1)
    |> Enum.take(n)
  end

  @doc """
  Build a start pool from the prior's OWN play: run `attacker` vs `defender`
  games in the sim and snapshot every `:every` frames (default 30) after
  frame 0, keeping states that pass the usability filter and the distance
  cap. On-distribution starts with real policy history — the random walk
  (`build_pool/3`) accepted ~4 % of walks under a 50-unit cap and warmed the
  policy with scripted junk (measured 2026-09-21). Options: `:seed`,
  `:warm` (30), `:max_distance` (50.0), `:every` (30), `:game_frames`
  (post-zero frames per game, 1800), `:stage`.
  """
  def build_pool_from_play(sim, attacker, defender, n, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    warm = Keyword.get(opts, :warm, 30)
    max_dist = Keyword.get(opts, :max_distance, 50.0)
    every = Keyword.get(opts, :every, 30)
    game_frames = Keyword.get(opts, :game_frames, 1800)
    stage = Keyword.get(opts, :stage, "final_destination")

    Stream.iterate(0, &(&1 + 1))
    |> Stream.flat_map(fn g ->
      {:ok, _} = Env.reinit(sim, %{stage: stage, players: @players, length: 256, seed: seed * 1000 + g})
      Agent.reset_buffer(attacker)
      if is_pid(defender), do: Agent.reset_buffer(defender)
      {:ok, [gs0]} = Env.frames(sim)

      {entries, _, _} =
        Enum.reduce_while(Stream.iterate(1, &(&1 + 1)), {[], gs0, []}, fn t, {acc, gs, hist} ->
          c1 = controller(attacker, gs, 1)
          c2 = controller(defender, gs, 2)
          hist = Enum.take([{gs, c1, c2} | hist], warm)

          case Env.step(sim, [[c1, c2]]) do
            {:ok, [next], [term]} ->
              acc =
                if next.frame > 0 and rem(next.frame, every) == 0 and usable?(next.players[1]) and usable?(next.players[2]) and next.distance <= max_dist do
                  {:ok, blob, sid} = Env.save(sim, 0, keep: true)
                  [%{id: g * 10_000 + next.frame, game: g, blob: blob, state_id: sid, frame: next.frame, summary: %{p1: ScenarioScan.player_summary(next.players[1]), p2: ScenarioScan.player_summary(next.players[2])}, history: Enum.reverse(hist)} | acc]
                else
                  acc
                end

              if term["done"] == 1 or next.frame >= game_frames - 123 or t > game_frames + 200, do: {:halt, {acc, next, hist}}, else: {:cont, {acc, next, hist}}

            {:error, reason} ->
              raise "sim step failed: #{inspect(reason)}"
          end
        end)

      Enum.reverse(entries)
    end)
    |> Enum.take(n)
  end

  @doc """
  Batched `build_pool_from_play/5`: `envs` self-play games advance together
  (one policy call per frame per side); every env snapshots on its own
  schedule. The SimPort must already be initialized with `batch_size: envs`.
  Games are re-initialized in rounds until `n` entries exist.
  """
  def build_pool_from_play_batch(sim, attacker, defender, n, envs, opts \\ []) do
    seed = Keyword.get(opts, :seed, 1)
    warm = Keyword.get(opts, :warm, 30)
    max_dist = Keyword.get(opts, :max_distance, 50.0)
    every = Keyword.get(opts, :every, 30)
    game_frames = Keyword.get(opts, :game_frames, 1800)
    stage = Keyword.get(opts, :stage, "final_destination")

    Stream.iterate(0, &(&1 + 1))
    |> Stream.flat_map(fn round ->
      {:ok, _} = Env.reinit(sim, %{stage: stage, players: @players, batch_size: envs, length: 256, seed: seed * 1000 + round})
      ensure_batch(attacker, envs)
      if is_pid(defender), do: ensure_batch(defender, envs)
      {:ok, states0} = Env.frames(sim)
      hists0 = List.duplicate([], envs)

      {entries, _, _} =
        Enum.reduce_while(Stream.iterate(1, &(&1 + 1)), {[], states0, hists0}, fn t, {acc, states, hists} ->
          {:ok, c1s} = Agent.batch_get_controllers(attacker, states, player_port: 1)

          c2s =
            if is_pid(defender) do
              {:ok, cs} = Agent.batch_get_controllers(defender, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
              cs
            else
              List.duplicate(@neutral, envs)
            end

          hists = Enum.zip([states, c1s, c2s, hists]) |> Enum.map(fn {gs, c1, c2, h} -> Enum.take([{gs, c1, c2} | h], warm) end)

          case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a, b] end)) do
            {:ok, nexts, terms} ->
              acc =
                Enum.zip([nexts, hists, 0..(envs - 1)])
                |> Enum.reduce(acc, fn {next, h, i}, acc ->
                  if next.frame > 0 and rem(next.frame, every) == 0 and usable?(next.players[1]) and usable?(next.players[2]) and next.distance <= max_dist do
                    {:ok, blob, sid} = Env.save(sim, i, keep: true)
                    [%{id: (round * envs + i) * 10_000 + next.frame, game: round * envs + i, blob: blob, state_id: sid, frame: next.frame, summary: %{p1: ScenarioScan.player_summary(next.players[1]), p2: ScenarioScan.player_summary(next.players[2])}, history: Enum.reverse(h)} | acc]
                  else
                    acc
                  end
                end)

              all_done = Enum.all?(terms, &(&1["done"] == 1))
              if all_done or hd(nexts).frame >= game_frames - 123 or t > game_frames + 200, do: {:halt, {acc, nexts, hists}}, else: {:cont, {acc, nexts, hists}}

            {:error, reason} ->
              raise "sim step failed: #{inspect(reason)}"
          end
        end)

      Enum.reverse(entries)
    end)
    |> Enum.take(n)
  end

  # Countdown with neutral inputs, then `walk` frames of random inputs; keep
  # the last `warm` frames of history; accept only usable states.
  defp run_walk(sim, walk, warm, i, max_dist) do
    {:ok, [gs0]} = Env.frames(sim)
    pre = max(-1 - gs0.frame, 0)

    hist =
      Enum.reduce(1..(pre + walk)//1, {[], gs0, {@neutral, @neutral}}, fn t, {h, gs, {c1, c2}} ->
        {c1, c2} = if t <= pre, do: {@neutral, @neutral}, else: {maybe_new(c1), maybe_new(c2)}
        {:ok, [next], _} = Env.step(sim, [[c1, c2]])
        {Enum.take([{gs, c1, c2} | h], warm), next, {c1, c2}}
      end)

    {history, final, _} = hist
    p1 = final.players[1]
    p2 = final.players[2]

    if usable?(p1) and usable?(p2) and final.distance <= max_dist do
      {:ok, blob} = Env.save(sim, 0)
      %{id: i, blob: blob, frame: final.frame, summary: %{p1: ScenarioScan.player_summary(p1), p2: ScenarioScan.player_summary(p2)}, history: Enum.reverse(history)}
    else
      nil
    end
  end

  defp usable?(p), do: p.stock == 4 and abs(p.x) < 70.0 and p.y > -5.0 and p.hitstun_frames_left == 0 and p.action not in 0..13

  # Random-walk vocabulary: hold the current input with p=0.9, else pick a new one.
  defp maybe_new(c) do
    if :rand.uniform() < 0.9 do
      c
    else
      case :rand.uniform(6) do
        1 -> @neutral
        2 -> %{@neutral | main_stick: %{x: 1.0, y: 0.5}}
        3 -> %{@neutral | main_stick: %{x: 0.0, y: 0.5}}
        4 -> %{@neutral | button_y: true}
        5 -> %{@neutral | main_stick: %{x: 0.7, y: 0.5}}
        6 -> %{@neutral | main_stick: %{x: 0.3, y: 0.5}}
      end
    end
  end

  @doc """
  Run one drill rollout from a pool entry. `attacker` is an Agent pid;
  `defender` is an Agent pid or `:idle`. Options: `:horizon` (default 120).
  Returns `%{frames: slim_frames, states: [GameState], fair: FairConversion.summary, chain: AerialChain.summary, converted?: boolean, contact?: boolean}`.
  """
  def rollout(sim, entry, attacker, defender, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, 120)
    {:ok, [_]} = Env.restore(sim, 0, restore_ref(entry))
    Agent.reset_buffer(attacker)
    if is_pid(defender), do: Agent.reset_buffer(defender)

    Enum.each(entry.history, fn {gs, c1, c2} ->
      :ok = Agent.observe(attacker, gs, c1, player_port: 1)
      if is_pid(defender), do: :ok = Agent.observe(defender, %{gs | own_port: 2}, c2, player_port: 2)
    end)

    {:ok, [gs0]} = Env.frames(sim)

    {states, _} =
      Enum.reduce_while(1..horizon, {[gs0], gs0}, fn _, {acc, gs} ->
        c1 = controller(attacker, gs, 1)
        c2 = controller(defender, gs, 2)

        case Env.step(sim, [[c1, c2]]) do
          {:ok, [next], [term]} ->
            if term["done"] == 1, do: {:halt, {[next | acc], next}}, else: {:cont, {[next | acc], next}}

          {:error, reason} ->
            raise "sim step failed: #{inspect(reason)}"
        end
      end)

    states = Enum.reverse(states)
    frames = Enum.map(states, fn s -> %{frame: s.frame, p1: ScenarioScan.player_summary(s.players[1]), p2: ScenarioScan.player_summary(s.players[2])} end)
    fair = FairConversion.summary(frames, window: horizon)
    chain = AerialChain.summary(frames)
    # Primary scorer: ANY opening (grab / smash / tilt / aerial / special),
    # converted = a second hit before the defender is actionable, or grab ->
    # throw (Opening; Bradley 2026-09-21). FairConversion stays as the
    # Mewtwo-specific secondary.
    openings = Opening.openings(frames, window: horizon)

    %{
      frames: frames,
      states: states,
      fair: fair,
      chain: chain,
      opening: Opening.summary(frames, window: horizon),
      openings: openings,
      contact?: openings != [],
      converted?: Enum.any?(openings, & &1.converted?),
      damage: (List.last(states).players[2].percent - gs0.players[2].percent) * 1.0
    }
  end

  @doc """
  Batched rollout: `entries` (one per sim env; `length(entries)` must equal
  the SimPort's batch size) are restored into envs 0..n-1, both agents are
  warmed with each entry's history through `Agent.batch_observe/3`, then
  `horizon` frames run with ONE policy call per frame per side
  (`Agent.batch_get_controllers/3`). Returns one result map per entry, same
  shape as `rollout/5`. ~75x the single-env decision rate (profile
  2026-09-21: 113 -> 8,600 decisions/s at batch 128).
  """
  def rollout_batch(sim, entries, attacker, defender, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, 120)
    n = length(entries)

    entries = Enum.map(entries, fn e -> {_, e} = ensure_cached(sim, e); e end)
    Enum.with_index(entries) |> Enum.each(fn {e, i} -> {:ok, _} = Env.restore(sim, i, restore_ref(e), frames: false) end)
    {:ok, _, _} = Env.observe(sim)

    ensure_batch(attacker, n)
    if is_pid(defender), do: ensure_batch(defender, n)

    warm = entries |> Enum.map(&length(&1.history)) |> Enum.min(fn -> 0 end)

    for t <- 0..(warm - 1)//1 do
      states = Enum.map(entries, fn e -> elem(Enum.at(e.history, t), 0) end)
      :ok = Agent.batch_observe(attacker, states, player_port: 1)
      if is_pid(defender), do: :ok = Agent.batch_observe(defender, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
    end

    {:ok, gs0s} = Env.frames(sim)

    {history, _} =
      Enum.reduce(1..horizon, {[gs0s], gs0s}, fn _, {acc, states} ->
        {:ok, c1s} = Agent.batch_get_controllers(attacker, states, player_port: 1)

        c2s =
          if is_pid(defender) do
            {:ok, cs} = Agent.batch_get_controllers(defender, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
            cs
          else
            List.duplicate(@neutral, n)
          end

        case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a, b] end)) do
          {:ok, nexts, _terms} -> {[nexts | acc], nexts}
          {:error, reason} -> raise "sim step failed: #{inspect(reason)}"
        end
      end)

    per_env = history |> Enum.reverse() |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

    Enum.map(per_env, fn states -> score_states(states, horizon) end)
  end

  defp ensure_batch(agent, n) do
    case Agent.batch_reset_rows(agent, Enum.to_list(0..(n - 1))) do
      :ok -> :ok
      {:error, :batch_not_initialized} -> :ok = Agent.batch_init(agent, n)
      {:error, other} -> raise "batch reset failed: #{inspect(other)}"
    end
  end

  # Shared scorer for single and batched rollouts.
  defp score_states(states, horizon) do
    gs0 = hd(states)
    frames = Enum.map(states, fn s -> %{frame: s.frame, p1: ScenarioScan.player_summary(s.players[1]), p2: ScenarioScan.player_summary(s.players[2])} end)
    openings = Opening.openings(frames, window: horizon)

    %{
      frames: frames,
      states: states,
      fair: FairConversion.summary(frames, window: horizon),
      chain: AerialChain.summary(frames),
      opening: Opening.summary(frames, window: horizon),
      openings: openings,
      contact?: openings != [],
      converted?: Enum.any?(openings, & &1.converted?),
      damage: (List.last(states).players[2].percent - gs0.players[2].percent) * 1.0
    }
  end

  @doc "Cached state id when the entry was saved with keep: true (or uploaded), else the blob."
  def restore_ref(%{state_id: id}) when is_integer(id), do: {:id, id}
  def restore_ref(%{blob: blob}), do: blob

  @doc "Upload an entry's blob once so later restores use the worker cache."
  def ensure_cached(sim, %{state_id: id} = e) when is_integer(id), do: {sim, e}
  def ensure_cached(sim, e) do
    {:ok, id} = Env.upload(sim, e.blob)
    {sim, Map.put(e, :state_id, id)}
  end

  defp controller(:idle, _gs, _port), do: @neutral

  defp controller(agent, gs, port) do
    case Agent.get_controller(agent, %{gs | own_port: port}, player_port: port) do
      {:ok, c} -> c
      {:error, _} -> @neutral
    end
  end

  @doc "Aggregate rollout results into the drill's headline numbers."
  def aggregate(results) do
    n = length(results)
    contacts = Enum.count(results, & &1.contact?)
    conv = Enum.count(results, & &1.converted?)
    outcomes = results |> Enum.flat_map(&Map.to_list(&1.fair.outcomes)) |> Enum.reduce(%{}, fn {k, v}, acc -> Map.update(acc, k, v, &(&1 + v)) end)
    families = results |> Enum.flat_map(&Map.to_list(Map.get(&1, :opening, %{by_family: %{}}).by_family)) |> Enum.reduce(%{}, fn {k, v}, acc -> Map.update(acc, k, v, &(&1 + v)) end)
    openers = results |> Enum.flat_map(&Map.to_list(Map.get(&1, :opening, %{by_opener: %{}}).by_opener)) |> Enum.reduce(%{}, fn {k, v}, acc -> Map.update(acc, k, v, &(&1 + v)) end)

    %{
      starts: n,
      contact_rate: if(n == 0, do: 0.0, else: contacts / n),
      conversion_rate: if(n == 0, do: 0.0, else: conv / n),
      conversion_given_contact: if(contacts == 0, do: 0.0, else: conv / contacts),
      mean_damage: if(n == 0, do: 0.0, else: Enum.sum(Enum.map(results, & &1.damage)) / n),
      mean_connected_aerials: results |> Enum.map(& &1.chain.mean_connected_aerials) |> then(&(if n == 0, do: 0.0, else: Enum.sum(&1) / n)),
      outcome_kinds: outcomes,
      opening_families: families,
      openers: openers,
      total_openings: Enum.sum(Map.values(families))
    }
  end

  @doc """
  Full pool (blobs + warm history) as one Erlang term file, so every arm of
  an experiment runs on the SAME starts with the SAME history (state_ids are
  dropped: they belong to the worker that made them; `ensure_cached/2`
  re-uploads on first use).
  """
  def pool_to_file(pool, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, :erlang.term_to_binary(Enum.map(pool, &Map.delete(&1, :state_id)), [:compressed]))
  end

  def pool_from_file(path), do: path |> File.read!() |> :erlang.binary_to_term()

  @doc "Pool entries without history (for saving); blobs base64."
  def pool_to_disk(pool, path) do
    File.mkdir_p!(Path.dirname(path))
    io = File.open!(path, [:write, :utf8])

    Enum.each(pool, fn e ->
      IO.write(io, Jason.encode!(%{id: e.id, frame: e.frame, summary: e.summary, blob: Base.encode64(e.blob)}) <> "\n")
    end)

    File.close(io)
    :ok
  end
end
