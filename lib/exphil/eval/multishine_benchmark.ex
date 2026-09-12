defmodule ExPhil.Eval.MultishineBenchmark do
  @moduledoc """
  Replay-based multishine benchmark. Recovery ends at a completed v3 cycle,
  not an isolated shine. Its readiness proxy is grounded locomotion or grounded
  reflector with zero reported hitstun; it is not an exact actionable-frame oracle.
  Gaps, deaths, repeated hits, and replay ends censor unfinished recovery episodes.
  """
  alias ExPhil.Eval.ShineChain
  alias ExPhil.Data.Peppi
  alias ExPhil.Agents.MultishineExpert

  def score(rows) when rows != [] do
    initial = %{
      previous: nil,
      bridge: [],
      grounded: false,
      episode: nil,
      episodes: [],
      onsets: 0,
      self_onsets: 0,
      last_hit: nil,
      cycles: 0,
      deaths: 0,
      segments: [],
      segment: []
    }

    state = Enum.reduce(rows, initial, &step/2) |> censor(:end_of_replay)
    segments = [Enum.reverse(state.segment) | state.segments]
    chains = Enum.flat_map(segments, &ShineChain.chains_detailed/1)
    episodes = Enum.reverse(state.episodes)
    completed = Enum.filter(episodes, &(&1.outcome == :reentered))

    %{
      frames: length(rows),
      ground_shine_onsets: state.onsets,
      ground_shines_per_minute: state.onsets * 3600 / length(rows),
      self_initiated_onsets: state.self_onsets,
      recent_hit_onsets: state.onsets - state.self_onsets,
      self_initiated_per_minute: state.self_onsets * 3600 / length(rows),
      completed_cycles: state.cycles,
      deaths: state.deaths,
      max_chain: chains |> Enum.map(& &1.length) |> Enum.max(fn -> 0 end),
      chains: chains,
      endings: Enum.frequencies_by(chains, & &1.ended_by),
      recovery: %{
        episodes: episodes,
        completed: length(completed),
        censored: length(episodes) - length(completed),
        ready_to_cycle_frames: Enum.map(completed, &(&1.finish - &1.ready)),
        disruption_to_cycle_frames: Enum.map(completed, &(&1.finish - &1.start))
      }
    }
  end

  def score([]), do: raise(ArgumentError, "No playable subject frames")

  defp step(row, state) do
    previous = state.previous
    gap = previous != nil and row.frame != previous.frame + 1
    death = previous != nil and row.stock < previous.stock

    state =
      if gap or death do
        state = censor(state, if(death, do: :death, else: :frame_gap))

        %{
          state
          | bridge: [],
            grounded: false,
            previous: nil,
            last_hit: nil,
            segments: [Enum.reverse(state.segment) | state.segments],
            segment: [],
            deaths: state.deaths + if(death, do: previous.stock - row.stock, else: 0)
        }
      else
        state
      end

    previous = state.previous
    metric_action = if row.hitstun > 0, do: 0, else: row.action
    family = ShineChain.family(metric_action)

    prior_family =
      if previous != nil and previous.hitstun == 0,
        do: ShineChain.family(previous.action),
        else: :other

    new_hit = row.hitstun > 0 and (previous == nil or previous.hitstun == 0)
    break = previous != nil and prior_family != :other and family == :other

    state =
      cond do
        new_hit ->
          state = censor(state, :interrupted_again)
          %{state | episode: %{start: row.frame, ready: nil, cause: :hitstun}}

        break and state.episode == nil ->
          %{state | episode: %{start: row.frame, ready: nil, cause: :loop_break}}

        true ->
          state
      end

    ready =
      row.hitstun == 0 and row.grounded and
        (row.action in 14..23 or family == :ground_reflect)

    state =
      if state.episode != nil and state.episode.ready == nil and ready,
        do: put_in(state.episode.ready, row.frame),
        else: state

    onset = family == :ground_reflect and prior_family != :ground_reflect

    self_onset =
      onset and previous != nil and row.hitstun == 0 and
        (state.last_hit == nil or row.frame - state.last_hit > 30)

    cycle =
      onset and state.grounded and row.hitstun == 0 and
        Enum.any?(state.bridge, &(ShineChain.family(&1) == :air_reflect)) and
        ShineChain.chains([360] ++ Enum.reverse(state.bridge) ++ [360]) == [2]

    state =
      if onset and state.grounded and not cycle and state.episode == nil,
        do: %{state | episode: %{start: row.frame, ready: row.frame, cause: :failed_cycle}},
        else: state

    state =
      if cycle and state.episode != nil and state.episode.ready != nil do
        event = Map.merge(state.episode, %{finish: row.frame, outcome: :reentered})
        %{state | episode: nil, episodes: [event | state.episodes]}
      else
        state
      end

    %{
      state
      | previous: row,
        onsets: state.onsets + if(onset and previous != nil, do: 1, else: 0),
        self_onsets: state.self_onsets + if(self_onset, do: 1, else: 0),
        last_hit: if(row.hitstun > 0, do: row.frame, else: state.last_hit),
        cycles: state.cycles + if(cycle, do: 1, else: 0),
        grounded:
          if(row.hitstun > 0, do: false, else: family == :ground_reflect or state.grounded),
        bridge:
          if(family == :ground_reflect,
            do: [],
            else: Enum.take([metric_action | state.bridge], 64)
          ),
        segment: [metric_action | state.segment]
    }
  end

  defp censor(%{episode: nil} = state, _reason), do: state

  defp censor(state, reason) do
    event =
      Map.merge(state.episode, %{
        finish: nil,
        outcome: reason,
        last_observed: if(state.previous, do: state.previous.frame, else: nil)
      })

    %{state | episode: nil, episodes: [event | state.episodes]}
  end

  def rows(replay, port) do
    replay.frames
    |> Enum.filter(&(&1.frame_number >= 0))
    |> Enum.map(fn frame ->
      player = Map.fetch!(frame.players, port)
      if player == nil, do: raise(ArgumentError, "Missing subject at frame #{frame.frame_number}")
      if trunc(player.character) != 1, do: raise(ArgumentError, "Subject must be Fox")

      %{
        frame: frame.frame_number,
        action: trunc(player.action),
        stock: player.stock,
        hitstun: player.hitstun_frames_left || 0,
        grounded: player.on_ground,
        player: player
      }
    end)
  end

  def teacher_audit(rows, expert) do
    pairs = Enum.chunk_every(rows, 2, 1, :discard)

    observations =
      Enum.flat_map(pairs, fn [row, next] ->
        if next.frame == row.frame + 1 and next.stock == row.stock do
          case MultishineExpert.label(expert, row.player, row.player.controller) do
            :skip ->
              []

            {:ok, target} ->
              key = {row.action, trunc(row.player.action_frame), row.grounded}

              [
                %{
                  frame: row.frame,
                  source: if(Map.has_key?(expert.table, key), do: :table, else: :fallback),
                  state: Tuple.to_list(key),
                  hitstun: row.hitstun,
                  agrees: buttons(target) == buttons(next.player.controller)
                }
              ]
          end
        else
          []
        end
      end)

    %{
      counts: Enum.frequencies_by(observations, & &1.source),
      state_counts:
        observations
        |> Enum.group_by(& &1.state)
        |> Enum.map(fn {key, values} ->
          %{
            state: key,
            frames: length(values),
            disagreements: Enum.count(values, &(not &1.agrees))
          }
        end)
        |> Enum.sort_by(& &1.state),
      disagreement_count: Enum.count(observations, &(not &1.agrees)),
      fallback_during_hitstun:
        Enum.count(observations, &(&1.source == :fallback and &1.hitstun > 0)),
      examples:
        observations |> Enum.filter(&(&1.source == :fallback or not &1.agrees)) |> Enum.take(100),
      interpretation:
        "Issued-input button agreement and teacher coverage only; not proof of valid corrections. No training-delay shift applied."
    }
  end

  defp buttons(controller) do
    Enum.map(
      [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r],
      &Map.get(controller, &1, false)
    )
  end

  def run_manifest(path) do
    manifest = path |> File.read!() |> Jason.decode!()
    if manifest["version"] != 1, do: raise(ArgumentError, "Expected manifest version 1")
    protocol = Map.fetch!(manifest, "protocol")

    Enum.each(
      ~w(runner frame_delay delay_id temperature buttons_temperature stage seconds),
      fn key ->
        if not Map.has_key?(protocol, key),
          do: raise(ArgumentError, "Missing protocol field #{key}")
      end
    )

    unless protocol["runner"] in ["async", "sync"], do: raise(ArgumentError, "Unknown runner")

    for key <- ~w(frame_delay delay_id) do
      unless is_integer(protocol[key]) and protocol[key] >= 0,
        do: raise(ArgumentError, "#{key} must be a nonnegative integer")
    end

    for key <- ~w(temperature buttons_temperature) do
      unless is_number(protocol[key]) and protocol[key] >= 0,
        do: raise(ArgumentError, "#{key} must be nonnegative")
    end

    runs = Map.fetch!(manifest, "runs")
    if runs == [], do: raise(ArgumentError, "Manifest has no runs")
    ids = Enum.map(runs, &Map.fetch!(&1, "id"))
    if length(Enum.uniq(ids)) != length(ids), do: raise(ArgumentError, "Duplicate run IDs")
    base = Path.dirname(Path.expand(path))
    fixture = Path.expand(Map.fetch!(manifest, "teacher_fixture"), base)

    if Map.fetch!(manifest, "teacher_fixture_sha256") != fingerprint(fixture),
      do: raise(ArgumentError, "Teacher fixture does not match frozen manifest")

    expert = MultishineExpert.from_fixture(fixture)

    results =
      Enum.map(runs, fn run ->
        replay_path = Path.expand(Map.fetch!(run, "replay"), base)
        policy_path = Path.expand(Map.fetch!(run, "policy"), base)
        scenario = Map.fetch!(run, "scenario")
        unless scenario in ["stand", "cpu", "human"], do: raise(ArgumentError, "Unknown scenario")
        port = Map.fetch!(run, "port")
        unless port in 1..4, do: raise(ArgumentError, "Explicit subject port required")
        {:ok, replay} = Peppi.parse(replay_path)

        for {key, artifact} <- [{"policy_sha256", policy_path}, {"replay_sha256", replay_path}] do
          if Map.fetch!(run, key) != fingerprint(artifact),
            do: raise(ArgumentError, "#{run["id"]}: #{key} does not match frozen manifest")
        end

        rows = rows(replay, port)

        seconds =
          if is_map(protocol["seconds"]),
            do: Map.fetch!(protocol["seconds"], scenario),
            else: protocol["seconds"]

        unless is_number(seconds) and seconds > 0,
          do: raise(ArgumentError, "Positive duration required")

        %{
          id: run["id"],
          scenario: scenario,
          port: port,
          policy: policy_path,
          observed_seconds: length(rows) / 60,
          expected_seconds: seconds,
          short_recording: length(rows) < seconds * 60 * 0.8,
          policy_sha256: fingerprint(policy_path),
          replay: replay_path,
          replay_sha256: fingerprint(replay_path),
          metrics: score(rows),
          teacher: teacher_audit(rows, expert)
        }
      end)

    hashes = Enum.map(results, & &1.replay_sha256)

    if length(Enum.uniq(hashes)) != length(hashes),
      do: raise(ArgumentError, "Duplicate replay content is not an independent run")

    %{
      version: 1,
      metric: "multishine_reentry_v1",
      protocol: protocol,
      manifest_sha256: fingerprint(path),
      source_sha256:
        Map.new(
          [
            "lib/exphil/eval/multishine_benchmark.ex",
            "lib/exphil/eval/shine_chain.ex",
            "lib/exphil/agents/multishine_expert.ex",
            "lib/exphil/data/peppi.ex"
          ],
          &{&1, fingerprint(&1)}
        ),
      protocol_verified: false,
      teacher_fixture_sha256: fingerprint(fixture),
      runs: results,
      note:
        "Protocol is declared, not inferred from replay. Report per run; do not select and confirm on the same games."
    }
  end

  defp fingerprint(path),
    do: :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
end
