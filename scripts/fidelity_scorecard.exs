# Fidelity scorecard: does the policy's own play look like the expert data? (2026-10-02)
#
# The policy plays a Fox ditto against ITSELF (or --opponent idle|POLICY) in the
# NIF sim; ExPhil.Eval.PlayStats is computed for port 1 exactly as
# scripts/expert_reference.exs computes it for expert Fox games, and the two
# are compared as distributions (total-variation distance, 0 = identical) and
# as rates. The reference file carries the expert split-half distance: the
# noise floor each distance is read against.
#
#   mix run scripts/fidelity_scorecard.exs --policy P --label L
#     [--reference eval_runs/1002_fidelity/expert_fd.json] [--opponent self|idle|POLICY]
#     [--envs 32] [--frames 3600] [--seeds 1001,1002,1003] [--stateful-step]
#     [--ablate-prev-action] [--out FILE.json] [--silence-reference eval_runs/1002_fidelity/expert_silence_map_fd.json]
#     [--decision-reference eval_runs/1002_fidelity/expert_decision_map_fd.json]
#
# Also writes silence_map.json beside --out: ExPhil.Eval.SilenceMap hazards by
# situation, compared with the expert map (where the bot lets go; 10-06).
#
# Several --seeds give mean ± sd per number (run-to-run noise of the scorecard).
# Caveat: the expert reference is Fox vs human opponents of many characters;
# the sim is a Fox ditto vs the same policy. Own-input and own-movement
# distributions transfer; damage/kill rates depend on the opponent.
alias ExPhil.Agents.Agent
alias ExPhil.Eval.{DecisionMap, PlayStats, SilenceMap}
# The decision map (10-07) is a new lib module; a queue running this script with
# --no-compile against an older _build must not fall over it.
decision_map? = Code.ensure_loaded?(DecisionMap)
alias ExPhil.Sim.{Env, Drill, GA}
alias ExPhil.Training.{Checkpoint, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, reference: :string, opponent: :string, envs: :integer,
             frames: :integer, seeds: :string, stateful_step: :boolean, ablate_prev_action: :boolean, out: :string,
             silence_reference: :string, decision_reference: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
ref = (opts[:reference] || "eval_runs/1002_fidelity/expert_fd.json") |> File.read!() |> Jason.decode!()
n = opts[:envs] || 32
frames = opts[:frames] || 3600
seeds = (opts[:seeds] || "1001") |> String.split(",") |> Enum.map(&String.to_integer/1)
stage = "final_destination"
edge = GA.stage_edge(32)
neutral = Drill.neutral()

start_agent = fn path, extra ->
  {:ok, export} = Checkpoint.load_policy(path)
  contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)
  {:ok, agent} =
    Agent.start_link([policy_path: path, deterministic: false, temperature: 1.0, af_convention: :parsed,
      frame_delay: 0, reaction_delay: 0, harness: :sync_runner,
      stateful_step: Keyword.get(extra, :stateful_step) || contract.recurrent_state == :carried_zero] ++ Keyword.delete(extra, :stateful_step))
  {:ok, _} = Agent.warmup(agent)
  :ok = Agent.batch_init(agent, n)
  agent
end

Output.banner("Fidelity scorecard: #{label}")
Output.config([{"Policy", policy}, {"Opponent", opts[:opponent] || "self"}, {"Envs x frames", "#{n} x #{frames}"},
  {"Seeds", inspect(seeds)}, {"Reference", "#{ref["games"]} expert games"}])

actor_opts = [ablate_prev_action: opts[:ablate_prev_action] || false, stateful_step: opts[:stateful_step] || false]
actor = start_agent.(policy, actor_opts)
opponent =
  case opts[:opponent] do
    "idle" -> :idle
    o when o in [nil, "self"] -> start_agent.(policy, actor_opts)
    path -> start_agent.(path, [])
  end

players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]

rollout = fn seed ->
  :ok = Agent.batch_reset_rows(actor, Enum.to_list(0..(n - 1)))
  if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, Enum.to_list(0..(n - 1)))
  {:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: seed)
  {:ok, _, _} = Env.observe(sim)
  {:ok, gs0} = Env.frames(sim)
  starts = for i <- 0..(n - 1), into: %{}, do: ({:ok, blob} = Env.save(sim, i); {i, blob})

  {games, open, _} =
    Enum.reduce(1..frames, {[], List.duplicate([], n), gs0}, fn _t, {games, open, states} ->
      {:ok, c1s} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
      c2s =
        case opponent do
          :idle -> List.duplicate(neutral, n)
          agent ->
            {:ok, cs} = Agent.batch_get_controllers(agent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
            cs
        end
      {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || neutral, b || neutral] end))
      open = Enum.zip_with([open, states, c1s], fn [h, s, c] -> [%{own: s.players[1], opp: s.players[2], controller: c || neutral} | h] end)
      finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

      if finished == [] do
        {games, open, nexts}
      else
        for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
        :ok = Agent.batch_reset_rows(actor, finished)
        if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, finished)
        {:ok, reset_states} = Env.frames(sim)
        done = for i <- finished, do: Enum.at(open, i)
        open = open |> Enum.with_index() |> Enum.map(fn {h, i} -> if i in finished, do: [], else: h end)
        {done ++ games, open, reset_states}
      end
    end)

  played = (open ++ games) |> Enum.reject(&(&1 == [])) |> Enum.map(&Enum.reverse/1)

  stats =
    played
    |> Enum.map(&PlayStats.from_game(&1, edge))
    |> Enum.reduce(PlayStats.empty(), &PlayStats.merge/2)
    |> PlayStats.summarize()

  # Where the bot lets go: input-change hazards by situation (SilenceMap, 10-06)
  silence = played |> Enum.map(&SilenceMap.from_game(&1, stage: 32, edge: edge)) |> Enum.reduce(SilenceMap.empty(), &SilenceMap.merge/2)
  # What the bot decides offstage, by height (DecisionMap, 10-07)
  decisions = if decision_map?, do: played |> Enum.map(&DecisionMap.from_game(&1, edge: edge)) |> Enum.reduce(DecisionMap.empty(), &DecisionMap.merge/2), else: %{}
  {stats, silence, decisions}
end

# Fox-specific derived technique rates from the histograms
derived = fn s ->
  lag = s["hists"]["landing_lag"] || %{}
  p = fn ks -> Enum.sum(Enum.map(ks, &Map.get(lag, "#{&1}", 0.0))) end
  hit = p.([7, 9, 10, 11])
  miss = p.([15, 18, 20, 22])
  peaks = s["hists"]["jump_peak"] || %{}
  low = peaks |> Enum.filter(fn {k, _} -> String.to_integer(k) in 5..19 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  real = peaks |> Enum.filter(fn {k, _} -> String.to_integer(k) >= 5 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  rr = fn a, b -> if b == 0 or b == 0.0, do: nil, else: Float.round(a / b, 3) end
  %{"l_cancel_rate" => rr.(hit, hit + miss), "short_hop_share" => rr.(low, real)}
end

jsonify = fn s -> s |> Jason.encode!() |> Jason.decode!() end
expert = ref["summary"]
expert_rates = Map.merge(expert["rates"], derived.(expert))
expert_struct = %{hists: expert["hists"]}

runs =
  for seed <- seeds do
    t0 = System.monotonic_time(:millisecond)
    {s, silence, decisions} = rollout.(seed)
    s = jsonify.(s)
    d = PlayStats.compare(%{hists: s["hists"]}, expert_struct)
    Output.puts("  seed #{seed}: #{s["rates"]["minutes"]} env-min in #{div(System.monotonic_time(:millisecond) - t0, 1000)} s")
    %{rates: Map.merge(s["rates"], derived.(s)), dist: d, hists: s["hists"], silence: silence, decisions: decisions}
  end

stat = fn vals ->
  vals = Enum.reject(vals, &is_nil/1)
  if vals == [] do
    {nil, nil}
  else
    m = Enum.sum(vals) / length(vals)
    sd = if length(vals) > 1, do: :math.sqrt(Enum.sum(Enum.map(vals, &((&1 - m) ** 2))) / (length(vals) - 1)), else: 0.0
    {Float.round(m * 1.0, 3), Float.round(sd, 3)}
  end
end

rate_keys = runs |> hd() |> Map.fetch!(:rates) |> Map.keys() |> Enum.sort()
dist_keys = runs |> hd() |> Map.fetch!(:dist) |> Map.keys() |> Enum.sort()
rates = Map.new(rate_keys, fn k -> {m, sd} = stat.(Enum.map(runs, & &1.rates[k])); {k, %{model: m, sd: sd, expert: expert_rates[k]}} end)
dists = Map.new(dist_keys, fn k -> {m, sd} = stat.(Enum.map(runs, & &1.dist[k])); {k, %{distance: m, sd: sd, noise_floor: ref["split_half_distance"][k]}} end)

headline = ~w(action_group position stick_zone stick_dwell hold_mean landing_lag jump_peak)
{fid, fid_sd} = stat.(Enum.map(runs, fn r -> Enum.sum(Enum.map(headline, &(r.dist[&1] || 0.0))) / length(headline) end))

result = %{label: label, policy: policy, opponent: opts[:opponent] || "self", envs: n, frames_per_env: frames, seeds: seeds,
  fidelity_distance: %{mean: fid, sd: fid_sd}, rates: rates, distances: dists, hists: hd(runs).hists}

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(result, pretty: true))
end

# Silence map: all seeds pooled; compared with the expert map when the reference exists
silence_counts = runs |> Enum.map(& &1.silence) |> Enum.reduce(SilenceMap.empty(), &SilenceMap.merge/2)
silence_model = SilenceMap.summarize(silence_counts)
silence_ref_path = opts[:silence_reference] || "eval_runs/1002_fidelity/expert_silence_map_fd.json"

silence_ref =
  case File.read(silence_ref_path) do
    {:ok, s} -> s |> Jason.decode!() |> Map.fetch!("summary") |> Map.new(fn {k, v} -> {k, Map.new(v, fn {a, b} -> {String.to_atom(a), b} end)} end)
    _ -> nil
  end

silence_cmp =
  if silence_ref do
    Map.new([:enter_silence, :change, :resume], fn h -> {h, SilenceMap.compare(silence_model, silence_ref, hazard: h, min_n: 200)} end)
  else
    %{}
  end

if out = opts[:out] do
  sm_out = Path.join(Path.dirname(out), "silence_map.json")
  File.write!(sm_out, Jason.encode!(%{label: label, policy: policy, seeds: seeds, summary: silence_model,
    reference: silence_ref && silence_ref_path, compare: silence_cmp}, pretty: true))
end

fmt_row = fn r -> "#{r.bucket} #{r.model}|#{r.expert} x#{r.ratio} (n=#{r.n})" end

# Decision map (10-07): recovery-decision onset hazards by height band × jumps
# left, all seeds pooled, against the expert map when the reference exists.
# The jump slope (hazard at -40..-60 over 0..-20, jump in hand) is the
# height-conditioning readout: expert ≈ 4, a flat head ≈ 1.
if decision_map? do
decision_counts = runs |> Enum.map(& &1.decisions) |> Enum.reduce(DecisionMap.empty(), &DecisionMap.merge/2)
decision_model = DecisionMap.summarize(decision_counts)
decision_ref_path = opts[:decision_reference] || "eval_runs/1002_fidelity/expert_decision_map_fd.json"

decision_ref =
  case File.read(decision_ref_path) do
    {:ok, s} -> s |> Jason.decode!() |> Map.fetch!("summary") |> Map.new(fn {k, v} -> {k, Map.new(v, fn {a, b} -> {String.to_atom(a), b} end)} end)
    _ -> nil
  end

decision_cmp =
  if decision_ref,
    do: Map.new(DecisionMap.decisions(), fn d -> {d, DecisionMap.compare(decision_model, decision_ref, decision: String.to_atom(d), min_n: 100)} end),
    else: %{}

if out = opts[:out] do
  dm_out = Path.join(Path.dirname(out), "decision_map.json")
  File.write!(dm_out, Jason.encode!(%{label: label, policy: policy, seeds: seeds, summary: decision_model,
    slope: Map.new(DecisionMap.decisions(), fn d -> {d, DecisionMap.slope(decision_model, String.to_atom(d))} end),
    reference: decision_ref && decision_ref_path, compare: decision_cmp}, pretty: true))
end

band_order = ~w(y>0 0..-20 -20..-40 -40..-60 <-60)
by_band = fn rows -> rows |> Enum.filter(&String.ends_with?(&1.bucket, ":j1+")) |> Enum.sort_by(fn r -> Enum.find_index(band_order, &(&1 == String.replace_suffix(r.bucket, ":j1+", ""))) end) end
slope_txt = fn d -> "#{d} #{inspect(DecisionMap.slope(decision_model, String.to_atom(d)))}|#{inspect(decision_ref && DecisionMap.slope(decision_ref, String.to_atom(d)))}" end
Output.puts("RESULT #{label} decision map height slope (-40..-60 / 0..-20, jump in hand) model|expert: " <>
  Enum.map_join(~w(jump special_up aerial airdodge), "  ", slope_txt))

if decision_ref do
  for d <- ~w(jump special_up special_side aerial airdodge) do
    rows = by_band.(decision_cmp[d])
    Output.puts("RESULT #{label} decision map #{d} hazard by height (jump in hand) model|expert xratio: " <> Enum.map_join(rows, "  ", fmt_row))
  end
else
  Output.warning("no expert decision map at #{decision_ref_path} (run scripts/expert_reference.exs --decision-map-out); model-only summary written")
end
else
  Output.warning("ExPhil.Eval.DecisionMap is not compiled into this build; decision map skipped (compile and rerun)")
end

if silence_ref do
  for {h, rows} <- silence_cmp do
    worst = rows |> Enum.filter(&(&1.z >= 3)) |> Enum.take(8)
    Output.puts("RESULT #{label} silence map #{h} model|expert xratio, worst buckets (z>=3): " <> Enum.map_join(worst, "  ", fmt_row))
  end

  states = silence_cmp[:enter_silence] |> Enum.filter(&String.starts_with?(&1.bucket, "state:")) |> Enum.sort_by(& &1.bucket)
  Output.puts("RESULT #{label} silence map enter_silence by state: " <> Enum.map_join(states, "  ", fmt_row))
  ages = silence_cmp[:enter_silence] |> Enum.filter(&String.starts_with?(&1.bucket, "age:")) |> Enum.sort_by(& &1.bucket)
  Output.puts("RESULT #{label} silence map enter_silence by age: " <> Enum.map_join(ages, "  ", fmt_row))
else
  Output.warning("no expert silence map at #{silence_ref_path} (run scripts/expert_reference.exs --silence-map-out); model-only summary written")
end

Output.puts("RESULT #{label} fidelity distance (mean of #{length(headline)} histograms, 0 = expert-like): #{fid} ± #{fid_sd}")
Output.puts("RESULT #{label} distances  " <> Enum.map_join(headline, "  ", fn k -> "#{k} #{dists[k].distance}±#{dists[k].sd} (floor #{dists[k].noise_floor})" end))
show = ~w(sd_per_min deaths_per_min offstage_eps_per_min offstage_return_rate damage_dealt_per_min input_repeat_share neutral_input_share
          dashes_per_min wavedashes_per_min ground_jumps_per_min short_hop_share l_cancel_rate tech_rate press_a_per_min press_b_per_min press_r_per_min)
Output.puts("RESULT #{label} rates model±sd | expert  " <> Enum.map_join(show, "  ", fn k -> "#{k} #{inspect(rates[k].model)}±#{inspect(rates[k].sd)}|#{inspect(rates[k].expert)}" end))
