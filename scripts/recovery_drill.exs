# Recovery drill: knocked offstage in a recoverable spot, does the policy get
# back? (2026-10-01)
#
# Cases are captured INSIDE the sim (replay seeding diverged silently on the
# held-out Fox-vs-Marth/Falco games, so replay-mined cases were not the states
# they claimed to be — see INPUT_COHERENCE_2026-10-01.md). A source policy
# plays a Fox ditto against itself; whenever port 1 comes out of hitstun
# airborne and offstage and the static ExPhil.Melee.Checkmate model says the
# position is recoverable, the sim state is saved together with the previous
# 79 states and port-1 controllers.
#
# Drill: restore a case into K envs, warm the policy's window and prev-action
# channel with the saved history, then let it drive Fox for up to --horizon
# frames against a NEUTRAL opponent (no edgeguard: pure recovery ability).
# Outcome per trial: recovered (on stage or on the ledge) / died / timeout.
#
#   build:  mix run scripts/recovery_drill.exs --build --source POLICY [--cases FILE] [--envs 32] [--frames 3600] [--max-cases 60]
#   run:    mix run scripts/recovery_drill.exs --policy P --label L [--cases FILE] [--trials 8] [--horizon 300]
#             [--ablate-prev-action] [--out FILE.json] [--trace FILE.json] [--warm-override SPEC]
#
# --trace writes every trial's per-frame record (position, action, jumps, the
# controller the policy emitted) so a failed recovery can be read frame by
# frame (interp Q3, 2026-10-02). --warm-override SPEC rewrites the LAST n
# controllers of the warm history before the drill starts — a counterfactual
# on the prev-action channel only (the game states stay real):
#   up:3   last 3 warm inputs = main stick full up, no buttons
#   upb:3  same plus B held (an up-B already in progress on the channel)
#   jump:3 same plus X held
#
# Every policy faces the identical saved starts. `--policy neutral` holds a
# neutral controller (floor: what doing nothing recovers).
alias ExPhil.Agents.Agent
alias ExPhil.Melee.Checkmate
alias ExPhil.Sim.{Env, Drill, GA}
alias ExPhil.Training.{Checkpoint, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [build: :boolean, source: :string, policy: :string, label: :string, cases: :string, envs: :integer,
             frames: :integer, max_cases: :integer, trials: :integer, horizon: :integer,
             ablate_prev_action: :boolean, out: :string, seed: :integer, trace: :string, warm_override: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

cases_path = opts[:cases] || "eval_runs/1001_recovery/cases_sim.bin"
neutral = Drill.neutral()
stage = "final_destination"
stage_id = 32
edge = GA.stage_edge(stage_id)
players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
ledge? = fn p -> (p.action || 0) in 252..263 end
offstage? = fn p -> not p.on_ground and (abs(p.x) > edge or p.y < -5.0) end

start_agent = fn path, n, extra ->
  {:ok, export} = Checkpoint.load_policy(path)
  contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)
  {:ok, agent} =
    Agent.start_link([policy_path: path, deterministic: false, temperature: 1.0, af_convention: :parsed,
      frame_delay: 0, reaction_delay: 0, harness: :sync_runner,
      stateful_step: contract.recurrent_state == :carried_zero] ++ extra)
  {:ok, _} = Agent.warmup(agent)
  :ok = Agent.batch_init(agent, n)
  agent
end

# ---- build -----------------------------------------------------------------------
if opts[:build] do
  source = opts[:source] || raise("--build needs --source POLICY")
  n = opts[:envs] || 32
  frames = opts[:frames] || 3600
  max_cases = opts[:max_cases] || 60
  Output.banner("Recovery drill: capturing cases in the sim")
  Output.config([{"Source policy (both ports)", source}, {"Envs", n}, {"Frames", frames}, {"Max cases", max_cases}])
  a1 = start_agent.(source, n, [])
  a2 = start_agent.(source, n, [])
  {:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: opts[:seed] || 2001)
  {:ok, _, _} = Env.observe(sim)
  {:ok, gs0} = Env.frames(sim)
  starts = for i <- 0..(n - 1), into: %{} do
    {:ok, blob} = Env.save(sim, i)
    {i, blob}
  end

  {cases, _, _} =
    Enum.reduce_while(1..frames, {[], List.duplicate([], n), gs0}, fn t, {cases, hists, states} ->
      {:ok, c1s} = Agent.batch_get_controllers(a1, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
      {:ok, c2s} = Agent.batch_get_controllers(a2, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
      {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || neutral, b || neutral] end))
      hists = Enum.zip_with([hists, states, c1s], fn [h, s, c] -> Enum.take([{s, c || neutral} | h], 79) end)

      new =
        [states, nexts, hists]
        |> Enum.zip()
        |> Enum.with_index()
        |> Enum.flat_map(fn {{s, nx, h}, i} ->
          p0 = s.players[1]
          p1 = nx.players[1]
          out_of_stun? = (p1.hitstun_frames_left || 0) == 0 and (p0.hitstun_frames_left || 0) > 0

          if out_of_stun? and offstage?.(p1) and (p1.stock || 0) == (p0.stock || 0) and length(h) == 79 do
            st = GA.checkmate_state(p1, stage_id)

            if Checkmate.checkmate?(st) do
              []
            else
              {:ok, blob} = Env.save(sim, i)
              [%{blob: blob, warm: Enum.reverse(h), x: p1.x, y: p1.y, jumps: p1.jumps_left, percent: p1.percent, t: t, env: i}]
            end
          else
            []
          end
        end)

      finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

      {nexts, hists} =
        if finished == [] do
          {nexts, hists}
        else
          for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
          for a <- [a1, a2], do: :ok = Agent.batch_reset_rows(a, finished)
          {:ok, reset_states} = Env.frames(sim)
          {reset_states, hists |> Enum.with_index() |> Enum.map(fn {h, i} -> if i in finished, do: [], else: h end)}
        end

      cases = cases ++ new
      if rem(t, 600) == 0, do: Output.puts("  frame #{t}/#{frames}: #{length(cases)} cases")
      if length(cases) >= max_cases, do: {:halt, {cases, hists, nexts}}, else: {:cont, {cases, hists, nexts}}
    end)

  File.mkdir_p!(Path.dirname(cases_path))
  File.write!(cases_path, :erlang.term_to_binary(%{source: source, stage: stage, players: players, edge: edge, cases: Enum.take(cases, max_cases)}))
  Output.success("#{min(length(cases), max_cases)} recoverable offstage cases -> #{cases_path}")
  System.halt(0)
end

# ---- run -------------------------------------------------------------------------
pool = cases_path |> File.read!() |> :erlang.binary_to_term()
cases = pool.cases
if cases == [], do: raise("no cases in #{cases_path}")
policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
k = opts[:trials] || 8
horizon = opts[:horizon] || 300
Output.banner("Recovery drill: #{label}")
Output.config([{"Policy", policy}, {"Cases", length(cases)}, {"Trials/case", k}, {"Horizon", horizon}])

idle? = policy == "neutral"
agent = if idle?, do: nil, else: start_agent.(policy, k, ablate_prev_action: opts[:ablate_prev_action] || false)
all_rows = Enum.to_list(0..(k - 1))
{:ok, sim} = Env.start(:nif, stage: pool.stage, players: pool.players, batch_size: k, seed: 7)

# counterfactual warm history: the last n channel inputs replaced
override_warm = fn warm ->
  case opts[:warm_override] do
    nil ->
      warm

    spec ->
      [kind, n] = String.split(spec, ":")
      n = String.to_integer(n)
      # x toward the stage from the case's side is unknown here; stick straight up
      ctrl = %{neutral | main_stick: %{x: 0.5, y: 1.0}}
      ctrl =
        case kind do
          "up" -> ctrl
          "upb" -> %{ctrl | button_b: true}
          "jump" -> %{ctrl | button_x: true}
        end
      {keep, last} = Enum.split(warm, length(warm) - n)
      keep ++ Enum.map(last, fn {gs, _} -> {gs, ctrl} end)
  end
end

trace? = opts[:trace] != nil
btn_bits = fn c -> for b <- [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r], do: if(Map.get(c, b) == true, do: 1, else: 0) end

{rows, traces} =
  cases
  |> Enum.with_index()
  |> Enum.map(fn {c, ci} ->
    {:ok, id} = Env.upload(sim, c.blob)
    for i <- all_rows, do: {:ok, _} = Env.restore(sim, i, {:id, id}, frames: false)
    # `frames: false` skips the re-observe, so Env.frames would still hold the
    # previous case's final states (found 2026-10-02: route verdicts differed
    # between runs on the same cases); observe once for all rows.
    {:ok, states, _} = Env.observe(sim)
    # surviving routes at the case frame (0 checkmate / 1 forced / 2+ mixup)
    verdict = Checkmate.routes(GA.checkmate_state(hd(states).players[1], stage_id))

    unless idle? do
      :ok = Agent.batch_reset_rows(agent, all_rows)
      for {gs, ctrl} <- override_warm.(c.warm) do
        :ok = Agent.batch_observe(agent, List.duplicate(%{gs | own_port: 1}, k), player_port: 1, controllers: List.duplicate(ctrl, k))
      end
    end

    stock0 = hd(states).players[1].stock

    {outcomes, finals, trace} =
      Enum.reduce_while(1..horizon, {List.duplicate(nil, k), states, []}, fn t, {outs, states, tr} ->
        cs =
          if idle? do
            List.duplicate(neutral, k)
          else
            {:ok, cs} = Agent.batch_get_controllers(agent, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
            cs
          end

        {:ok, nexts, _} = Env.step(sim, Enum.map(cs, fn a -> [a || neutral, neutral] end))

        # per-frame record: [t, x, y, action, jumps_left, main_x, main_y, c_y, buttons a..r] per trial
        tr =
          if trace? do
            step =
              Enum.zip_with([all_rows, outs, states, cs], fn [_i, o, s, a] ->
                if o != nil do
                  nil
                else
                  p = s.players[1]
                  a = a || neutral
                  [t, Float.round(p.x * 1.0, 1), Float.round(p.y * 1.0, 1), p.action, p.jumps_left,
                   Float.round(a.main_stick.x * 1.0, 3), Float.round(a.main_stick.y * 1.0, 3), Float.round(a.c_stick.y * 1.0, 3) | btn_bits.(a)]
                end
              end)
            [step | tr]
          else
            tr
          end

        outs =
          Enum.zip_with(outs, nexts, fn o, nx ->
            p = nx.players[1]
            cond do
              o != nil -> o
              # death states 0..10, rebirth 12/13: the stock field alone missed deaths
              # (a neutral Fox ended on the respawn platform as a "timeout")
              (p.stock || 0) < (stock0 || 0) or (p.action || 99) in 0..13 -> {:died, t}
              (p.on_ground and abs(p.x) <= pool.edge + 1.0) or ledge?.(p) -> {:recovered, t}
              true -> nil
            end
          end)

        if Enum.all?(outs, &(&1 != nil)), do: {:halt, {outs, nexts, tr}}, else: {:cont, {outs, nexts, tr}}
      end)

    rec = Enum.count(outcomes, &match?({:recovered, _}, &1))
    died = Enum.count(outcomes, &match?({:died, _}, &1))
    # where unresolved trials ended (action id, y) — to audit the outcome rule
    stuck = for {o, nx} <- Enum.zip(outcomes, finals), o == nil, do: {nx.players[1].action, round(nx.players[1].y)}
    if System.get_env("DRILL_TRACE") == "1" and stuck != [], do: IO.puts("TRACE stuck #{inspect(Enum.frequencies(stuck))} start=(#{round(c.x)},#{round(c.y)})")
    row = %{x: c.x, y: c.y, jumps: c.jumps, percent: c.percent, recovered: rec, died: died, timeout: k - rec - died,
            routes: verdict.count, verdict: verdict.verdict,
            route_means: Enum.map(verdict.routes, &"#{Enum.join(&1.means, "+")}->#{&1.outcome}")}

    trace_entry =
      if trace? do
        # per trial: outcome + its frame rows (steps were prepended; trials are columns)
        steps = Enum.reverse(trace)
        trials =
          Enum.map(all_rows, fn i ->
            frames = steps |> Enum.map(&Enum.at(&1, i)) |> Enum.reject(&is_nil/1)
            outcome = case Enum.at(outcomes, i) do
              {kind, t} -> %{result: kind, t: t}
              nil -> %{result: :timeout, t: horizon}
            end
            Map.put(outcome, :frames, frames)
          end)
        %{case: ci, x: c.x, y: c.y, jumps: c.jumps, percent: c.percent, trials: trials}
      end

    {row, trace_entry}
  end)
  |> Enum.unzip()

if trace? do
  File.mkdir_p!(Path.dirname(opts[:trace]))
  File.write!(opts[:trace], Jason.encode!(%{label: label, policy: policy, warm_override: opts[:warm_override],
    columns: ~w(t x y action jumps main_x main_y c_y a b x_btn y_btn z l r), cases: traces}))
  Output.puts("trace -> #{opts[:trace]}")
end

trials = length(rows) * k
rec = Enum.sum(Enum.map(rows, & &1.recovered))
died = Enum.sum(Enum.map(rows, & &1.died))
# recovery by how many routes the position left open: 1 = forced (the opponent
# has one thing to cover), 2-3 = thin mixup, 4+ = open
route_bucket = fn n -> cond do n <= 1 -> "1"; n <= 3 -> "2-3"; true -> "4+" end end
by_routes =
  rows
  |> Enum.group_by(&route_bucket.(&1.routes))
  |> Map.new(fn {b, rs} ->
    {b, %{cases: length(rs), recovery_rate: Float.round(Enum.sum(Enum.map(rs, & &1.recovered)) / (length(rs) * k), 3)}}
  end)

result = %{label: label, policy: policy, cases: length(rows), trials: trials,
  recovery_rate: Float.round(rec / trials, 3), died_rate: Float.round(died / trials, 3),
  timeout_rate: Float.round((trials - rec - died) / trials, 3),
  cases_never_recovered: Enum.count(rows, &(&1.recovered == 0)),
  cases_always_recovered: Enum.count(rows, &(&1.recovered == k)), by_routes: by_routes, rows: rows}

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(result, pretty: true))
end

Output.puts("RESULT #{label}: #{length(rows)} cases x #{k} trials  recovery #{result.recovery_rate}  died #{result.died_rate}  timeout #{result.timeout_rate}  " <>
  "never #{result.cases_never_recovered}/#{length(rows)}  always #{result.cases_always_recovered}/#{length(rows)}")
Output.puts("RESULT #{label} by routes: " <>
  Enum.map_join(["1", "2-3", "4+"], "  ", fn b ->
    case by_routes[b] do
      nil -> "#{b}: -"
      %{cases: n, recovery_rate: r} -> "#{b} routes: #{r} (#{n} cases)"
    end
  end))
