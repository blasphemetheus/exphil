# Sim recovery DAgger with the EXPERT-DISTRIBUTION labeler (2026-10-08,
# INPUT_COHERENCE "10-08 13:45"; Bradley's ok for expert-labeled relabel on
# the bot's own offstage states). Same rollout + trip export as
# scripts/sim_recovery_dagger.exs (states = the bot's own; input history =
# the previous LABEL; 90 input-only context frames per trip), but the label
# for each offstage frame is the input a real expert gave in the nearest
# expert states (scripts/lib/expert_recovery_labeler.exs, index from
# scripts/build_expert_recovery_index.exs), sampled so the expert's hazards
# stay hazards. Labels are produced in order inside a trip because each one
# conditions on the previous label.
#
#   mix run scripts/sim_recovery_dagger_expert.exs --policy P.bin --index data/silent_fall/expert_recovery_index.bin \
#     --out data/silent_fall/sim_dagger_expert_r1.frames [--envs 32 --frames 3600 --seeds 2001,2002,2003]
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.{Agent, ExpertRecoveryLabeler}
alias ExPhil.Sim.{Drill, Env, GA}
alias ExPhil.Training.{Checkpoint, Output, SilentFallWeighting}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, out: :string, envs: :integer, frames: :integer, seeds: :string, opponent: :string,
             report: :string, action_delay: :integer, only_silent: :boolean, index: :string, label_seed: :integer,
             max_d2: :float, keep_actual: :boolean])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

# --only-silent (16:50): relabel ONLY the frames where the policy's actual
# press was neutral — the diagnosed defect ("silent-falling -> act") and
# nothing else. The full set (every offstage frame, 15.5k frames x4 = +50 %
# of the corpus's offstage data in a robotic style: full deflection every
# frame, B on 35 % of frames) shifted the policy globally even
# teacher-forced (neutral share 0.26 -> 0.10, B presses 14 -> 40/min).
only_silent = opts[:only_silent] || false
# --keep-actual (10-10): keep the bot's own input on every frame (`:actual`)
# so training can weight the frames where it pressed and the label holds
# (`--veto-weight`, SilentFallWeighting.veto?/3). Off = the old sets.
keep_actual = opts[:keep_actual] || false

policy = opts[:policy] || raise("--policy required")
out = opts[:out] || raise("--out required")
n = opts[:envs] || 32
frames_per = opts[:frames] || 3600
seeds = (opts[:seeds] || "2001,2002,2003") |> String.split(",") |> Enum.map(&String.to_integer/1)
action_delay = opts[:action_delay] || 0
max_d2 = opts[:max_d2]
edge = GA.stage_edge(32)
neutral = Drill.neutral()
index = ExpertRecoveryLabeler.load(opts[:index] || "data/silent_fall/expert_recovery_index.bin")
:rand.seed(:exsss, {opts[:label_seed] || 7, 11, 13})

Output.banner("Sim recovery DAgger (expert labeler, #{index.n} rows / #{index.meta[:games]} games): #{Path.basename(Path.dirname(policy))}")
Output.config([{"Policy", policy}, {"Opponent", opts[:opponent] || "self"}, {"Envs x frames", "#{n} x #{frames_per}"}, {"Seeds", inspect(seeds)}, {"Out", out}])

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

actor = start_agent.(policy, stateful_step: false)
opponent =
  case opts[:opponent] do
    "idle" -> :idle
    o when o in [nil, "self"] -> start_agent.(policy, stateful_step: false)
    path -> start_agent.(path, [])
  end
players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]

# one env's history -> relabeled frame lists: one per contiguous offstage run,
# each preceded by up to `prefix` frames of the SAME env's history as
# input-only context (the bot's own approach; never supervised — the lazy
# layout skips input_only targets, Data.from_frame_lists keeps the boundary)
prefix = 90

relabel = fn history ->
  frames =
    history
    |> Enum.reverse()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [{s0, c0}, {s1, c1}] ->
      p = s1.players[1]
      offstage_or_below = p.on_ground != true and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0)
      ledge = (p.action || 0) in 252..263

      _ = offstage_or_below
      _ = ledge
      labeled = ExpertRecoveryLabeler.labelable?(p, edge, index.window) and (not only_silent or SilentFallWeighting.neutral?(c1))
      # the label is filled in per trip below (sequentially: it conditions on the previous label)
      %{game_state: %{s1 | own_port: 1}, controller: c1, prev_controller: c0, player_tag: nil, actual: c1, labeled: labeled, prev_player: s0.players[1]}
    end)
    |> List.to_tuple()

  n_frames = tuple_size(frames)
  consecutive? = fn i -> i > 0 and elem(frames, i).game_state.frame == elem(frames, i - 1).game_state.frame + 1 end

  # runs of labeled, frame-consecutive frames
  {runs, cur} =
    Enum.reduce(0..(n_frames - 1)//1, {[], nil}, fn i, {runs, cur} ->
      f = elem(frames, i)

      cond do
        f.labeled and cur != nil and consecutive?.(i) -> {runs, {elem(cur, 0), i}}
        f.labeled -> {if(cur, do: [cur | runs], else: runs), {i, i}}
        true -> {if(cur, do: [cur | runs], else: runs), nil}
      end
    end)

  runs = Enum.reverse(if cur, do: [cur | runs], else: runs)

  Enum.map(runs, fn {a, b} ->
    # prefix: walk back while frame numbers stay consecutive
    start =
      Enum.reduce_while((a - 1)..max(a - prefix, 0)//-1, a, fn i, s ->
        if consecutive?.(i + 1), do: {:cont, i}, else: {:halt, s}
      end)

    ctx = for i <- start..(a - 1)//1, i < a, do: elem(frames, i) |> Map.put(:input_only, true) |> Map.delete(:labeled)
    trip = for i <- a..b, do: elem(frames, i) |> Map.delete(:labeled)

    # prev = the PREVIOUS LABEL, not the policy's actual press (first attempt,
    # 16:25): with event heads the previous input is only the hold/change
    # selector (the trunk's copy is zeroed), and "policy press -> expert
    # label" made 98 % of the set change events (expert corpus ~76 % holds).
    # 108 interleaved batches of that taught the bot to change its input
    # every frame everywhere: closed-loop repeat share 0.75 -> 0.36, dashes
    # 40 -> 3/min, SDs 1.3 -> 6.5/min, while teacher-forced metrics were
    # untouched. The state is still the bot's own; the input history is the
    # expert-consistent one (82 % holds). The first trip frame keeps the
    # actual press at t-1 (what really preceded it).
    # expert label per frame, in order: prev = the previous label (the first
    # frame's prev is the real press at t-1), state = the bot's own
    # --max-d2 (22:30): a state farther than this from every expert row gets
    # NO label — it stays in the trip as input-only context (the bot's own
    # press, unsupervised) and the next frame's prev is that real press.
    {trip, _} =
      Enum.map_reduce(trip, nil, fn f, prev_label ->
        prev = prev_label || f.prev_controller
        p = f.game_state.players[1]

        case ExpertRecoveryLabeler.label(index, p, prev, f.game_state.players[2], edge, max_d2: max_d2, prev_p: f.prev_player) do
          nil -> {f |> Map.put(:input_only, true) |> Map.put(:gated, true), f.controller}
          label -> {%{f | controller: label, prev_controller: prev}, label}
        end
      end)

    ctx ++ trip
  end)
end

rollout = fn seed ->
  :ok = Agent.batch_reset_rows(actor, Enum.to_list(0..(n - 1)))
  if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, Enum.to_list(0..(n - 1)))
  {:ok, sim} = Env.start(:nif, stage: "final_destination", players: players, batch_size: n, seed: seed)
  {:ok, _, _} = Env.observe(sim)
  {:ok, gs0} = Env.frames(sim)
  starts = for i <- 0..(n - 1), into: %{}, do: ({:ok, blob} = Env.save(sim, i); {i, blob})

  {histories, open, _} =
    Enum.reduce(1..frames_per, {[], List.duplicate([], n), gs0}, fn _t, {done, open, states} ->
      {:ok, c1s} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
      c2s =
        case opponent do
          :idle -> List.duplicate(neutral, n)
          agent ->
            {:ok, cs} = Agent.batch_get_controllers(agent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
            cs
        end
      {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || neutral, b || neutral] end))
      # history entry: the state the policy saw and what it pressed there
      open = Enum.zip_with([open, states, c1s], fn [h, s, c] -> [{s, c || neutral} | h] end)
      finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

      if finished == [] do
        {done, open, nexts}
      else
        for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
        :ok = Agent.batch_reset_rows(actor, finished)
        if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, finished)
        {:ok, reset_states} = Env.frames(sim)
        closed = for i <- finished, do: Enum.at(open, i)
        open = open |> Enum.with_index() |> Enum.map(fn {h, i} -> if i in finished, do: [], else: h end)
        {closed ++ done, open, reset_states}
      end
    end)

  Env.stop(sim)
  lists = (open ++ histories) |> Enum.reject(&(&1 == [])) |> Enum.flat_map(relabel)
  Output.puts("  seed #{seed}: #{length(lists)} offstage runs, #{lists |> Enum.map(&length/1) |> Enum.sum()} relabeled frames")
  lists
end

frame_lists = Enum.flat_map(seeds, rollout)
all = frame_lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true))
total = length(all)
gated = frame_lists |> List.flatten() |> Enum.count(&(&1[:gated] == true))
ctx_total = (frame_lists |> List.flatten() |> length()) - total - gated

# what the relabel changed: the policy was silent, the expert says act
silent_actual = Enum.count(all, &SilentFallWeighting.neutral?(&1.actual))
# hold share as the event heads will read it (label vs prev): expert corpus ~0.76
same_action = fn a, b -> ExPhil.Training.Data.controller_to_action(a, axis_buckets: 16) == ExPhil.Training.Data.controller_to_action(b, axis_buckets: 16) end
hold_share = Enum.count(all, &same_action.(&1.controller, &1.prev_controller)) / max(total, 1)
label_b = Enum.count(all, & &1.controller.button_b)
label_jump = Enum.count(all, &(&1.controller.button_x or &1.controller.button_y))
runs_len = frame_lists |> Enum.map(&length/1) |> Enum.sort()
median = fn l -> if l == [], do: nil, else: Enum.at(l, div(length(l), 2)) end

File.mkdir_p!(Path.dirname(out))
File.write!(out, :erlang.term_to_binary(%{
  expert: "expert_recovery_labeler_knn",
  index: opts[:index] || "data/silent_fall/expert_recovery_index.bin",
  exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  policy: policy,
  action_delay: action_delay,
  label_convention: ExPhil.Data.LabelConvention.current(),
  keep_actual: keep_actual,
  frame_lists: if(keep_actual, do: frame_lists, else: Enum.map(frame_lists, fn l -> Enum.map(l, &Map.delete(&1, :actual)) end))
}, [:compressed]))

if report = opts[:report] do
  File.mkdir_p!(Path.dirname(report))
  File.write!(report, Jason.encode!(%{
    "policy" => policy, "seeds" => seeds, "runs" => length(frame_lists), "frames" => total,
    "max_d2" => max_d2, "gated_frames" => gated, "gated_share" => Float.round(gated / max(total + gated, 1), 3),
    "policy_silent_share" => Float.round(silent_actual / max(total, 1), 3),
    "hold_share" => Float.round(hold_share, 3),
    "label_b_share" => Float.round(label_b / max(total, 1), 3), "label_jump_share" => Float.round(label_jump / max(total, 1), 3),
    "run_len_median" => median.(runs_len)
  }, pretty: true))
end

Output.puts("RESULT sim dagger: #{length(frame_lists)} offstage runs (median #{median.(runs_len)} f incl. prefix), #{total} relabeled frames + #{ctx_total} input-only context" <>
  if(max_d2, do: " + #{gated} gated (d2 > #{max_d2}, #{Float.round(100 * gated / max(total + gated, 1), 1)} % of labelable)", else: "") <> "; " <>
  "policy was silent on #{Float.round(100 * silent_actual / max(total, 1), 1)} %; expert label: B #{Float.round(100 * label_b / max(total, 1), 1)} %, jump #{Float.round(100 * label_jump / max(total, 1), 1)} %; hold share (label vs prev) #{Float.round(hold_share, 3)} -> #{out}")
