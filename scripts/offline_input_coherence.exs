# Offline input-coherence scoreboard (2026-10-01).
#
# Feeds held-out EXPERT game states frame by frame through the live Agent API
# (sampled exactly as in play; a prev-action policy feeds back its OWN output)
# and compares the emitted controller stream with the expert's on the same
# frames. No Dolphin; minutes per policy. Answers:
#
#   flicker   — press edges per minute, model vs expert, per button; and A
#               press-edges inside expert jab-1 episodes (experts: ~0)
#   copying   — on frames where the expert's buttons CHANGE, how often the
#               model's buttons equal the expert's new state (change recall),
#               vs on frames where the expert holds (hold agreement)
#   freezing  — share of frames repeating the model's previous output, and
#               share fully neutral
#
#   mix run scripts/offline_input_coherence.exs --policy P --label L \
#     [--games 3] [--stateful-step] [--out FILE.json] [--split SPLIT.json]
#
# Games come from the "validation" list of the split (default: the Fox Mamba
# split), so no policy trained on them.
alias ExPhil.{Agents.Agent, Data.Peppi, Training.Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, games: :integer, stateful_step: :boolean, out: :string,
             split: :string, ablate_prev_action: :boolean, temperature: :float])

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
split = opts[:split] || "checkpoints/fox_mamba_v1_20260925/split.json"
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation") |> Enum.take(opts[:games] || 3)

buttons = [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]
bt = fn c -> Enum.map(buttons, &(Map.get(c, &1) == true)) end
neutral? = fn c ->
  not Enum.any?(bt.(c)) and abs(c.main_stick.x - 0.5) < 0.1 and abs(c.main_stick.y - 0.5) < 0.1
end
key = fn c -> {bt.(c), Float.round(c.main_stick.x * 1.0, 2), Float.round(c.main_stick.y * 1.0, 2),
               Float.round(c.c_stick.x * 1.0, 2), Float.round(c.c_stick.y * 1.0, 2)} end

Output.banner("Offline input coherence: #{label}")
{:ok, agent} =
  Agent.start_link(policy_path: policy, deterministic: false, temperature: opts[:temperature] || 1.0,
    af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0,
    stateful_step: opts[:stateful_step] || false, ablate_prev_action: opts[:ablate_prev_action] || false)
{:ok, _} = Agent.warmup(agent)

games =
  for path <- files do
    {:ok, meta} = Peppi.metadata(path)
    own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    opp = Enum.find(meta.players, &(&1.port != own.port))
    {:ok, replay} = Peppi.parse(path, player_port: own.port)
    frames =
      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
    Agent.reset_buffer(agent)
    outs = Enum.map(frames, fn f ->
      {:ok, c} = Agent.get_controller(agent, f.game_state, player_port: own.port)
      c
    end)
    Output.puts("  #{Path.basename(path)}: #{length(frames)} frames")
    {frames, outs, own.port}
  end

# ---- aggregate ---------------------------------------------------------------
edges = fn seq, idx ->
  seq |> Enum.map(&Enum.at(&1, idx)) |> Enum.chunk_every(2, 1, :discard) |> Enum.count(fn [a, b] -> b and not a end)
end

acc =
  Enum.reduce(games, %{n: 0, exp_edges: List.duplicate(0, 7), mod_edges: List.duplicate(0, 7), change: 0, change_hit: 0,
                       hold: 0, hold_hit: 0, same_prev: 0, neutral: 0, exp_same_prev: 0, exp_neutral: 0,
                       jab_eps: 0, jab_model_a_edges: 0, jab_exp_a_edges: 0}, fn {frames, outs, port}, a ->
    exp = Enum.map(frames, &bt.(&1.controller))
    mod = Enum.map(outs, bt)
    n = length(frames)
    pairs = Enum.zip([exp, tl(exp), tl(mod)])
    {ch, chh, ho, hoh} =
      Enum.reduce(pairs, {0, 0, 0, 0}, fn {e0, e1, m1}, {ch, chh, ho, hoh} ->
        if e1 != e0, do: {ch + 1, chh + if(m1 == e1, do: 1, else: 0), ho, hoh},
          else: {ch, chh, ho + 1, hoh + if(m1 == e1, do: 1, else: 0)}
      end)
    mk = Enum.map(outs, key)
    ek = Enum.map(frames, &key.(&1.controller))
    same = fn ks -> Enum.zip(ks, tl(ks)) |> Enum.count(fn {x, y} -> x == y end) end
    # expert jab-1 episodes: runs of action 44 for the subject
    acts = Enum.map(frames, &(&1.game_state.players[port].action || 0))
    idx = Enum.with_index(acts)
    eps = idx |> Enum.chunk_by(fn {x, _} -> x == 44 end) |> Enum.filter(fn [{x, _} | _] -> x == 44 end)
    a_edges_in = fn seq ->
      t = List.to_tuple(Enum.map(seq, &hd/1))
      Enum.sum(Enum.map(eps, fn ep ->
        Enum.count(ep, fn {_, i} -> i > 0 and elem(t, i) and not elem(t, i - 1) end)
      end))
    end
    %{a |
      n: a.n + n,
      exp_edges: Enum.zip_with(a.exp_edges, Enum.map(0..6, &edges.(exp, &1)), &+/2),
      mod_edges: Enum.zip_with(a.mod_edges, Enum.map(0..6, &edges.(mod, &1)), &+/2),
      change: a.change + ch, change_hit: a.change_hit + chh, hold: a.hold + ho, hold_hit: a.hold_hit + hoh,
      same_prev: a.same_prev + same.(mk), exp_same_prev: a.exp_same_prev + same.(ek),
      neutral: a.neutral + Enum.count(outs, neutral?), exp_neutral: a.exp_neutral + Enum.count(frames, &neutral?.(&1.controller)),
      jab_eps: a.jab_eps + length(eps), jab_model_a_edges: a.jab_model_a_edges + a_edges_in.(mod),
      jab_exp_a_edges: a.jab_exp_a_edges + a_edges_in.(exp)}
  end)

# ---- change events with a timing window (2026-10-02) ---------------------------
# Same-frame change recall is harsh on a stochastic expert: a press one frame
# early scores as a miss. Here every input CHANGE is an event — press(button),
# release(button), or the main stick entering a new zone (neutral + 8
# directions) — and an expert event at frame t is recalled at tolerance k if
# the model has the SAME event within t±k. Precision is the mirror: model
# events that some expert event explains.
zone = fn c ->
  dx = c.main_stick.x - 0.5
  dy = c.main_stick.y - 0.5
  if :math.sqrt(dx * dx + dy * dy) < 0.14, do: :n, else: round(:math.atan2(dy, dx) / (:math.pi() / 4))
end
events = fn ctrls ->
  ctrls
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {[p, q], t} ->
    bs = Enum.zip([buttons, bt.(p), bt.(q)]) |> Enum.flat_map(fn
      {b, false, true} -> [{{:press, b}, t}]
      {b, true, false} -> [{{:release, b}, t}]
      _ -> []
    end)
    zq = zone.(q)
    if zone.(p) != zq, do: [{{:stick, zq}, t} | bs], else: bs
  end)
end
kind = fn {{k, _}, _} -> k end
match = fn evs, other, k ->
  set = MapSet.new(other)
  Enum.group_by(evs, kind) |> Map.new(fn {kd, es} ->
    {kd, {Enum.count(es, fn {ty, t} -> Enum.any?(-k..k, &MapSet.member?(set, {ty, t + &1})) end), length(es)}}
  end)
end
window_scores =
  for k <- [0, 2, 5], into: %{} do
    {rec, prec} =
      Enum.reduce(games, {%{}, %{}}, fn {frames, outs, _port}, {ra, pa} ->
        e = events.(Enum.map(frames, & &1.controller))
        m = events.(outs)
        add = fn a, b -> Map.merge(a, b, fn _, {x1, y1}, {x2, y2} -> {x1 + x2, y1 + y2} end) end
        {add.(ra, match.(e, m, k)), add.(pa, match.(m, e, k))}
      end)
    rr = fn m, kd -> case m[kd] do {h, n} when n > 0 -> Float.round(h / n, 3); _ -> nil end end
    {"k#{k}", Map.new([:press, :release, :stick], fn kd -> {kd, %{recall: rr.(rec, kd), precision: rr.(prec, kd)}} end)}
  end

minutes = acc.n / 3600
r = fn x, d -> if d == 0, do: nil, else: Float.round(x / d, 3) end
names = ~w(A B X Y Z L R)
result = %{
  label: label, policy: policy, frames: acc.n, games: length(games),
  press_edges_per_min: Map.new(Enum.zip(names, Enum.zip_with(acc.mod_edges, acc.exp_edges, fn m, e ->
    %{model: Float.round(m / minutes, 2), expert: Float.round(e / minutes, 2), ratio: r.(m, e)} end))),
  change_recall: r.(acc.change_hit, acc.change), hold_agreement: r.(acc.hold_hit, acc.hold),
  change_frames: acc.change,
  change_events: window_scores,
  repeat_prev: %{model: r.(acc.same_prev, acc.n), expert: r.(acc.exp_same_prev, acc.n)},
  neutral_share: %{model: r.(acc.neutral, acc.n), expert: r.(acc.exp_neutral, acc.n)},
  jab1: %{episodes: acc.jab_eps, model_a_edges_per_ep: r.(acc.jab_model_a_edges, acc.jab_eps),
          expert_a_edges_per_ep: r.(acc.jab_exp_a_edges, acc.jab_eps)}
}

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(result, pretty: true))
end

Output.puts("RESULT #{label}: frames=#{acc.n} change_recall=#{result.change_recall} hold_agreement=#{result.hold_agreement} " <>
  "repeat_prev=#{result.repeat_prev.model} (expert #{result.repeat_prev.expert}) neutral=#{result.neutral_share.model} (expert #{result.neutral_share.expert})")
Output.puts("RESULT #{label} press edges/min model:expert  " <>
  Enum.map_join(names, "  ", fn b -> e = result.press_edges_per_min[b]; "#{b} #{e.model}:#{e.expert}" end))
Output.puts("RESULT #{label} jab1 episodes=#{acc.jab_eps} A-edges/episode model=#{result.jab1.model_a_edges_per_ep} expert=#{result.jab1.expert_a_edges_per_ep}")
Output.puts("RESULT #{label} change events recall/precision  " <>
  Enum.map_join(["k0", "k2", "k5"], "  |  ", fn k ->
    "±#{String.trim_leading(k, "k")}: " <> Enum.map_join([:press, :release, :stick], " ", fn kd ->
      v = window_scores[k][kd]; "#{kd} #{inspect(v.recall)}/#{inspect(v.precision)}" end) end))
