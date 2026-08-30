# Why does mode-of-N walk off the stage and never recover? — the interp test.
#
# Hypothesis: button presses are SPARSE IN TIME (a jump or B press lasts 1–2
# frames out of many), so at any single frame every button's marginal
# probability is < 0.5, and the MODE of the joint per-frame distribution is
# "stick held, no buttons". Held stick + no jump = walk/run off the edge;
# no B press offstage = no up-B/side-B = death. Sampling escapes this because
# a 20%-per-frame press fires within a few frames; the mode never fires it.
#
# Test: at expert frames in the states that matter (standing/neutral on
# stage; cornered near the edge; offstage / recovery_low / being_edgeguarded),
# run one forward and report per-frame button marginals (A, B, X, Y, Z, L, R,
# d-up) at the deploy buttons T=0.5, the fraction of frames where ANY button
# marginal exceeds 0.5 (the only way a majority vote can ever press it), the
# modal main-stick bucket, and — empirically — the majority vote of 16 draws:
# how often it contains a jump (X/Y) or B at all.
#
#   mix run scripts/interp_mode_mechanism.exs --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --limit-files 20 \
#     --frames-per-label 150 --out eval_runs/0830_mode_mechanism/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, port: :integer, limit_files: :integer,
             frames_per_label: :integer, n: :integer, out: :string]
  )

policy_path = opts[:policy] || raise("--policy required")
glob = opts[:replays] || raise("--replays required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
limit_files = opts[:limit_files] || 20
per_label = opts[:frames_per_label] || 150
n_draw = opts[:n] || 16

labels = [:neutral, :cornered, :edge_danger, :offstage, :recovery_low, :recovery_high, :being_edgeguarded]
files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files)

Output.banner("Mode-of-N mechanism — button marginals by situation")
Output.config([{"Policy", Path.basename(policy_path)}, {"Files", length(files)}, {"Frames / label", per_label}, {"Draws", n_draw}])

{by_label, frames_by_path} =
  Enum.reduce(files, {%{}, %{}}, fn path, {bl, bp} ->
    case Peppi.parse(Path.expand(path)) do
      {:ok, replay} ->
        frames = replay |> Peppi.to_training_frames(player_port: port, opponent_port: opp) |> Enum.reject(&(&1.game_state.frame < 0))
        sits = Situations.label_states(Enum.map(frames, & &1.game_state), port, as: :set) |> List.to_tuple()
        arr = List.to_tuple(frames)

        bl =
          Enum.reduce(60..(tuple_size(arr) - 1), bl, fn i, acc ->
            Enum.reduce(MapSet.intersection(elem(sits, i), MapSet.new(labels)), acc, fn l, a -> Map.update(a, l, [{path, i}], &[{path, i} | &1]) end)
          end)

        {bl, Map.put(bp, path, arr)}

      _ -> {bl, bp}
    end
  end)

spread = fn list, k -> list = Enum.reverse(list); if length(list) > k, do: list |> Enum.take_every(max(div(length(list), k), 1)) |> Enum.take(k), else: list end

{:ok, agent} = Agent.start_link(policy_path: policy_path, deterministic: false, temperature: %{buttons: 0.5, main: 0.5, c: 0.5, shoulder: 0.5}, delay_id: 0)
Agent.warmup(agent)
cfg = Agent.get_config(agent)
window = if cfg.temporal, do: cfg.window_size || 60, else: 0

names = ~w(A B X Y Z L R dup)
rng = Nx.Random.key(830)

rows =
  Enum.map(labels, fn label ->
    sel = spread.(Map.get(by_label, label, []), per_label)
    Output.puts("#{label}: #{length(sel)} frames")

    {res, _} =
      Enum.map_reduce(sel, rng, fn {path, i}, key ->
        arr = frames_by_path[path]
        Agent.reset_buffer(agent)
        for h <- max(i - window, 0)..(i - 1)//1, do: Agent.get_controller(agent, elem(arr, h).game_state, player_port: port)

        case Agent.get_action_with_confidence(agent, elem(arr, i).game_state, player_port: port) do
          {:ok, action, _} ->
            probs = action.logits.buttons |> Nx.squeeze() |> Nx.divide(0.5) |> Nx.sigmoid() |> Nx.to_flat_list()
            mx = action.logits.main_x |> Nx.squeeze() |> Nx.argmax() |> Nx.to_number()
            my = action.logits.main_y |> Nx.squeeze() |> Nx.argmax() |> Nx.to_number()
            # 16 joint draws of buttons (Bernoulli at T=0.5) + sticks; majority vote on the joint key
            {u, key} = Nx.Random.uniform(key, shape: {n_draw, 8})
            draws = Nx.less(u, Nx.tensor(probs) |> Nx.new_axis(0)) |> Nx.as_type(:u8) |> Nx.to_list()
            sx = action.logits.main_x |> Nx.squeeze() |> Nx.divide(0.5)
            {g, key} = Nx.Random.uniform(key, shape: {n_draw, Nx.size(sx)})
            picks = Nx.argmax(Nx.add(Nx.new_axis(sx, 0), Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(g, 1.0e-10)))))), axis: 1) |> Nx.to_flat_list()
            keys = Enum.zip(draws, picks)
            counts = Enum.frequencies(keys)
            {mode_btn, _} = keys |> Enum.with_index() |> Enum.max_by(fn {k, _} -> counts[k] end) |> elem(0)
            expert_btn = elem(arr, i).controller
            {%{probs: probs, mx: mx, my: my, mode_btn: mode_btn, expert: expert_btn}, key}

          _ -> {nil, key}
        end
      end)

    res = Enum.reject(res, &is_nil/1)
    n = max(length(res), 1)
    marg = Enum.map(0..7, fn b -> Enum.sum(Enum.map(res, &Enum.at(&1.probs, b))) / n end)
    any_over_half = Enum.count(res, fn r -> Enum.any?(r.probs, &(&1 > 0.5)) end) / n
    mode_jump = Enum.count(res, fn r -> Enum.at(r.mode_btn, 2) == 1 or Enum.at(r.mode_btn, 3) == 1 end) / n
    mode_b = Enum.count(res, fn r -> Enum.at(r.mode_btn, 1) == 1 end) / n
    mode_none = Enum.count(res, fn r -> Enum.sum(r.mode_btn) == 0 end) / n
    exp_jump = Enum.count(res, fn r -> r.expert.button_x or r.expert.button_y end) / n
    exp_b = Enum.count(res, fn r -> r.expert.button_b end) / n
    exp_none = Enum.count(res, fn r -> not (r.expert.button_a or r.expert.button_b or r.expert.button_x or r.expert.button_y or r.expert.button_z or r.expert.button_l or r.expert.button_r) end) / n
    mode_stick = Enum.frequencies_by(res, fn r -> {r.mx, r.my} end) |> Enum.max_by(&elem(&1, 1), fn -> {{8, 8}, 0} end)
    %{label: label, n: length(res), marg: marg, any_over_half: any_over_half, mode_jump: mode_jump, mode_b: mode_b, mode_none: mode_none,
      exp_jump: exp_jump, exp_b: exp_b, exp_none: exp_none, mode_stick: mode_stick}
  end)

pct = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end

table =
  "| situation | n | " <> Enum.map_join(names, " | ", &"P(#{&1})") <> " | frames with any button > 0.5 | mode has jump | mode has B | mode presses NOTHING | expert pressing jump / B / nothing (this frame) | modal stick (x,y bucket of 17; 8=center) |\n|---|---:|" <> String.duplicate("---:|", 8) <> "---:|---:|---:|---:|---|---|\n" <>
    Enum.map_join(rows, "\n", fn r ->
      {{mx, my}, c} = r.mode_stick
      "| #{r.label} | #{r.n} | " <> Enum.map_join(r.marg, " | ", &pct.(&1)) <> " | #{pct.(r.any_over_half)} | #{pct.(r.mode_jump)} | #{pct.(r.mode_b)} | **#{pct.(r.mode_none)}** | #{pct.(r.exp_jump)} / #{pct.(r.exp_b)} / #{pct.(r.exp_none)} | (#{mx},#{my}) in #{pct.(c / max(r.n, 1))}% |"
    end)

report = """
# Mode-of-N mechanism — per-frame button marginals by situation

Policy `#{Path.basename(policy_path)}`, #{length(files)} expert files, port #{port}; buttons at T=0.5, sticks T=0.5,
#{n_draw} joint draws for the majority vote. Marginals are the policy's per-frame press probabilities at
the expert's states. A majority vote over joint draws can only press a button when its marginal is
> 0.5 on that frame; "mode presses NOTHING" is the empirical vote outcome.

#{table}

Read: if every button marginal sits well under 0.5 in offstage / recovery states while the expert
presses B or jump on 20–40% of those frames, the mode is structurally unable to recover — the
press is a 1–2-frame event whose per-frame probability never crosses one half. Sampling fires it
within a few frames; the vote never does. The same in neutral explains the walk-off: modal stick
held, no jump.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
