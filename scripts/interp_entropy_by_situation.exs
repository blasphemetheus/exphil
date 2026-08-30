# B3 — policy entropy per situation vs the expert's diversity (EVAL_DIRECTIONS B3).
#
# For expert frames grouped by situation label, give the policy the same
# 60-frame history, run one forward, and read the heads: per-head entropy
# (bits) at T=1.0 — buttons as 8 independent Bernoullis, sticks/shoulder as
# categoricals — plus the entropy at the deploy decode (T=0.5 sticks,
# buttons 0.5). Next to it: the EXPERT's option-histogram entropy in the same
# situation (from Options events, the A1 table). Low policy entropy where
# the expert is diverse = an over-confident state = a loop waiting to happen;
# high policy entropy where the expert is decisive = dithering.
#
#   mix run scripts/interp_entropy_by_situation.exs --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 --limit-files 20 \
#     --frames-per-label 80 --out eval_runs/0829_entropy/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Options
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, port: :integer, limit_files: :integer,
             frames_per_label: :integer, labels: :string, temperature: :float, out: :string]
  )

policy_path = opts[:policy] || raise("--policy required")
glob = opts[:replays] || raise("--replays required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
limit_files = opts[:limit_files] || 20
per_label = opts[:frames_per_label] || 80
temperature = opts[:temperature] || 0.5

default_labels = ~w(neutral approach retreat advantage disadvantage conversion_open combo_active
  tech_chase edgeguard pummel_throw_decision shield_pressure_theirs being_edgeguarded
  recovery_low recovery_high cornered offstage respawn_invincible percent_lead percent_deficit)a

labels =
  case opts[:labels] do
    nil -> default_labels
    s -> s |> String.split(",") |> Enum.map(&String.to_atom(String.trim(&1)))
  end

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files)
if files == [], do: raise("no replays matched")

Output.banner("Entropy by situation (B3)")
Output.config([{"Policy", Path.basename(policy_path)}, {"Replays", "#{length(files)} files"},
               {"Labels", length(labels)}, {"Frames / label", per_label}, {"Deploy T", temperature}])

# ---- collect frames per label + expert option histograms per label -------
{by_label, expert_hist, frames_by_path} =
  Enum.reduce(files, {%{}, %{}, %{}}, fn path, {bl, eh, bp} ->
    case Peppi.parse(Path.expand(path)) do
      {:ok, replay} ->
        frames =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
          |> Enum.reject(&(&1.game_state.frame < 0))

        states = Enum.map(frames, & &1.game_state)
        sits = Situations.label_states(states, port, as: :set) |> List.to_tuple()
        events = Options.events(states, port)
        arr = List.to_tuple(frames)
        n = tuple_size(arr)

        bl =
          Enum.reduce(60..(n - 1), bl, fn i, acc ->
            Enum.reduce(MapSet.intersection(elem(sits, i), MapSet.new(labels)), acc, fn l, a ->
              Map.update(a, l, [{path, i}], &[{path, i} | &1])
            end)
          end)

        eh =
          Enum.reduce(events, eh, fn %{index: i, option: o}, acc ->
            ctx = elem(sits, max(i - 1, 0))
            Enum.reduce(MapSet.intersection(ctx, MapSet.new(labels)), acc, fn l, a ->
              Map.update(a, l, %{o => 1}, fn h -> Map.update(h, o, 1, &(&1 + 1)) end)
            end)
          end)

        {bl, eh, Map.put(bp, path, arr)}

      _ ->
        {bl, eh, bp}
    end
  end)

spread = fn list, k ->
  list = Enum.reverse(list)
  if length(list) > k, do: list |> Enum.take_every(max(div(length(list), k), 1)) |> Enum.take(k), else: list
end

{:ok, agent} =
  Agent.start_link(policy_path: policy_path, deterministic: false,
    temperature: %{buttons: 0.5, main: temperature, c: temperature, shoulder: temperature}, delay_id: 0)

Agent.warmup(agent)
cfg = Agent.get_config(agent)
window = if cfg.temporal, do: cfg.window_size || 60, else: 0

log2 = fn x -> if x <= 0.0, do: 0.0, else: :math.log(x) / :math.log(2) end
h_cat = fn probs -> -Enum.sum(Enum.map(probs, fn p -> p * log2.(p) end)) end
softmax = fn logits, t -> l = logits |> Nx.squeeze() |> Nx.divide(t); l |> Nx.subtract(Nx.reduce_max(l)) |> Nx.exp() |> then(&Nx.divide(&1, Nx.sum(&1))) |> Nx.to_flat_list() end
h_bern = fn logits, t -> logits |> Nx.squeeze() |> Nx.divide(t) |> Nx.sigmoid() |> Nx.to_flat_list() |> Enum.map(fn p -> -(p * log2.(p) + (1 - p) * log2.(1 - p)) end) |> Enum.sum() end

entropies = fn action, t ->
  %{
    buttons: h_bern.(action.logits.buttons, t),
    main: (h_cat.(softmax.(action.logits.main_x, t)) + h_cat.(softmax.(action.logits.main_y, t))) / 2,
    c: (h_cat.(softmax.(action.logits.c_x, t)) + h_cat.(softmax.(action.logits.c_y, t))) / 2,
    shoulder: h_cat.(softmax.(action.logits.shoulder, t))
  }
end

rows =
  Enum.map(labels, fn label ->
    sel = spread.(Map.get(by_label, label, []), per_label)
    Output.puts("#{label}: #{length(sel)} frames")

    hs =
      Enum.map(sel, fn {path, i} ->
        arr = frames_by_path[path]
        Agent.reset_buffer(agent)
        for h <- max(i - window, 0)..(i - 1)//1, do: Agent.get_controller(agent, elem(arr, h).game_state, player_port: port)

        case Agent.get_action_with_confidence(agent, elem(arr, i).game_state, player_port: port) do
          {:ok, action, _} -> {entropies.(action, 1.0), entropies.(action, temperature)}
          _ -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)

    n = length(hs)
    avg = fn which, k -> if n == 0, do: 0.0, else: Enum.sum(Enum.map(hs, fn t -> Map.fetch!(elem(t, which), k) end)) / n end
    eh = Map.get(expert_hist, label, %{})
    et = eh |> Map.values() |> Enum.sum()
    e_ent = if et == 0, do: 0.0, else: h_cat.(Enum.map(Map.values(eh), &(&1 / et)))

    %{label: label, n: n, expert_options: et, expert_entropy: e_ent,
      b1: avg.(0, :buttons), m1: avg.(0, :main), c1: avg.(0, :c), s1: avg.(0, :shoulder),
      b5: avg.(1, :buttons), m5: avg.(1, :main), c5: avg.(1, :c)}
  end)

f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end

table =
  "| situation | frames | expert option-entropy (bits) | policy H buttons T=1 | H main T=1 | H c T=1 | H shoulder T=1 | H buttons @0.5 | H main @#{temperature} | H c @#{temperature} |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n" <>
    Enum.map_join(rows, "\n", fn r ->
      "| #{r.label} | #{r.n} | #{f2.(r.expert_entropy)} (n=#{r.expert_options}) | #{f2.(r.b1)} | #{f2.(r.m1)} | #{f2.(r.c1)} | #{f2.(r.s1)} | #{f2.(r.b5)} | #{f2.(r.m5)} | #{f2.(r.c5)} |"
    end)

report = """
# Entropy by situation (B3)

Policy `#{Path.basename(policy_path)}`, #{length(files)} expert files, port #{port}, #{per_label} frames per label.
Policy entropies in bits from the head logits at the expert's states (T=1 = the learned
distribution; @deploy = what sampling actually draws from). Buttons = sum of 8 Bernoulli
entropies (max 8); main/c = mean of x,y categorical entropies over 17 buckets (max 4.09);
shoulder max 2. Expert option-entropy = entropy of the expert's next-option histogram in
that situation (max ≈ 4.4 over ~21 options) — a different quantity, shown as the
"how diverse is correct play here" reference, comparable across rows not across columns.

#{table}

Read: rows where the policy's main/buttons entropy is LOW while the expert's option
entropy is HIGH are over-confident states (loop candidates). Rows where the policy is
high-entropy while the expert is decisive are dithering candidates.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
