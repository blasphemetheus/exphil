# G7 of INTERP_GEN_V1: blind input audit — which conditioning inputs does the
# policy actually READ?
#
# Perturbs one input family at a time (opponent character, stage, percents,
# stocks) and measures per-head output divergence vs baseline. Near-zero
# divergence = dead channel (a dead opponent-character channel would doom
# matchup work); large = load-bearing. opp_x_far is the POSITIVE CONTROL: the
# model demonstrably approaches and converts, so it MUST read opponent
# position — if opp_x_far scores ~0 the audit itself is broken, not the model.
#
#   mix run scripts/interp_blind_audit.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays replays/erickfm_ranked/FOX/extracted --max-files 5
#
# Options: --max-files N [5]  --sample-stride N [7]  --batch-size N [128]
#          --port N [1]  --out PATH [eval_runs/0826_gen_v1_sweep/blind_audit.txt]
#
# NO-MIX LAW: run only with no other live beam. Mirrors the G1 script's
# model-reconstruction (do NOT route through Activations.load_heads — it reads
# :hidden_size not :hidden_sizes and would build the wrong GRU).

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Networks.Policy
alias ExPhil.Training.{Checkpoint, Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, replays: :string, max_files: :integer,
      sample_stride: :integer, batch_size: :integer, port: :integer, out: :string
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
replay_dir = opts[:replays] || raise "--replays required"
max_files = opts[:max_files] || 5
sample_stride = opts[:sample_stride] || 7
batch_size = opts[:batch_size] || 128
port = opts[:port] || 1
opp_port = 3 - port
out_path = opts[:out] || "eval_runs/0826_gen_v1_sweep/blind_audit.txt"

Output.banner("Blind input audit (INTERP_GEN_V1 G7)")

{:ok, export} = Checkpoint.load_policy(policy_path)
config = export.config
params = export.params
window = config[:window_size] || 60

embed_config = ExPhil.Embeddings.config(Map.to_list(config))
embed_size = ExPhil.Embeddings.embedding_size(embed_config)
Output.puts("policy: #{Path.basename(policy_path)} (embed #{embed_size}, window #{window})")

policy_model =
  Policy.build_temporal(
    embed_size: embed_size,
    backbone: String.to_atom(to_string(config[:backbone] || "gru")),
    hidden_size: config[:hidden_size] || 512,
    num_layers: config[:num_layers] || 2,
    num_heads: config[:num_heads] || 4,
    head_dim: config[:head_dim] || 64,
    window_size: window,
    state_size: config[:state_size] || 16,
    expand_factor: config[:expand_factor] || 2,
    conv_size: config[:conv_size] || 4,
    dropout: 0.0,
    axis_buckets: config[:axis_buckets] || 16,
    shoulder_buckets: config[:shoulder_buckets] || 4
  )

{_init, predict_fn} = Axon.build(policy_model, compiler: EXLA)

files = Path.wildcard(Path.join(replay_dir, "*.slp")) |> Enum.take(max_files)
Output.puts("#{length(files)} files, stride #{sample_stride}")

# --- Perturbations: {name, fn game_state -> game_state'} ---
# Swap-style character/stage perturbations guarantee a real change (a fixed
# target would be a no-op on frames already at that value and understate
# sensitivity). Fox=2, Marth=9; BF=31, FD=32.
update_player = fn gs, p, f ->
  case Map.fetch(gs.players, p) do
    {:ok, player} -> %{gs | players: Map.put(gs.players, p, f.(player))}
    :error -> gs
  end
end

update_own = fn gs, f -> update_player.(gs, port, f) end
update_opp = fn gs, f -> update_player.(gs, opp_port, f) end

perturbations = [
  {:opp_x_far, fn gs -> update_opp.(gs, &%{&1 | x: &1.x + 120.0}) end},
  {:opp_char_swap, fn gs -> update_opp.(gs, &%{&1 | character: if(&1.character == 2, do: 9, else: 2)}) end},
  {:stage_swap, fn gs -> %{gs | stage: if(gs.stage == 31, do: 32, else: 31)} end},
  {:own_percent_hi, fn gs -> update_own.(gs, &%{&1 | percent: 150.0}) end},
  {:opp_percent_hi, fn gs -> update_opp.(gs, &%{&1 | percent: 150.0}) end},
  {:own_stock_last, fn gs -> update_own.(gs, &%{&1 | stock: 1}) end},
  {:opp_stock_last, fn gs -> update_opp.(gs, &%{&1 | stock: 1}) end}
]

heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

# Logits for a set of frames → %{head => {n,k} tensor} over sampled windows.
# Same machinery as interp_entropy_map.exs (teacher-forced windows, fresh
# embed — the perturbed variants must NOT hit the embedding cache).
logits_for = fn frames ->
  n = length(frames)
  dataset = Data.from_frames(frames, embed_config: embed_config)
  embedded = Data.precompute_frame_embeddings(dataset, show_progress: false)
  flat = embedded.embedded_frames

  chunks =
    Enum.to_list(window..(n - 1)//sample_stride)
    |> Enum.chunk_every(batch_size)
    |> Enum.map(fn idxs ->
      batch =
        idxs
        |> Enum.map(fn i -> Nx.slice(flat, [i - window + 1, 0], [window, embed_size]) end)
        |> Nx.stack()

      predict_fn.(params, %{"state_sequence" => batch})
    end)

  heads
  |> Enum.with_index()
  |> Map.new(fn {h, i} -> {h, Nx.concatenate(Enum.map(chunks, fn ch -> elem(ch, i) end))} end)
end

# --- Collect baseline + variant logits across files ---
# collected: %{variant_name => %{head => [tensor, ...]}} (per-file lists)
empty_collected = Map.new([:baseline | Enum.map(perturbations, &elem(&1, 0))], &{&1, %{}})

collected =
  Enum.reduce(Enum.with_index(files, 1), empty_collected, fn {file, fidx}, acc ->
    IO.write(:stderr, "\r  file #{fidx}/#{length(files)}\e[K")

    with {:ok, replay} <- Peppi.parse(file) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port)
        |> Enum.reject(&(&1.game_state.frame < 0))

      if length(frames) <= window do
        acc
      else
        variants =
          [{:baseline, frames} | Enum.map(perturbations, fn {name, pf} ->
            {name, Enum.map(frames, fn f -> %{f | game_state: pf.(f.game_state)} end)}
          end)]

        Enum.reduce(variants, acc, fn {name, fs}, acc2 ->
          lm = logits_for.(fs)

          Map.update!(acc2, name, fn head_lists ->
            Map.new(heads, fn h -> {h, [lm[h] | Map.get(head_lists, h, [])]} end)
          end)
        end)
      end
    else
      _ -> acc
    end
  end)

IO.write(:stderr, "\n")

# Concatenate per-file tensors → %{variant => %{head => {n,k}}}
finalized =
  Map.new(collected, fn {name, head_lists} ->
    {name, Map.new(head_lists, fn {h, ts} -> {h, Nx.concatenate(Enum.reverse(ts))} end)}
  end)

base = finalized[:baseline]
n_windows = base[:buttons] |> Nx.axis_size(0)
Output.puts("windows sampled: #{n_windows}")

# --- Divergence vs baseline ---
delta_fn = fn b0, b1 -> Nx.mean(Nx.abs(Nx.subtract(b0, b1))) |> Nx.to_number() end

flip_fn = fn h, b0, b1 ->
  if h == :buttons do
    Nx.mean(Nx.not_equal(Nx.greater(b0, 0.0), Nx.greater(b1, 0.0))) |> Nx.to_number()
  else
    Nx.mean(Nx.not_equal(Nx.argmax(b0, axis: -1), Nx.argmax(b1, axis: -1))) |> Nx.to_number()
  end
end

results =
  Enum.map(perturbations, fn {name, _pf} ->
    var = finalized[name]
    per_head = Map.new(heads, &{&1, delta_fn.(base[&1], var[&1])})
    flips = Map.new(heads, &{&1, flip_fn.(&1, base[&1], var[&1])})
    {name, per_head, flips}
  end)

control_delta = results |> Enum.find_value(fn {n, ph, _} -> n == :opp_x_far && Enum.sum(Map.values(ph)) end)

# --- Output ---
lines =
  [
    "policy: #{Path.basename(policy_path)}  files: #{length(files)}  stride: #{sample_stride}  windows: #{n_windows}",
    "per-head mean |logit delta| vs baseline (nats of logit):",
    String.pad_trailing("perturbation", 16) <>
      Enum.map_join(heads, "", &String.pad_leading(to_string(&1), 10)) <>
      String.pad_leading("sum", 10) <>
      String.pad_leading("ratio", 9) <>
      String.pad_leading("flip%", 9)
  ] ++ Enum.map(results, fn {name, ph, flips} ->
        sum = ph |> Map.values() |> Enum.sum()

        ratio =
          if control_delta && control_delta > 0 do
            :erlang.float_to_binary(sum / control_delta, decimals: 2)
          else
            "-"
          end

        mean_flip = flips |> Map.values() |> Enum.sum() |> Kernel./(length(heads))

        String.pad_trailing(to_string(name), 16) <>
          Enum.map_join(heads, "", fn h ->
            String.pad_leading(:erlang.float_to_binary(ph[h], decimals: 4), 10)
          end) <>
          String.pad_leading(:erlang.float_to_binary(sum, decimals: 4), 10) <>
          String.pad_leading(ratio, 9) <>
          String.pad_leading(:erlang.float_to_binary(mean_flip * 100, decimals: 1), 9)
      end) ++ [
    "",
    "ratio = perturbation sum / opp_x_far (positive-control) sum. A ratio ~1 = as",
    "load-bearing as opponent position; ~0.0x = dead channel. flip% = mean fraction",
    "of frames whose sampled action changed (buttons: on/off toggle; sticks: argmax",
    "bucket change), averaged over the six heads."
  ]

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Enum.join(lines, "\n") <> "\n")
Enum.each(lines, &Output.puts/1)
Output.success("written to #{out_path}")
