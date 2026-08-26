# G1+G6 of INTERP_GEN_V1: per-head output entropy conditioned on
# ExPhil.Situations labels, over corpus states (teacher-forced windows).
#
#   mix run scripts/interp_entropy_map.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays replays/erickfm_ranked/FOX/extracted --max-files 20
#
# Options: --max-files N [20]  --sample-stride N [7]  --batch-size N [128]
#          --port N [1]  --out PATH [eval_runs/0826_gen_v1_sweep/entropy_map.txt]
#
# Decisions this feeds (INTERP_GEN_V1 G1/G6): global + per-head temperature,
# state-adaptive T schedule, v2 curation targets (high-entropy labels).
# NO-MIX LAW: run only with no other live beam.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Networks.Policy
alias ExPhil.Situations
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
max_files = opts[:max_files] || 20
sample_stride = opts[:sample_stride] || 7
batch_size = opts[:batch_size] || 128
port = opts[:port] || 1
out_path = opts[:out] || "eval_runs/0826_gen_v1_sweep/entropy_map.txt"

Output.banner("Entropy-by-situation map (INTERP_GEN_V1 G1/G6)")

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
Output.puts("#{length(files)} files, sample stride #{sample_stride}")

# Entropies, all in nats. Categorical: softmax over last dim.
cat_entropy = fn logits ->
  p = Axon.Activations.softmax(logits)
  Nx.negate(Nx.sum(Nx.multiply(p, Nx.log(Nx.add(p, 1.0e-9))), axes: [-1]))
end

# Buttons: 8 independent Bernoullis; sum of their entropies.
btn_entropy = fn logits ->
  p = Nx.sigmoid(logits)
  h1 = Nx.multiply(p, Nx.log(Nx.add(p, 1.0e-9)))
  h0 = Nx.multiply(Nx.subtract(1.0, p), Nx.log(Nx.add(Nx.subtract(1.0, p), 1.0e-9)))
  Nx.negate(Nx.sum(Nx.add(h1, h0), axes: [-1]))
end

heads = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

# acc: %{label => %{n: int, sums: %{head => float}}}
acc =
  Enum.reduce(Enum.with_index(files, 1), %{}, fn {file, fidx}, acc ->
    IO.write(:stderr, "\r  file #{fidx}/#{length(files)}\e[K")

    with {:ok, replay} <- Peppi.parse(file) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: 3 - port)
        |> Enum.reject(&(&1.game_state.frame < 0))

      n = length(frames)

      if n <= window do
        acc
      else
        gss = Enum.map(frames, & &1.game_state)

        label_sets =
          gss
          |> Situations.label_states(port)
          |> Enum.map(fn l -> if is_integer(l), do: Situations.from_mask(l), else: l end)
          |> List.to_tuple()

        dataset = Data.from_frames(frames, embed_config: embed_config)
        embedded = Data.precompute_frame_embeddings(dataset, show_progress: false)
        flat = embedded.embedded_frames

        idxs = Enum.to_list(window..(n - 1)//sample_stride)

        idxs
        |> Enum.chunk_every(batch_size)
        |> Enum.reduce(acc, fn chunk, acc2 ->
          batch =
            chunk
            |> Enum.map(fn i -> Nx.slice(flat, [i - window + 1, 0], [window, embed_size]) end)
            |> Nx.stack()

          {b, mx, my, cx, cy, sh} = predict_fn.(params, %{"state_sequence" => batch})

          ent = %{
            buttons: btn_entropy.(b) |> Nx.to_flat_list(),
            main_x: cat_entropy.(mx) |> Nx.to_flat_list(),
            main_y: cat_entropy.(my) |> Nx.to_flat_list(),
            c_x: cat_entropy.(cx) |> Nx.to_flat_list(),
            c_y: cat_entropy.(cy) |> Nx.to_flat_list(),
            shoulder: cat_entropy.(sh) |> Nx.to_flat_list()
          }

          chunk
          |> Enum.with_index()
          |> Enum.reduce(acc2, fn {frame_idx, k}, acc3 ->
            labels = elem(label_sets, frame_idx) |> MapSet.put(:__all__)

            Enum.reduce(labels, acc3, fn label, acc4 ->
              entry = Map.get(acc4, label, %{n: 0, sums: Map.new(heads, &{&1, 0.0})})

              sums =
                Map.new(heads, fn h ->
                  {h, entry.sums[h] + Enum.at(ent[h], k)}
                end)

              Map.put(acc4, label, %{n: entry.n + 1, sums: sums})
            end)
          end)
        end)
      end
    else
      _ -> acc
    end
  end)

IO.write(:stderr, "\n")

rows =
  acc
  |> Enum.map(fn {label, %{n: n, sums: sums}} ->
    {label, n, Map.new(sums, fn {h, s} -> {h, s / max(n, 1)} end)}
  end)
  |> Enum.sort_by(fn {_l, n, _} -> -n end)

lines =
  [
    "policy: #{Path.basename(policy_path)}  files: #{length(files)}  stride: #{sample_stride}",
    "entropy in NATS. uniform ref: buttons(8 bern)=5.55, 17-bucket cat=2.83, 5-bucket=1.61",
    String.pad_trailing("label", 26) <>
      Enum.map_join(heads, "", &String.pad_leading(to_string(&1), 10)) <>
      String.pad_leading("n", 9)
    | Enum.map(rows, fn {label, n, means} ->
        String.pad_trailing(to_string(label), 26) <>
          Enum.map_join(heads, "", fn h ->
            String.pad_leading(:erlang.float_to_binary(means[h], decimals: 3), 10)
          end) <> String.pad_leading(Integer.to_string(n), 9)
      end)
  ]

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Enum.join(lines, "\n") <> "\n")
Enum.each(lines, &Output.puts/1)
Output.success("written to #{out_path}")
