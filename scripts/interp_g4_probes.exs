# G4 of INTERP_GEN_V1: linear probes on the trunk — is the BC-then-RL bet sound?
#
# Trains a linear probe on the GRU trunk's hidden state (and on the raw
# current-frame embedding as the input floor) to test whether the trunk
# linearly encodes the game-state features a value model (D2) would sharpen.
#
# Features: opponent/own percent (4 buckets), offstage (binary), hitstun
# (binary), stage identity (7), opponent-is-Fox (binary — the G7 character
# disambiguation). Trunk >> input floor = the trunk DISCARDS info that was in
# the input (the bad case); trunk ~ input = preserved. Shuffled-label control
# is the probe-soundness floor.
#
#   mix run scripts/interp_g4_probes.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays replays/erickfm_ranked/FOX/extracted --max-files 12
#
# Options: --max-files N [12]  --sample-stride N [7]  --batch-size N [128]
#          --out PATH [eval_runs/0826_gen_v1_sweep/g4_probes.txt]
#
# NO-MIX LAW: one beam; never run beside a live training. Mirrors the G1
# script's embedding (296-dim stage-internals layout), NOT Activations
# .capture_replay (which defaults to the 288-dim layout for this policy).

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.{Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, replays: :string, max_files: :integer,
      sample_stride: :integer, batch_size: :integer, out: :string
    ]
  )

policy_path = opts[:policy] || "checkpoints/fox_gen_v1_20260825_210355_ep10.bin"
replay_dir = opts[:replays] || "replays/erickfm_ranked/FOX/extracted"
max_files = opts[:max_files] || 12
sample_stride = opts[:sample_stride] || 7
batch_size = opts[:batch_size] || 128
out_path = opts[:out] || "eval_runs/0826_gen_v1_sweep/g4_probes.txt"

Output.banner("G4 linear probes (INTERP_GEN_V1)")

trunk = Activations.load_trunk(policy_path)
config = trunk.config
window = trunk.window

embed_config = ExPhil.Embeddings.config(Map.to_list(config))
embed_size = ExPhil.Embeddings.embedding_size(embed_config)
Output.puts("policy: #{Path.basename(policy_path)} (embed #{embed_size}, window #{window}, hidden #{trunk.hidden_size})")

files = Path.wildcard(Path.join(replay_dir, "*.slp")) |> Enum.take(max_files)
Output.puts("#{length(files)} files, stride #{sample_stride}")

# --- Features and per-frame label computation ---
features = [
  :opp_percent, :own_percent, :opp_offstage, :own_offstage,
  :opp_hitstun, :own_hitstun, :stage_identity, :opp_character
]

num_classes = %{
  opp_percent: 4, own_percent: 4, opp_offstage: 2, own_offstage: 2,
  opp_hitstun: 2, own_hitstun: 2, stage_identity: 7, opp_character: 8
}

pct_bucket = fn p ->
  cond do
    p < 40 -> 0
    p < 80 -> 1
    p < 120 -> 2
    true -> 3
  end
end

offstage? = fn p -> p != nil and (abs(p.x) > 85.0 or p.y < 0.0) end

stage_class = fn s ->
  case s do
    2 -> 0
    3 -> 1
    8 -> 2
    28 -> 3
    31 -> 4
    32 -> 5
    _ -> 6
  end
end

# Opponent character classes: the top matchups in the FOX corpus (the tracked
# player is Fox; opponent is a variety, never a ditto) + "other". Raw Melee
# char IDs: 10=Mewtwo 25=Ganon 9=Marth 12=Peach 13=Pikachu 14=ICs 17=Yoshi.
char_class = fn c ->
  case c do
    10 -> 0
    25 -> 1
    9 -> 2
    12 -> 3
    13 -> 4
    14 -> 5
    17 -> 6
    _ -> 7
  end
end

labels_for = fn gs ->
  own = gs.players[1]
  opp = gs.players[2]

  %{
    opp_percent: pct_bucket.(opp.percent || 0.0),
    own_percent: pct_bucket.(own.percent || 0.0),
    opp_offstage: if(offstage?.(opp), do: 1, else: 0),
    own_offstage: if(offstage?.(own), do: 1, else: 0),
    opp_hitstun: if((opp.hitstun_frames_left || 0) > 0, do: 1, else: 0),
    own_hitstun: if((own.hitstun_frames_left || 0) > 0, do: 1, else: 0),
    stage_identity: stage_class.(gs.stage),
    opp_character: char_class.(opp.character || 0)
  }
end

# Capture activations + per-feature label tensors for one replay, under the
# given trunk (real trunk or input_trunk). Returns {activations {n,d},
# labels %{feature => {n} s64}, n}.
capture_one = fn trunk, path ->
  {:ok, replay} = Peppi.parse(path)

  frames =
    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))

  per_frame = Enum.map(frames, fn f -> labels_for.(f.game_state) end)
  n_frames = length(frames)

  dataset = Data.from_frames(frames, embed_config: embed_config)
  embedded = Data.precompute_frame_embeddings(dataset, show_progress: false)
  flat = embedded.embedded_frames

  idxs = Enum.to_list(window..(n_frames - 1)//sample_stride)

  activations =
    idxs
    |> Enum.chunk_every(batch_size)
    |> Enum.map(fn chunk ->
      batch =
        chunk
        |> Enum.map(fn i -> Nx.slice(flat, [i - window + 1, 0], [window, embed_size]) end)
        |> Nx.stack()

      trunk.predict_fn.(trunk.params, batch)
    end)
    |> Nx.concatenate()

  labels =
    Map.new(features, fn f ->
      {f, idxs |> Enum.map(fn i -> Map.fetch!(Enum.at(per_frame, i), f) end) |> Nx.tensor(type: :s64)}
    end)

  {activations, labels, length(idxs)}
end

# --- Capture over all replays (trunk + input floor) ---
capture_all = fn trunk ->
  Enum.map(Enum.with_index(files, 1), fn {path, fidx} ->
    IO.write(:stderr, "\r  file #{fidx}/#{length(files)}\e[K")
    {acts, labels, n} = capture_one.(trunk, path)
    {acts, labels, n}
  end)
end

Output.puts("capturing trunk activations...")
trunk_caps = capture_all.(trunk)

window_ua = Map.get(trunk.config, :use_prev_action, false)
input_trunk = Activations.input_trunk(window: window, use_prev_action: window_ua)
Output.puts("capturing input-floor (raw last-frame embedding)...")
input_caps = capture_all.(input_trunk)
IO.write(:stderr, "\n")

# --- Split by replay: hold out the last ~20% (never split by frame) ---
n_replays = length(files)
n_eval = max(1, div(n_replays, 5))
n_train = n_replays - n_eval

concat_acts = fn caps -> caps |> Enum.map(&elem(&1, 0)) |> Nx.concatenate() end

build_split = fn caps ->
  x_all = concat_acts.(caps)

  labels_all =
    Map.new(features, fn f ->
      {f, caps |> Enum.map(&(elem(&1, 1)[f])) |> Nx.concatenate()}
    end)

  # row -> replay index (activations concatenated in replay order)
  ri =
    caps
    |> Enum.with_index()
    |> Enum.flat_map(fn {{_, _, n}, r} -> List.duplicate(r, n) end)
    |> Enum.with_index()

  train_rows = ri |> Enum.reject(fn {r, _} -> r >= n_train end) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)
  eval_rows = ri |> Enum.filter(fn {r, _} -> r >= n_train end) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)

  x_train = Nx.take(x_all, train_rows, axis: 0)
  x_eval = Nx.take(x_all, eval_rows, axis: 0)

  labels_train = Map.new(features, fn f -> {f, Nx.take(labels_all[f], train_rows, axis: 0)} end)
  labels_eval = Map.new(features, fn f -> {f, Nx.take(labels_all[f], eval_rows, axis: 0)} end)

  %{x_train: x_train, x_eval: x_eval, labels_train: labels_train, labels_eval: labels_eval}
end

trunk_split = build_split.(trunk_caps)
input_split = build_split.(input_caps)

Output.puts("split: #{n_train} train replays / #{n_eval} eval replays; train rows=#{Nx.axis_size(trunk_split.x_train, 0)} eval rows=#{Nx.axis_size(trunk_split.x_eval, 0)}")

# --- Probe each feature ---
probe_feature = fn split, feature ->
  k = num_classes[feature]
  y_tr = split.labels_train[feature]
  y_ev = split.labels_eval[feature]
  Probe.fit_eval(split.x_train, y_tr, split.x_eval, y_ev, k)
end

shuffled = fn split, feature ->
  Probe.shuffled_control(split, feature, num_classes[feature])
end

# --- Output ---
header =
  "policy: #{Path.basename(policy_path)}  files: #{length(files)}  stride: #{sample_stride}\n" <>
    "train rows #{Nx.axis_size(trunk_split.x_train, 0)} / eval rows #{Nx.axis_size(trunk_split.x_eval, 0)}\n" <>
    "bal_acc: trunk vs INPUT-floor (raw embedding) vs SHUFFLE floor vs majority\n" <>
    String.pad_trailing("feature", 16) <>
    String.pad_leading("trunk", 9) <>
    String.pad_leading("input", 9) <>
    String.pad_leading("shuffle", 9) <>
    String.pad_leading("majority", 10) <>
    "  verdict\n"

rows =
  Enum.map(features, fn f ->
    t = probe_feature.(trunk_split, f)
    i = probe_feature.(input_split, f)
    s = shuffled.(trunk_split, f)

    ft = fn x -> if x, do: :erlang.float_to_binary(x, decimals: 3), else: "  -  " end

    verdict =
      cond do
        is_nil(t.balanced_accuracy) or is_nil(i.balanced_accuracy) ->
          "no eval rows"

        t.balanced_accuracy < i.balanced_accuracy - 0.15 ->
          "DISCARDED (trunk lost input info)"

        t.balanced_accuracy > i.balanced_accuracy + 0.1 and t.balanced_accuracy > 1.5 / num_classes[f] ->
          "ENRICHED (trunk > raw input)"

        t.balanced_accuracy > (t.majority_baseline || 0) + 0.15 and
            t.balanced_accuracy > (s.balanced_accuracy || 0) + 0.1 ->
          "PRESERVED (decodable, ~ input)"

        true ->
          "WEAK/AMBIGUOUS"
      end

    String.pad_trailing(to_string(f), 16) <>
      String.pad_leading(ft.(t.balanced_accuracy), 9) <>
      String.pad_leading(ft.(i.balanced_accuracy), 9) <>
      String.pad_leading(ft.(s.balanced_accuracy), 9) <>
      String.pad_leading(ft.(t.majority_baseline), 10) <>
      "  #{verdict}"
  end)

lines = [header | rows]

File.mkdir_p!(Path.dirname(out_path))
File.write!(out_path, Enum.join(lines, "\n") <> "\n")
Enum.each(lines, &Output.puts/1)
Output.success("written to #{out_path}")
