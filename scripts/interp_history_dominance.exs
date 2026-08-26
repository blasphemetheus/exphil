# G2 of INTERP_GEN_V1: history-vs-state dominance + crouch-attractor probe.
#
#   mix run scripts/interp_history_dominance.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays replays/erickfm_ranked/FOX/extracted --max-files 8
#
# Two measurements over real corpus windows:
#
#  A. PREFIX vs CURRENT sensitivity: for window W, compare the policy's
#     output to (i) W with its first 50 frames swapped for another game's
#     frames (prefix influence) and (ii) W with only its LAST frame swapped
#     (current-state influence). Per-head KL, summed. If prefix-KL >>
#     last-frame-KL, the policy is history-dominated -> absorbing basins
#     are structural -> v2 trains with scheduled sampling.
#
#  B. FROZEN-STATE attractor: tile ONE real frame 60x (physically valid
#     for idle/crouch states — that IS what a crouch loop looks like) and
#     measure the probability mass the policy puts on "continue" (no
#     buttons + neutral/down stick). Compares idle-tiled vs movement
#     windows: how sticky is stillness?
#
# NO-MIX LAW: run only with no other live beam.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Networks.Policy
alias ExPhil.Training.{Checkpoint, Data, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, max_files: :integer, samples: :integer, port: :integer]
  )

policy_path = opts[:policy] || raise "--policy required"
replay_dir = opts[:replays] || raise "--replays required"
max_files = opts[:max_files] || 8
n_samples = opts[:samples] || 256
port = opts[:port] || 1

Output.banner("History-vs-state dominance (INTERP_GEN_V1 G2)")

{:ok, export} = Checkpoint.load_policy(policy_path)
config = export.config
params = export.params
window = config[:window_size] || 60

embed_config = ExPhil.Embeddings.config(Map.to_list(config))
embed_size = ExPhil.Embeddings.embedding_size(embed_config)

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

# Flat embedded tensors per file + the parsed frames (for action-state reads)
embedded_files =
  files
  |> Enum.map(fn file ->
    with {:ok, replay} <- Peppi.parse(file) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: 3 - port)
        |> Enum.reject(&(&1.game_state.frame < 0))

      if length(frames) > window + 10 do
        ds = Data.from_frames(frames, embed_config: embed_config)
        emb = Data.precompute_frame_embeddings(ds, show_progress: false)
        {emb.embedded_frames, frames}
      end
    else
      _ -> nil
    end
  end)
  |> Enum.reject(&is_nil/1)

Output.puts("#{length(embedded_files)} usable files")
if length(embedded_files) < 2, do: raise("need >=2 files for prefix swaps")

:rand.seed(:exsss, {42, 42, 42})

slice_win = fn flat, last_idx ->
  Nx.slice(flat, [last_idx - window + 1, 0], [window, embed_size])
end

# --- A. prefix vs current sensitivity ---------------------------------
prefix_keep = 10  # keep the last 10 frames; swap the first 50

samples =
  for _ <- 1..n_samples do
    {flat_a, frames_a} = Enum.random(embedded_files)
    {flat_b, _} = Enum.random(embedded_files)
    {n_a, _} = Nx.shape(flat_a)
    {n_b, _} = Nx.shape(flat_b)
    i = Enum.random(window..(n_a - 2))
    j = Enum.random(window..(n_b - 2))

    orig = slice_win.(flat_a, i)

    donor = slice_win.(flat_b, j)

    prefix_swapped =
      Nx.concatenate([
        Nx.slice(donor, [0, 0], [window - prefix_keep, embed_size]),
        Nx.slice(orig, [window - prefix_keep, 0], [prefix_keep, embed_size])
      ])

    last_swapped =
      Nx.concatenate([
        Nx.slice(orig, [0, 0], [window - 1, embed_size]),
        Nx.slice(donor, [window - 1, 0], [1, embed_size])
      ])

    {orig, prefix_swapped, last_swapped, frames_a, i}
  end

batchify = fn tensors -> Nx.stack(tensors) end

p0_in = batchify.(Enum.map(samples, &elem(&1, 0)))
p1_in = batchify.(Enum.map(samples, &elem(&1, 1)))
p2_in = batchify.(Enum.map(samples, &elem(&1, 2)))

softmax_heads = fn {b, mx, my, cx, cy, sh} ->
  %{
    buttons: Nx.sigmoid(b),
    main_x: Axon.Activations.softmax(mx),
    main_y: Axon.Activations.softmax(my),
    c_x: Axon.Activations.softmax(cx),
    c_y: Axon.Activations.softmax(cy),
    shoulder: Axon.Activations.softmax(sh)
  }
end

p0 = softmax_heads.(predict_fn.(params, %{"state_sequence" => p0_in}))
p1 = softmax_heads.(predict_fn.(params, %{"state_sequence" => p1_in}))
p2 = softmax_heads.(predict_fn.(params, %{"state_sequence" => p2_in}))

cat_kl = fn p, q ->
  Nx.sum(Nx.multiply(p, Nx.log(Nx.divide(Nx.add(p, 1.0e-9), Nx.add(q, 1.0e-9)))), axes: [-1])
end

bern_kl = fn p, q ->
  a = Nx.multiply(p, Nx.log(Nx.divide(Nx.add(p, 1.0e-9), Nx.add(q, 1.0e-9))))
  b = Nx.multiply(Nx.subtract(1.0, p),
        Nx.log(Nx.divide(Nx.add(Nx.subtract(1.0, p), 1.0e-9), Nx.add(Nx.subtract(1.0, q), 1.0e-9))))
  Nx.sum(Nx.add(a, b), axes: [-1])
end

total_kl = fn pa, pb ->
  [
    bern_kl.(pa.buttons, pb.buttons),
    cat_kl.(pa.main_x, pb.main_x),
    cat_kl.(pa.main_y, pb.main_y),
    cat_kl.(pa.c_x, pb.c_x),
    cat_kl.(pa.c_y, pb.c_y),
    cat_kl.(pa.shoulder, pb.shoulder)
  ]
  |> Enum.reduce(&Nx.add/2)
end

prefix_kl = total_kl.(p0, p1) |> Nx.to_flat_list()
last_kl = total_kl.(p0, p2) |> Nx.to_flat_list()

mean = fn l -> Enum.sum(l) / length(l) end
med = fn l -> Enum.sort(l) |> Enum.at(div(length(l), 2)) end

Output.puts("")
Output.puts("A. sensitivity (summed per-head KL, n=#{n_samples}):")
Output.puts("   prefix-swap (50 frames of history): mean #{Float.round(mean.(prefix_kl), 3)}  median #{Float.round(med.(prefix_kl), 3)}")
Output.puts("   last-frame swap (current state):    mean #{Float.round(mean.(last_kl), 3)}  median #{Float.round(med.(last_kl), 3)}")
ratio = mean.(prefix_kl) / max(mean.(last_kl), 1.0e-9)
Output.puts("   HISTORY/CURRENT ratio: #{Float.round(ratio, 3)}  (>1 = history-dominated)")

# --- B. frozen-state attractor ---------------------------------------
# Tile single frames 60x; classify source frame idle vs active by its
# controller (no buttons + |stick| < 0.2 = idle-ish proxy).
idle? = fn f ->
  c = f.controller
  not (c.button_a or c.button_b or c.button_x or c.button_y or c.button_z) and
    abs(c.main_stick.x - 0.5) < 0.1 and abs(c.main_stick.y - 0.5) < 0.1
end

tiled = fn flat, idx ->
  Nx.slice(flat, [idx, 0], [1, embed_size]) |> Nx.tile([window, 1])
end

{idle_wins, active_wins} =
  Enum.reduce(1..400, {[], []}, fn _, {is, as} ->
    {flat, frames} = Enum.random(embedded_files)
    {n, _} = Nx.shape(flat)
    i = Enum.random(window..(n - 2))
    f = Enum.at(frames, i)

    cond do
      idle?.(f) and length(is) < 128 -> {[tiled.(flat, i) | is], as}
      not idle?.(f) and length(as) < 128 -> {is, [tiled.(flat, i) | as]}
      true -> {is, as}
    end
  end)

report_stay = fn wins, name ->
  if length(wins) < 8 do
    Output.puts("B. #{name}: too few samples (#{length(wins)})")
  else
    probs = softmax_heads.(predict_fn.(params, %{"state_sequence" => Nx.stack(wins)}))
    # "continue doing nothing" mass: no-button prob x argmax-stick-is-modal
    no_btn = Nx.reduce_max(probs.buttons, axes: [-1]) |> Nx.negate() |> Nx.add(1.0)
    top_my = Nx.reduce_max(probs.main_y, axes: [-1])
    top_mx = Nx.reduce_max(probs.main_x, axes: [-1])
    stay = Nx.multiply(no_btn, Nx.multiply(top_my, top_mx)) |> Nx.to_flat_list()
    Output.puts(
      "B. #{name} (n=#{length(wins)}): mean stay-mass #{Float.round(mean.(stay), 3)}, " <>
        "mean top main_y prob #{Nx.to_flat_list(top_my) |> mean.() |> Float.round(3)}"
    )
  end
end

Output.puts("")
report_stay.(idle_wins, "IDLE-tiled windows  ")
report_stay.(active_wins, "ACTIVE-tiled windows")
Output.puts("")
Output.puts("Interpretation: ratio >> 1 in A + high idle stay-mass in B = exposure")
Output.puts("bias is structural -> v2 trains with scheduled sampling (doc G2).")
