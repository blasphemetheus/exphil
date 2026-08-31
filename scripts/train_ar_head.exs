# AUTOREGRESSIVE_HEAD_PLAN item 8a — head-only fit on a FROZEN trunk.
#
# Pass 1 (capture, STREAMED): run the policy's trunk over every 60-frame
# window of the corpus (Activations.capture_replay — cached embeddings,
# config-aware) and APPEND each replay's {hidden} features + target
# controller components to raw binary files as it lands. Crash-safe and
# free of the 2 GB term_to_binary limit (the 08-30 first run captured 856
# replays into a 19 GB map and died at save — never build the giant term).
#
# Ports: per-file fox detection (--char-id, erickfm masters sit on varying
# ports). DITTOS ARE CAPTURED ON BOTH PORTS; both halves carry the same
# game id so they never straddle the train/val split.
#
# Pass 2 (fit): train ONLY a new controller head on the cached features —
#   --head autoregressive -> v1.1-ARhead   (the recipe change under test)
#   --head independent    -> v1.1-INDhead  (control)
#   --head both           -> both, one capture, one invocation
# Minibatches are gathered by :binary.part row slicing (the 19 GB feature
# blob stays a plain binary; no BinaryBackend tensor ops on it).
#
# Export: source checkpoint params + the new head's params merged in and
# config.head set — deployable via the standard Agent.
#
# Usage (the 8a run):
#   mix run scripts/train_ar_head.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --char-id 2 --limit-files 1000 --head both \
#     --out-ar checkpoints/fox_gen_v1.1_ARhead_policy.bin \
#     --out-ind checkpoints/fox_gen_v1.1_INDhead_policy.bin
#
# Options:
#   --policy PATH       source exported policy .bin (temporal)      (required)
#   --replays GLOB      training replays                            (required)
#   --port N            pin the master's port; omit to auto-detect by --char-id
#   --char-id N         per-file port auto-detect (default 2 = Fox); dittos -> both ports
#   --limit-files N     default 1000
#   --head TYPE         autoregressive | independent | both (default autoregressive)
#   --epochs N          head-fit epochs (default 4)
#   --batch-size N      default 1024
#   --lr F              default 1.0e-3
#   --val-frac F        fraction of GAMES held out (default 0.05)
#   --features DIR      capture cache dir (reused when meta.term exists)
#   --out PATH          output .bin (single-head mode)
#   --out-ar / --out-ind  outputs for --head both
#   --seed N            default 828
require Logger
Logger.configure(level: :warning)

alias ExPhil.Interp.Activations
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, Output, Utils}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, replays: :string, port: :integer, char_id: :integer,
      limit_files: :integer, head: :string, epochs: :integer, batch_size: :integer,
      lr: :float, val_frac: :float, features: :string, out: :string,
      out_ar: :string, out_ind: :string, seed: :integer
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
glob = opts[:replays] || raise "--replays required"
char_id = opts[:char_id] || 2
limit = opts[:limit_files] || 1000
epochs = opts[:epochs] || 4
batch_size = opts[:batch_size] || 1024
lr = opts[:lr] || 1.0e-4
val_frac = opts[:val_frac] || 0.05
seed = opts[:seed] || 828

heads_to_fit =
  case opts[:head] || "autoregressive" do
    "both" -> [:autoregressive, :independent]
    "autoregressive" -> [:autoregressive]
    "independent" -> [:independent]
    other -> raise "--head must be autoregressive|independent|both (got #{other})"
  end

out_for = fn head ->
  case {heads_to_fit, head} do
    {[_single], _} -> opts[:out] || raise("--out required for single-head mode")
    {_, :autoregressive} -> opts[:out_ar] || raise("--out-ar required with --head both")
    {_, :independent} -> opts[:out_ind] || raise("--out-ind required with --head both")
  end
end

# validate outputs early (fail before hours of capture)
Enum.each(heads_to_fit, out_for)

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("AR head-only fit (plan item 8a)")

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Heads", inspect(heads_to_fit)},
  {"Replays", "#{length(files)} files"},
  {"Port", if(opts[:port], do: "pinned #{opts[:port]}", else: "auto by char #{char_id} (dittos: both ports)")},
  {"Epochs", epochs},
  {"Batch", batch_size},
  {"LR", lr}
])

trunk = Activations.load_trunk(policy_path)
window = trunk.window
hidden = trunk.hidden_size
axis_buckets = Map.get(trunk.config, :axis_buckets, 16)
shoulder_buckets = Map.get(trunk.config, :shoulder_buckets, 4)

# v1's LOSS RECIPE — the 08-30 first bracket died at ~35s/game in BOTH arms
# because the heads were fit with plain CE at 1e-3: no button pos-weights
# (rare buttons collapse — no jumps), no smoothing/focal/edge-weight
# (miscalibrated sticks). Mirror the source run's *_config.json.
recipe_path =
  policy_path
  |> String.replace(~r/(_ep\d+)?(_policy)?\.bin$/, "")
  |> Kernel.<>("_config.json")

recipe =
  if File.exists?(recipe_path) do
    Output.puts("Loss recipe from #{Path.basename(recipe_path)}")
    Jason.decode!(File.read!(recipe_path))
  else
    Output.warning("no #{Path.basename(recipe_path)} — using v1-style loss defaults")
    %{}
  end

label_smoothing = recipe["label_smoothing"] || 0.1
focal_loss = if is_nil(recipe["focal_loss"]), do: true, else: recipe["focal_loss"]
focal_gamma = recipe["focal_gamma"] || 3.0
button_weight = recipe["button_weight"] || 2.0
stick_edge_weight = recipe["stick_edge_weight"] || 2.0
entropy_weight = recipe["entropy_weight"] || 0.01
pos_weight_auto? = recipe["button_pos_weight"] in ["auto", nil]

Output.puts("loss: smooth=#{label_smoothing} focal=#{focal_loss}/#{focal_gamma} " <>
  "btn_w=#{button_weight} pos_w=#{if pos_weight_auto?, do: "auto", else: "off"} " <>
  "edge=#{stick_edge_weight} entropy=#{entropy_weight} lr=#{lr}")

port_tag = if opts[:port], do: "p#{opts[:port]}", else: "c#{char_id}"

features_dir =
  opts[:features] ||
    "cache/ar_head/#{Path.basename(policy_path, ".bin")}_#{length(files)}f_#{port_tag}"

meta_path = Path.join(features_dir, "meta.term")
components = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]
bin_path = fn name -> Path.join(features_dir, "#{name}.bin") end

# ---------------------------------------------------------------------------
# Pass 1: streamed capture (skipped when meta.term exists)
# ---------------------------------------------------------------------------

resolve_ports = fn path ->
  if opts[:port] do
    [opts[:port]]
  else
    case Peppi.metadata(path) do
      {:ok, meta} -> for p <- meta.players, p.character == char_id, do: p.port
      _ -> []
    end
  end
end

unless File.exists?(meta_path) do
  File.mkdir_p!(features_dir)
  Output.puts("Capturing trunk features (window #{window}, hidden #{hidden}) -> #{features_dir}")

  ios =
    Map.new([:features | components], fn name ->
      {name, File.open!(bin_path.(name), [:write, :raw, :binary, :delayed_write])}
    end)

  capture_one = fn path, port ->
    cap =
      Activations.capture_replay(trunk, path,
        player_port: port,
        opponent_port: if(port == 1, do: 2, else: 1),
        labels: false
      )

    {:ok, replay} = Peppi.parse(Path.expand(path))

    actions =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
      |> Enum.reject(&(&1.game_state.frame < 0))
      # Row r of the capture <-> the window ENDING at frame r + window - 1;
      # the target is that frame's controller (what training predicts).
      |> Enum.drop(window - 1)
      |> Enum.take(cap.n)
      |> Enum.map(&Data.controller_to_action(&1.controller,
           axis_buckets: axis_buckets, shoulder_buckets: shoulder_buckets))
      |> Data.actions_to_tensors()

    n_actions = Nx.axis_size(actions.main_x, 0)
    if n_actions != cap.n, do: raise("row/target mismatch: #{cap.n} vs #{n_actions}")

    IO.binwrite(ios.features, Nx.to_binary(Nx.as_type(cap.activations, :f32)))
    # all components fit in u8 (buttons multi-hot 0/1; buckets <= 16)
    Enum.each(components, fn c ->
      IO.binwrite(ios[c], Nx.to_binary(Nx.as_type(actions[c], :u8)))
    end)

    cap.n
  end

  {games, skipped} =
    files
    |> Enum.with_index()
    |> Enum.reduce({[], 0}, fn {path, gid}, {games, skipped} ->
      Output.progress_bar(gid + 1, length(files), label: "replays")

      case resolve_ports.(path) do
        [] ->
          {games, skipped + 1}

        ports ->
          spans =
            Enum.flat_map(ports, fn port ->
              try do
                [{gid, capture_one.(path, port)}]
              rescue
                err ->
                  Output.warning("skip #{Path.basename(path)} p#{port}: #{Exception.message(err)}")
                  []
              end
            end)

          {spans ++ games, if(spans == [], do: skipped + 1, else: skipped)}
      end
    end)

  Output.progress_done()
  Enum.each(ios, fn {_, io} -> File.close(io) end)
  games = Enum.reverse(games)
  if games == [], do: raise("nothing captured")

  meta = %{
    # [{game_id, n_rows}] in row order; ditto halves share a game_id
    games: games,
    rows: games |> Enum.map(&elem(&1, 1)) |> Enum.sum(),
    hidden: hidden,
    window: window,
    axis_buckets: axis_buckets,
    shoulder_buckets: shoulder_buckets,
    skipped: skipped,
    policy: policy_path
  }

  File.write!(meta_path, :erlang.term_to_binary(meta))
  Output.success("Captured #{meta.rows} rows / #{length(games)} port-streams -> #{features_dir}")
end

meta = meta_path |> File.read!() |> :erlang.binary_to_term()
n = meta.rows
if meta.hidden != hidden, do: raise("cache hidden #{meta.hidden} != policy hidden #{hidden}")

Output.puts("Dataset: #{n} rows, #{length(Enum.uniq(Enum.map(meta.games, &elem(&1, 0))))} games " <>
  "(#{length(meta.games)} port-streams; #{meta.skipped} files skipped)")

feats_bin = File.read!(bin_path.(:features))
comp_bins = Map.new(components, fn c -> {c, File.read!(bin_path.(c))} end)
row_bytes = hidden * 4

expected = n * row_bytes
if byte_size(feats_bin) != expected do
  raise "features.bin is #{byte_size(feats_bin)} bytes, expected #{expected} — partial capture? delete #{features_dir} and recapture"
end

# resolve the recipe's :auto button pos-weights from the captured targets
button_pos_weight =
  if pos_weight_auto? do
    rates =
      comp_bins.buttons
      |> Nx.from_binary(:u8)
      |> Nx.reshape({n, 8})
      |> Nx.as_type(:f32)
      |> Nx.mean(axes: [0])

    w = ExPhil.Networks.Policy.Loss.compute_pos_weights_from_rates(rates, 30.0)
    Output.puts("button pos-weights: #{inspect(w |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 1)))}")
    Nx.backend_copy(w, Nx.BinaryBackend)
  else
    nil
  end

loss_opts = [
  label_smoothing: label_smoothing,
  focal_loss: focal_loss,
  focal_gamma: focal_gamma,
  button_weight: button_weight,
  button_pos_weight: button_pos_weight,
  stick_edge_weight: stick_edge_weight,
  entropy_weight: entropy_weight
]

# ---------------------------------------------------------------------------
# Split BY GAME (ditto halves share a game id -> same side)
# ---------------------------------------------------------------------------

key = Nx.Random.key(seed)
game_ids = meta.games |> Enum.map(&elem(&1, 0)) |> Enum.uniq()

val_games =
  if length(game_ids) < 3 or val_frac <= 0.0 do
    MapSet.new()
  else
    n_val = max(trunc(length(game_ids) * val_frac), 1)
    {shuffled, _} = Nx.Random.shuffle(key, Nx.tensor(game_ids, type: :s64))
    shuffled |> Nx.to_flat_list() |> Enum.take(n_val) |> MapSet.new()
  end

{train_ids, val_ids, _offset} =
  Enum.reduce(meta.games, {[], [], 0}, fn {gid, rows}, {tr, va, off} ->
    ids = Enum.to_list(off..(off + rows - 1))

    if MapSet.member?(val_games, gid) do
      {tr, [ids | va], off + rows}
    else
      {[ids | tr], va, off + rows}
    end
  end)

train_ids = train_ids |> Enum.reverse() |> List.flatten()
val_ids = val_ids |> Enum.reverse() |> List.flatten()

Output.puts("Split: #{length(train_ids)} train / #{length(val_ids)} val rows " <>
  "(#{MapSet.size(val_games)} games held out)")

# ---------------------------------------------------------------------------
# Row gather straight from the binaries (never a full-size tensor)
# ---------------------------------------------------------------------------

take_rows = fn ids ->
  k = length(ids)

  feats =
    ids
    |> Enum.map(&:binary.part(feats_bin, &1 * row_bytes, row_bytes))
    |> IO.iodata_to_binary()
    |> Nx.from_binary(:f32)
    |> Nx.reshape({k, hidden})

  actions =
    Map.new(components, fn c ->
      bin = comp_bins[c]
      width = if c == :buttons, do: 8, else: 1

      t =
        ids
        |> Enum.map(&:binary.part(bin, &1 * width, width))
        |> IO.iodata_to_binary()
        |> Nx.from_binary(:u8)

      t = if c == :buttons, do: Nx.reshape(t, {k, 8}), else: t
      {c, Nx.as_type(t, :s64)}
    end)

  {feats, actions}
end

# ---------------------------------------------------------------------------
# Pass 2: fit each requested head on the frozen features
# ---------------------------------------------------------------------------

trunk_input = Axon.input("trunk", shape: {nil, hidden})

{:ok, %{params: src_params, config: src_config}} =
  ExPhil.Training.Checkpoint.load_policy(policy_path)

src_data =
  case src_params do
    %Axon.ModelState{data: d} -> d
    m when is_map(m) -> m
  end

to_bin_backend = fn t -> Nx.backend_copy(t, Nx.BinaryBackend) end

fit_one = fn head ->
  out = out_for.(head)
  Output.puts("=== Fitting #{head} head -> #{out} ===")

  model =
    case head do
      :autoregressive ->
        Heads.build_autoregressive_head(trunk_input,
          axis_buckets: axis_buckets, shoulder_buckets: shoulder_buckets)

      :independent ->
        Heads.build_controller_head(trunk_input, axis_buckets, shoulder_buckets)
    end

  # mode: :train build only for init; the head has no dropout, so the
  # inference forward IS the training forward.
  {init_fn, _} = Utils.build_compiled(model, mode: :train)
  {_, predict_fn} = Utils.build_compiled(model, mode: :inference)

  init_template =
    case head do
      :autoregressive ->
        Map.merge(%{"trunk" => Nx.template({1, hidden}, :f32)}, Heads.tf_templates(1))

      :independent ->
        Nx.template({1, hidden}, :f32)
    end

  head_params = init_fn.(init_template, Axon.ModelState.empty())
  {opt_init, opt_update} = Polaris.Optimizers.adam(learning_rate: lr)
  opt_state = opt_init.(head_params.data)

  build_inputs =
    case head do
      :autoregressive -> fn feats, actions -> Map.put(Heads.tf_inputs(actions), "trunk", feats) end
      :independent -> fn feats, _actions -> feats end
    end

  loss_for = fn params, feats, actions ->
    {b, mx, my, cx, cy, sh} =
      predict_fn.(Utils.ensure_model_state(params), build_inputs.(feats, actions))

    ExPhil.Networks.Policy.imitation_loss(
      %{buttons: b, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh},
      actions,
      loss_opts
    )
  end

  train_step =
    Nx.Defn.jit(
      fn params, opt_state, feats, actions ->
        {loss, grads} = Nx.Defn.value_and_grad(fn p -> loss_for.(p, feats, actions) end).(params)
        {updates, opt_state} = opt_update.(grads, opt_state, params)
        {Polaris.Updates.apply_updates(params, updates), opt_state, loss}
      end,
      compiler: EXLA, on_conflict: :reuse
    )

  eval_loss =
    Nx.Defn.jit(fn params, feats, actions -> loss_for.(params, feats, actions) end,
      compiler: EXLA, on_conflict: :reuse)

  steps_per_epoch = div(length(train_ids), batch_size)
  val_sample = Enum.take(val_ids, 50_000)
  Output.puts("⏳ JIT compiling #{head} head fit (first batch)...")

  {params_data, _opt} =
    Enum.reduce(1..epochs, {head_params.data, opt_state}, fn epoch, {pd, os} ->
      shuffled = :rand.seed(:exsss, {seed, epoch, 7}) && Enum.shuffle(train_ids)

      {pd, os, losses} =
        shuffled
        |> Enum.chunk_every(batch_size, batch_size, :discard)
        |> Enum.with_index()
        |> Enum.reduce({pd, os, []}, fn {ids, s}, {pd, os, acc} ->
          {feats, actions} = take_rows.(ids)
          {pd, os, loss} = train_step.(pd, os, feats, actions)

          if rem(s, 200) == 0 do
            IO.write(:stderr, "\r  #{head} epoch #{epoch} #{s}/#{steps_per_epoch} loss #{Float.round(Nx.to_number(loss), 4)}\e[K")
          end

          {pd, os, [Nx.to_number(loss) | acc]}
        end)

      IO.write(:stderr, "\n")

      val =
        if val_sample != [] do
          {vf, va} = take_rows.(val_sample)
          Nx.to_number(eval_loss.(pd, vf, va))
        end

      train_mean = Enum.sum(losses) / max(length(losses), 1)

      Output.puts("#{head} epoch #{epoch}: train #{Float.round(train_mean, 4)}" <>
        if(val, do: " val #{Float.round(val, 4)}", else: ""))

      {pd, os}
    end)

  new_head_data =
    params_data
    |> Map.new(fn {layer, ps} -> {layer, Map.new(ps, fn {k, t} -> {k, to_bin_backend.(t)} end)} end)

  merged = Map.merge(src_data, new_head_data)
  config = Map.put(src_config, :head, head)
  spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config), external: true)
  File.mkdir_p!(Path.dirname(out))
  Edifice.Checkpoint.save(merged, out, spec: spec, metadata: %{config: config})
  Output.success("Exported #{head} head policy -> #{out}")
end

Enum.each(heads_to_fit, fit_one)
Output.success("8a fit complete.")
