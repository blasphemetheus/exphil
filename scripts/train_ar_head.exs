# AUTOREGRESSIVE_HEAD_PLAN item 8a — head-only fit on a FROZEN trunk.
#
# Pass 1 (capture): run the policy's trunk over every 60-frame window of the
# corpus (Activations.capture_replay — cached embeddings, config-aware) and
# pair each row's {hidden} features with the master's controller components
# at the window's last frame (the frame the policy predicts).
#
# Pass 2 (fit): train ONLY a new controller head on those features —
#   --head autoregressive  -> v1.1-ARhead  (the recipe change under test)
#   --head independent     -> v1.1-INDhead (control: "any fresh head fit helps")
# Adam + optional val split by replay; minutes on the 5090, not a retrain.
#
# Export: the source checkpoint's params with the new head's params merged in
# (AR: ar_* keys added; IND: buttons_hidden/... replaced) and config.head set,
# saved in the Edifice manifest format — deployable via the standard Agent.
#
# Usage:
#   mix run scripts/train_ar_head.exs \
#     --policy checkpoints/fox_gen_v1_20260825_210355_ep10.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' --port 1 \
#     --limit-files 800 --head autoregressive \
#     --out checkpoints/fox_gen_v1.1_ARhead_policy.bin
#
# Options:
#   --policy PATH       source exported policy .bin (temporal)      (required)
#   --replays GLOB      training replays                            (required)
#   --port N            pin the master's port; omit to auto-detect by --char-id
#   --char-id N         character for per-file port auto-detect (default 2 = Fox;
#                       erickfm ranked masters sit on VARYING ports — never pin)
#   --limit-files N     default 800
#   --head TYPE         autoregressive | independent (default autoregressive)
#   --epochs N          head-fit epochs over the cached features (default 4)
#   --batch-size N      default 1024
#   --lr F              default 1.0e-3
#   --val-frac F        fraction of REPLAYS held out (default 0.05)
#   --features PATH     cache file for pass-1 output (default derives from
#                       policy+corpus; reused when present — delete to recapture)
#   --out PATH          output policy .bin (required)
#   --seed N            default 828
require Logger
Logger.configure(level: :warning)

alias ExPhil.Interp.Activations
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Training.{Data, Output, Utils}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, replays: :string, port: :integer, char_id: :integer,
      limit_files: :integer,
      head: :string, epochs: :integer, batch_size: :integer, lr: :float,
      val_frac: :float, features: :string, out: :string, seed: :integer
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
glob = opts[:replays] || raise "--replays required"
out = opts[:out] || raise "--out required"
char_id = opts[:char_id] || 2
limit = opts[:limit_files] || 800
head = String.to_existing_atom(opts[:head] || "autoregressive")
unless head in [:autoregressive, :independent], do: raise("--head must be autoregressive|independent")
epochs = opts[:epochs] || 4
batch_size = opts[:batch_size] || 1024
lr = opts[:lr] || 1.0e-3
val_frac = opts[:val_frac] || 0.05
seed = opts[:seed] || 828

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("AR head-only fit (plan item 8a)")

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Head", head},
  {"Replays", "#{length(files)} files"},
  {"Port", if(opts[:port], do: "pinned #{opts[:port]}", else: "auto by char #{char_id}")},
  {"Epochs", epochs},
  {"Batch", batch_size},
  {"LR", lr},
  {"Out", out}
])

trunk = Activations.load_trunk(policy_path)
window = trunk.window
hidden = trunk.hidden_size
axis_buckets = Map.get(trunk.config, :axis_buckets, 16)
shoulder_buckets = Map.get(trunk.config, :shoulder_buckets, 4)

# ---------------------------------------------------------------------------
# Pass 1: capture trunk features + aligned controller targets, per replay
# ---------------------------------------------------------------------------

port_tag = if opts[:port], do: "p#{opts[:port]}", else: "c#{char_id}"

features_path =
  opts[:features] ||
    "cache/ar_head/#{Path.basename(policy_path, ".bin")}_#{length(files)}f_#{port_tag}.nx"

File.mkdir_p!(Path.dirname(features_path))

# erickfm ranked masters sit on varying ports — resolve per file by
# character unless --port pins it (same rule as critic_extract.exs)
resolve_port = fn path ->
  if opts[:port] do
    {:ok, opts[:port]}
  else
    case ExPhil.Data.Peppi.metadata(path) do
      {:ok, meta} ->
        case Enum.filter(meta.players, &(&1.character == char_id)) do
          [%{port: p}] -> {:ok, p}
          _ -> :skip
        end

      _ ->
        :skip
    end
  end
end

capture_one = fn path, port ->
  cap = Activations.capture_replay(trunk, path, player_port: port,
    opponent_port: if(port == 1, do: 2, else: 1), labels: false)

  {:ok, replay} = ExPhil.Data.Peppi.parse(Path.expand(path))

  actions =
    replay
    |> ExPhil.Data.Peppi.to_training_frames(
      player_port: port,
      opponent_port: if(port == 1, do: 2, else: 1)
    )
    |> Enum.reject(&(&1.game_state.frame < 0))
    # Row r of the capture <-> the window ENDING at frame r + window - 1;
    # the target is that frame's controller (what training predicts).
    |> Enum.drop(window - 1)
    |> Enum.take(cap.n)
    |> Enum.map(&Data.controller_to_action(&1.controller,
         axis_buckets: axis_buckets, shoulder_buckets: shoulder_buckets))
    |> Data.actions_to_tensors()

  n_actions = Nx.axis_size(actions.main_x, 0)

  if n_actions != cap.n do
    raise "row/target mismatch for #{path}: #{cap.n} features vs #{n_actions} targets"
  end

  to_bin = &Nx.backend_copy(&1, Nx.BinaryBackend)

  %{
    features: to_bin.(cap.activations),
    actions: Map.new(actions, fn {k, v} -> {k, to_bin.(v)} end),
    n: cap.n
  }
end

data =
  if File.exists?(features_path) do
    Output.puts("Reusing cached features: #{features_path}")
    features_path |> File.read!() |> :erlang.binary_to_term()
  else
    Output.puts("Capturing trunk features (window #{window}, hidden #{hidden})...")

    caps =
      files
      |> Enum.with_index(1)
      |> Enum.flat_map(fn {path, i} ->
        Output.progress_bar(i, length(files), label: "replays")

        case resolve_port.(path) do
          {:ok, port} ->
            try do
              [capture_one.(path, port)]
            rescue
              err ->
                Output.warning("skip #{Path.basename(path)}: #{Exception.message(err)}")
                []
            end

          :skip ->
            []
        end
      end)

    Output.progress_done()
    if caps == [], do: raise("nothing captured")

    replay_index =
      caps
      |> Enum.with_index()
      |> Enum.flat_map(fn {c, i} -> List.duplicate(i, c.n) end)
      |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

    merged = %{
      features: Nx.concatenate(Enum.map(caps, & &1.features), axis: 0),
      actions:
        Map.new([:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder], fn k ->
          {k, Nx.concatenate(Enum.map(caps, & &1.actions[k]), axis: 0)}
        end),
      replay_index: replay_index,
      num_replays: length(caps)
    }

    File.write!(features_path, :erlang.term_to_binary(merged))
    Output.success("Captured #{Nx.axis_size(merged.features, 0)} rows -> #{features_path}")
    merged
  end

n = Nx.axis_size(data.features, 0)
Output.puts("Dataset: #{n} rows from #{data.num_replays} replays")

# Split BY REPLAY (frames within a game are correlated); <3 replays = no val
key = Nx.Random.key(seed)

{train_idx, val_idx} =
  if data.num_replays < 3 or val_frac <= 0.0 do
    Output.puts("Split: all #{n} rows train (too few replays for a val split)")
    {Nx.iota({n}, type: :s64), nil}
  else
    n_val_replays = max(trunc(data.num_replays * val_frac), 1)
    {perm, _key} = Nx.Random.shuffle(key, Nx.iota({data.num_replays}))

    val_replays =
      perm |> Nx.slice_along_axis(0, n_val_replays, axis: 0) |> Nx.to_flat_list() |> MapSet.new()

    val_mask =
      data.replay_index
      |> Nx.to_flat_list()
      |> Enum.map(&if(MapSet.member?(val_replays, &1), do: 1, else: 0))

    train_idx = val_mask |> Enum.with_index()
                |> Enum.filter(fn {v, _} -> v == 0 end) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)
    val_idx = val_mask |> Enum.with_index()
              |> Enum.filter(fn {v, _} -> v == 1 end) |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :s64)

    Output.puts("Split: #{Nx.size(train_idx)} train / #{Nx.size(val_idx)} val rows " <>
      "(#{n_val_replays} replays held out)")

    {train_idx, val_idx}
  end

# ---------------------------------------------------------------------------
# Pass 2: train the head on frozen features
# ---------------------------------------------------------------------------

trunk_input = Axon.input("trunk", shape: {nil, hidden})

model =
  case head do
    :autoregressive ->
      Heads.build_autoregressive_head(trunk_input,
        axis_buckets: axis_buckets, shoulder_buckets: shoulder_buckets)

    :independent ->
      Heads.build_controller_head(trunk_input, axis_buckets, shoulder_buckets)
  end

# mode: :train build only for init (its predict returns %{prediction, state});
# the head has no dropout, so the inference forward is the training forward.
{init_fn, _train_mode_fn} = Utils.build_compiled(model, mode: :train)
{_init2, infer_fn} = Utils.build_compiled(model, mode: :inference)
predict_fn = infer_fn

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

loss_for = fn fwd, params, feats, actions ->
  {b, mx, my, cx, cy, sh} = fwd.(Utils.ensure_model_state(params), build_inputs.(feats, actions))

  ExPhil.Networks.Policy.imitation_loss(
    %{buttons: b, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh},
    actions
  )
end

train_step =
  Nx.Defn.jit(
    fn params, opt_state, feats, actions ->
      {loss, grads} = Nx.Defn.value_and_grad(fn p -> loss_for.(predict_fn, p, feats, actions) end).(params)
      {updates, opt_state} = opt_update.(grads, opt_state, params)
      {Polaris.Updates.apply_updates(params, updates), opt_state, loss}
    end,
    compiler: EXLA, on_conflict: :reuse
  )

eval_loss = Nx.Defn.jit(
  fn params, feats, actions -> loss_for.(infer_fn, params, feats, actions) end,
  compiler: EXLA, on_conflict: :reuse
)

take_rows = fn idx ->
  {Nx.take(data.features, idx),
   Map.new(data.actions, fn {k, v} -> {k, Nx.take(v, idx)} end)}
end

n_train = Nx.size(train_idx)
steps_per_epoch = div(n_train, batch_size)
Output.puts("⏳ JIT compiling head fit (first batch)...")

{params_data, _opt, _key} =
  Enum.reduce(1..epochs, {head_params.data, opt_state, key}, fn epoch, {pd, os, key} ->
    {perm, key} = Nx.Random.shuffle(key, train_idx)

    {pd, os, losses} =
      Enum.reduce(0..(steps_per_epoch - 1), {pd, os, []}, fn s, {pd, os, acc} ->
        idx = Nx.slice_along_axis(perm, s * batch_size, batch_size, axis: 0)
        {feats, actions} = take_rows.(idx)
        {pd, os, loss} = train_step.(pd, os, feats, actions)

        if rem(s, 100) == 0 do
          IO.write(:stderr, "\r  epoch #{epoch} #{s}/#{steps_per_epoch} loss #{Float.round(Nx.to_number(loss), 4)}\e[K")
        end

        {pd, os, [Nx.to_number(loss) | acc]}
      end)

    IO.write(:stderr, "\n")

    val =
      if val_idx != nil and Nx.size(val_idx) > 0 do
        {vf, va} = take_rows.(Nx.slice_along_axis(val_idx, 0, min(Nx.size(val_idx), 50_000), axis: 0))
        Nx.to_number(eval_loss.(pd, vf, va))
      else
        nil
      end

    train_mean = Enum.sum(losses) / max(length(losses), 1)
    Output.puts("epoch #{epoch}: train #{Float.round(train_mean, 4)}" <>
      if(val, do: " val #{Float.round(val, 4)}", else: ""))

    {pd, os, key}
  end)

# ---------------------------------------------------------------------------
# Export: source params + new head params, config.head updated
# ---------------------------------------------------------------------------

{:ok, %{params: src_params, config: src_config}} = ExPhil.Training.Checkpoint.load_policy(policy_path)

src_data =
  case src_params do
    %Axon.ModelState{data: d} -> d
    m when is_map(m) -> m
  end

to_bin = fn t -> Nx.backend_copy(t, Nx.BinaryBackend) end

new_head_data =
  params_data
  |> Map.new(fn {layer, ps} -> {layer, Map.new(ps, fn {k, t} -> {k, to_bin.(t)} end)} end)

merged =
  case head do
    :autoregressive ->
      # Keep the old independent head params in place (harmless; the agent
      # dispatches on config.head) and ADD the ar_* layers.
      Map.merge(src_data, new_head_data)

    :independent ->
      # Control: REPLACE the six head layers with the re-fit ones.
      Map.merge(src_data, new_head_data)
  end

config = src_config |> Map.put(:head, head)

spec = Edifice.Spec.new(:exphil_policy, Map.to_list(config), external: true)
File.mkdir_p!(Path.dirname(out))
Edifice.Checkpoint.save(merged, out, spec: spec, metadata: %{config: config})

Output.success("Exported #{head} head policy -> #{out}")
Output.puts("Deploy: mix run scripts/play_dolphin_async.exs --policy #{out} ...")
