# Jab-continuation probe (Bradley 09-05; labels SUCCESSOR-ALIGNED 09-09, GOTCHA #113).
#
# The prev-action-channel hypothesis is DEAD at the config: the fox_gen
# line trains with use_prev_action=false. The live suspect is
# RATE-VS-EVENT COMPOUNDING: continuation fires if A is pressed on ANY
# actionable jab1 frame (~12 of them), so expert 12.7% continuation
# implies a per-frame press rate under 1%/frame — a model that learned a
# "low-looking" 10%/frame still chains ~75% of the time. Same math
# family as the taunt floor (V2_PREP item 0), one level up: per-state
# instead of global.
#
# Measurement: run the policy over EXPERT windows ending on jab1 frames
# (subject in action 44, A currently up) and read the button head's
# per-frame p(A); compare to the experts' per-frame press rate on the
# same frames. Control: p(A) on WAIT (standing, action 14) frames.
#
#   mix run scripts/probe_jab_conditional.exs \
#     --policy checkpoints/fox_gen_v16a_cleanloss_20260905_183735_best_policy.bin \
#     --replays 'replays/erickfm_ranked/shakeout_v15/*.slp' --limit-files 40
#
# Options: --policy PATH (required) · --replays GLOB · --limit-files N [40]
#          --window N [60] · --max-windows N [768] · --batch N [128]
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Embeddings
alias ExPhil.Networks.Policy
alias ExPhil.Training.{Checkpoint, Data, Output, Streaming}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, limit_files: :integer,
             window: :integer, max_windows: :integer, batch: :integer]
  )

policy_path = opts[:policy] || raise "--policy required"
glob = opts[:replays] || "replays/erickfm_ranked/shakeout_v15/*.slp"
limit = opts[:limit_files] || 40
window = opts[:window] || 60
max_windows = opts[:max_windows] || 768
batch_size = opts[:batch] || 128

Output.banner("Jab-continuation probe")

# ---- Load checkpoint ----------------------------------------------------
{:ok, %{params: params, config: config}} = Checkpoint.load_policy(policy_path)

# The .bin's embedded metadata is impoverished (it lacked
# num_player_names, 16:26 run) — the sidecar _config.json is the full
# record; its values win.
json_config =
  [
    String.replace(policy_path, "_best_policy.bin", "_config.json"),
    String.replace(policy_path, "_policy.bin", "_config.json"),
    String.replace(policy_path, ~r/\.(axon|bin)$/, "_config.json")
  ]
  |> Enum.uniq()
  |> Enum.find_value(%{}, fn p ->
    with true <- File.exists?(p),
         {:ok, raw} <- File.read(p),
         {:ok, cfg} <- Jason.decode(raw) do
      cfg
    else
      _ -> nil
    end
  end)

get = fn key, default ->
  Map.get(json_config, to_string(key)) ||
    Map.get(config, key, Map.get(config, to_string(key), default))
end

to_atom = fn
  v when is_atom(v) -> v
  v when is_binary(v) -> String.to_atom(v)
end

# ---- Embed config (eval_model.exs recipe, condensed) --------------------
embed_opts =
  [
    action_mode: to_atom.(get.(:action_mode, :learned)),
    character_mode: to_atom.(get.(:character_mode, :learned)),
    stage_mode: to_atom.(get.(:stage_mode, :one_hot_compact)),
    nana_mode: to_atom.(get.(:nana_mode, :compact)),
    stage_internals: get.(:stage_internals, false) in [true, "true"],
    # INVARIANTS item 4: new checkpoints have NO projectile block
    with_projectiles: get.(:with_projectiles, true) in [true, "true"],
    num_player_names: get.(:num_player_names, 0) || 0
  ]

embed_config = Embeddings.config(embed_opts)
embed_size = Embeddings.embedding_size(embed_config)

canary = get.(:embed_canary, nil)

if is_list(canary) and length(canary) != embed_size do
  raise "embed size mismatch: canary #{length(canary)} vs reconstructed #{embed_size} — embed_opts wrong"
end

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Embed size", embed_size},
  {"Window", window},
  {"Files (max)", limit}
])

# ---- Model (imitation.ex bptt build, verbatim opts) ---------------------
model =
  Policy.build_temporal_bptt(
    head: :autoregressive,
    embed_size: embed_size,
    backbone: :gru,
    window_size: window,
    hidden_size: get.(:hidden_size, nil) || 256,
    num_layers: get.(:num_layers, 2),
    dropout: get.(:dropout, 0.1),
    axis_buckets: get.(:axis_buckets, 16),
    shoulder_buckets: get.(:shoulder_buckets, 4)
  )

{_init_fn, predict_fn} = Axon.build(model, mode: :inference, compiler: EXLA)

hidden_size = get.(:hidden_size, nil) || 256
num_layers = get.(:num_layers, 2)

# ---- Collect expert windows --------------------------------------------
files =
  glob
  |> Path.wildcard()
  |> Enum.sort()
  |> Enum.take(limit * 3)

# Resolve the fox port per file (CSS char id 2); skip dittos/absent.
tuples =
  files
  |> Enum.flat_map(fn path ->
    case Peppi.metadata(path) do
      {:ok, meta} ->
        case Enum.filter(meta.players, &(&1.character == 2)) do
          [%{port: p}] -> [{path, p}]
          _ -> []
        end

      _ ->
        []
    end
  end)
  |> Enum.take(limit)

Output.puts("  #{length(tuples)} files with a unique Fox")

jab1_action = 44
wait_action = 14

# Per file: parse (subject remapped to port 1), embed with the training
# path, then index jab1/wait frames. Windows never cross file bounds.
collect = fn {path, port}, acc ->
  {jab_acc, wait_acc} = acc

  if length(jab_acc) >= max_windows and length(wait_acc) >= max_windows do
    acc
  else
    case Streaming.parse_chunk([{path, port}],
           subject_character: "Fox",
           show_progress: false
         ) do
      {:ok, frames, _errors} when length(frames) > window + 10 ->
        dataset =
          frames
          |> Data.from_frames(embed_config: embed_config)
          |> Data.precompute_frame_embeddings(show_progress: false, use_prev_action: false)

        embedded = dataset.embedded_frames
        n = length(frames)
        farr = :array.from_list(frames)

        # state only — inputs come from ExPhil.Interp.Labels (item 9)
        subj = fn i -> {:array.get(i, farr).game_state.players[1], nil} end

        idxs = window..(n - 1)

        {jabs, waits} =
          Enum.reduce(idxs, {[], []}, fn i, {js, ws} ->
            # SUCCESSOR-aligned (GOTCHA #113 / INVARIANTS item 9, 09-09): the
            # A the player pressed FROM frame i is recorded on i+1; the
            # producing input (frame i's own controller) is the "A up" gate.
            {p, _ctrl} = subj.(i)
            action = p && trunc(p.action || 0)
            producing = ExPhil.Interp.Labels.producing_input(farr, i)
            issued = ExPhil.Interp.Labels.issued_input(farr, i)
            prev_a = producing && producing.button_a
            label = if issued && issued.button_a, do: 1, else: 0

            af = (p && trunc(p.action_frame || 0)) || 0

            cond do
              action == jab1_action and prev_a == false ->
                {[{i, label, af} | js], ws}

              action == wait_action and prev_a == false and rem(i, 37) == 0 ->
                {js, [{i, label, af} | ws]}

              true ->
                {js, ws}
            end
          end)

        take_windows = fn picks ->
          Enum.map(picks, fn {i, label, af} ->
            w = Nx.slice_along_axis(embedded, i - window + 1, window, axis: 0)
            {w, label, af}
          end)
        end

        {take_windows.(jabs) ++ jab_acc, take_windows.(waits) ++ wait_acc}

      _ ->
        acc
    end
  end
end

{jab_windows, wait_windows} =
  Enum.reduce(tuples, {[], []}, fn tuple, acc ->
    {j, w} = collect.(tuple, acc)
    IO.write(:stderr, "\r  windows: jab1 #{length(j)} | wait #{length(w)}\e[K")
    {j, w}
  end)

IO.write(:stderr, "\n")

jab_windows = Enum.take(jab_windows, max_windows)
wait_windows = Enum.take(wait_windows, max_windows)

if jab_windows == [], do: raise("no jab1 windows found")

# ---- Run the model ------------------------------------------------------
run = fn windows ->
  windows
  |> Enum.chunk_every(batch_size)
  |> Enum.flat_map(fn chunk ->
    b = length(chunk)
    seq = chunk |> Enum.map(&elem(&1, 0)) |> Nx.stack()

    inputs = %{
      "state_sequence" => seq,
      "initial_hidden" => Nx.broadcast(0.0, {b, num_layers, hidden_size}),
      "tf_buttons" => Nx.broadcast(0.0, {b, window, 8}),
      "tf_main_x" => Nx.broadcast(Nx.tensor(0, type: :s64), {b, window}),
      "tf_main_y" => Nx.broadcast(Nx.tensor(0, type: :s64), {b, window}),
      "tf_c_x" => Nx.broadcast(Nx.tensor(0, type: :s64), {b, window}),
      "tf_c_y" => Nx.broadcast(Nx.tensor(0, type: :s64), {b, window})
    }

    {logits, _final_hidden} = predict_fn.(params, inputs)
    buttons = elem(logits, 0)

    p_a =
      buttons
      |> Nx.slice_along_axis(window - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
      |> Nx.slice_along_axis(0, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
      |> Nx.sigmoid()
      |> Nx.to_flat_list()

    Enum.zip_with([p_a, chunk], fn [p, {_w, label, af}] -> {p, label, af} end)
  end)
end

jab_results = run.(jab_windows)
wait_results = run.(wait_windows)

# ---- Report -------------------------------------------------------------
stats = fn results ->
  ps = Enum.map(results, &elem(&1, 0))
  labels = Enum.map(results, &elem(&1, 1))
  n = length(ps)
  sorted = Enum.sort(ps)

  %{
    n: n,
    expert_press_rate: Float.round(Enum.sum(labels) / n, 4),
    model_mean_p: Float.round(Enum.sum(ps) / n, 4),
    model_median_p: Float.round(Enum.at(sorted, div(n, 2)), 4),
    model_p90: Float.round(Enum.at(sorted, trunc(n * 0.9)), 4)
  }
end

# Timing structure within jab1 (action_frame buckets): jab2 only chains
# from presses in the LATE window, so where the presses land decides
# whether they convert. Jab1 IASA ~frame 12-16 for Fox.
bucket_report = fn results ->
  results
  |> Enum.group_by(fn {_p, _l, af} ->
    cond do
      af <= 5 -> "jab1 f0-5 (dead-early)"
      af <= 11 -> "jab1 f6-11 (chain window)"
      true -> "jab1 f12+ (late/IASA)"
    end
  end)
  |> Enum.sort()
  |> Enum.map(fn {bucket, rows} ->
    s = stats.(rows)
    "| #{bucket} | #{s.n} | #{s.expert_press_rate} | #{s.model_mean_p} | #{s.model_median_p} | #{s.model_p90} |"
  end)
end

jab = stats.(jab_results)
wait = stats.(wait_results)

# ~12 actionable continuation frames in jab1
implied = fn per_frame -> Float.round(1.0 - :math.pow(1.0 - per_frame, 12), 3) end

IO.puts("""

| context | n frames | expert press/frame | model mean p(A) | median | p90 | implied continuation (12f): expert vs model |
|---|---:|---:|---:|---:|---:|---|
| jab1, A up | #{jab.n} | #{jab.expert_press_rate} | #{jab.model_mean_p} | #{jab.model_median_p} | #{jab.model_p90} | #{implied.(jab.expert_press_rate)} vs #{implied.(jab.model_mean_p)} |
| WAIT (control) | #{wait.n} | #{wait.expert_press_rate} | #{wait.model_mean_p} | #{wait.model_median_p} | #{wait.model_p90} | — |

By frame-within-jab1 (expert presses that convert to jab2 come from the late window):

| bucket | n | expert press/frame | model mean p(A) | median | p90 |
|---|---:|---:|---:|---:|---:|
#{Enum.join(bucket_report.(jab_results), "\n")}

Corpus action-level jab1->jab2 continuation: 0.127 (pathology_scan 09-05).
Reading: if the model's implied continuation lands near the OBSERVED bot
continuation (~0.88) while expert-implied lands near 0.127, per-frame
rate compounding IS the mechanism — the per-frame conditional is
miscalibrated upward, not a self-conditioning loop (no prev-action
channel exists in this line).
""")
