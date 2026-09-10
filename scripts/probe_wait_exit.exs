# WAIT-exit calibration probe (2026-09-08; labels SUCCESSOR-ALIGNED 09-09 —
# the 09-08 same-frame labels were leaked, GOTCHA #113) — which KIND of
# passivity is v2's?
#
# v2 idles 6.5x the corpus. Two mechanisms predict that, with opposite
# fixes:
#   (a) OBJECTIVE miscalibration: on EXPERT standing contexts the model's
#       per-frame p(leave WAIT) is already too low -> frame reweighting
#       (neutral_weight / transition_weight) can fix it.
#   (b) COMPOUNDING: calibrated on expert contexts, wrong only on the
#       self-generated off-distribution states its own standing creates
#       -> reweighting cannot fix it; needs interaction data (DAgger) or
#       consequences (AWBC/RL).
#
# Measurement: expert windows ending on a WAIT (action 14) frame; label =
# the expert's controller at that frame is NON-neutral (any button, or
# main stick > 0.3 from center) = "leaves standing now". Model p(leave) =
# 1 - p(no buttons) * p(main_x center | no buttons) * p(main_y center | ..)
# from the AR head, teacher-forced down the all-neutral branch. Bucketed
# by frames-in-WAIT (action_frame) so long idles are read separately.
#
#   mix run scripts/probe_wait_exit.exs --policy checkpoints/<policy>.bin
# Options: --replays GLOB · --limit-files N [40] · --window N [60]
#          --max-windows N [1200] · --batch N [128]
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Embeddings
alias ExPhil.Networks.Policy
alias ExPhil.Training.{Checkpoint, Data, Output, Streaming}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, limit_files: :integer,
             window: :integer, max_windows: :integer, batch: :integer, port: :integer,
             min_af: :integer, center_only: :boolean, carry: :boolean]
  )

# --carry (09-08): evaluate with the hidden state CARRIED across the whole
# game (chunks of --window threaded through final_hidden, exactly as bptt
# training does) instead of a fresh zero-init window per pick. Live
# stateful-step inference carries state; if p(full deflection) collapses
# only under carry, the drift lives in the recurrent state on the bot's
# self-generated standing, not in the sampler.
carry? = opts[:carry] || false

# --min-af N: minimum frames-in-WAIT for a window (default 3 = settled
# standing). 0 includes the ENTRY frames — where experts do most of their
# leaving (corpus dwell = 5 frames/entry, 2026-09-08 pathology_scan).
min_af = opts[:min_af] || 3

# --port N pins the subject port (bot replays: port 1) instead of resolving
# the unique Fox per file — lets the probe run on the bot's OWN games, so
# p(leave) on self-generated standing states can be compared with the
# expert-context number (the compounding test).

policy_path = opts[:policy] || raise "--policy required"
glob = opts[:replays] || "replays/erickfm_ranked/shakeout_v15/*.slp"
limit = opts[:limit_files] || 40
window = opts[:window] || 60
max_windows = opts[:max_windows] || 1200
batch_size = opts[:batch] || 128

Output.banner("WAIT-exit calibration probe")

{:ok, %{params: params, config: config}} = Checkpoint.load_policy(policy_path)

json_config =
  [
    String.replace(policy_path, "_best_policy.bin", "_config.json"),
    String.replace(policy_path, "_policy.bin", "_config.json"),
    String.replace(policy_path, ~r/\.(axon|bin)$/, "_config.json")
  ]
  |> Enum.uniq()
  |> Enum.find_value(%{}, fn p ->
    with true <- File.exists?(p), {:ok, raw} <- File.read(p), {:ok, cfg} <- Jason.decode(raw) do
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

embed_opts = [
  action_mode: to_atom.(get.(:action_mode, :learned)),
  character_mode: to_atom.(get.(:character_mode, :learned)),
  stage_mode: to_atom.(get.(:stage_mode, :one_hot_compact)),
  nana_mode: to_atom.(get.(:nana_mode, :compact)),
  stage_internals: get.(:stage_internals, false) in [true, "true"],
  action_frame_buckets: get.(:action_frame_buckets, 0) || 0,
  # INVARIANTS item 4: new checkpoints have NO projectile block
  with_projectiles: get.(:with_projectiles, true) in [true, "true"],
  num_player_names: get.(:num_player_names, 0) || 0
]

embed_config = Embeddings.config(embed_opts)
embed_size = Embeddings.embedding_size(embed_config)
canary = get.(:embed_canary, nil)

if is_list(canary) and length(canary) != embed_size do
  raise "embed size mismatch: canary #{length(canary)} vs reconstructed #{embed_size}"
end

axis_buckets = get.(:axis_buckets, 16)
center = div(axis_buckets, 2)
hidden_size = get.(:hidden_size, nil) || 256
num_layers = get.(:num_layers, 2)

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Embed size", embed_size},
  {"Center bucket", center},
  {"Files (max)", limit}
])

model =
  Policy.build_temporal_bptt(
    head: :autoregressive,
    embed_size: embed_size,
    backbone: :gru,
    window_size: window,
    hidden_size: hidden_size,
    num_layers: num_layers,
    dropout: get.(:dropout, 0.1),
    axis_buckets: axis_buckets,
    shoulder_buckets: get.(:shoulder_buckets, 4)
  )

{_init_fn, predict_fn} = Axon.build(model, mode: :inference, compiler: EXLA)

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit * 3)

tuples =
  files
  |> Enum.flat_map(fn path ->
    if opts[:port] do
      [{path, opts[:port]}]
    else
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == 2)) do
            [%{port: p}] -> [{path, p}]
            _ -> []
          end

        _ ->
          []
      end
    end
  end)
  |> Enum.take(limit)

Output.puts("  #{length(tuples)} files with a unique Fox")

wait_action = 14

non_neutral? = fn
  nil ->
    false

  c ->
    buttons =
      [c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r]
      |> Enum.any?(&(&1 == true))

    stick =
      case c.main_stick do
        %{x: x, y: y} when is_number(x) and is_number(y) -> abs(x - 0.5) > 0.15 or abs(y - 0.5) > 0.15
        _ -> false
      end

    buttons or stick
end

# expert FULL deflection (>= 0.75 of range on either axis) at this frame
full_deflect? = fn
  nil -> false
  c ->
    case c.main_stick do
      %{x: x, y: y} when is_number(x) and is_number(y) -> abs(x - 0.5) * 2 >= 0.75 or abs(y - 0.5) * 2 >= 0.75
      _ -> false
    end
end

collect = fn {path, port}, acc ->
  if length(acc) >= max_windows do
    acc
  else
    case Streaming.parse_chunk([{path, port}], subject_character: "Fox", show_progress: false) do
      {:ok, frames, _errors} when length(frames) > window + 10 ->
        dataset =
          frames
          |> Data.from_frames(embed_config: embed_config)
          |> Data.precompute_frame_embeddings(show_progress: false, use_prev_action: false)

        embedded = dataset.embedded_frames
        n = length(frames)
        farr = :array.from_list(frames)

        picks =
          Enum.reduce(window..(n - 1), [], fn i, ps ->
            f = :array.get(i, farr)
            p = f.game_state.players[1]
            action = p && trunc(p.action || 0)
            af = (p && trunc(p.action_frame || 0)) || 0
            # settled standing only (>= 3 frames in WAIT); subsample so
            # one long idle doesn't dominate
            # entry frames (af < 3) are rare per episode: keep them all;
            # subsample the long settled stretches
            keep = af < 3 or rem(i, 5) == 0

            if action == wait_action and af >= min_af and keep do
              # SUCCESSOR-aligned labels (GOTCHA #113 / INVARIANTS item 9): the
              # input issued FROM this frame is recorded on frame i+1. The
              # 09-08 version labelled with f.controller (same frame) and was
              # blind to a 1000x miscalibration.
              issued = ExPhil.Interp.Labels.issued_input(farr, i)
              label = if non_neutral?.(issued), do: 1, else: 0
              full = if full_deflect?.(issued), do: 1, else: 0
              [{i, label, af, full} | ps]
            else
              ps
            end
          end)

        windows =
          Enum.map(picks, fn {i, label, af, full} ->
            {Nx.slice_along_axis(embedded, i - window + 1, window, axis: 0), label, af, full}
          end)

        windows ++ acc

      _ ->
        acc
    end
  end
end

windows =
  if carry? do
    []
  else
    Enum.reduce(tuples, [], fn tuple, acc ->
      acc = collect.(tuple, acc)
      IO.write(:stderr, "\r  WAIT windows: #{length(acc)}\e[K")
      acc
    end)
    |> Enum.take(max_windows)
  end

IO.write(:stderr, "\n")
if not carry? and windows == [], do: raise("no WAIT windows found")

softmax_last = fn t -> Nx.exp(Nx.subtract(t, Nx.reduce_max(t, axes: [-1], keep_axes: true))) |> then(&Nx.divide(&1, Nx.sum(&1, axes: [-1], keep_axes: true))) end

# Per-timestep decode of one predict output into {p_leave, p_full} lists
# (shared by the batched-window path, last timestep only, and the carry
# path, every timestep).
band = if opts[:center_only], do: 1, else: 5
band_start = if opts[:center_only], do: center, else: center - 2

per_step = fn logits ->
  # logits tuple from the model: {buttons {b,t,8}, main_x {b,t,17}, main_y, ...}
  p_no_buttons =
    logits |> elem(0) |> Nx.sigmoid() |> then(&Nx.subtract(1.0, &1)) |> Nx.product(axes: [2])

  band_mass = fn t ->
    t |> softmax_last.() |> Nx.slice_along_axis(band_start, band, axis: 2) |> Nx.sum(axes: [2])
  end

  outer = fn t ->
    sm = softmax_last.(t)
    lo = sm |> Nx.slice_along_axis(0, center - 5, axis: 2) |> Nx.sum(axes: [2])
    hi = sm |> Nx.slice_along_axis(center + 6, axis_buckets + 1 - (center + 6), axis: 2) |> Nx.sum(axes: [2])
    Nx.add(lo, hi)
  end

  p_leave = Nx.subtract(1.0, Nx.multiply(p_no_buttons, Nx.multiply(band_mass.(elem(logits, 1)), band_mass.(elem(logits, 2)))))
  fx = outer.(elem(logits, 1))
  fy = outer.(elem(logits, 2))
  p_full = Nx.subtract(1.0, Nx.multiply(Nx.subtract(1.0, fx), Nx.subtract(1.0, fy)))
  {p_leave, p_full}
end

carry_results = fn ->
  Enum.flat_map(tuples, fn {path, port} ->
    case Streaming.parse_chunk([{path, port}], subject_character: "Fox", show_progress: false) do
      {:ok, frames, _} when length(frames) > window + 10 ->
        dataset =
          frames
          |> Data.from_frames(embed_config: embed_config)
          |> Data.precompute_frame_embeddings(show_progress: false, use_prev_action: false)

        embedded = dataset.embedded_frames
        n = length(frames)
        farr = :array.from_list(frames)
        nchunks = div(n, window)
        center_t = Nx.broadcast(Nx.tensor(center, type: :s64), {1, window})
        h0 = Nx.broadcast(0.0, {1, num_layers, hidden_size})

        {per_chunk, _h} =
          Enum.map_reduce(0..(nchunks - 1), h0, fn c, h ->
            seq = embedded |> Nx.slice_along_axis(c * window, window, axis: 0) |> Nx.new_axis(0)

            inputs = %{
              "state_sequence" => seq,
              "initial_hidden" => h,
              "tf_buttons" => Nx.broadcast(0.0, {1, window, 8}),
              "tf_main_x" => center_t,
              "tf_main_y" => center_t,
              "tf_c_x" => center_t,
              "tf_c_y" => center_t
            }

            {logits, h_final} = predict_fn.(params, inputs)
            {pl, pf} = per_step.(logits)
            {Enum.zip(Nx.to_flat_list(pl), Nx.to_flat_list(pf)), h_final}
          end)

        flat = List.flatten(per_chunk)
        IO.write(:stderr, "\r  carry: #{Path.basename(path)} (#{n} frames)\e[K")

        flat
        |> Enum.with_index()
        |> Enum.flat_map(fn {{pl, pf}, i} ->
          f = :array.get(i, farr)
          p = f.game_state.players[1]
          action = p && trunc(p.action || 0)
          af = (p && trunc(p.action_frame || 0)) || 0
          keep = af < 3 or rem(i, 5) == 0

          if action == wait_action and af >= min_af and keep do
            issued = ExPhil.Interp.Labels.issued_input(farr, i)
            label = if non_neutral?.(issued), do: 1, else: 0
            full = if full_deflect?.(issued), do: 1, else: 0
            [{pl, label, af, pf, full}]
          else
            []
          end
        end)

      _ ->
        []
    end
  end)
  |> Enum.take(max_windows)
end

batched_results = fn ->
  windows
  |> Enum.chunk_every(batch_size)
  |> Enum.flat_map(fn chunk ->
    b = length(chunk)
    seq = chunk |> Enum.map(&elem(&1, 0)) |> Nx.stack()
    center_t = Nx.broadcast(Nx.tensor(center, type: :s64), {b, window})

    inputs = %{
      "state_sequence" => seq,
      "initial_hidden" => Nx.broadcast(0.0, {b, num_layers, hidden_size}),
      # teacher-force the ALL-NEUTRAL branch: no buttons, sticks centered
      "tf_buttons" => Nx.broadcast(0.0, {b, window, 8}),
      "tf_main_x" => center_t,
      "tf_main_y" => center_t,
      "tf_c_x" => center_t,
      "tf_c_y" => center_t
    }

    {logits, _} = predict_fn.(params, inputs)
    last = fn t -> t |> Nx.slice_along_axis(window - 1, 1, axis: 1) |> Nx.squeeze(axes: [1]) end

    p_no_buttons =
      logits |> elem(0) |> last.() |> Nx.sigmoid() |> then(&Nx.subtract(1.0, &1)) |> Nx.product(axes: [1])

    # ALIGNED neutral band (09-08 fix): the label calls a stick "neutral"
    # within 0.15 of center; with 16 buckets over [0,1] that is +-2 buckets
    # (0.125). Counting only the exact center bucket as neutral overcounted
    # model "leaves" — mass on the adjacent buckets is a 6% deflection that
    # does not move Fox. --center-only restores the old (loose) definition.
    band = if opts[:center_only], do: 1, else: 5
    band_start = if opts[:center_only], do: center, else: center - 2

    p_mx_center = logits |> elem(1) |> last.() |> softmax_last.() |> Nx.slice_along_axis(band_start, band, axis: 1) |> Nx.sum(axes: [1])
    p_my_center = logits |> elem(2) |> last.() |> softmax_last.() |> Nx.slice_along_axis(band_start, band, axis: 1) |> Nx.sum(axes: [1])

    p_leave =
      Nx.subtract(1.0, Nx.multiply(p_no_buttons, Nx.multiply(p_mx_center, p_my_center)))
      |> Nx.to_flat_list()

    # Stick MAGNITUDE breakdown (09-08 failed-exit finding: the bot never
    # full-deflects out of WAIT). Mass on the outer buckets (|b - center|
    # >= 6 of 8 = >= 0.75 deflection) per axis; "full" = either axis.
    outer_mass = fn t ->
      sm = t |> last.() |> softmax_last.()
      lo = sm |> Nx.slice_along_axis(0, center - 5, axis: 1) |> Nx.sum(axes: [1])
      hi = sm |> Nx.slice_along_axis(center + 6, axis_buckets + 1 - (center + 6), axis: 1) |> Nx.sum(axes: [1])
      Nx.add(lo, hi)
    end

    fx = outer_mass.(elem(logits, 1))
    fy = outer_mass.(elem(logits, 2))
    p_full = Nx.subtract(1.0, Nx.multiply(Nx.subtract(1.0, fx), Nx.subtract(1.0, fy))) |> Nx.to_flat_list()

    Enum.zip_with([p_leave, p_full, chunk], fn [p, pf, {_w, label, af, full}] -> {p, label, af, pf, full} end)
  end)
end

results = if carry?, do: carry_results.(), else: batched_results.()
IO.write(:stderr, "\n")
if results == [], do: raise("no WAIT windows found")

stats = fn rows ->
  ps = Enum.map(rows, &elem(&1, 0))
  labels = Enum.map(rows, &elem(&1, 1))
  pfull = Enum.map(rows, &elem(&1, 3))
  full_labels = Enum.map(rows, &elem(&1, 4))
  n = length(ps)
  sorted = Enum.sort(ps)

  %{
    n: n,
    expert: Float.round(Enum.sum(labels) / n, 4),
    model_mean: Float.round(Enum.sum(ps) / n, 4),
    model_median: Float.round(Enum.at(sorted, div(n, 2)), 4),
    model_p10: Float.round(Enum.at(sorted, div(n, 10)), 4),
    expert_full: Float.round(Enum.sum(full_labels) / n, 4),
    model_full: Float.round(Enum.sum(pfull) / n, 4)
  }
end

bucket = fn af ->
  cond do
    af == 0 -> "WAIT f0 (entry)"
    af <= 2 -> "WAIT f1-2"
    af <= 10 -> "WAIT f3-10 (just stopped)"
    af <= 30 -> "WAIT f11-30"
    af <= 90 -> "WAIT f31-90"
    true -> "WAIT f91+ (long idle)"
  end
end

all = stats.(results)

full_rows =
  results
  |> Enum.group_by(fn {_, _, af, _, _} -> bucket.(af) end)
  |> Enum.sort()
  |> Enum.map(fn {b, rs} ->
    s = stats.(rs)
    "| #{b} | #{s.n} | #{s.expert_full} | #{s.model_full} | #{Float.round(s.model_full / max(s.expert_full, 1.0e-4), 2)} |"
  end)

rows =
  results
  |> Enum.group_by(fn {_, _, af, _, _} -> bucket.(af) end)
  |> Enum.sort()
  |> Enum.map(fn {b, rs} ->
    s = stats.(rs)
    "| #{b} | #{s.n} | #{s.expert} | #{s.model_mean} | #{s.model_median} | #{s.model_p10} | #{Float.round(s.model_mean / max(s.expert, 1.0e-4), 2)} |"
  end)

IO.puts("""

Policy: #{Path.basename(policy_path)}

| context | n frames | expert leave/frame | model mean p(leave) | median | p10 | model/expert |
|---|---:|---:|---:|---:|---:|---:|
| ALL settled WAIT | #{all.n} | #{all.expert} | #{all.model_mean} | #{all.model_median} | #{all.model_p10} | #{Float.round(all.model_mean / max(all.expert, 1.0e-4), 2)} |
#{Enum.join(rows, "\n")}

FULL stick deflection (>= 0.75 on either axis — a dash out of standing):

| context | n | expert full/frame | model p(full) | model/expert |
|---|---:|---:|---:|---:|
| ALL | #{all.n} | #{all.expert_full} | #{all.model_full} | #{Float.round(all.model_full / max(all.expert_full, 1.0e-4), 2)} |
#{Enum.join(full_rows, "\n")}

Reading: model/expert ~1 on EXPERT standing contexts = calibrated ->
passivity is (b) compounding on self-generated states (reweighting won't
fix it). model/expert well below 1 = (a) objective miscalibration ->
neutral_weight / transition_weight are the right levers.
""")
