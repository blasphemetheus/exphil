# Sampler probe (2026-09-09): does the LIVE decode path ever emit a full
# stick deflection from standing?
#
# Chain so far (V2_PREP passivity verdict): the model's distribution says
# ~0.5-1.2%/frame full deflection on the bot's own WAIT frames; the bot
# realized ZERO in ~7,600 frames across live + headless. Training, carry,
# projectiles, same-frame re-stepping all cleared. This reproduces the
# actual live inference path offline — Edifice.Recurrent.init_state/step
# (the --stateful-step trunk) + Policy.Sampling.sample_autoregressive_
# from_features (the real sampler) + Networks.to_controller_state (the
# real bucket->analog decode) — on the bot's own replay, and draws K
# samples per WAIT frame to histogram what comes out.
#
#   mix run scripts/probe_sampler_wait.exs --policy checkpoints/<p>.bin \
#     --replay eval_runs/0907_v2_live/2026-09-Mainline/Game_....slp [--port 1] [--k 100]
require Logger
Logger.configure(level: :warning)

alias ExPhil.Embeddings
alias ExPhil.Networks
alias ExPhil.Networks.Policy.Sampling
alias ExPhil.Training.{Checkpoint, Data, Output, Streaming}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replay: :string, port: :integer, k: :integer, every: :integer, max_frames: :integer,
             min_af: :integer, max_af: :integer]
  )

# --min-af/--max-af: frames-in-WAIT window to probe (default 3..inf =
# settled standing). Experts leave WAIT at f0-2 (dwell 5 frames), so the
# dash-out comparison lives at --min-af 0 --max-af 2.
min_af = opts[:min_af] || 3
max_af = opts[:max_af] || 1_000_000

policy_path = opts[:policy] || raise "--policy required"
replay = opts[:replay] || raise "--replay required"

# --port N pins the subject; otherwise resolve the unique Fox (CSS char 2)
# so expert corpus replays work too.
port =
  opts[:port] ||
    (case ExPhil.Data.Peppi.metadata(replay) do
       {:ok, meta} ->
         case Enum.filter(meta.players, &(&1.character == 2)) do
           [%{port: p}] -> p
           _ -> raise "no unique Fox in #{replay}; pass --port"
         end

       _ ->
         raise "cannot read metadata for #{replay}"
     end)
k = opts[:k] || 100
every = opts[:every] || 5
max_frames = opts[:max_frames] || 120

Output.banner("Sampler probe (live decode path, offline)")

{:ok, %{params: params, config: config}} = Checkpoint.load_policy(policy_path)

json_config =
  [String.replace(policy_path, "_best_policy.bin", "_config.json"), String.replace(policy_path, "_policy.bin", "_config.json")]
  |> Enum.uniq()
  |> Enum.find_value(%{}, fn p ->
    with true <- File.exists?(p), {:ok, raw} <- File.read(p), {:ok, cfg} <- Jason.decode(raw), do: cfg, else: (_ -> nil)
  end)

get = fn key, default -> Map.get(json_config, to_string(key)) || Map.get(config, key, Map.get(config, to_string(key), default)) end
to_atom = fn v when is_atom(v) -> v; v when is_binary(v) -> String.to_atom(v) end

embed_config =
  Embeddings.config(
    action_mode: to_atom.(get.(:action_mode, :learned)),
    character_mode: to_atom.(get.(:character_mode, :learned)),
    stage_mode: to_atom.(get.(:stage_mode, :one_hot_compact)),
    nana_mode: to_atom.(get.(:nana_mode, :compact)),
    stage_internals: get.(:stage_internals, false) in [true, "true"],
    # INVARIANTS item 4: new checkpoints have NO projectile block
    with_projectiles: get.(:with_projectiles, true) in [true, "true"],
    num_player_names: get.(:num_player_names, 0) || 0
  )

axis_buckets = get.(:axis_buckets, 16)
center = div(axis_buckets, 2)
hidden_size = get.(:hidden_size, nil) || 256
num_layers = get.(:num_layers, 2)

raw = case params do
  %Axon.ModelState{data: d} -> d
  m -> m
end

trunk_params =
  raw
  |> Map.filter(fn {kk, _} -> is_binary(kk) and (kk == "input_ln" or String.starts_with?(kk, "gru_")) end)

state0 =
  Edifice.Recurrent.init_state(trunk_params,
    batch_size: 1, hidden_size: hidden_size, num_layers: num_layers, cell_type: :gru)

{:ok, frames, _} = Streaming.parse_chunk([{replay, port}], subject_character: "Fox", show_progress: false)

dataset =
  frames
  |> Data.from_frames(embed_config: embed_config)
  |> Data.precompute_frame_embeddings(show_progress: false, use_prev_action: false)

embedded = dataset.embedded_frames
n = length(frames)
farr = :array.from_list(frames)
Output.puts("  #{Path.basename(replay)}: #{n} frames; K=#{k} samples per probed frame")

# Walk the game through the REAL step path; at WAIT frames draw K samples.
sample_opts = [temperature: 1.0, deterministic: false]
wait_action = 14

full_bucket? = fn idx -> abs(idx - center) >= 6 end

{_st, rows, printed?} =
  Enum.reduce(0..(n - 1), {state0, [], false}, fn i, {st, rows, printed?} ->
    frame = embedded |> Nx.slice_along_axis(i, 1, axis: 0)
    {features, st2} = Edifice.Recurrent.step(trunk_params, st, frame)

    f = :array.get(i, farr)
    p = f.game_state.players[1]
    action = p && trunc(p.action || 0)
    af = (p && trunc(p.action_frame || 0)) || 0

    probe_kind =
      cond do
        length(rows) >= max_frames -> nil
        action == wait_action and af >= min_af and af <= max_af and rem(i, every) == 0 -> :wait
        action in [20, 21] and rem(i, every * 7) == 0 -> :dash_control
        true -> nil
      end

    if probe_kind do
      samples =
        Enum.map(1..k, fn _ ->
          a = Sampling.sample_autoregressive_from_features(raw, features, sample_opts)
          mx = Nx.to_number(Nx.squeeze(a.main_x))
          my = Nx.to_number(Nx.squeeze(a.main_y))
          cs = Networks.to_controller_state(a, axis_buckets: axis_buckets)
          mag = max(abs(cs.main_stick.x - 0.5), abs(cs.main_stick.y - 0.5)) * 2.0
          {mx, my, mag}
        end)

      printed? =
        if not printed? do
          a = Sampling.sample_autoregressive_from_features(raw, features, sample_opts)
          IO.puts("  sampler return keys: #{inspect(Map.keys(a))}")
          true
        else
          printed?
        end

      full_b = Enum.count(samples, fn {mx, my, _} -> full_bucket?.(mx) or full_bucket?.(my) end)
      full_a = Enum.count(samples, fn {_, _, mag} -> mag >= 0.75 end)
      centered = Enum.count(samples, fn {mx, my, _} -> abs(mx - center) <= 2 and abs(my - center) <= 2 end)
      full_x = Enum.count(samples, fn {mx, _, _} -> full_bucket?.(mx) end)
      hist = samples |> Enum.map(fn {mx, _, _} -> mx end) |> Enum.frequencies()

      # What the SUBJECT actually did: Slippi records the input that ENDS a
      # state on the first frame of the NEXT state (measured 09-09: expert
      # full-X on WAIT frames is 0.0 even at entry), so the exit label is
      # the NEXT frame's controller. full-X next = a dash-out from here.
      # INVARIANTS.md item 9: the ONE definition of "input issued from frame i"
      subj_full_x =
        if ExPhil.Interp.Labels.full_x?(ExPhil.Interp.Labels.issued_input(farr, i)), do: 1, else: 0

      {st2, [{probe_kind, i, af, full_b / k, full_a / k, centered / k, hist, full_x / k, subj_full_x} | rows], printed?}
    else
      {st2, rows, printed?}
    end
  end)

rows = Enum.reverse(rows)

summ = fn kind ->
  rs = Enum.filter(rows, &(elem(&1, 0) == kind))
  nn = length(rs)
  if nn == 0 do
    "| #{kind} | 0 | - | - | - | - |"
  else
    mean = fn idx -> Float.round(Enum.sum(Enum.map(rs, &elem(&1, idx))) / nn, 4) end
    "| #{kind} | #{nn} | #{nn * k} | #{mean.(3)} | #{mean.(4)} | #{mean.(5)} | #{mean.(7)} | #{mean.(8)} |"
  end
end

agg_hist =
  rows
  |> Enum.filter(&(elem(&1, 0) == :wait))
  |> Enum.map(&elem(&1, 6))
  |> Enum.reduce(%{}, fn h, acc -> Map.merge(acc, h, fn _, a, b -> a + b end) end)
  |> Enum.sort()
  |> Enum.map(fn {b, c} -> "#{b}:#{c}" end)
  |> Enum.join(" ")

IO.puts("""

| context | frames probed | samples | full (either axis, bucket) | full (analog >= 0.75) | centered (+-2) | MODEL full-X/frame | SUBJECT full-X NEXT frame |
|---|---:|---:|---:|---:|---:|---:|---:|
#{summ.(:wait)}
#{summ.(:dash_control)}

main_x bucket histogram over all WAIT samples (bucket:count, center=#{center}):
#{agg_hist}

Reading: WAIT "full" ~0.005-0.012 = sampler matches the model (paradox
moves downstream: controller send / game-side). WAIT full = 0 while
dash_control full >> 0 = the sampler/decode suppresses outer buckets
in WAIT contexts specifically. Bucket>=6 vs analog>=0.75 disagreeing =
the undiscretize mapping.
""")
