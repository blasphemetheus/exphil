# F4 — combo-depth probe (09-01, Bradley's live-look question: "the only
# punish it gets is 1-2 hits; 10-hit combos are in the data; how do we show
# it the deeper layers?").
#
# Splits "knowledge missing" from "expression missing" the way the
# coincidence probe did for the AR wire: put the MODEL on the EXPERT'S own
# states MID-PUNISH (teacher-forced states, no state-visitation confound)
# and measure whether its sampled actions match the expert's continuation,
# STRATIFIED BY COMBO DEPTH (how many hits into the punish the state is).
#
#   match(depth 2+) ~ match(depth 1) ~ match(neutral)
#       -> the continuations ARE learned; live shallowness is
#          STATE-VISITATION (it never creates/holds the follow-up state)
#       -> levers: drills/DAgger from hit-confirm savestates, closed-loop
#   match collapses with depth
#       -> BC never learned deep continuations (rare-frame underweighting)
#       -> levers: conversion-window curation / AWBC (damage RTG upweights
#          exactly these frames), oversampling deep-punish windows
#
# Depth = number of opponent-hitstun RISING EDGES since the current
# conversion window opened (Situations :conversion_open/:combo_active).
#
#   mix run scripts/combo_depth_probe.exs \
#     --policy checkpoints/fox_gen_v1.3_ARrefit_policy.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --limit-files 24 --k 16 --out eval_runs/0901_combo_depth/RESULTS.md
#
# Options: --char-id (2) · --temperature (0.5) · --seed (20260901) ·
#   --max-per-stratum (2500)
require Logger
Logger.configure(level: :warning)
Code.require_file("lib/critic_features.exs", __DIR__)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, limit_files: :integer, k: :integer,
             char_id: :integer, temperature: :float, seed: :integer,
             max_per_stratum: :integer, out: :string]
  )

policy = opts[:policy] || raise "--policy required"
glob = opts[:replays] || raise "--replays required"
limit_files = opts[:limit_files] || 24
k = opts[:k] || 16
char_id = opts[:char_id] || 2
temperature = opts[:temperature] || 0.5
seed = opts[:seed] || 20_260_901
cap = opts[:max_per_stratum] || 2500

Output.banner("F4 combo-depth probe")

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files * 4)

resolve = fn path ->
  case Peppi.metadata(path) do
    {:ok, meta} ->
      case Enum.filter(meta.players, &(&1.character == char_id)) do
        [%{port: p}] -> {:ok, p}
        _ -> :skip
      end

    _ -> :skip
  end
end

picked =
  files
  |> Enum.flat_map(fn f ->
    case resolve.(f) do
      {:ok, p} -> [{f, p}]
      _ -> []
    end
  end)
  |> Enum.take(limit_files)

if picked == [], do: raise("no fox-resolved files")

trunk = Activations.load_trunk(policy)
heads = Activations.load_heads_only(policy)
window = trunk.window
key = Nx.Random.key(seed)

Output.config([
  {"Policy", Path.basename(policy)},
  {"Files", length(picked)},
  {"K samples", k},
  {"Temperature", temperature}
])

hitstun? = fn pl -> pl != nil and (pl.hitstun_frames_left || 0) > 0 end

# Per replay: capture trunk feats (clean :r2 cache) + per-frame stratum.
# Stratum: :neutral | {:combo, depth} — depth = opp-hitstun rising edges
# since the conversion opened; nil = neither (skipped).
rows =
  picked
  |> Enum.with_index()
  |> Enum.flat_map(fn {{path, port}, fi} ->
    opp = if port == 1, do: 2, else: 1
    cap_r = Activations.capture_replay(trunk, path, player_port: port, opponent_port: opp, labels: false)

    {:ok, replay} = Peppi.parse(path)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: port, opponent_port: opp, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))

    states = Enum.map(frames, & &1.game_state)
    # frames are remapped: subject = 1, opponent = 2
    sits = Situations.label_states(states, 1, as: :set)

    {strata, _} =
      Enum.zip(states, sits)
      |> Enum.map_reduce({0, false}, fn {gs, set}, {depth, prev_hs} ->
        in_conv = set != nil and (MapSet.member?(set, :combo_active) or MapSet.member?(set, :conversion_open))
        hs = hitstun?.(gs.players[2])

        depth =
          cond do
            not in_conv -> 0
            hs and not prev_hs -> depth + 1
            true -> depth
          end

        stratum =
          cond do
            in_conv and depth > 0 -> {:combo, min(depth, 4)}
            set != nil and MapSet.member?(set, :neutral) -> :neutral
            true -> nil
          end

        {stratum, {depth, hs}}
      end)

    strata_t = List.to_tuple(strata)
    frames_t = List.to_tuple(frames)
    n = cap_r.n
    off = cap_r.frame_offset

    # decision-frame filter (input changed) within each stratum
    prevs = [nil | Enum.map(Enum.drop(frames, -1), & &1.controller)] |> List.to_tuple()

    kept =
      0..(n - 1)
      |> Enum.flat_map(fn r ->
        i = off + r

        with s when s != nil <- elem(strata_t, i),
             prev when prev != nil <- elem(prevs, i),
             f = elem(frames_t, i),
             false <- CriticFeatures.controller_match?(prev, f.controller) do
          [{s, r, f}]
        else
          _ -> []
        end
      end)

    if kept == [] do
      []
    else
      idx = Nx.tensor(Enum.map(kept, &elem(&1, 1)))
      feats = Nx.take(cap_r.activations, idx, axis: 0)
      kframes = Enum.map(kept, &elem(&1, 2))
      sub = Nx.Random.fold_in(key, fi)
      {_samples, match, _} = CriticFeatures.sample_candidates(heads, feats, kframes, k, temperature, sub)
      match_b = Nx.backend_copy(match, Nx.BinaryBackend)

      kept
      |> Enum.with_index()
      |> Enum.map(fn {{s, _r, _f}, j} -> {s, match_b[j]} end)
    end
  end)

Output.puts("  #{length(rows)} decision rows collected")

by_stratum =
  rows
  |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
  |> Map.new(fn {s, ms} -> {s, Enum.take(ms, cap)} end)

stat = fn ms ->
  n = length(ms)

  if n == 0 do
    nil
  else
    m = Nx.stack(ms)
    p1 = Nx.to_number(Nx.mean(Nx.as_type(m, :f32)))
    pk = Nx.to_number(Nx.mean(Nx.as_type(Nx.greater(Nx.sum(m, axes: [1]), 0), :f32)))
    {n, p1, pk}
  end
end

order = [:neutral, {:combo, 1}, {:combo, 2}, {:combo, 3}, {:combo, 4}]
labels = %{:neutral => "neutral", {:combo, 1} => "punish hit 1", {:combo, 2} => "hit 2",
           {:combo, 3} => "hit 3", {:combo, 4} => "hit 4+"}

pct = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end

table_rows =
  order
  |> Enum.map(fn s -> {s, stat.(Map.get(by_stratum, s, []))} end)
  |> Enum.map_join("\n", fn
    {s, nil} -> "| #{labels[s]} | 0 | — | — |"
    {s, {n, p1, pk}} -> "| #{labels[s]} | #{n} | #{pct.(p1)} | #{pct.(pk)} |"
  end)

report = """
# F4 combo-depth probe — RESULTS

Policy #{Path.basename(policy)}, #{length(picked)} expert files, K=#{k}
coherent samples at T=#{temperature}, Leg S match rule, teacher-forced
expert states (no state-visitation confound). Depth = opponent-hitstun
rising edges since the conversion opened.

| stratum | rows | sampling pass@1 % | pass@#{k} % |
|---|---:|---:|---:|
#{table_rows}

Reading (pre-declared):
- flat across depth (and ~neutral level) -> continuations ARE learned;
  live 1-2-hit punishes are STATE-VISITATION (it never creates/holds the
  follow-up state) -> levers: hit-confirm savestate drills / DAgger,
  closed-loop work.
- collapses with depth -> BC never learned the deep layer (rare frames
  underweighted) -> levers: conversion-window curation / AWBC standard
  RTG (damage return-to-go upweights exactly these frames), deep-punish
  oversampling.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
