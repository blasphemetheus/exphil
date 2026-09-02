# Edgeguard pass@K probe (09-02, Bradley's live look: "it doesn't even
# seem to know you're below stage" — bot does on-stage behavior while the
# human recovers; never takes ledge).
#
# The F4-analog discriminator for the EDGEGUARD gap. Unlike hit-confirms
# this is NOT a visitation question — the state occurred live and the
# corpus is dense there (expert 4.34 edgeguard eps/min, dash-first 41.5%
# vs bot 3.9% in score24). So: put the MODEL on the EXPERT'S own
# opponent-offstage states (teacher-forced, no visitation confound) and
# measure action match, stratified by edgeguard phase/position:
#
#   match(edgeguard strata) ~ match(neutral)
#       -> the conditioning IS learned; live blindness is selection
#          (marginal-collapse at T=0.5) -> the critic-selector experiment
#          covers it
#   match collapses in edgeguard strata (esp. opp-low)
#       -> BC lost the opponent-offstage conditioning -> levers: curation/
#          AWBC on edgeguard windows + an edgeguard drill (scripted
#          recover-low dummy on the drill driver)
#
# Strata: :neutral | {:eg, :onset} (first 60 f after :edgeguard rises —
# the edge_scorecard first-option window) | {:eg, :low} (sustained,
# opponent below stage level, Bradley's case) | {:eg, :high} (sustained,
# opponent at/above stage level).
#
#   mix run scripts/edgeguard_passk_probe.exs \
#     --policy checkpoints/fox_gen_v1.3_ARrefit_policy.bin \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --limit-files 24 --k 16 --out eval_runs/0902_edgeguard_passk/RESULTS.md
#
# Options: --char-id (2) · --temperature (0.5) · --seed (20260902) ·
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
seed = opts[:seed] || 20_260_902
cap = opts[:max_per_stratum] || 2500

# Below-stage threshold matches Situations' @offstage_y sign convention;
# -15 keeps "low" clearly under stage level rather than at the ledge lip.
low_y = -15.0
onset_frames = 60

Output.banner("Edgeguard pass@K probe")

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
key = Nx.Random.key(seed)

Output.config([
  {"Policy", Path.basename(policy)},
  {"Files", length(picked)},
  {"K samples", k},
  {"Temperature", temperature}
])

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
      |> Enum.map_reduce(false, fn {gs, set}, prev_eg ->
        eg = set != nil and MapSet.member?(set, :edgeguard)
        opp_pl = gs.players[2]

        stratum =
          cond do
            eg and not prev_eg -> {:eg, :onset_start}
            eg -> {:eg, (opp_pl && (opp_pl.y || 0.0) < low_y && :low) || :high}
            set != nil and MapSet.member?(set, :neutral) -> :neutral
            true -> nil
          end

        {stratum, eg}
      end)

    # Expand {:eg, :onset_start} into an onset window: the first
    # onset_frames of each episode are {:eg, :onset} regardless of height.
    {strata, _} =
      Enum.map_reduce(strata, 0, fn s, left ->
        case s do
          {:eg, :onset_start} -> {{:eg, :onset}, onset_frames - 1}
          {:eg, _pos} when left > 0 -> {{:eg, :onset}, left - 1}
          {:eg, pos} -> {{:eg, pos}, 0}
          other -> {other, 0}
        end
      end)

    strata_t = List.to_tuple(strata)
    frames_t = List.to_tuple(frames)
    n = cap_r.n
    off = cap_r.frame_offset

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

order = [:neutral, {:eg, :onset}, {:eg, :high}, {:eg, :low}]

labels = %{
  :neutral => "neutral",
  {:eg, :onset} => "edgeguard onset (first 60 f)",
  {:eg, :high} => "edgeguard sustained, opp high",
  {:eg, :low} => "edgeguard sustained, opp BELOW stage"
}

pct = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 1) end

table_rows =
  order
  |> Enum.map(fn s -> {s, stat.(Map.get(by_stratum, s, []))} end)
  |> Enum.map_join("\n", fn
    {s, nil} -> "| #{labels[s]} | 0 | — | — |"
    {s, {n, p1, pk}} -> "| #{labels[s]} | #{n} | #{pct.(p1)} | #{pct.(pk)} |"
  end)

report = """
# Edgeguard pass@K probe — RESULTS

Policy #{Path.basename(policy)}, #{length(picked)} expert files, K=#{k}
coherent samples at T=#{temperature}, Leg S match rule, teacher-forced
expert states (no state-visitation confound). :edgeguard = Situations
label (opp offstage, subject onstage, opp not on cliff); low = opp y <
#{low_y}.

| stratum | rows | sampling pass@1 % | pass@#{k} % |
|---|---:|---:|---:|
#{table_rows}

Reading (pre-declared):
- edgeguard strata ~ neutral level -> the opponent-offstage conditioning
  IS learned; live blindness is SELECTION (marginal-collapse at decode)
  -> the critic-selector live experiment covers it.
- collapses in edgeguard strata (esp. opp-low) -> BC lost the
  conditioning -> levers: edgeguard-window curation / AWBC + an
  edgeguard drill (scripted recover-low dummy on the drill driver;
  reference behavior: ledge-refresh shine, proactive ledge-take).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
