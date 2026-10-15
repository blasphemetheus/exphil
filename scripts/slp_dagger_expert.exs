# Relabel the bot's own frames from a .slp session with the expert-distribution
# labeler (2026-10-15; Bradley: "both offstage and onstage"). The sim round
# (scripts/sim_recovery_dagger_expert.exs) gives volume against itself; a
# live session gives the states a HUMAN put the bot in — the 31 games vs
# Bradley (eval_runs/fox_mamba_live_v3s906ep2_20261014_131550/) are the only
# ones where the onstage deficits (OOS rolls, missed techs, in-threat
# smashes) were measured. Same trip logic (ExpertRecoveryLabeler.relabel_history/4),
# same export format as the sim round, so dagger_set_concat / hazards /
# split_gated take the result unchanged. FD games only (the index's stage).
#
#   mix run --no-compile scripts/slp_dagger_expert.exs --index data/silent_fall/expert_recovery_index_v6.bin \
#     --set 'eval_runs/<live>/2026-10-Mainline/*.slp' --port 1 --max-d2 G --max-run 90 --keep-actual \
#     --out data/silent_fall/slp_dagger_expert_live1014.frames --report eval_runs/1001_queue/slp_dagger_live1014.json
Code.require_file("scripts/lib/expert_recovery_labeler.exs")
alias ExPhil.Agents.ExpertRecoveryLabeler
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.{Output, SilentFallWeighting}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, port: :integer, index: :string, out: :string, report: :string, max_d2: :float,
             max_run: :integer, label_seed: :integer, keep_actual: :boolean, stage: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

out = opts[:out] || raise("--out required")
port = opts[:port] || 1
opp_port = if port == 1, do: 2, else: 1
stage = opts[:stage] || 32
edge = GA.stage_edge(stage)
max_d2 = opts[:max_d2]
keep_actual = opts[:keep_actual] || false
index = ExpertRecoveryLabeler.load(opts[:index] || raise("--index required"))
:rand.seed(:exsss, {opts[:label_seed] || 7, 11, 13})

files =
  Keyword.get_values(opts, :set)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.sort()
  |> Enum.reject(&(File.stat!(&1).size < 150_000))
if files == [], do: raise("--set GLOB matched no .slp files")

Output.banner("Replay DAgger (expert labeler v6, #{index.n} rows / #{index.meta[:games]} games, window #{index.window})")
Output.config([{"Files", length(files)}, {"Bot port", port}, {"Stage", stage}, {"max_d2", inspect(max_d2)}, {"max_run", inspect(opts[:max_run])}, {"Out", out}])

frame_lists =
  Enum.flat_map(files, fn path ->
    case Peppi.metadata(path) do
      {:ok, %{stage: ^stage}} ->
        {:ok, replay} = Peppi.parse(path, player_port: port)
        # history = NEWEST FIRST [{state, press}], own player remapped to port 1 like the index
        history =
          replay
          |> Peppi.to_training_frames(player_port: port, opponent_port: opp_port, remap_ports: true)
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(&{&1.game_state, &1.controller})
          |> Enum.reverse()

        lists = ExpertRecoveryLabeler.relabel_history(history, index, edge, max_d2: max_d2, max_run: opts[:max_run])
        Output.puts("  #{Path.basename(path)}: #{length(lists)} runs, #{lists |> List.flatten() |> Enum.count(&(&1[:input_only] != true))} relabeled frames")
        lists

      {:ok, %{stage: s}} ->
        Output.warning("skip #{Path.basename(path)}: stage #{s}")
        []

      _ ->
        Output.warning("skip #{Path.basename(path)}: no metadata")
        []
    end
  end)

all = frame_lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true))
total = length(all)
gated = frame_lists |> List.flatten() |> Enum.count(&(&1[:gated] == true))
ctx_total = (frame_lists |> List.flatten() |> length()) - total - gated
silent_actual = Enum.count(all, &SilentFallWeighting.neutral?(&1.actual))
same_action = fn a, b -> ExPhil.Training.Data.controller_to_action(a, axis_buckets: 16) == ExPhil.Training.Data.controller_to_action(b, axis_buckets: 16) end
hold_share = Enum.count(all, &same_action.(&1.controller, &1.prev_controller)) / max(total, 1)
agree = Enum.count(all, &same_action.(&1.controller, &1.actual)) / max(total, 1)
runs_len = frame_lists |> Enum.map(&length/1) |> Enum.sort()
median = fn l -> if l == [], do: nil, else: Enum.at(l, div(length(l), 2)) end

File.mkdir_p!(Path.dirname(out))
File.write!(out, :erlang.term_to_binary(%{
  expert: "expert_recovery_labeler_knn",
  index: opts[:index],
  exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  policy: "slp:" <> Enum.join(Keyword.get_values(opts, :set), ","),
  action_delay: 0,
  label_convention: ExPhil.Data.LabelConvention.current(),
  keep_actual: keep_actual,
  frame_lists: if(keep_actual, do: frame_lists, else: Enum.map(frame_lists, fn l -> Enum.map(l, &Map.delete(&1, :actual)) end))
}, [:compressed]))

if report = opts[:report] do
  File.mkdir_p!(Path.dirname(report))
  File.write!(report, Jason.encode!(%{
    "files" => length(files), "runs" => length(frame_lists), "frames" => total,
    "max_d2" => max_d2, "gated_frames" => gated, "gated_share" => Float.round(gated / max(total + gated, 1), 3),
    "policy_silent_share" => Float.round(silent_actual / max(total, 1), 3),
    "hold_share" => Float.round(hold_share, 3), "agreement_16" => Float.round(agree, 3),
    "run_len_median" => median.(runs_len)
  }, pretty: true))
end

Output.puts("RESULT slp dagger: #{length(frame_lists)} runs (median #{median.(runs_len)} f incl. prefix), #{total} relabeled frames + #{ctx_total} input-only context" <>
  if(max_d2, do: " + #{gated} gated (d2 > #{max_d2}, #{Float.round(100 * gated / max(total + gated, 1), 1)} % of labelable)", else: "") <>
  "; label = bot's own press on #{Float.round(100 * agree, 1)} %; hold share (label vs prev) #{Float.round(hold_share, 3)} -> #{out}")
