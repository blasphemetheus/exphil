# Full-cycle offline multishine simulator (task #5) — Dolphin-free
# closed-loop eval + break forensics through the fixture transition graph.
#
#   mix run scripts/cycle_sim.exs --policies "checkpoints/ms_open_z*.bin" \
#     [--fixture test/fixtures/replays/fox_multishine_closed.slp] \
#     [--max-frames 600]
#
# Validation gate (pre-registered): champion ms_open_z chains, metronome
# ms_open_zz doesn't. Reads heads via Interp.Activations.load_heads (same
# contract as probe_cycle_margins.exs).

alias ExPhil.Interp.{Activations, CycleSim}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, fixture: :string, graph_replays: :string, max_frames: :integer]
  )

fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
max_frames = opts[:max_frames] || 600
policies = Path.wildcard(opts[:policies] || "checkpoints/ms_open_z*.bin") |> Enum.sort()

# --graph-replays: comma list of globs of SUCCESSFUL bot replays whose
# observed dynamics enrich the graph beyond the teacher's exact timing
# (default: the champion-lineage stand-dummy eval replays).
graph_replays =
  (opts[:graph_replays] || "eval_runs/0728_open_z_idle*/r*.slp")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)

Output.banner("Cycle simulator")
Output.puts("Graph sources: fixture + #{length(graph_replays)} replays")
{entry, table} = CycleSim.from_fixture(fixture, graph_replays: graph_replays)

Output.puts(
  "Fixture graph: #{map_size(table.transitions)} transitions, " <>
    "#{MapSet.size(table.states)} states, #{table.conflicts} conflicted keys"
)

for path <- policies do
  seed = Path.basename(path, ".bin")
  loaded = Activations.load_heads(path)

  r = CycleSim.rollout(loaded.predict_fn, loaded.params, entry, table, max_frames: max_frames)

  chains = if r.chains == [], do: "-", else: Enum.join(r.chains, ",")

  brk =
    case r.break do
      nil -> "survived #{r.frames}f"
      b -> "break@#{b.at} in #{b.family} af#{b.af} buttons=#{inspect(b.buttons)}"
    end

  Output.puts("#{String.pad_trailing(seed, 20)} chains=[#{chains}] soft=#{r.soft} #{brk}")
end
