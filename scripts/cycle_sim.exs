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
    strict: [
      policies: :string,
      fixture: :string,
      graph_replays: :string,
      max_frames: :integer,
      delay_id: :integer,
      decode_lag: :integer
    ]
  )

fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
max_frames = opts[:max_frames] || 600
# Comma-separated globs (single-wildcard on a comma list silently matches
# nothing — the GOTCHA-2026-08-04 --replays failure mode).
policies =
  (opts[:policies] || "checkpoints/ms_open_z*.bin")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)
  |> Enum.sort()

if policies == [], do: raise("--policies matched nothing")

# --graph-replays: comma list of globs of SUCCESSFUL bot replays whose
# observed dynamics enrich the graph beyond the teacher's exact timing
# (default: the champion-lineage stand-dummy eval replays).
graph_replays =
  (opts[:graph_replays] || "eval_runs/0728_open_z_idle*/r*.slp")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard/1)

Output.banner("Cycle simulator")
Output.puts("Graph sources: fixture + #{length(graph_replays)} replays")
{_default_entry, table} = CycleSim.from_fixture(fixture, graph_replays: graph_replays)
fixture_frames = CycleSim.load_frames(fixture)

Output.puts(
  "Fixture graph: #{map_size(table.transitions)} transitions, " <>
    "#{MapSet.size(table.states)} states, #{table.conflicts} conflicted keys"
)

for path <- policies do
  seed = Path.basename(path, ".bin")
  loaded = Activations.load_heads(path)

  # Per-policy entry + rollout in the policy's own embed layout
  # (queue/delay-id policies need --delay-id, cf. Activations.embed_frames).
  layout_opts = [config: loaded.config, delay_id: opts[:delay_id]]

  entry =
    ExPhil.Interp.BasinRollout.entry_from_frames(
      Enum.take(fixture_frames, 16),
      layout_opts
    )

  # decode_lag: measured live pipeline is N+2 at --frame-delay N (default 2
  # is the delay-0-era value; d3 policies need 5).
  lag_opts = if opts[:decode_lag], do: [decode_lag: opts[:decode_lag]], else: []

  r =
    CycleSim.rollout(loaded.predict_fn, loaded.params, entry, table,
      [max_frames: max_frames] ++ layout_opts ++ lag_opts
    )

  chains = if r.chains == [], do: "-", else: Enum.join(r.chains, ",")

  brk =
    case r.break do
      nil -> "survived #{r.frames}f"
      b -> "break@#{b.at} in #{b.family} af#{b.af} buttons=#{inspect(b.buttons)}"
    end

  Output.puts("#{String.pad_trailing(seed, 20)} chains=[#{chains}] soft=#{r.soft} #{brk}")
end
