# P4 verdict instrument: does the policy READ techs or guess?
#
#   mix run scripts/tech_reaction_curve.exs \
#     --policies checkpoints/a_policy.bin,checkpoints/b_policy.bin \
#     --replays "probes/.../p1/*.slp,..." [--out logs/tech_curve.json]
#
# For every opponent tech-lifecycle ENTRY (tech in place / roll F / roll
# B / missed), probe the tech choice from the trunk state at frame
# offsets around the animation start (t0). Decodability vs offset is the
# verdict:
#   - rises AFTER t0 (+visibility lag): the policy reads the animation
#   - high BEFORE t0: it predicts/exploits (or a label leak — see the
#     next_kd mystery in INTERP_ROADMAP)
#   - flat at chance: it never represents the choice at all
#
# Events are sparse (~10/game), so curves are coarse — the SHAPE across
# offsets is the signal, not any single point. Split is by replay.
#
# NO-MIX: inference-only beam (#67).

alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, replays: :string, out: :string, offsets: :string]
  )

policies = (opts[:policies] || "") |> String.split(",", trim: true) |> Enum.map(&Path.expand/1)

replays =
  (opts[:replays] || "")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))
  |> Enum.filter(fn f -> File.stat!(f).size > 512_000 end)

offsets =
  case opts[:offsets] do
    nil -> Enum.to_list(-15..30//3)
    s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_integer/1)
  end

if policies == [], do: raise("--policies required")
if length(replays) < 3, do: raise("need >= 3 replays for a by-replay split")

# Tech-lifecycle entry classes (mirrors ReplayStats/GroundTruth)
lifecycle = MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
entry_class = fn
  a when a in [183, 191] -> 0
  199 -> 1
  200 -> 2
  201 -> 3
  _ -> nil
end

# Opponent tech entries per replay: [{frame_idx, class}]
find_events = fn path ->
  {:ok, replay} = Peppi.parse(Path.expand(path))

  p2_actions =
    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> Enum.map(fn f ->
      pl = f.game_state.players[2]
      trunc((pl && pl.action) || 0)
    end)

  p2_actions
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.with_index(1)
  |> Enum.flat_map(fn {[a, b], i} ->
    cls = entry_class.(b)

    if cls != nil and not MapSet.member?(lifecycle, a), do: [{i, cls}], else: []
  end)
end

Output.banner("Tech reaction curve (P4)")
Output.config([
  {"Policies", Enum.map(policies, &Path.basename/1)},
  {"Replays", length(replays)},
  {"Offsets", "#{List.first(offsets)}..#{List.last(offsets)}"}
])

events_by_replay = Enum.map(replays, find_events)
total_events = events_by_replay |> Enum.map(&length/1) |> Enum.sum()
Output.puts("#{total_events} tech-entry events across #{length(replays)} replays")
if total_events < 24, do: Output.warning("very few events — curve will be noisy")

n_eval_replays = max(div(length(replays), 3), 1)
eval_set = MapSet.new((length(replays) - n_eval_replays)..(length(replays) - 1))

curves =
  Enum.map(policies, fn policy ->
    trunk = Activations.load_trunk(policy)
    window = trunk.window

    caps = Enum.map(replays, fn r -> Activations.capture_replay(trunk, r, labels: false) end)

    curve =
      Enum.map(offsets, fn k ->
        # Collect (activation_row, class) per replay at offset k
        {train, eval} =
          caps
          |> Enum.zip(events_by_replay)
          |> Enum.with_index()
          |> Enum.reduce({{[], []}, {[], []}}, fn {{cap, events}, pos}, {{xt, yt}, {xe, ye}} ->
            rows =
              Enum.flat_map(events, fn {frame, cls} ->
                row = frame + k - (window - 1)

                if row >= 0 and row < cap.n,
                  do: [{Nx.slice_along_axis(cap.activations, row, 1, axis: 0), cls}],
                  else: []
              end)

            {xs, ys} = Enum.unzip(rows)

            if MapSet.member?(eval_set, pos),
              do: {{xt, yt}, {xe ++ xs, ye ++ ys}},
              else: {{xt ++ xs, yt ++ ys}, {xe, ye}}
          end)

        {xt, yt} = train
        {xe, ye} = eval

        if length(yt) < 12 or length(ye) < 6 do
          %{offset: k, balanced_accuracy: nil, n_train: length(yt), n_eval: length(ye)}
        else
          r =
            Probe.fit_eval(
              Nx.concatenate(xt),
              Nx.tensor(yt, type: :s64),
              Nx.concatenate(xe),
              Nx.tensor(ye, type: :s64),
              4,
              steps: 300
            )

          %{offset: k, balanced_accuracy: r.balanced_accuracy, n_train: r.n_train, n_eval: r.n_eval}
        end
      end)

    Output.puts("")
    Output.puts(Path.basename(policy))
    Output.puts("  offset(f)   BA      n_eval   (t0 = tech animation start; chance = 0.25)")

    Enum.each(curve, fn p ->
      ba = if p.balanced_accuracy, do: :erlang.float_to_binary(p.balanced_accuracy, decimals: 3), else: "  -  "
      Output.puts("  #{String.pad_leading(to_string(p.offset), 6)}     #{ba}   #{p.n_eval}")
    end)

    %{policy: Path.basename(policy), curve: curve}
  end)

out = opts[:out] || "logs/tech_reaction_curve_#{Date.utc_today() |> Date.to_iso8601(:basic)}.json"
File.write!(out, Jason.encode!(%{offsets: offsets, events: total_events, curves: curves}, pretty: true))
Output.success("curves -> #{out}")
