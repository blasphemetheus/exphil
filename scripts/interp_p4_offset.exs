# P4 verdict artifact (task #7, 2026-08-02): decodability vs lead time.
#
# For each policy, probe the opponent's TECH CHOICE from the trunk state
# at each frame offset relative to the tech-episode start:
#
#   accuracy(k) high only for k >= 0   -> policy READS the tech (reaction)
#   accuracy(k) ~ chance everywhere    -> policy guesses
#   accuracy(k) high for k < 0         -> policy exploits a pre-visibility
#                                         tell (distribution leak — check
#                                         the dummy/RNG before celebrating;
#                                         cf. the v1 saturation lesson)
#
#   mix run scripts/interp_p4_offset.exs \
#     --policies "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin" \
#     --replays "eval_runs/probe_replays/*.slp" \
#     [--offset-range "-30:30"] [--out eval_runs/p4_offset.json]
#
# Split is BY REPLAY (frames within a game are correlated); a policy needs
# tech episodes in >= 4 replays for a meaningful curve. Uses final-timestep
# capture (per-frame trunk states); the within-window logit-lens variant
# rides Activations.load_trunk(all_timesteps: true) separately.

alias ExPhil.Interp.{Activations, Probe}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policies: :string, replays: :string, offset_range: :string, out: :string]
  )

policies = Path.wildcard(opts[:policies] || "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin")
replays = Path.wildcard(opts[:replays] || "eval_runs/probe_replays/*.slp") |> Enum.sort()

{off_lo, off_hi} =
  case String.split(opts[:offset_range] || "-30:30", ":") do
    [a, b] -> {String.to_integer(a), String.to_integer(b)}
  end

Output.banner("P4: decodability vs lead time (tech choice)")
Output.puts("#{length(policies)} policies x #{length(replays)} replays, offsets #{off_lo}..#{off_hi}")

results =
  for path <- policies do
    seed = Path.basename(path, ".bin")
    trunk = Activations.load_trunk(path)

    # Per replay: activations + tech-episode entries (start frame, choice)
    caps =
      replays
      |> Enum.with_index()
      |> Enum.flat_map(fn {replay, ri} ->
        try do
          cap = Activations.capture_replay(trunk, replay)
          tech = cap.labels.tech_choice |> Nx.to_flat_list()

          entries =
            tech
            |> Enum.with_index()
            |> Enum.filter(fn {c, i} ->
              c >= 0 and (i == 0 or Enum.at(tech, i - 1) < 0)
            end)
            |> Enum.map(fn {c, i} -> {i, c} end)

          if entries == [], do: [], else: [{ri, cap.activations, entries, cap.n}]
        rescue
          e ->
            Output.warning("#{Path.basename(replay)}: #{Exception.message(e)} — skipped")
            []
        end
      end)

    n_events = caps |> Enum.map(fn {_, _, es, _} -> length(es) end) |> Enum.sum()
    n_replays = length(caps)
    Output.puts("#{seed}: #{n_events} tech episodes across #{n_replays} replays")

    if n_replays < 2 do
      Output.warning("#{seed}: need episodes in >= 2 replays for a by-replay split — skipping")
      %{policy: seed, curve: []}
    else
      # By-replay split: last quarter of replay indices (>= 1) is eval
      eval_ris =
        caps
        |> Enum.map(&elem(&1, 0))
        |> Enum.sort()
        |> Enum.take(-max(div(n_replays, 4), 1))
        |> MapSet.new()

      curve =
        for k <- off_lo..off_hi do
          {xs_t, ys_t, xs_e, ys_e} =
            Enum.reduce(caps, {[], [], [], []}, fn {ri, acts, entries, n}, acc ->
              Enum.reduce(entries, acc, fn {i, c}, {xt, yt, xe, ye} ->
                row = i + k

                if row >= 0 and row < n do
                  x = Nx.slice_along_axis(acts, row, 1, axis: 0)

                  if MapSet.member?(eval_ris, ri),
                    do: {xt, yt, [x | xe], [c | ye]},
                    else: {[x | xt], [c | yt], xe, ye}
                else
                  {xt, yt, xe, ye}
                end
              end)
            end)

          if xs_t == [] or xs_e == [] do
            %{offset: k, balanced_accuracy: nil, n_train: length(ys_t), n_eval: length(ys_e)}
          else
            num_classes = Enum.max(ys_t ++ ys_e) + 1
            x = Nx.concatenate(xs_t)
            y = Nx.tensor(ys_t, type: :s64)
            xe = Nx.concatenate(xs_e)
            ye = Nx.tensor(ys_e, type: :s64)

            r = Probe.fit_eval(x, y, xe, ye, num_classes)

            %{
              offset: k,
              balanced_accuracy: r.balanced_accuracy,
              majority_baseline: r.majority_baseline,
              n_train: r.n_train,
              n_eval: r.n_eval
            }
          end
        end

      # Terminal sparkline: one row per offset with a bar
      for %{offset: k, balanced_accuracy: ba} <- curve, ba != nil do
        bar = String.duplicate("█", round(ba * 40))
        Output.puts("  #{String.pad_leading(to_string(k), 4)}  #{Float.round(ba, 3)} #{bar}")
      end

      %{policy: seed, curve: curve}
    end
  end

out = opts[:out] || "eval_runs/p4_offset_#{System.os_time(:second)}.json"
File.write!(out, Jason.encode!(%{results: results}, pretty: true))
Output.success("Curves -> #{out}")
