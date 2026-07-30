# Decision-boundary comparison across policies on the canonical multishine
# cycle (head-swap mechanism test, offline half — 2026-07-29).
#
# For each policy, run its heads over a FIXTURE replay stream (the clean
# teacher cycle, d0 and d1 variants) and tabulate mean B/X logits by
# {phase bucket, action_frame}. The d1 teacher's rules are the d0 rules
# with trigger windows shifted one frame earlier — so a d1-adapted policy
# should show its X-press (jump-cancel) boundary one af earlier in the
# grounded-reflector states, and its B boundaries shifted likewise.
#
# Applied to the spliced hybrids this answers WHERE the adaptation lives:
#   champTrunk+d1Heads shows the shifted boundary  -> heads carry it
#   d1Trunk+champHeads shows the shifted boundary  -> trunk carries it
#
#   mix run scripts/probe_cycle_boundary.exs
#
# Offline; safe on battery. NO-MIX beside live training.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.{Data, Output}

# Hybrids deleted 2026-07-29 (cross-lineage head-swap is degenerate — see
# LATENCY_ARCHITECTURE.md); r4e10 = R4 margin round's epoch-10 export.
policies =
  [
    {"champion", "checkpoints/ms_open_z.bin"},
    {"dagger3", "checkpoints/ms_d1_dagger3_policy.bin"},
    {"dagger3_r4e10", "checkpoints/ms_d1_dagger3_policy_latest.bin"}
  ]
  |> Enum.filter(fn {name, path} ->
    File.exists?(path) || (Output.warning("skipping #{name}: #{path} missing"); false)
  end)

fixtures = [
  {"d0", "test/fixtures/replays/fox_multishine_closed.slp"},
  {"d1", "test/fixtures/replays/fox_multishine_closed_d1.slp"}
]

# Fox landmarks (see MultishineExpert / record_multishine)
reflector_ground = 360..365
reflector_air = 366..369
jumpsquat = 24

Output.banner("Cycle decision boundaries: champion / dagger / hybrids")

phase_of = fn player ->
  af = min(max(player.action_frame || 0, 0), 6)

  cond do
    player.action in reflector_ground and player.on_ground -> {:ground_reflector, af}
    player.action == jumpsquat -> {:jumpsquat, af}
    player.action in reflector_air -> {:air_reflector, af}
    not player.on_ground -> {:airborne_other, min(af, 3)}
    true -> nil
  end
end

for {fix_name, fix_path} <- fixtures do
  {:ok, parsed} = Peppi.parse(fix_path)

  frames =
    parsed
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))

  Output.puts("")
  Output.puts("=== fixture #{fix_name} (#{fix_path}, #{length(frames)} frames)")

  ds =
    frames
    |> Data.from_frames()
    |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

  emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
  {total, _} = Nx.shape(emb)
  frames_arr = List.to_tuple(frames)

  for {name, path} <- policies do
    loaded = Activations.load_heads(path)
    window = loaded.window

    logits =
      (window - 1)..(total - 1)
      |> Enum.chunk_every(512)
      |> Enum.flat_map(fn ts ->
        wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
        out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
        buttons = elem(out, 0)
        b = buttons[[.., 1]] |> Nx.to_flat_list()
        x = buttons[[.., 2]] |> Nx.to_flat_list()
        Enum.zip(b, x)
      end)

    by_phase =
      logits
      |> Enum.with_index(window - 1)
      |> Enum.reduce(%{}, fn {{b, x}, t}, acc ->
        player = elem(frames_arr, t).game_state.players[1]

        case phase_of.(player) do
          nil -> acc
          key -> Map.update(acc, key, [{b, x}], &[{b, x} | &1])
        end
      end)

    mean = fn vals -> Float.round(Enum.sum(vals) / max(length(vals), 1), 2) end

    row = fn phase, max_af ->
      cells =
        Enum.map_join(0..max_af, " ", fn af ->
          case by_phase[{phase, af}] do
            nil ->
              "     .    "

            pairs ->
              bs = pairs |> Enum.map(&elem(&1, 0)) |> mean.()
              xs = pairs |> Enum.map(&elem(&1, 1)) |> mean.()
              String.pad_leading("#{bs}/#{xs}", 10)
          end
        end)

      Output.puts("    #{String.pad_trailing(to_string(phase), 17)} #{cells}")
    end

    Output.puts("  #{name} — mean B/X logit by af (af0..af6):")
    row.(:ground_reflector, 6)
    row.(:jumpsquat, 6)
    row.(:air_reflector, 6)
    row.(:airborne_other, 3)
  end
end

Output.puts("")
Output.puts("Read: the JC trigger = first ground_reflector af with X>0; the d1")
Output.puts("teacher's windows sit one af EARLIER. Whichever hybrid shows the")
Output.puts("shifted boundary carries the adaptation.")
