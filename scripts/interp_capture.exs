# Interp Phase 0: capture trunk activations + ground-truth labels.
#
#   mix run scripts/interp_capture.exs \
#     --policy checkpoints/mewtwo_combo_poolgrow_r1_policy.bin \
#     --replays ~/Slippi/Game_A.slp,~/Slippi/Game_B.slp \
#     --out interp/poolgrow_r1
#
# Writes to --out dir:
#   capture.bin  - term_to_binary of %{activations, labels, replay_index,
#                  replays, policy, window, hidden_size} (all tensors on
#                  BinaryBackend; load with :erlang.binary_to_term)
#   summary.txt  - human-readable shapes + label base rates
#
# NO-MIX: this runs a beam — never start it while a training/play beam is
# live on this machine.

alias ExPhil.Interp.{Activations, GroundTruth}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replays: :string,
      out: :string,
      player_port: :integer,
      opponent_port: :integer,
      batch_size: :integer
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
out_dir = opts[:out] || raise "--out required"

replay_paths =
  (opts[:replays] || raise("--replays required (comma-separated .slp paths)"))
  |> String.split(",", trim: true)
  |> Enum.map(&Path.expand/1)

for p <- replay_paths, not File.exists?(p), do: raise("replay not found: #{p}")

Output.banner("Interp Phase 0: activation capture")

Output.config([
  {"Policy", policy_path},
  {"Replays", length(replay_paths)},
  {"Out", out_dir}
])

trunk = Activations.load_trunk(policy_path)
Output.puts("Trunk loaded: window=#{trunk.window} hidden=#{trunk.hidden_size}")

result =
  Activations.capture(trunk, replay_paths,
    player_port: opts[:player_port] || 1,
    opponent_port: opts[:opponent_port] || 2,
    batch_size: opts[:batch_size] || 256
  )

n = Nx.axis_size(result.activations, 0)
File.mkdir_p!(out_dir)

payload =
  result
  |> Map.merge(%{
    policy: policy_path,
    window: trunk.window,
    hidden_size: trunk.hidden_size,
    captured_at: DateTime.utc_now() |> DateTime.to_iso8601()
  })

File.write!(Path.join(out_dir, "capture.bin"), :erlang.term_to_binary(payload))

base_rates =
  GroundTruth.binary_features()
  |> Enum.map(fn f ->
    rate = result.labels[f] |> Nx.mean() |> Nx.to_number()
    "  #{f}: #{Float.round(rate * 100, 2)}%"
  end)

kd_rows = result.labels[:tech_choice] |> Nx.greater_equal(0) |> Nx.sum() |> Nx.to_number()

summary = """
policy: #{policy_path}
replays (#{length(replay_paths)}):
#{Enum.map_join(result.replays, "\n", &"  #{&1}")}
activations: {#{n}, #{trunk.hidden_size}} f32 (window #{trunk.window}, stride 1)
label base rates:
#{Enum.join(base_rates, "\n")}
frames inside knockdown lifecycle: #{kd_rows}
"""

File.write!(Path.join(out_dir, "summary.txt"), summary)
Output.puts(summary)
Output.success("Capture written to #{out_dir}/capture.bin")
