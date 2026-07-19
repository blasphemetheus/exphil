# Tier-0 shield-lock steering vector (BRAINSTORM_2026-07-18, Tier-0 #3).
#
# Offline contrast extraction from the r14 probe replays — no Dolphin:
#   v_raw = mean(trunk | P1 action in 178-182)          # shield family
#         - mean(trunk | P1 grounded, action not in 178-182)
#   v     = v_raw / |v_raw|
#
# Trunk features are captured with the SAME embed config the policy
# carries (Activations.load_trunk reads it from the export) and the same
# windowed pipeline live inference uses. Saved as
# :erlang.term_to_binary(%{v: unit f32 {hidden} (BinaryBackend), meta: ...})
# for --steer-vector (ExPhil.Interp.Steering.load!/1).
#
#   mix run scripts/extract_steer_vector.exs [--policy PATH] [--out PATH] [replay.slp ...]
#
# Default replays: the six >500KB probe replays under probes/newera8/r14
# (largest .slp per arm dir — size, not mtime, per GOTCHAS #69).
# NO-MIX: one beam.

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, argv, _} = OptionParser.parse(System.argv(), strict: [policy: :string, out: :string])
policy = opts[:policy] || "checkpoints/mewtwo_combo_newera_r14_policy.bin"
out = opts[:out] || "checkpoints/r14_shield_steer.bin"

replays =
  if argv != [] do
    argv
  else
    for arm <- ["plain", "debounce"], p <- ["p1", "p2", "p3"] do
      "probes/newera8/r14/#{arm}/#{p}/*.slp"
      |> Path.wildcard()
      |> Enum.max_by(fn f -> File.stat!(f).size end, fn -> nil end)
    end
    |> Enum.reject(&is_nil/1)
    |> Enum.filter(fn f -> File.stat!(f).size > 512_000 end)
  end

if replays == [], do: raise("no usable replays found")

# Shield family per the report-card OOS classification (GuardOn 178,
# Guard 179, GuardOff 180, GuardSetOff 181, GuardReflect 182)
shield_set = MapSet.new(178..182)

trunk = Activations.load_trunk(policy)
window = trunk.window

Output.banner("Steering vector extraction (shield-lock contrast)")

Output.config([
  {"Policy", policy},
  {"Backbone", inspect(Map.get(trunk.config, :backbone))},
  {"Window", window},
  {"Hidden", trunk.hidden_size},
  {"Replays", length(replays)},
  {"Out", out}
])

{act_parts, shield_parts, ground_parts} =
  replays
  |> Enum.map(fn path ->
    cap = Activations.capture_replay(trunk, path, labels: false)

    {:ok, replay} = Peppi.parse(path)

    p1 =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.game_state.players[1])

    # Activation row i corresponds to a decision AT frame i + window - 1
    # (Activations.capture_replay :frame_offset)
    decision = Enum.slice(p1, window - 1, cap.n)

    if length(decision) != cap.n,
      do: raise("activation/frame misalignment on #{path}: #{length(decision)} vs #{cap.n}")

    shield =
      Enum.map(decision, fn pl ->
        if MapSet.member?(shield_set, pl.action), do: 1, else: 0
      end)

    ground =
      Enum.map(decision, fn pl ->
        if pl.on_ground and not MapSet.member?(shield_set, pl.action), do: 1, else: 0
      end)

    n_sh = Enum.sum(shield)
    Output.puts("  #{Path.basename(path)}: n=#{cap.n} shield=#{n_sh} grounded-nonshield=#{Enum.sum(ground)}")

    {cap.activations,
     Nx.tensor(shield, type: :f32, backend: Nx.BinaryBackend),
     Nx.tensor(ground, type: :f32, backend: Nx.BinaryBackend)}
  end)
  |> Enum.reduce({[], [], []}, fn {a, s, g}, {as, ss, gs} -> {[a | as], [s | ss], [g | gs]} end)

x = Nx.concatenate(Enum.reverse(act_parts), axis: 0)
shield_mask = Nx.concatenate(Enum.reverse(shield_parts), axis: 0)
ground_mask = Nx.concatenate(Enum.reverse(ground_parts), axis: 0)

n = Nx.axis_size(x, 0)
n_shield = shield_mask |> Nx.sum() |> Nx.to_number() |> round()
n_ground = ground_mask |> Nx.sum() |> Nx.to_number() |> round()

Output.puts("total: #{n} frames — shield #{n_shield}, grounded non-shield #{n_ground}")

if n_shield == 0, do: raise("no shield frames — nothing to contrast")
if n_ground == 0, do: raise("no grounded non-shield frames — nothing to contrast")

masked_mean = fn mask, count ->
  x
  |> Nx.multiply(Nx.new_axis(mask, -1))
  |> Nx.sum(axes: [0])
  |> Nx.divide(count)
end

diff = Nx.subtract(masked_mean.(shield_mask, n_shield), masked_mean.(ground_mask, n_ground))
raw_norm = diff |> Nx.LinAlg.norm() |> Nx.to_number()

if raw_norm == 0, do: raise("contrast vector has zero norm")

v =
  diff
  |> Nx.divide(raw_norm)
  |> Nx.as_type(:f32)
  |> Nx.backend_transfer(Nx.BinaryBackend)

unit_check = v |> Nx.LinAlg.norm() |> Nx.to_number()

meta = %{
  policy: policy,
  replays: replays,
  window: window,
  hidden: trunk.hidden_size,
  shield_frames: n_shield,
  ground_frames: n_ground,
  total_frames: n,
  raw_contrast_norm: raw_norm,
  shield_set: Enum.sort(shield_set),
  extracted_at: DateTime.utc_now() |> DateTime.to_iso8601()
}

File.write!(out, :erlang.term_to_binary(%{v: v, meta: meta}))

Output.success(
  "saved #{out}: |v|=#{Float.round(unit_check, 6)} (raw contrast norm #{Float.round(raw_norm, 4)}), " <>
    "#{n_shield} shield vs #{n_ground} grounded frames over #{length(replays)} replays"
)
