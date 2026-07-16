# Interp P3 offline stage: shield dead-zone occupancy + jump-spam
# prev-action ablation, from per-head logits (no Dolphin needed).
#
#   mix run scripts/interp_shield_jump_offline.exs [--policy PATH]
#
# 1. SHIELD-LOCK diagnosis: with hysteresis press=0.45 / release=0.3, a
#    held shield button (l/r) only releases when its probability drops
#    below 0.3. Measure how often the model's raw shield probs sit in the
#    [0.3, 0.45) dead zone and how long >=0.3 runs last. Long runs PREDICT
#    the shim A/B outcome: run length ~ hold length ~ shield breaks.
# 2. JUMP-SPAM ablation: capture the same replays with the prev-action
#    channel intact vs zeroed. If the self-reinforcing loop drives the
#    spam, jump probs (x/y) should drop sharply under ablation,
#    especially on frames right after jumps.
#
# Button order: a=0 b=1 x=2 y=3 z=4 l=5 r=6 d_up=7.
# NO-MIX: one beam.

alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string, replays: :string])
policy = opts[:policy] || "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin"

slippi = Path.expand("~/Slippi")

# Default: mix of old-era (CPU dummy) and new-era (real tech dummy) games.
# Override with --replays comma,separated,paths (relative to ~/Slippi or absolute).
replays =
  case opts[:replays] do
    nil ->
      [
        "Game_20260713T015257.slp",
        "Game_20260713T070100.slp",
        "Game_20260714T115716.slp"
      ]

    list ->
      String.split(list, ",", trim: true)
  end
  |> Enum.map(fn p -> if String.starts_with?(p, "/"), do: p, else: Path.join(slippi, p) end)
  |> Enum.filter(&File.exists?/1)

press_t = 0.45
release_t = 0.3
jump_idx = [2, 3]
shield_idx = [5, 6]

Output.banner("Interp P3 offline: shield dead-zone + jump ablation")
Output.config([{"Policy", policy}, {"Replays", length(replays)}])

heads = Activations.load_heads(policy)

capture = fn ablate? ->
  Activations.capture(heads, replays, labels: not ablate?, use_prev_action: not ablate?)
end

intact = capture.(false)
ablated = capture.(true)

btn_probs = fn cap -> Nx.sigmoid(cap.activations.buttons) end
p_intact = btn_probs.(intact)
p_ablated = btn_probs.(ablated)
n = Nx.axis_size(p_intact, 0)

# ---------------------------------------------------------------------------
# 1. Shield dead-zone occupancy (intact capture — live conditions)
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("=== SHIELD (hysteresis dead zone [#{release_t}, #{press_t})) ===")

shield_p = Nx.take(p_intact, Nx.tensor(shield_idx), axis: 1) |> Nx.reduce_max(axes: [1])

frac = fn t -> Nx.mean(Nx.as_type(t, :f32)) |> Nx.to_number() |> then(&Float.round(&1 * 100, 2)) end

above_press = Nx.greater_equal(shield_p, press_t)
in_dead_zone = Nx.logical_and(Nx.greater_equal(shield_p, release_t), Nx.less(shield_p, press_t))
holdable = Nx.greater_equal(shield_p, release_t)

Output.puts("frames with max(l,r) prob >= press (would press): #{frac.(above_press)}%")
Output.puts("frames in dead zone (would NOT release once held): #{frac.(in_dead_zone)}%")

# Run lengths of consecutive holdable frames (>= release): once pressed,
# the shim holds shield through the whole run.
runs =
  holdable
  |> Nx.to_flat_list()
  |> Enum.chunk_by(& &1)
  |> Enum.filter(&(hd(&1) == 1))
  |> Enum.map(&length/1)

if runs != [] do
  sorted = Enum.sort(runs)
  pct = fn p -> Enum.at(sorted, min(trunc(p * length(sorted)), length(sorted) - 1)) end

  Output.puts(
    "holdable runs: n=#{length(runs)} p50=#{pct.(0.5)}f p90=#{pct.(0.9)}f " <>
      "p99=#{pct.(0.99)}f max=#{Enum.max(runs)}f " <>
      "(shield break ~ 450f of full hold; runs >= 450: #{Enum.count(runs, &(&1 >= 450))})"
  )
else
  Output.puts("no holdable runs at all")
end

# ---------------------------------------------------------------------------
# 2. Jump-spam prev-action ablation
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("=== JUMP (x/y) prev-action ablation ===")

jump_p = fn probs -> Nx.take(probs, Nx.tensor(jump_idx), axis: 1) |> Nx.reduce_max(axes: [1]) end
ji = jump_p.(p_intact)
ja = jump_p.(p_ablated)

mean = fn t -> Nx.mean(t) |> Nx.to_number() |> Float.round(4) end
above = fn t -> frac.(Nx.greater_equal(t, press_t)) end

Output.puts("mean jump prob:   intact=#{mean.(ji)}  ablated=#{mean.(ja)}")
Output.puts("frames >= press:  intact=#{above.(ji)}%  ablated=#{above.(ja)}%")

# Conditional on the policy being airborne (the loop's habitat)
airborne = intact.labels.own_airborne |> Nx.as_type(:u8)
n_air = Nx.sum(airborne) |> Nx.to_number()

cond_mean = fn t, mask ->
  Nx.sum(Nx.multiply(t, mask)) |> Nx.divide(Nx.max(Nx.sum(mask), 1)) |> Nx.to_number() |> Float.round(4)
end

air_f = Nx.as_type(airborne, :f32)
ground_f = Nx.subtract(1.0, air_f)

Output.puts(
  "airborne frames (#{n_air}): intact=#{cond_mean.(ji, air_f)} ablated=#{cond_mean.(ja, air_f)} | " <>
    "grounded: intact=#{cond_mean.(ji, ground_f)} ablated=#{cond_mean.(ja, ground_f)}"
)

Output.puts("")
Output.puts("READINGS: jump probs collapsing under ablation => prev-action loop is")
Output.puts("causally load-bearing (proceed to live A/B). Unchanged => spam lives in")
Output.puts("the game-state mapping. Long shield runs => shim conviction predicted.")

File.write!(
  "interp/p3_offline_results.bin",
  :erlang.term_to_binary(%{
    policy: policy,
    replays: replays,
    n: n,
    shield_runs: runs,
    jump_mean_intact: mean.(ji),
    jump_mean_ablated: mean.(ja)
  })
)

Output.success("Saved interp/p3_offline_results.bin")
