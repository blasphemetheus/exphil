# P3 follow-up studies (tasks #17 + #30), offline, one beam session.
#
#   mix run scripts/interp_p3_followups.exs
#
# 1. FAIR FOLLOW-UP (#17): does tap-inhibition kill combo continuations?
#    P(A-press | prev frame pressed A, opponent in hitstun) with the
#    prev-action channel intact vs ablated. Prediction (tap-inhibition):
#    ablated >> intact on exactly those frames.
# 2. L-CANCEL ORIGIN (#30, user hypothesis): are shield-run onsets
#    clustered right after aerial landings (metastasized L-cancel taps)
#    rather than in neutral? Plus: trigger-press durations in the training
#    fixtures (user's own play) — expect 1-4f taps, no sustained shields.

alias ExPhil.Interp.{Activations, ReplayStats}
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

slippi = Path.expand("~/Slippi")
policy = "checkpoints/mewtwo_combo_poolgrow_r1_policy.bin"
new_era_replay = Path.join(slippi, "Game_20260714T115716.slp")
baseline_replays = ["Game_20260713T015257.slp", "Game_20260713T070100.slp"] |> Enum.map(&Path.join(slippi, &1))

fixtures =
  ~w(mewtwo_fair_chains.slp mewtwo_shfair_only.slp mewtwo_approach_fair.slp)
  |> Enum.map(&Path.join("test/fixtures/replays", &1))

Output.banner("P3 follow-ups: fair follow-up (#17) + L-cancel origin (#30)")

# ---------------------------------------------------------------------------
# #17: fair follow-up tap inhibition
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("=== #17 fair follow-up: P(A | prevA, opp hitstun) intact vs ablated ===")

heads = Activations.load_heads(policy)
replays17 = [new_era_replay | baseline_replays]

probs_for = fn ablate? ->
  cap = Activations.capture(heads, replays17, labels: true, use_prev_action: not ablate?)
  a_probs = Nx.sigmoid(cap.activations.buttons) |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])
  {cap, a_probs}
end

{cap_i, a_i} = probs_for.(false)
{_cap_a, a_a} = probs_for.(true)

# prev-A from controllers, aligned like the LEACE script
window = heads.config[:window_size] || 60

prev_a =
  replays17
  |> Enum.flat_map(fn path ->
    {:ok, replay} = Peppi.parse(path)

    controllers =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(& &1.controller)

    n = length(controllers) - window + 1
    controllers |> Enum.slice(window - 2, n) |> Enum.map(&if(Map.get(&1, :button_a), do: 1, else: 0))
  end)
  |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)

hitstun = cap_i.labels.opp_hitstun |> Nx.as_type(:f32)
combo_mask = Nx.multiply(prev_a, hitstun)

cmean = fn probs, mask ->
  Nx.sum(Nx.multiply(probs, mask)) |> Nx.divide(Nx.max(Nx.sum(mask), 1)) |> Nx.to_number() |> Float.round(4)
end

n_combo = Nx.sum(combo_mask) |> Nx.to_number() |> trunc()

Output.puts(
  "combo frames (prevA ∧ opp hitstun): n=#{n_combo} | " <>
    "P(A) intact=#{cmean.(a_i, combo_mask)} ablated=#{cmean.(a_a, combo_mask)} | " <>
    "all-frames P(A): intact=#{Nx.mean(a_i) |> Nx.to_number() |> Float.round(4)} " <>
    "ablated=#{Nx.mean(a_a) |> Nx.to_number() |> Float.round(4)}"
)

Output.puts("READING: ablated >> intact on combo frames => tap-inhibition blocks follow-ups.")

# ---------------------------------------------------------------------------
# #30: L-cancel origin of shield onsets
# ---------------------------------------------------------------------------
Output.puts("")
Output.puts("=== #30 L-cancel origin: shield onsets vs aerial landings ===")

aerial_attack = MapSet.new(65..70)
aerial_landing = MapSet.new(71..75)

for path <- baseline_replays do
  d = ReplayStats.load(path)
  acts = d.p1.actions

  shield_onsets =
    acts
    |> Enum.with_index()
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [{a, _}, {b, _}] -> b in [178, 179, 180] and a not in [178, 179, 180] end)
    |> Enum.map(fn [_, {_, i}] -> i end)

  arr = List.to_tuple(acts)

  near_aerial =
    Enum.count(shield_onsets, fn i ->
      lo = max(i - 12, 0)

      Enum.any?(lo..(i - 1)//1, fn j ->
        a = elem(arr, j)
        MapSet.member?(aerial_attack, a) or MapSet.member?(aerial_landing, a)
      end)
    end)

  n_on = length(shield_onsets)

  Output.puts(
    "#{Path.basename(path)}: shield onsets=#{n_on}, within 12f after aerial=#{near_aerial} " <>
      "(#{if n_on > 0, do: Float.round(100 * near_aerial / n_on, 1), else: 0.0}%)"
  )
end

Output.puts("")
Output.puts("--- training fixtures: trigger-press durations (user's play) ---")

for path <- fixtures, File.exists?(path) do
  d = ReplayStats.load(path)
  presses = ReplayStats.trigger_presses(d.p1.controllers)
  durs = Enum.map(presses, &elem(&1, 1))
  sorted = Enum.sort(durs)
  pct = fn p -> if sorted == [], do: 0, else: Enum.at(sorted, min(trunc(p * length(sorted)), length(sorted) - 1)) end

  Output.puts(
    "#{Path.basename(path)}: presses=#{length(durs)} p50=#{pct.(0.5)}f p90=#{pct.(0.9)}f " <>
      "max=#{if durs == [], do: 0, else: Enum.max(durs)}f taps(<=4f)=#{Enum.count(durs, &(&1 <= 4))}"
  )
end

Output.puts("READING: shield onsets clustered post-aerial + fixture presses mostly 1-4f taps")
Output.puts("=> 'shield-lock' was metastasized L-cancels; no shield concept in the data.")
