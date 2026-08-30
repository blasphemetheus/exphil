# Joint-head audit — how much within-frame dependency between controller
# components exists in EXPERT play that a conditionally-independent head
# cannot represent? (EVAL_DIRECTIONS, 2026-08-30.)
#
# fox_gen_v1's six heads are computed in parallel from the trunk: the joint
# per-frame action is modelled as p(buttons|s)·p(main_x|s)·p(main_y|s)·…
# The chain rule says the true joint is p(buttons|s)·p(main_x|s,buttons)·…
# The gap is the TOTAL CORRELATION (multi-information) of the components
# given the state:  TC = Σ H(component) − H(joint), in bits per frame.
# We cannot condition on the full state, so we report it three ways:
#   unconditional              — an upper bound on what the state could explain
#   | action-state             — conditioned on the player's current action id
#                                (a coarse state proxy the trunk certainly has)
#   | action-state, situation  — plus the Situations label set (coarser bins,
#                                more of the state); the residual is a LOWER
#                                bound-ish estimate of what an independent head
#                                loses even with a perfect trunk
# plus the specific pairs that matter for recovery and grabs:
#   P(stick up | B)  vs P(stick up)          — up-B needs both
#   P(B | stick up)  vs P(B)
#   P(stick side | L or R) vs P(stick side)  — airdodge direction / wavedash
#   P(A | stick down) vs P(A)                — dtilt vs dash-attack coupling
# and the up-B specific: of expert frames with B pressed while offstage, the
# stick-y distribution, vs the product prediction.
#
#   mix run scripts/joint_head_audit.exs --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --port 1 --limit-files 1500 --out eval_runs/0830_joint_head_audit/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Embeddings.Controller, as: CE
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, limit_files: :integer, concurrency: :integer, out: :string]
  )

glob = opts[:replays] || raise("--replays required")
port = opts[:port] || 1
opp = if port == 1, do: 2, else: 1
files = glob |> Path.wildcard() |> Enum.sort()
files = if opts[:limit_files], do: Enum.take(files, opts[:limit_files]), else: files

Output.banner("Joint-head audit — within-frame dependency in expert play")
Output.config([{"Replays", "#{length(files)} files"}, {"Port", port}])

buckets = 16
# coarse stick zones for the pair stats: 17 buckets -> up (>=13), down (<=3), side (x<=3 or x>=13), centre
sit_keys = [:offstage, :recovery_low, :recovery_high, :being_edgeguarded, :neutral, :advantage, :disadvantage, :cornered, :edge_danger]

scan = fn path ->
  try do
    with {:ok, replay} <- Peppi.parse(path, player_port: port) do
      frames = replay |> Peppi.to_training_frames(player_port: port, opponent_port: opp) |> Enum.reject(&(&1.game_state.frame < 0))
      states = Enum.map(frames, & &1.game_state)
      if length(states) < 300 do
        nil
      else
        sits = Situations.label_states(states, port, as: :set) |> List.to_tuple()

        frames
        |> Enum.with_index()
        |> Enum.reduce(%{}, fn {f, i}, acc ->
          c = f.controller
          p = f.game_state.players[port]
          b = for k <- [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r], do: if(Map.get(c, k), do: 1, else: 0)
          mx = CE.discretize_axis(c.main_stick.x, buckets)
          my = CE.discretize_axis(c.main_stick.y, buckets)
          a = trunc(p.action || 0)
          sit = elem(sits, i) |> MapSet.to_list() |> Enum.filter(&(&1 in sit_keys)) |> Enum.sort()
          key = {b, mx, my, a, sit}
          Map.update(acc, key, 1, &(&1 + 1))
        end)
      end
    else
      _ -> nil
    end
  rescue
    _ -> nil
  end
end

counts =
  files
  |> Task.async_stream(scan, max_concurrency: opts[:concurrency] || 12, timeout: :infinity, ordered: false)
  |> Stream.with_index(1)
  |> Enum.reduce(%{}, fn {{:ok, r}, i}, acc ->
    if rem(i, 100) == 0, do: Output.progress_bar(i, length(files), label: "replays")
    if r, do: Map.merge(acc, r, fn _, x, y -> x + y end), else: acc
  end)

Output.progress_done()
n_total = counts |> Map.values() |> Enum.sum()
Output.puts("#{n_total} frames, #{map_size(counts)} distinct (buttons, mx, my, action, situation) keys")

log2 = fn x -> :math.log(x) / :math.log(2) end

entropy = fn hist ->
  t = hist |> Map.values() |> Enum.sum()
  if t == 0, do: 0.0, else: -Enum.sum(Enum.map(Map.values(hist), fn c -> p = c / t; p * log2.(p) end))
end

# total correlation of (b, mx, my) within a group of frames
tc_of = fn group_counts ->
  hb = Enum.reduce(group_counts, %{}, fn {{b, _, _, _, _}, c}, a -> Map.update(a, b, c, &(&1 + c)) end)
  hx = Enum.reduce(group_counts, %{}, fn {{_, mx, _, _, _}, c}, a -> Map.update(a, mx, c, &(&1 + c)) end)
  hy = Enum.reduce(group_counts, %{}, fn {{_, _, my, _, _}, c}, a -> Map.update(a, my, c, &(&1 + c)) end)
  hj = Enum.reduce(group_counts, %{}, fn {{b, mx, my, _, _}, c}, a -> Map.update(a, {b, mx, my}, c, &(&1 + c)) end)
  %{tc: entropy.(hb) + entropy.(hx) + entropy.(hy) - entropy.(hj), hb: entropy.(hb), hx: entropy.(hx), hy: entropy.(hy), hj: entropy.(hj),
    n: group_counts |> Enum.map(&elem(&1, 1)) |> Enum.sum()}
end

conditional_tc = fn key_fn ->
  groups = Enum.group_by(counts, fn {k, _} -> key_fn.(k) end)
  Enum.reduce(groups, 0.0, fn {_, g}, acc ->
    r = tc_of.(g)
    acc + r.n / n_total * r.tc
  end)
end

uncond = tc_of.(Map.to_list(counts))
tc_action = conditional_tc.(fn {_, _, _, a, _} -> a end)
tc_action_sit = conditional_tc.(fn {_, _, _, a, s} -> {a, s} end)

# ---- pair statistics ---------------------------------------------------------
sum_where = fn pred -> counts |> Enum.filter(fn {k, _} -> pred.(k) end) |> Enum.map(&elem(&1, 1)) |> Enum.sum() end
p = fn pred -> sum_where.(pred) / n_total end
b_pressed = fn {b, _, _, _, _} -> Enum.at(b, 1) == 1 end
a_pressed = fn {b, _, _, _, _} -> Enum.at(b, 0) == 1 end
lr_pressed = fn {b, _, _, _, _} -> Enum.at(b, 5) == 1 or Enum.at(b, 6) == 1 end
up = fn {_, _, my, _, _} -> my >= 13 end
down = fn {_, _, my, _, _} -> my <= 3 end
side = fn {_, mx, _, _, _} -> mx <= 3 or mx >= 13 end
offstage = fn {_, _, _, _, s} -> :offstage in s or :recovery_low in s or :recovery_high in s end
cond_p = fn pred, given -> d = sum_where.(given); if d == 0, do: 0.0, else: sum_where.(fn k -> pred.(k) and given.(k) end) / d end

f = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 2) end
f3 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

pairs = [
  {"P(stick up | B pressed)", cond_p.(up, b_pressed), "P(stick up)", p.(up)},
  {"P(B pressed | stick up)", cond_p.(b_pressed, up), "P(B pressed)", p.(b_pressed)},
  {"P(stick up | B pressed, offstage)", cond_p.(up, fn k -> b_pressed.(k) and offstage.(k) end), "P(stick up | offstage)", cond_p.(up, offstage)},
  {"P(B pressed | stick up, offstage)", cond_p.(b_pressed, fn k -> up.(k) and offstage.(k) end), "P(B pressed | offstage)", cond_p.(b_pressed, offstage)},
  {"P(stick side | L/R pressed)", cond_p.(side, lr_pressed), "P(stick side)", p.(side)},
  {"P(stick side | L/R pressed, offstage)", cond_p.(side, fn k -> lr_pressed.(k) and offstage.(k) end), "P(stick side | offstage)", cond_p.(side, offstage)},
  {"P(A pressed | stick down)", cond_p.(a_pressed, down), "P(A pressed)", p.(a_pressed)},
  {"P(stick down | A pressed)", cond_p.(down, a_pressed), "P(stick down)", p.(down)}
]

pair_rows = Enum.map_join(pairs, "\n", fn {l1, v1, l2, v2} -> "| #{l1} | #{f.(v1)}% | #{l2} | #{f.(v2)}% | #{:erlang.float_to_binary(if(v2 > 0, do: v1 / v2, else: 0.0), decimals: 2)}× |" end)

# stick-y distribution when B is pressed offstage: joint vs product prediction
sy_given_b_off = for y <- 0..16, do: cond_p.(fn {_, _, my, _, _} -> my == y end, fn k -> b_pressed.(k) and offstage.(k) end)
sy_off = for y <- 0..16, do: cond_p.(fn {_, _, my, _, _} -> my == y end, offstage)
tv_upb = 0.5 * Enum.sum(Enum.zip_with(sy_given_b_off, sy_off, fn a, b -> abs(a - b) end))

report = """
# Joint-head audit — within-frame dependency the independent head cannot represent

#{length(files)} expert files, port #{port}, #{n_total} frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | #{f3.(uncond.hb)} | #{f3.(uncond.hx)} | #{f3.(uncond.hy)} | #{f3.(uncond.hj)} | **#{f3.(uncond.tc)}** |
| given action-state id | | | | | **#{f3.(tc_action)}** |
| given action-state + situation labels | | | | | **#{f3.(tc_action_sit)}** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
#{pair_rows}

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **#{f3.(tv_upb)}**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
#{Enum.map_join(Enum.with_index(sy_given_b_off), " ", fn {v, i} -> "#{i}:#{f.(v)}" end)}
P(stick_y bucket | offstage):
#{Enum.map_join(Enum.with_index(sy_off), " ", fn {v, i} -> "#{i}:#{f.(v)}" end)}

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
