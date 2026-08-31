# HEADROOM TRIAD, Leg S — selection headroom (pass@k vs pass@1).
#
# See docs/planning/HEADROOM_TRIAD.md. The question this answers:
#
#   Does the policy CONTAIN the right action and fail to select it, or does
#   it not contain it at all?
#
# pass@1 = P(one sample matches the master's action)
# pass@k = P(at least one of k samples matches)
# pass@k - pass@1 = what a PERFECT selector could recover without touching
#                   a single weight. The upper bound on a value model.
#
# Two design choices carry the whole thing:
#
# 1. DECISION FRAMES ONLY. The existing fixture-agreement metric saturated
#    at 0.974 flat across ten epochs and has zero ranking power, because
#    most Melee frames are trivial (holding, or stuck in an animation) and
#    an average over them is decided by frames where nothing is at stake.
#    A frame counts here only if it carries a real situation label
#    (ExPhil.Situations) AND the master's input actually CHANGED on it —
#    the moments of commitment.
#
# 2. THE UNBIASED pass@k ESTIMATOR (Codex/HumanEval). Draw n samples once
#    per frame, count c correct, then for every k <= n:
#        pass@k = 1 - C(n-c, k) / C(n, k)
#    One sampling pass yields the entire curve, and the estimate is
#    unbiased rather than "did any of my first k happen to hit".
#
# Usage:
#   mix run scripts/interp_passk.exs --policy checkpoints/fox_gen_v1_..._ep10.bin \
#     --replays 'replays/huggingface/*.slp' --n 16 --temperature 0.5 --limit-frames 2000
#
# Options:
#   --policy PATH        policy .bin (required)
#   --replays GLOB       master replays to score against (required)
#   --port N             pin the port (use for a port-NORMALIZED corpus);
#                        omit to auto-detect per file by character
#   --char-id N          character to score when auto-detecting (default 2 = Fox)
#   --n N                samples per frame (default 16); pass@k reported for k<=n
#   --temperature T      decode temperature (default 0.5, the deploy decode)
#   --buttons-temperature T  per-head button temperature (optional)
#   --limit-frames N     cap decision frames scored (default 2000)
#   --limit-files N      cap replays parsed (default 20)
#   --stick-tol F        stick match tolerance, stick units (default 0.0625 = 1/16)
#   --all-frames         DIAGNOSTIC: score every frame, not just decision frames
#                        (use to demonstrate the saturation this metric avoids)
#   --out PATH           write a markdown report
#   --seed N             RNG seed for the per-frame samples (default 20260828)

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replays: :string,
      port: :integer,
      char_id: :integer,
      n: :integer,
      temperature: :float,
      buttons_temperature: :float,
      limit_frames: :integer,
      limit_files: :integer,
      stick_tol: :float,
      all_frames: :boolean,
      out: :string,
      seed: :integer
    ]
  )

policy_path = opts[:policy] || raise "--policy required"
glob = opts[:replays] || raise "--replays required (glob)"
port = opts[:port] || 1
char_id = opts[:char_id] || 2
n_samples = opts[:n] || 16
temperature = opts[:temperature] || 0.5
limit_frames = opts[:limit_frames] || 2000
limit_files = opts[:limit_files] || 20
stick_tol = opts[:stick_tol] || 0.0625
decision_only = !opts[:all_frames]

# Labels that mark a frame as carrying a real decision. Geometry-only
# labels (:onstage_center, :on_platform, ...) are deliberately excluded —
# they describe where you are, not that anything is being decided.
decision_labels =
  MapSet.new([
    :neutral,
    :advantage,
    :disadvantage,
    :approach,
    :retreat,
    :juggle,
    :tech_chase,
    :ledge_trap,
    :shield_pressure_ours,
    :shield_break_confirm,
    :pummel_throw_decision,
    :edgeguard,
    :conversion_open,
    :combo_active,
    :in_hitstun,
    :tumble,
    :being_juggled,
    :being_tech_chased,
    :being_edgeguarded,
    :recovery_low,
    :recovery_high,
    :cornered,
    :shield_pressure_theirs,
    :jc_window,
    :shine_cancellable
  ])

legal_buttons = [
  :button_a,
  :button_b,
  :button_x,
  :button_y,
  :button_z,
  :button_l,
  :button_r,
  :button_d_up
]

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files)
if files == [], do: raise("no replays matched #{glob}")

Output.banner("HEADROOM TRIAD Leg S — selection headroom (pass@k)")

Output.config([
  {"Policy", Path.basename(policy_path)},
  {"Replays", "#{length(files)} files"},
  {"Port", if(opts[:port], do: "pinned #{port}", else: "auto by char #{char_id}")},
  {"Samples/frame (n)", n_samples},
  {"Temperature", temperature},
  {"Frames", if(decision_only, do: "DECISION only", else: "ALL (diagnostic)")},
  {"Stick tolerance", stick_tol}
])

# ---- collect candidate frames ----------------------------------------------

pressed_set = fn c ->
  Enum.reduce(legal_buttons, MapSet.new(), fn b, acc ->
    if Map.get(c, b), do: MapSet.put(acc, b), else: acc
  end)
end

# "The master committed to something here": the pressed-button set changed,
# or a stick moved by more than the match tolerance.
input_changed? = fn prev, cur ->
  cond do
    is_nil(prev) ->
      false

    not MapSet.equal?(pressed_set.(prev), pressed_set.(cur)) ->
      true

    abs(prev.main_stick.x - cur.main_stick.x) > stick_tol ->
      true

    abs(prev.main_stick.y - cur.main_stick.y) > stick_tol ->
      true

    abs(prev.c_stick.x - cur.c_stick.x) > stick_tol ->
      true

    abs(prev.c_stick.y - cur.c_stick.y) > stick_tol ->
      true

    true ->
      false
  end
end

Output.puts("Collecting frames...")

# Resolve WHICH PORT holds the master we are scoring against, per file.
#
# This is not a detail. The training corpus
# (replays/erickfm_ranked/FOX/extracted) is normalized to port 1, but other
# Fox corpora are not: replays/fox_il_v1 has filenames like
# "Captain Falcon + Fox (FD).slp", so a fixed --port would compare the
# policy's Fox actions against CAPTAIN FALCON's inputs on half the corpus
# and report the resulting garbage as a pass@k number.
#
# --port pins it explicitly (use for a normalized corpus). Otherwise the
# port is detected per file by character id (Fox = 2, the id
# analyze_shine_source.exs:56 uses). Dittos are SKIPPED under auto-detect:
# with two Foxes there is no principled way to know which one the metric
# means, and guessing would quietly corrupt the number.
resolve_port = fn path ->
  cond do
    opts[:port] ->
      {:ok, port}

    true ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == char_id)) do
            [%{port: p}] -> {:ok, p}
            [_ | _] -> {:skip, :ditto}
            [] -> {:skip, :character_absent}
          end

        _ ->
          {:skip, :unreadable}
      end
  end
end

# Candidates are {path, index, port}; the per-file frame lists are kept so
# each decision frame can be scored WITH ITS OWN GAME'S HISTORY (see the
# sampling section — the policy is a windowed temporal model).
{candidates, total_frames, frames_by_path} =
  Enum.reduce(files, {[], 0, %{}}, fn path, {acc, total, by_path} ->
    with {:ok, file_port} <- resolve_port.(path),
         {:ok, replay} <- Peppi.parse(Path.expand(path)) do
      opp = if file_port == 1, do: 2, else: 1

      frames =
        replay
        |> Peppi.to_training_frames(player_port: file_port, opponent_port: opp)
        |> Enum.reject(&(&1.game_state.frame < 0))

      masks =
        if decision_only do
          Situations.label_states(Enum.map(frames, & &1.game_state), file_port, as: :set)
        else
          List.duplicate(nil, length(frames))
        end

      # Pair each frame with the PREVIOUS frame's controller in one pass
      # (Enum.at/2 inside a filter would make this quadratic).
      prevs = [nil | Enum.map(Enum.drop(frames, -1), & &1.controller)]

      # Each candidate carries its own port: the corpus is not guaranteed
      # port-normalized, and inference must be run for the SAME player the
      # ground-truth controller came from.
      picked =
        [frames, masks, prevs]
        |> Enum.zip()
        |> Enum.with_index()
        |> Enum.filter(fn {{f, set, prev}, _idx} ->
          situational? =
            not decision_only or
              (set && not MapSet.disjoint?(set, decision_labels))

          situational? and (not decision_only or input_changed?.(prev, f.controller))
        end)
        |> Enum.map(fn {_, idx} -> {path, idx, file_port} end)

      {[picked | acc], total + length(frames), Map.put(by_path, path, List.to_tuple(frames))}
    else
      _ -> {acc, total, by_path}
    end
  end)

candidates = candidates |> Enum.reverse() |> Enum.concat()

# Spread the sample across the whole corpus rather than taking a prefix
# (a prefix is one player, one stretch of one game).
selected =
  if length(candidates) > limit_frames do
    stride = max(div(length(candidates), limit_frames), 1)
    candidates |> Enum.take_every(stride) |> Enum.take(limit_frames)
  else
    candidates
  end

Output.puts(
  "  #{total_frames} frames -> #{length(candidates)} candidates " <>
    "(#{Float.round(100.0 * length(candidates) / max(total_frames, 1), 1)}%) -> " <>
    "#{length(selected)} scored"
)

if selected == [], do: raise("no frames selected — loosen the decision-frame filter")

# ---- sample the policy ------------------------------------------------------

agent_opts = [
  policy_path: policy_path,
  deterministic: false,
  temperature:
    if(opts[:buttons_temperature],
      do: %{
        buttons: opts[:buttons_temperature],
        main: temperature,
        c: temperature,
        shoulder: temperature
      },
      else: temperature
    ),
  delay_id: 0
]

{:ok, agent} = Agent.start_link(agent_opts)
Agent.warmup(agent)

agent_config = Agent.get_config(agent)
window = if agent_config.temporal, do: agent_config.window_size || 60, else: 0

# AR-head checkpoints sample SEQUENTIALLY (buttons condition the sticks).
# Drawing n components independently from one forward — the independent-head
# path below — would both ignore the conditioning and reuse one path's
# conditional logits, silently scoring the AR head as if it were independent.
ar_head? = Map.get(agent_config, :head) == :autoregressive
Output.puts("  Head: #{Map.get(agent_config, :head, :independent)}")

# ---- WHY the sampling below looks the way it does (first-run bug, 08-28) ---
#
# The first execution of this script returned pass@k IDENTICAL to pass@1 at
# every k for every head. Two defects, both in how the policy was queried:
#
# 1. `Agent.get_controller/3` enforces ONE DECISION PER GAME FRAME (the
#    async runner polls faster than Dolphin produces frames, so a repeated
#    frame re-sends the cached action). Calling it 16x on the same frame
#    yielded ONE sample and 15 copies of it — pass@k could not move.
#
# 2. The policy is a windowed temporal model (GRU, window 60) and was fed
#    isolated frames strided across the corpus, so its window held frames
#    from unrelated games. Every logit was conditioned on garbage.
#
# Now: for each decision frame, reset the agent, replay the preceding
# `window` frames OF THE SAME GAME through the normal per-frame path (each
# has a distinct frame number, so no debounce), run ONE forward at the
# decision frame via `get_action_with_confidence` (returns the raw head
# logits), and draw n samples from those logits here — the same
# Bernoulli / Gumbel-max decode the Agent uses, at the same temperatures.

head_temps =
  case agent_config.temperature do
    %{} = m -> m
    t when is_number(t) -> %{buttons: t, main: t, c: t, shoulder: t}
    _ -> %{buttons: 1.0, main: 1.0, c: 1.0, shoulder: 1.0}
  end

axis_buckets = 16

flat = fn t -> t |> Nx.squeeze() |> Nx.to_flat_list() end

# Draw n samples from one set of head logits. Returns a list of n
# ControllerStates.
draw_samples = fn action, n, key ->
  b_logits = Nx.squeeze(action.logits.buttons)
  b_probs = Nx.sigmoid(Nx.divide(b_logits, head_temps.buttons))
  {u, key} = Nx.Random.uniform(key, shape: {n, Nx.size(b_probs)})
  buttons = Nx.greater(Nx.new_axis(b_probs, 0), u)

  categorical = fn logits, temp, key ->
    scaled = Nx.divide(Nx.squeeze(logits), temp)
    {r, key} = Nx.Random.uniform(key, shape: {n, Nx.size(scaled)})
    gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(r, 1.0e-10)))))
    {Nx.argmax(Nx.add(Nx.new_axis(scaled, 0), gumbel), axis: 1), key}
  end

  {main_x, key} = categorical.(action.logits.main_x, head_temps.main, key)
  {main_y, key} = categorical.(action.logits.main_y, head_temps.main, key)
  {c_x, key} = categorical.(action.logits.c_x, head_temps.c, key)
  {c_y, key} = categorical.(action.logits.c_y, head_temps.c, key)
  {shoulder, key} = categorical.(action.logits.shoulder, head_temps.shoulder, key)

  samples =
    for i <- 0..(n - 1) do
      ExPhil.Networks.Policy.to_controller_state(
        %{
          buttons: buttons[i],
          main_x: main_x[i],
          main_y: main_y[i],
          c_x: c_x[i],
          c_y: c_y[i],
          shoulder: shoulder[i]
        },
        axis_buckets: axis_buckets
      )
    end

  {samples, key}
end

_ = flat

match_buttons? = fn a, b -> MapSet.equal?(pressed_set.(a), pressed_set.(b)) end

match_stick? = fn a, b, which ->
  pa = Map.get(a, which)
  pb = Map.get(b, which)
  abs(pa.x - pb.x) <= stick_tol and abs(pa.y - pb.y) <= stick_tol
end

match_shoulder? = fn a, b ->
  abs((a.l_shoulder || 0.0) - (b.l_shoulder || 0.0)) <= 0.25
end

Output.puts(
  "Sampling #{n_samples}x per frame from one forward each, with #{window} frames of " <>
    "same-game history (#{length(selected) * (window + 1)} inferences)..."
)

rng = Nx.Random.key(opts[:seed] || 20_260_828)

# correct-counts per head, per frame
{counts, _rng} =
  selected
  |> Enum.with_index(1)
  |> Enum.map_reduce(rng, fn {{path, fi, f_port}, idx}, key ->
    if rem(idx, 100) == 0 do
      Output.progress_bar(idx, length(selected), label: "pass@k")
    end

    frames = frames_by_path[path]
    f = elem(frames, fi)
    truth = f.controller

    # Replay this game's preceding window through the agent so the temporal
    # buffer holds the real context, then one forward at the decision frame.
    Agent.reset_buffer(agent)

    for hi <- max(fi - window, 0)..(fi - 1)//1 do
      Agent.get_controller(agent, elem(frames, hi).game_state, player_port: f_port)
    end

    {samples, key} =
      if ar_head? do
        # n coherent sequential samples via the agent's instrument path
        # (side-effect-free: no debounce, no buffer mutation).
        split = Nx.Random.split(key)
        {sub, key} = {split[0], split[1]}

        case Agent.get_action_samples(agent, f.game_state,
               player_port: f_port,
               n: n_samples,
               key: sub
             ) do
          {:ok, actions} ->
            {Enum.map(
               actions,
               &ExPhil.Networks.Policy.to_controller_state(&1, axis_buckets: axis_buckets)
             ), key}

          _ ->
            {[], key}
        end
      else
        case Agent.get_action_with_confidence(agent, f.game_state, player_port: f_port) do
          {:ok, action, _conf} -> draw_samples.(action, n_samples, key)
          _ -> {[], key}
        end
      end

    %{
      n: length(samples),
      buttons: Enum.count(samples, &match_buttons?.(&1, truth)),
      main: Enum.count(samples, &match_stick?.(&1, truth, :main_stick)),
      c: Enum.count(samples, &match_stick?.(&1, truth, :c_stick)),
      shoulder: Enum.count(samples, &match_shoulder?.(&1, truth)),
      joint:
        Enum.count(samples, fn s ->
          match_buttons?.(s, truth) and match_stick?.(s, truth, :main_stick) and
            match_stick?.(s, truth, :c_stick) and match_shoulder?.(s, truth)
        end)
    }
    |> then(&{&1, key})
  end)

Output.progress_done()

# Instrument self-check: with n independent samples per frame, the per-frame
# correct count must vary across frames unless the policy is (near-)
# deterministic. All-or-nothing counts on every frame is the 08-28 bug
# signature (one sample copied n times) — refuse to report a verdict on it.
degenerate? =
  Enum.all?(counts, fn row -> row.buttons in [0, row.n] and row.main in [0, row.n] end)

if degenerate? do
  Output.error(
    "Every frame scored all-or-nothing on buttons AND main stick: the samples are " <>
      "not independent draws (debounce/caching?). Verdict withheld."
  )
end

# ---- unbiased pass@k --------------------------------------------------------
#
#   pass@k = 1 - C(n-c, k)/C(n, k) = 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
#
# computed as a product to stay away from factorials.

pass_at_k = fn n, c, k ->
  cond do
    c >= n -> 1.0
    n - c < k -> 1.0
    k > n -> 1.0
    true -> 1.0 - Enum.reduce(0..(k - 1)//1, 1.0, fn i, acc -> acc * (n - c - i) / (n - i) end)
  end
end

ks = [1, 2, 4, 8, 16] |> Enum.filter(&(&1 <= n_samples))
heads = [:joint, :buttons, :main, :c, :shoulder]

curve =
  Map.new(heads, fn head ->
    per_k =
      Map.new(ks, fn k ->
        vals =
          Enum.map(counts, fn row -> pass_at_k.(row.n, Map.fetch!(row, head), k) end)

        {k, Enum.sum(vals) / length(vals)}
      end)

    {head, per_k}
  end)

fmt = fn v -> :erlang.float_to_binary(v * 100, decimals: 1) end

header = "| head | " <> Enum.map_join(ks, " | ", &"pass@#{&1}") <> " | **headroom** |"
sep = "|" <> String.duplicate("---|", 2 + length(ks))

rows =
  Enum.map(heads, fn head ->
    per_k = curve[head]
    p1 = per_k[1]
    pmax = per_k[Enum.max(ks)]

    "| #{head} | " <>
      Enum.map_join(ks, " | ", &fmt.(per_k[&1])) <>
      " | **+#{fmt.(pmax - p1)}** |"
  end)

table = Enum.join([header, sep | rows], "\n")
IO.puts("\n" <> table <> "\n")

joint_gap = curve[:joint][Enum.max(ks)] - curve[:joint][1]

verdict =
  cond do
    curve[:joint][1] > 0.5 and joint_gap < 0.05 ->
      "The policy is already right at decision points and barely improves with k. " <>
        "The ceiling is NOT selection and NOT capacity — look at closed-loop drift, " <>
        "the harness, or delay."

    joint_gap >= 0.10 ->
      "SELECTION headroom is large (+#{fmt.(joint_gap)} pts). The right action is in " <>
        "the distribution and the decode is not picking it. A value model has real room; " <>
        "this is the upper bound on what it could recover."

    curve[:joint][Enum.max(ks)] < 0.15 ->
      "The right action is NOT in the distribution even at k=#{Enum.max(ks)}. No selector " <>
        "can help. Pay for Leg C (capacity ladder) and Leg D (data ladder) to find out which."

    true ->
      "Ambiguous: gap +#{fmt.(joint_gap)} pts. Not clearly selection-limited. Consider " <>
        "tightening the decision-frame filter or widening n before paying for Legs C/D."
  end

Output.puts("")
Output.warning(verdict)

Output.puts("")

Output.puts(
  "REMINDER: pass@k assumes a PERFECT selector, so this is an UPPER BOUND. " <>
    "It is also open-loop (teacher-forced) and cannot see closed-loop drift. " <>
    "And a master's exact action is not the only correct play, so absolute " <>
    "pass@1 is not a skill score — only differences and the gap mean anything."
)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))

  File.write!(out, """
  # Leg S — selection headroom (pass@k)

  Policy: `#{Path.basename(policy_path)}`
  Replays: #{length(files)} files, port #{port}
  Frames: #{if(decision_only, do: "decision only", else: "ALL (diagnostic)")} —
  #{length(candidates)} candidates from #{total_frames} frames
  (#{Float.round(100.0 * length(candidates) / max(total_frames, 1), 1)}%),
  #{length(selected)} scored at n=#{n_samples} (one forward per frame, #{window} frames of same-game history), temperature #{temperature},
  stick tolerance #{stick_tol}.

  #{table}

  **Headroom** = pass@#{Enum.max(ks)} - pass@1: what a perfect selector could
  recover without changing a weight.

  ## Verdict

  #{verdict}

  ## Limits

  pass@k assumes a perfect selector and is therefore an UPPER BOUND on what a
  value model can buy. It is open-loop and cannot see closed-loop drift. A
  master's exact action is not the only correct play, so absolute pass@1 is
  not a skill score — only differences and the gap are meaningful.
  """)

  Output.success("Wrote #{out}")
end
