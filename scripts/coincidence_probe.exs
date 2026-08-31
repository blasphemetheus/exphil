# Teacher-forced coincidence probe — settles the 0831 "coincidence inversion"
# (eval_runs/0831_v11_score/RESULTS.md): 8a's frozen-trunk AR head lifted
# offstage P(up|B) 2.2-2.65x while IND sat at 1.05x; after the unfreeze the
# ARMS' LIVE PLAY showed AR 1.28x vs IND 1.93x. Live play confounds the
# model's conditioning with which states each arm reaches. This probe asks
# the MODELS directly, on the SAME expert offstage states:
#
#   R_state = E_s[P(B|s) * P(up|s)] / (E_s P(B|s) * E_s P(up|s))
#     state-mediated coincidence — how much the trunk alone co-locates
#     "press B" and "hold up" onto the same frames. Both heads have this.
#   L_cond  = E_s P(up | s, B=1) / E_s P(up | s, B=0)      (AR head only)
#     the head's WIRING lift: teacher-force tf_buttons B on/off (main_x
#     forced neutral bucket 8) and read the main_y logits.
#
#   predicted offstage lift ~ R_state * L_cond (AR) or R_state (IND).
#
# If the unfreeze moved the dependency INTO the trunk ("absorption"),
# v1.1-AR shows L_cond near 1 with R_state up vs the 8a ARhead; if the small-n
# story is right, v1.1-AR keeps L_cond >> 1 and the live inversion was noise.
#
#   mix run scripts/coincidence_probe.exs \
#     --set v11_AR=checkpoints/fox_gen_v1.1_AR_..._policy.bin \
#     --set v11_IND=checkpoints/... --set 8a_ARhead=checkpoints/... \
#     --replays 'replays/erickfm_ranked/FOX/extracted/*.slp' \
#     --limit-files 8 --out eval_runs/0831_coincidence_probe/RESULTS.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Situations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, replays: :string, limit_files: :integer, char_id: :integer,
             max_states: :integer, up_bucket: :integer, out: :string]
  )

sets = Keyword.get_values(opts, :set) |> Enum.map(fn s -> [n, p] = String.split(s, "=", parts: 2); {n, p} end)
if sets == [], do: raise("--set NAME=POLICY_PATH required")
glob = opts[:replays] || raise("--replays required")
limit_files = opts[:limit_files] || 8
char_id = opts[:char_id] || 2
max_states = opts[:max_states] || 4000
up_from = opts[:up_bucket] || 12

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files * 4)

Output.banner("Teacher-forced coincidence probe")
Output.config(Enum.map(sets, fn {n, p} -> {n, Path.basename(p)} end) ++
  [{"Replays", "#{limit_files} fox-resolved from #{length(files)} candidates"},
   {"Up buckets", "#{up_from}..16"}, {"Max offstage states", max_states}])

# Fox port per file (E1); skip dittos
resolve = fn path ->
  case Peppi.metadata(path) do
    {:ok, meta} ->
      case Enum.filter(meta.players, &(&1.character == char_id)) do
        [%{port: p}] -> {:ok, p}
        _ -> :skip
      end

    _ -> :skip
  end
end

picked =
  files
  |> Enum.flat_map(fn f ->
    case resolve.(f) do
      {:ok, p} -> [{f, p}]
      _ -> []
    end
  end)
  |> Enum.take(limit_files)

if picked == [], do: raise("no fox-resolved files")

# B is button index 1 in the (a,b,x,y,z,l,r,d_up) order (load_heads doc).
b_idx = 1

probe_one = fn {name, policy} ->
  Output.puts("== #{name}: loading trunk + heads")
  trunk = Activations.load_trunk(policy)
  heads = Activations.load_heads_only(policy)

  feats =
    picked
    |> Enum.flat_map(fn {path, fox_port} ->
      opp = if fox_port == 1, do: 2, else: 1
      cap = Activations.capture_replay(trunk, path, player_port: fox_port, opponent_port: opp, labels: false)

      {:ok, replay} = Peppi.parse(path)

      states =
        replay
        |> Peppi.to_training_frames(player_port: fox_port, opponent_port: opp)
        |> Enum.reject(&(&1.game_state.frame < 0))
        |> Enum.map(& &1.game_state)

      sits = Situations.label_states(states, fox_port, as: :set) |> List.to_tuple()

      off_rows =
        0..(cap.n - 1)
        |> Enum.filter(fn r ->
          idx = cap.frame_offset + r
          idx < tuple_size(sits) and MapSet.member?(elem(sits, idx), :offstage)
        end)

      if off_rows == [] do
        []
      else
        idx_t = Nx.tensor(off_rows)
        [Nx.take(cap.activations, idx_t, axis: 0)]
      end
    end)

  feats = Nx.concatenate(feats, axis: 0)
  n = min(elem(Nx.shape(feats), 0), max_states)
  feats = Nx.slice_along_axis(feats, 0, n, axis: 0)
  Output.puts("   #{name}: #{n} offstage states")

  softmax_up = fn my_l ->
    p = Axon.Activations.softmax(my_l, axis: -1)
    width = elem(Nx.shape(p), 1)
    Nx.sum(Nx.slice_along_axis(p, up_from, width - up_from, axis: 1), axes: [1])
  end

  tf_for = fn b_on ->
    buttons = if b_on, do: Nx.tensor([[0, 1, 0, 0, 0, 0, 0, 0]]) |> Nx.broadcast({n, 8}), else: Nx.broadcast(0, {n, 8})
    neutral = Nx.broadcast(8, {n})

    Heads.tf_inputs(%{
      buttons: buttons,
      main_x: neutral,
      main_y: neutral,
      c_x: neutral,
      c_y: neutral,
      shoulder: Nx.broadcast(0, {n})
    })
  end

  {p_b, p_up_base, p_up_b1} =
    case heads.head do
      :independent ->
        {b_l, _mx, my_l, _cx, _cy, _sh} = heads.predict_fn.(heads.params, %{"trunk" => feats})
        p_b = Nx.sigmoid(b_l[[.., b_idx]])
        p_up = softmax_up.(my_l)
        {p_b, p_up, nil}

      :autoregressive ->
        in0 = Map.merge(%{"trunk" => feats}, tf_for.(false))
        in1 = Map.merge(%{"trunk" => feats}, tf_for.(true))
        {b_l, _mx0, my_l0, _cx0, _cy0, _sh0} = heads.predict_fn.(heads.params, in0)
        {_b1, _mx1, my_l1, _cx1, _cy1, _sh1} = heads.predict_fn.(heads.params, in1)
        p_b = Nx.sigmoid(b_l[[.., b_idx]])
        {p_b, softmax_up.(my_l0), softmax_up.(my_l1)}
    end

  m = fn t -> Nx.to_number(Nx.mean(t)) end
  mean_b = m.(p_b)
  mean_up = m.(p_up_base)
  r_state = m.(Nx.multiply(p_b, p_up_base)) / max(mean_b * mean_up, 1.0e-9)
  l_cond = if p_up_b1, do: m.(p_up_b1) / max(mean_up, 1.0e-9), else: nil

  %{name: name, head: heads.head, n: n, mean_b: mean_b, mean_up: mean_up,
    r_state: r_state, l_cond: l_cond}
end

results = Enum.map(sets, probe_one)

f2 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end
f3 = fn v -> :erlang.float_to_binary(v * 100.0, decimals: 2) end

rows =
  Enum.map(results, fn r ->
    pred = r.r_state * (r.l_cond || 1.0)

    "| #{r.name} | #{r.head} | #{r.n} | #{f3.(r.mean_b)} | #{f3.(r.mean_up)} | " <>
      "#{f2.(r.r_state)} | #{if r.l_cond, do: f2.(r.l_cond), else: "—"} | #{f2.(pred)} |"
  end)

table =
  Enum.join(
    [
      "| checkpoint | head | states | mean P(B) % | mean P(up) % | R_state | L_cond | predicted lift |",
      "|---|---|---:|---:|---:|---:|---:|---:|" | rows
    ],
    "\n"
  )

report = """
# Teacher-forced coincidence probe

Same #{length(picked)} expert files (fox-resolved ports), offstage decision
states, all checkpoints see IDENTICAL states — live-play state-distribution
confounds removed. "up" = main_y buckets #{up_from}..16. R_state =
state-mediated coincidence (trunk); L_cond = the AR head's wiring lift
(teacher-forced B on/off, main_x forced neutral); predicted offstage
P(up|B)/P(up) ~ R_state x L_cond.

#{table}

Reference points: 8a live lift AR 2.20-2.65x vs IND 1.05x (frozen trunk);
post-unfreeze ARM replays AR 1.28x vs IND 1.93x (state-confounded, small n).
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end
