# Expert-distribution recovery labeler (2026-10-08, INPUT_COHERENCE "10-08
# 13:45"; Bradley's ok for expert-labeled relabel on the bot's own offstage
# states). The 10-05 sim DAgger failed on its LABELER: a rules expert has a
# style (full deflection every frame, B taps) and mistakes (the Illusion
# press) and a small trunk carries both onstage. This labeler has neither:
# the label for a bot state is the input a real expert produced in the
# nearest expert states — sampled, not voted, so the expert's hazards stay
# hazards (the aim is 9 % at a decision frame, not 0 and not 100).
#
#   index  = every offstage frame of N expert FD Fox games: a mirrored,
#            scaled feature vector + that frame's input + "was it a hold"
#   label  = among the k nearest index rows to (bot state, bot's previous
#            label), sample one: a hold -> hold the bot's previous label;
#            a change -> that expert input, un-mirrored to the bot's side
#
# Mirroring: every x-quantity is multiplied by `toward` = the direction of
# the stage from the player, so "drifting toward the stage with the stick
# toward the stage" is one state whichever side the player fell off.
# Loaded with Code.require_file by scripts/build_expert_recovery_index.exs
# and scripts/sim_recovery_dagger_expert.exs (a .exs so it can be written
# while a queue's mix stages are recompiling lib/).
defmodule ExPhil.Agents.ExpertRecoveryLabeler do
  alias ExPhil.Sim.GA

  @k 8
  # feature scales: a difference of one scale unit costs 1.0 in the distance
  # (weights > 1 make a feature count more than its natural unit)
  @dims [
    {:y, 40.0, 1.0},
    {:dist, 30.0, 1.0},
    {:vx, 2.0, 1.0},
    {:vy, 2.0, 1.0},
    {:jumps, 1.0, 3.0},
    {:facing, 1.0, 1.0},
    {:special, 1.0, 2.0},
    {:hitstun, 1.0, 2.0},
    {:airdodge, 1.0, 1.5},
    {:jumping, 1.0, 1.5},
    {:action_frame, 30.0, 0.7},
    {:prev_sx, 1.0, 1.5},
    {:prev_sy, 1.0, 1.5},
    {:prev_b, 1.0, 1.5},
    {:prev_jump, 1.0, 1.5},
    {:opp_dx, 60.0, 0.5},
    {:opp_dy, 60.0, 0.5}
  ]

  def k, do: @k
  def dim_names, do: Enum.map(@dims, &elem(&1, 0))

  @doc "Direction of the stage from the player: +1 (player is left of centre) or -1."
  def toward(p), do: if((p.x || 0.0) < 0.0, do: 1.0, else: -1.0)

  @doc "Offstage, airborne, not on the ledge, not dead/helpless: a labelable state."
  def labelable?(p, edge) do
    p.on_ground != true and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -5.0) and
      (p.action || 0) > 13 and (p.action || 0) != 35 and (p.action || 0) not in 252..263
  end

  @doc """
  Raw (unscaled) feature map for a player state, previous input and opponent.
  `prev_p` = the player's state one frame earlier: velocity is the POSITION
  DELTA (v2, 10-08 22:30). v1 read `speed_y_self` / `speed_air_x_self`, which
  Slippi only records from 3.5 — on this corpus the index columns were all
  zero while the sim fills them, so the bot's every state looked "far" (77 %
  beyond the expert's q95, d2 carried by vy 1.35 + vx 0.58). Ranking was
  unaffected (a constant per query), the distances were not.
  """
  def features(p, prev, opp, edge, prev_p \\ nil) do
    t = toward(p)
    a = p.action || 0
    ms = prev.main_stick || %{x: 0.5, y: 0.5}

    %{
      y: p.y || 0.0,
      dist: abs(p.x || 0.0) - edge,
      vx: if(prev_p, do: ((p.x || 0.0) - (prev_p.x || 0.0)) * t, else: 0.0),
      vy: if(prev_p, do: (p.y || 0.0) - (prev_p.y || 0.0), else: 0.0),
      jumps: p.jumps_left || 0,
      facing: (p.facing || 1) * t,
      special: if(a >= 341, do: 1.0, else: 0.0),
      hitstun: if((p.hitstun_frames_left || 0) > 0, do: 1.0, else: 0.0),
      airdodge: if(a == 236, do: 1.0, else: 0.0),
      jumping: if(a in 25..28, do: 1.0, else: 0.0),
      action_frame: p.action_frame || 0,
      prev_sx: ((ms[:x] || 0.5) - 0.5) * 2.0 * t,
      prev_sy: ((ms[:y] || 0.5) - 0.5) * 2.0,
      prev_b: if(prev.button_b, do: 1.0, else: 0.0),
      prev_jump: if(prev.button_x or prev.button_y, do: 1.0, else: 0.0),
      opp_dx: if(opp, do: ((opp.x || 0.0) - (p.x || 0.0)) * t, else: 0.0),
      opp_dy: if(opp, do: (opp.y || 0.0) - (p.y || 0.0), else: 0.0)
    }
  end

  @doc "Scaled, weighted feature list in `@dims` order."
  def vector(feats), do: for({name, scale, w} <- @dims, do: feats[name] / scale * w)

  @doc """
  A controller as a flat row in the TOWARD frame (sticks' x mirrored by `t`):
  [msx, msy, csx, csy, a, b, x, y, z, l, r, shoulder].
  """
  def controller_row(c, t) do
    ms = c.main_stick || %{x: 0.5, y: 0.5}
    cs = c.c_stick || %{x: 0.5, y: 0.5}

    [
      0.5 + ((ms[:x] || 0.5) - 0.5) * t, ms[:y] || 0.5,
      0.5 + ((cs[:x] || 0.5) - 0.5) * t, cs[:y] || 0.5,
      b(c.button_a), b(c.button_b), b(c.button_x), b(c.button_y), b(c.button_z), b(c.button_l), b(c.button_r),
      max(c.l_shoulder || 0.0, c.r_shoulder || 0.0)
    ]
  end

  defp b(true), do: 1.0
  defp b(_), do: 0.0

  @doc "Controller map from a row, un-mirrored to the side `t` (inverse of controller_row/2)."
  def row_controller([msx, msy, csx, csy, a, bb, x, y, z, l, r, sh], t, template) do
    %{
      template
      | main_stick: %{x: 0.5 + (msx - 0.5) * t, y: msy},
        c_stick: %{x: 0.5 + (csx - 0.5) * t, y: csy},
        button_a: a > 0.5, button_b: bb > 0.5, button_x: x > 0.5, button_y: y > 0.5,
        button_z: z > 0.5, button_l: l > 0.5, button_r: r > 0.5,
        l_shoulder: sh, r_shoulder: 0.0
    }
  end

  @doc "Same input at the 16-bucket resolution the event heads use (a hold)."
  def same_input?(a, b2) do
    ExPhil.Training.Data.controller_to_action(a, axis_buckets: 16) ==
      ExPhil.Training.Data.controller_to_action(b2, axis_buckets: 16)
  end

  # ---- index -------------------------------------------------------------------

  @doc """
  Build the index from training frames of one game (`Peppi.to_training_frames`
  output, own player = port 1): rows for every labelable frame t with the
  input at t as the label and the input at t-1 as the previous input.
  Returns `[{vector, controller_row, hold?}]`.
  """
  def index_rows(frames, stage) do
    edge = GA.stage_edge(stage)

    frames
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.flat_map(fn [f0, f1] ->
      p = f1.game_state.players[1]

      if f1.game_state.frame == f0.game_state.frame + 1 and labelable?(p, edge) do
        t = toward(p)
        v = features(p, f0.controller, f1.game_state.players[2], edge, f0.game_state.players[1]) |> vector()
        [{v, controller_row(f1.controller, t), same_input?(f1.controller, f0.controller)}]
      else
        []
      end
    end)
  end

  @doc "Pack rows into tensors: %{x: {n, d} f32, labels: {n, 12} f32, hold: {n} u8, n: n}."
  def pack(rows) do
    n = length(rows)

    %{
      x: Nx.tensor(Enum.map(rows, &elem(&1, 0)), type: :f32),
      labels: Nx.tensor(Enum.map(rows, &elem(&1, 1)), type: :f32),
      hold: Nx.tensor(Enum.map(rows, fn {_, _, h} -> if(h, do: 1, else: 0) end), type: :u8),
      n: n
    }
  end

  def save(index, path) do
    File.mkdir_p!(Path.dirname(path))

    File.write!(path, :erlang.term_to_binary(%{
      dims: dim_names(), k: @k, n: index.n,
      x: Nx.to_binary(index.x), labels: Nx.to_binary(index.labels), hold: Nx.to_binary(index.hold),
      meta: Map.get(index, :meta, %{})
    }, [:compressed]))
  end

  def load(path) do
    m = path |> File.read!() |> :erlang.binary_to_term()
    d = length(m.dims)
    backend = Nx.default_backend()

    %{
      x: Nx.from_binary(m.x, :f32) |> Nx.reshape({m.n, d}) |> Nx.backend_transfer(backend),
      # labels / holds are read per neighbour on the Elixir side: tuples
      labels: Nx.from_binary(m.labels, :f32) |> Nx.reshape({m.n, 12}) |> Nx.to_list() |> List.to_tuple(),
      hold: Nx.from_binary(m.hold, :u8) |> Nx.to_list() |> List.to_tuple(),
      n: m.n, k: m.k, dims: m.dims, meta: m.meta
    }
  end

  # ---- labelling -----------------------------------------------------------------

  @doc """
  k-nearest index rows for a batch of query vectors: `{batch, k}` s64 indices
  (brute force; the index is ~10^5 rows and the query batches are small).
  """
  def nearest(index, queries, k \\ @k), do: nearest_d2(index, queries, k) |> elem(0)

  @doc """
  Like `nearest/3` but also returns the squared distance to the NEAREST row,
  `{idx {batch, k}, d2 {batch}}` — the coverage readout: a query far from
  every expert row is a state the expert never visits, and its sampled label
  is an extrapolation (10-08 22:12: `airdodge_with_jump` labels on the bot's
  low, jump-in-hand states).
  """
  def nearest_d2(index, queries, k \\ @k) do
    q = Nx.tensor(queries, type: :f32)
    # one jitted call: the {batch, n} distance matrix (256 x 214k f32 = 219 MB)
    # lives only inside the executable. Eagerly it was a device buffer per
    # chunk freed only at the beam's next GC — the 10-08 22:16 RESOURCE_EXHAUSTED
    # after ~100 chunks (and the 12-seed rollout's death at seed 8).
    {idx, d2} = Nx.Defn.jit(&nearest_kernel/3).(q, index.x, k: k)
    :erlang.garbage_collect()
    {idx, d2}
  end

  import Nx.Defn

  defn nearest_kernel(q, x, opts \\ []) do
    k = opts[:k]
    # squared distances via |q|^2 - 2 q.x + |x|^2
    qq = Nx.sum(q * q, axes: [1]) |> Nx.new_axis(1)
    xx = Nx.sum(x * x, axes: [1]) |> Nx.new_axis(0)
    d2 = qq - 2.0 * Nx.dot(q, [1], x, [1]) + xx
    {neg, idx} = Nx.top_k(-d2, k: k)
    {idx, -(neg |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1]))}
  end

  @doc "Squared distance from each `{player, prev, opponent}` state to its nearest expert row."
  def nearest_d2_batch(index, states, edge) do
    states
    |> Enum.map(&(query_features(&1, edge) |> vector()))
    |> Enum.chunk_every(256)
    |> Enum.flat_map(fn q -> nearest_d2(index, q, 1) |> elem(1) |> Nx.to_list() end)
  end

  @doc """
  Labels for a batch of `{player, prev_label, opponent}` states: for each, one of
  its k nearest expert rows is sampled; a hold keeps `prev_label`, a change
  becomes that expert input on the player's side. Returns `[controller]`.
  Sampling uses the process `:rand` state (seed it for reproducibility).
  `max_d2:` (optional) — a state whose nearest expert row is farther than
  this gets `nil` (no label: the expert has not been there).
  """
  def label_batch(index, states, edge, opts \\ []) do
    max_d2 = opts[:max_d2]
    queries = Enum.map(states, &(query_features(&1, edge) |> vector()))
    # distance matrices in chunks of 256 queries (256 x n floats each)
    {idx, d2} =
      queries
      |> Enum.chunk_every(256)
      |> Enum.map(fn q -> nearest_d2(index, q, index.k) end)
      |> Enum.reduce({[], []}, fn {i, d}, {is, ds} -> {is ++ Nx.to_list(i), ds ++ Nx.to_list(d)} end)

    labels = index.labels
    holds = index.hold

    Enum.zip([states, idx, d2])
    |> Enum.map(fn {state, neighbours, best} ->
      {p, prev} = {elem(state, 0), elem(state, 1)}
      j = Enum.at(neighbours, :rand.uniform(length(neighbours)) - 1)

      cond do
        max_d2 != nil and best > max_d2 -> nil
        elem(holds, j) == 1 -> prev
        true -> row_controller(elem(labels, j), toward(p), prev)
      end
    end)
  end

  @doc "One label (see label_batch/4); `nil` when gated by `max_d2:`."
  def label(index, p, prev, opp, edge, opts \\ []), do: hd(label_batch(index, [{p, prev, opp, opts[:prev_p]}], edge, opts))

  @doc "A query state is `{player, prev_input, opponent}` or `{player, prev_input, opponent, prev_player}`."
  def query_features({p, prev, opp}, edge), do: features(p, prev, opp, edge)
  def query_features({p, prev, opp, prev_p}, edge), do: features(p, prev, opp, edge, prev_p)
end
