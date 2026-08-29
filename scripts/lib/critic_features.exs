# Shared feature code for the D2 critic (critic_extract / critic_train /
# interp_bestofn). Loaded with Code.require_file — a script-local module,
# deliberately NOT under lib/ yet (written 2026-08-29 while the AWBC arm
# chain was recompiling lib/ at every arm start; promote to
# lib/exphil/critic/ once the chain is done and it has run once).
#
# Feature layout per decision row:
#
#   phi(s) = trunk(s) (+) raw(s)
#     trunk(s): the policy's own GRU trunk output for the 60-frame window
#               ending at s ({hidden} = 256 for fox_gen_v1)
#     raw(s):   scalars the trunk DISCARDS or half-discards (G4: percent
#               R^2 0.46-0.52 vs 0.85 from input; stage half; opponent
#               character dead) — percents, stocks, positions, hitstun,
#               shield, stage one-hot (compact 7), opp character one-hot
#               (33). @raw_size below.
#
#   a        = continuous controller encoding, 13 dims, IDENTICAL to
#              Embeddings.Controller.embed_continuous/1 (buttons 8 +
#              main 2 + c 2 + shoulder 1).
#
# Two heads are trained on these (critic_train.exs):
#   V(s)      linear ridge on phi(s) -> discounted return-to-go
#   S(s, a)   bilinear selector: a^T W phi(s) + v^T a  (softmax over the
#             master's action vs K policy samples). This is the head that
#             cashes in Leg S: pick, among N samples, the one the master
#             would have chosen.
defmodule CriticFeatures do
  alias ExPhil.Bridge.ControllerState
  alias ExPhil.Embeddings.Controller, as: ControllerEmbed
  alias ExPhil.Interp.Activations
  alias ExPhil.Networks.Policy
  alias ExPhil.Situations
  alias ExPhil.Training.AdvantageWeighting

  # competitive stages -> compact index (same set as stage_mode
  # :one_hot_compact): FoD 2, PS 3, YS 8, DL 28, BF 31, FD 32, other 6
  @stage_index %{2 => 0, 3 => 1, 8 => 2, 28 => 3, 31 => 4, 32 => 5}
  @num_chars 33
  @scalars 22
  def raw_size, do: @scalars + 7 + @num_chars

  def decision_labels do
    MapSet.new([
      :neutral, :advantage, :disadvantage, :approach, :retreat, :juggle,
      :tech_chase, :ledge_trap, :shield_pressure_ours, :shield_break_confirm,
      :pummel_throw_decision, :edgeguard, :conversion_open, :combo_active,
      :in_hitstun, :tumble, :being_juggled, :being_tech_chased,
      :being_edgeguarded, :recovery_low, :recovery_high, :cornered,
      :shield_pressure_theirs, :jc_window, :shine_cancellable
    ])
  end

  @legal_buttons [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r, :button_d_up]

  def pressed_set(c) do
    Enum.reduce(@legal_buttons, MapSet.new(), fn b, acc ->
      if Map.get(c, b), do: MapSet.put(acc, b), else: acc
    end)
  end

  # Leg S match rule (interp_passk.exs): buttons exact, sticks within tol,
  # shoulder within 0.25.
  def match?(a, b, tol \\ 0.0625) do
    MapSet.equal?(pressed_set(a), pressed_set(b)) and
      abs(a.main_stick.x - b.main_stick.x) <= tol and
      abs(a.main_stick.y - b.main_stick.y) <= tol and
      abs(a.c_stick.x - b.c_stick.x) <= tol and
      abs(a.c_stick.y - b.c_stick.y) <= tol and
      abs((a.l_shoulder || 0.0) - (b.l_shoulder || 0.0)) <= 0.25
  end

  # ---- raw scalars -----------------------------------------------------------

  def raw_features(game_state, port, opp) do
    me = game_state.players[port]
    op = game_state.players[opp]
    f = fn v, d -> (v || 0) / d end
    b = fn v -> if v, do: 1.0, else: 0.0 end

    scalars = [
      f.(me.percent, 100.0), f.(op.percent, 100.0),
      f.(me.stock, 4.0), f.(op.stock, 4.0),
      f.(me.x, 100.0), f.(me.y, 100.0), f.(op.x, 100.0), f.(op.y, 100.0),
      f.(me.x - op.x, 100.0), f.(me.y - op.y, 100.0),
      b.(me.on_ground), b.(op.on_ground),
      f.(me.hitstun_frames_left, 30.0), f.(op.hitstun_frames_left, 30.0),
      f.(me.shield_strength, 60.0), f.(op.shield_strength, 60.0),
      f.(me.jumps_left, 2.0), f.(op.jumps_left, 2.0),
      b.(me.invulnerable), b.(op.invulnerable),
      f.(me.facing, 1.0), f.(op.facing, 1.0)
    ]

    stage = List.duplicate(0.0, 7) |> List.replace_at(Map.get(@stage_index, game_state.stage, 6), 1.0)

    char =
      List.duplicate(0.0, @num_chars)
      |> List.replace_at(min(max(op.character || 0, 0), @num_chars - 1), 1.0)

    scalars ++ stage ++ char
  end

  # ---- replay -> rows ---------------------------------------------------------

  @doc """
  Extract everything the critic needs from one replay.

  Returns nil when the port cannot be resolved, else a map of BinaryBackend
  tensors:
    phi {n, hidden+raw}  f32   state features (rows aligned to decision frames)
    a_master {n, 13}     f32   the master's action
    a_samples {n, K, 13} f32   K policy samples at `temperature`
    match {n, K}         u8    sample matches master under the Leg S rule
    rtg {n}              f32   discounted return-to-go (standard reward)
    decision {n}         u8    decision-frame flag (situation label + input changed)
    frame {n}            s64   game frame number
  """
  def extract_replay(trunk, heads, path, opts) do
    port = Keyword.fetch!(opts, :port)
    opp = if port == 1, do: 2, else: 1
    k = Keyword.get(opts, :k, 8)
    temperature = Keyword.get(opts, :temperature, 0.5)
    stride = Keyword.get(opts, :stride, 2)
    gamma = Keyword.get(opts, :gamma, 0.99)
    horizon = Keyword.get(opts, :horizon, 600)
    key = Keyword.get(opts, :key, Nx.Random.key(7))

    cap = Activations.capture_replay(trunk, path, player_port: port, opponent_port: opp, labels: false)
    {:ok, replay} = ExPhil.Data.Peppi.parse(Path.expand(path))

    frames =
      replay
      |> ExPhil.Data.Peppi.to_training_frames(player_port: port, opponent_port: opp)
      |> Enum.reject(&(&1.game_state.frame < 0))

    n = cap.n
    off = cap.frame_offset
    aligned = Enum.slice(frames, off, n)
    if length(aligned) != n, do: raise("alignment: #{length(aligned)} frames for #{n} rows")

    # rewards / return-to-go over the FULL frame list, then aligned
    rtg_all =
      frames
      |> AdvantageWeighting.standard_rewards(port)
      |> AdvantageWeighting.return_to_go(gamma, horizon)

    rtg = Enum.slice(rtg_all, off, n)

    # decision flag: situation label AND master's input changed
    masks = Situations.label_states(Enum.map(frames, & &1.game_state), port, as: :set)
    prevs = [nil | Enum.map(Enum.drop(frames, -1), & &1.controller)]
    dl = decision_labels()

    decision_all =
      Enum.zip([frames, masks, prevs])
      |> Enum.map(fn {f, set, prev} ->
        situational = set && not MapSet.disjoint?(set, dl)
        changed = prev != nil and not match?(prev, f.controller)
        if situational and changed, do: 1, else: 0
      end)

    decision = Enum.slice(decision_all, off, n)

    # subsample rows by stride (keeps every decision frame). Tuples: O(1)
    # access — Enum.at on 10k-frame lists inside these loops is O(n^2).
    decision_t = List.to_tuple(decision)
    aligned_t = List.to_tuple(aligned)
    rtg_t = List.to_tuple(rtg)

    keep =
      0..(n - 1)
      |> Enum.filter(fn i -> rem(i, stride) == 0 or elem(decision_t, i) == 1 end)

    idx = Nx.tensor(keep, type: :s64, backend: Nx.BinaryBackend)
    acts = cap.activations |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.take(idx, axis: 0)
    rows = Enum.map(keep, &elem(aligned_t, &1))

    raw =
      rows
      |> Enum.map(&raw_features(&1.game_state, port, opp))
      |> Nx.tensor(type: :f32, backend: Nx.BinaryBackend)

    phi = Nx.concatenate([acts, raw], axis: 1)

    a_master = ControllerEmbed.embed_continuous_batch(Enum.map(rows, & &1.controller))

    # K samples per row from the policy heads on the same trunk activations
    {a_samples, match, _key} = sample_candidates(heads, acts, rows, k, temperature, key)

    %{
      phi: phi,
      a_master: Nx.backend_copy(a_master, Nx.BinaryBackend),
      a_samples: a_samples,
      match: match,
      rtg: Nx.tensor(Enum.map(keep, &elem(rtg_t, &1)), type: :f32, backend: Nx.BinaryBackend),
      decision: Nx.tensor(Enum.map(keep, &elem(decision_t, &1)), type: :u8, backend: Nx.BinaryBackend),
      frame: Nx.tensor(Enum.map(rows, & &1.game_state.frame), type: :s64, backend: Nx.BinaryBackend)
    }
  end

  # Run the six heads on {n, hidden} trunk activations and draw K samples per
  # row with the agent's own decode (Bernoulli buttons, Gumbel-max sticks).
  def sample_candidates(heads, acts, rows, k, temperature, key) do
    out = heads.predict_fn.(heads.params, %{"trunk" => acts})

    {b, mx, my, cx, cy, sh} =
      case out do
        {_, _, _, _, _, _} = t -> t
        %{buttons: b, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh} -> {b, mx, my, cx, cy, sh}
      end

    n = Nx.axis_size(acts, 0)

    {u, key} = Nx.Random.uniform(key, shape: {k, n, 8})
    buttons = Nx.greater(Nx.new_axis(Nx.sigmoid(Nx.divide(b, temperature)), 0), u)

    gumbel_argmax = fn logits, key ->
      scaled = Nx.divide(logits, temperature)
      {r, key} = Nx.Random.uniform(key, shape: {k, n, Nx.axis_size(logits, 1)})
      g = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(r, 1.0e-10)))))
      {Nx.argmax(Nx.add(Nx.new_axis(scaled, 0), g), axis: 2), key}
    end

    {i_mx, key} = gumbel_argmax.(mx, key)
    {i_my, key} = gumbel_argmax.(my, key)
    {i_cx, key} = gumbel_argmax.(cx, key)
    {i_cy, key} = gumbel_argmax.(cy, key)
    {i_sh, key} = gumbel_argmax.(sh, key)

    # to Elixir once, then per (row, sample) build the ControllerState
    # {k, n, ...} -> tuple-of-tuples for O(1) (s, i) access (Enum.at in the
    # n*k loop below would be O(n^2 k)).
    tt = fn t -> t |> Nx.to_list() |> Enum.map(&List.to_tuple/1) |> List.to_tuple() end
    bl = tt.(Nx.as_type(buttons, :u8))
    lx = tt.(i_mx)
    ly = tt.(i_my)
    lcx = tt.(i_cx)
    lcy = tt.(i_cy)
    lsh = tt.(i_sh)

    axis_buckets = Map.get(heads.config, :axis_buckets, 16)
    shoulder_buckets = Map.get(heads.config, :shoulder_buckets, 4)

    masters = rows |> Enum.map(& &1.controller) |> List.to_tuple()

    per_row =
      for i <- 0..(n - 1) do
        master = elem(masters, i)

        for s <- 0..(k - 1) do
          cs =
            Policy.to_controller_state(
              %{
                buttons: Nx.tensor(elem(elem(bl, s), i), type: :u8, backend: Nx.BinaryBackend),
                main_x: Nx.tensor(elem(elem(lx, s), i), backend: Nx.BinaryBackend),
                main_y: Nx.tensor(elem(elem(ly, s), i), backend: Nx.BinaryBackend),
                c_x: Nx.tensor(elem(elem(lcx, s), i), backend: Nx.BinaryBackend),
                c_y: Nx.tensor(elem(elem(lcy, s), i), backend: Nx.BinaryBackend),
                shoulder: Nx.tensor(elem(elem(lsh, s), i), backend: Nx.BinaryBackend)
              },
              axis_buckets: axis_buckets,
              shoulder_buckets: shoulder_buckets
            )

          {cs, if(match?(cs, master), do: 1, else: 0)}
        end
      end

    states = per_row |> List.flatten() |> Enum.map(&elem(&1, 0))
    flags = per_row |> Enum.map(fn r -> Enum.map(r, &elem(&1, 1)) end)

    a_samples =
      states
      |> ControllerEmbed.embed_continuous_batch()
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.reshape({n, k, 13})

    {a_samples, Nx.tensor(flags, type: :u8, backend: Nx.BinaryBackend), key}
  end

  # ---- (de)serialization ----------------------------------------------------

  def save!(path, map) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Nx.serialize(map))
  end

  def load!(path), do: path |> File.read!() |> Nx.deserialize()

  # Concatenate per-replay extracts, adding a replay_index column.
  def concat(extracts) do
    keys = [:phi, :a_master, :a_samples, :match, :rtg, :decision, :frame]

    base =
      Map.new(keys, fn k ->
        {k, extracts |> Enum.map(&Map.fetch!(&1, k)) |> Nx.concatenate(axis: 0)}
      end)

    ri =
      extracts
      |> Enum.with_index()
      |> Enum.flat_map(fn {e, i} -> List.duplicate(i, Nx.axis_size(e.phi, 0)) end)
      |> Nx.tensor(type: :s64, backend: Nx.BinaryBackend)

    Map.put(base, :replay_index, ri)
  end

  def to_controller_state(%ControllerState{} = cs), do: cs
end
