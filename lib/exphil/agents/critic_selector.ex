defmodule ExPhil.Agents.CriticSelector do
  @moduledoc """
  Live critic-selector decode (EVAL_DIRECTIONS task 2, wired 2026-09-02).

  Loads the D2 bilinear critic (`scripts/critic_train.exs` artifact) and
  picks among K coherent AR samples per decision by

      S(s, a) = a . (W' phi_std(s)) + v . a

  where `phi(s) = [trunk features | raw scalars]` standardized with the
  critic's stored train-time mean/std, and `a` is the 13-dim continuous
  controller encoding (`Embeddings.Controller.embed_continuous/1`).

  The agent wires this through `Sampling.sample_autoregressive_from_features`'
  `:select_n`/`:select_fn` hook — the same tiled K-sample machinery
  mode-of-N uses, with the vote replaced by the critic argmax.

  `raw_features/3` is a VERBATIM copy of the training-side implementation
  in `scripts/lib/critic_features.exs` — the standardization was fit on
  that exact layout; any drift silently corrupts scores. Change BOTH or
  neither. (That file's header planned promotion to lib/; this is the
  live half of it.)
  """

  import Nx.Defn

  require Logger

  # competitive stages -> compact index (same set as stage_mode
  # :one_hot_compact): FoD 2, PS 3, YS 8, DL 28, BF 31, FD 32, other 6
  @stage_index %{2 => 0, 3 => 1, 8 => 2, 28 => 3, 31 => 4, 32 => 5}
  @num_chars 33
  @raw_size 22 + 7 + @num_chars

  @doc """
  Load a critic.nx artifact (term_to_binary map from critic_train.exs).
  Returns `%{mean, std, w, v, phi_size}` with BinaryBackend tensors.
  Trusted-file load, same posture as checkpoint loading.
  """
  def load!(path) do
    m = path |> File.read!() |> :erlang.binary_to_term()

    unless m[:kind] == :d2_critic_v1 do
      raise ArgumentError, "#{path} is not a d2_critic_v1 artifact (kind=#{inspect(m[:kind])})"
    end

    # Transfer to the default (EXLA) backend once at load — per-frame
    # BinaryBackend->device transfers would tax the 60 Hz loop.
    %{
      mean: Nx.backend_transfer(m.mean),
      std: Nx.backend_transfer(m.std),
      w: Nx.backend_transfer(m.selector.w),
      v: Nx.backend_transfer(m.selector.v),
      phi_size: m.phi_size
    }
  end

  @doc """
  Pick the best of K sampled actions for the current state.

  * `features` — trunk output `{1, hidden}` (the same tensor the AR head
    samples from)
  * `raw` — `raw_features/3` list for the current game state
  * sample tensors — the `{k, 8}` buttons / `{k}` component tensors from
    `sample_autoregressive_from_features`' tiled n>1 branch (live batch=1
    folded into the tile axis)

  Returns the row index of the argmax-scored candidate.
  """
  def pick(critic, features, raw, {buttons, mx, my, cx, cy, sh}, _k, embed_opts) do
    d = Nx.axis_size(features, 1) + length(raw)

    if d != critic.phi_size do
      raise ArgumentError,
            "critic phi_size mismatch: live #{d} vs trained #{critic.phi_size} — " <>
              "wrong checkpoint/critic pairing (the critic is trunk-specific)"
    end

    ab = Keyword.get(embed_opts, :axis_buckets, 16) * 1.0
    sb = Keyword.get(embed_opts, :shoulder_buckets, 4) * 1.0

    pick_impl(
      features,
      Nx.tensor(raw, type: :f32),
      critic.mean,
      critic.std,
      critic.w,
      critic.v,
      Nx.as_type(buttons, :f32),
      mx,
      my,
      cx,
      cy,
      sh,
      Nx.tensor(ab, type: :f32),
      Nx.tensor(sb, type: :f32)
    )
    |> Nx.to_number()
  end

  # One fused XLA program per decision — the eager per-candidate
  # to_controller_state/embed loop blew the 60 Hz budget (59.7% stale,
  # 0902_critic_sanity). Candidate encoding replicates
  # to_controller_state + Embeddings.Controller.embed_continuous EXACTLY:
  # buttons 0/1, sticks (bucket/buckets - 0.5) * 2, shoulder bucket/buckets
  # (pinned by the parity test on the training side).
  defn pick_impl(features, raw, mean, std, w, v, buttons, mx, my, cx, cy, sh, ab, sb) do
    phi = Nx.concatenate([Nx.reshape(features, {:auto}), raw])
    phi_std = (phi - Nx.reshape(mean, {:auto})) / Nx.reshape(std, {:auto})

    cands =
      Nx.concatenate(
        [
          buttons,
          Nx.new_axis((mx / ab - 0.5) * 2.0, 1),
          Nx.new_axis((my / ab - 0.5) * 2.0, 1),
          Nx.new_axis((cx / ab - 0.5) * 2.0, 1),
          Nx.new_axis((cy / ab - 0.5) * 2.0, 1),
          Nx.new_axis(sh / sb, 1)
        ],
        axis: 1
      )

    proj = Nx.dot(phi_std, w)
    scores = Nx.dot(cands, proj) + Nx.dot(cands, v)
    Nx.argmax(scores)
  end

  @doc """
  Compile the pick program at agent warmup (dummy shapes) so the first
  live decision doesn't pay the XLA compile.
  """
  def warmup(critic, k) do
    hidden = critic.phi_size - @raw_size

    pick(
      critic,
      Nx.broadcast(0.0, {1, hidden}),
      List.duplicate(0.0, @raw_size),
      {Nx.broadcast(0, {k, 8}), Nx.broadcast(0, {k}), Nx.broadcast(0, {k}),
       Nx.broadcast(0, {k}), Nx.broadcast(0, {k}), Nx.broadcast(0, {k})},
      k,
      []
    )

    :ok
  end

  @doc """
  Raw state scalars the trunk discards — VERBATIM from
  scripts/lib/critic_features.exs (see moduledoc). 22 scalars + 7 stage
  one-hot + 33 opponent-character one-hot.
  """
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
end
