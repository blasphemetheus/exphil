defmodule ExPhil.Inspect do
  @moduledoc """
  Moment inspection: everything inference exposes about one frame, as
  one plain map (coach roadmap #2; the rewind scrubber's backend).

  Unifies what the probe scripts each rebuild by hand: situation labels
  (`ExPhil.Situations`), the policy's per-head distributions, entropy,
  argmax-vs-recorded agreement, and window-consistent counterfactuals
  (the `probe_edge_attribution` pattern).

  ## Usage

      {:ok, session} = Inspect.open(
        "checkpoints/fox_il_v2_edgeB_..._policy.bin",
        "eval_runs/0810_edgeB_pool/r1/Game_x.slp",
        player_port: 1
      )

      moment = Inspect.moment(session, 4200)
      moment.situations           #=> [:neutral, :onstage_corner, ...]
      moment.policy.buttons.b     #=> 0.03  (press probability)
      moment.policy.main_x.probs  #=> [...] (softmax over stick buckets)

      # Counterfactual: shove the fox 25 units toward the edge across the
      # whole window (self-consistent history — the absorber lesson)
      cf = Inspect.counterfactual(session, 4200, fn p -> %{p | x: p.x + 25.0} end)
      cf.policy.main_x.argmax

  The session parses + embeds ONCE (the expensive part); `moment/3` is
  one forward pass. Everything in the returned map is JSON-encodable
  (`Jason.encode!/1` works directly) for the scrubber.

  Not here yet (roadmap): trunk activations + probe-zoo outputs (wire
  `Activations.load_trunk` in behind an opt), OOD-vs-corpus flags
  (needs the per-feature corpus stats job).
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Interp.Activations
  alias ExPhil.Situations

  @buttons ~w(a b x y z l r d_up)a
  @axis_heads [:main_x, :main_y, :c_x, :c_y, :shoulder]

  defstruct [:loaded, :frames, :states, :embedded, :situations, :window, :port, :total]

  @type t :: %__MODULE__{}

  # ==========================================================================
  # Session construction
  # ==========================================================================

  @doc """
  Open a session from a policy .bin and a .slp replay.

  Options: `:player_port` (default 1), `:delay_id` (REQUIRED for
  with_delay_id policies — Activations.embed_frames raises otherwise).
  """
  @spec open(Path.t(), Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def open(policy_path, replay_path, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)
    opp = if port == 1, do: 2, else: 1

    with {:ok, replay} <- Peppi.parse(replay_path, player_port: port) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
        |> Enum.reject(&(&1.game_state.frame < 0))

      loaded = Activations.load_heads(policy_path)
      {:ok, from_frames(loaded, frames, opts)}
    end
  end

  @doc """
  Core constructor from an `Activations.load_heads/1`-shaped map and a
  training-frames list. Split out so tests (and live ring buffers) can
  inject both.
  """
  @spec from_frames(map(), [map()], keyword()) :: t()
  def from_frames(loaded, frames, opts \\ []) do
    port = Keyword.get(opts, :player_port, 1)

    ds = Activations.embed_frames(frames, loaded.config, Keyword.take(opts, [:delay_id]))
    embedded = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)

    states = Enum.map(frames, & &1.game_state)

    %__MODULE__{
      loaded: loaded,
      frames: List.to_tuple(frames),
      states: states,
      embedded: embedded,
      situations: Situations.label_states(states, port, as: :set),
      window: loaded.window,
      port: port,
      total: length(frames)
    }
  end

  # ==========================================================================
  # Moment
  # ==========================================================================

  @doc """
  Inspect frame `t` (0-based index into the parsed frames). Returns a
  plain map; `policy` is nil for the first `window - 1` frames (not
  enough history for the temporal trunk).
  """
  @spec moment(t(), non_neg_integer(), keyword()) :: map()
  def moment(%__MODULE__{} = s, t, _opts \\ []) when is_integer(t) do
    if t < 0 or t >= s.total, do: raise(ArgumentError, "frame #{t} outside 0..#{s.total - 1}")

    frame = elem(s.frames, t)

    %{
      index: t,
      frame: frame.game_state.frame,
      stage: frame.game_state.stage,
      situations: s.situations |> Enum.at(t) |> Enum.sort(),
      players: players_summary(frame.game_state, s.port),
      recorded: controller_summary(frame.controller),
      policy: policy_at(s, t, s.embedded)
    }
  end

  @doc """
  Re-run the policy at `t` with `patch_fn` applied to OUR player on
  EVERY frame of the trunk window (self-consistent history — patching
  one frame teaches nothing, the GRU smooths it away). Returns the same
  shape as `moment/2` plus `:baseline_policy` for direct comparison.
  """
  @spec counterfactual(t(), non_neg_integer(), (map() -> map()), keyword()) :: map()
  def counterfactual(%__MODULE__{} = s, t, patch_fn, opts \\ []) do
    lo = max(t - s.window + 1, 0)

    patched_frames =
      for i <- lo..t do
        f = elem(s.frames, i)
        p = f.game_state.players[s.port]

        if p == nil do
          f
        else
          players = Map.put(f.game_state.players, s.port, patch_fn.(p))
          %{f | game_state: %{f.game_state | players: players}}
        end
      end

    ds = Activations.embed_frames(patched_frames, s.loaded.config, Keyword.take(opts, [:delay_id]))
    emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)

    base = moment(s, t)

    Map.merge(base, %{
      policy: policy_at(s, t - lo, emb),
      baseline_policy: base.policy
    })
  end

  # ==========================================================================
  # Export (the standalone viewer's food)
  # ==========================================================================

  @doc """
  Export the whole session to one JSON file for the standalone viewer
  (`priv/viewer/rewind_viewer.html`): per-frame player states, situation
  bitmasks + the label registry, per-frame policy summaries (batched
  forward passes over every frame), and the stage geometry the labels
  were computed from.

  Policy rows are compact: button sigmoids (8), main-stick softmaxes
  (buckets), argmaxes for the c-stick/shoulder heads. Frames before the
  trunk window carry `nil`.
  """
  @spec export_session(t(), Path.t(), keyword()) :: :ok
  def export_session(%__MODULE__{} = s, path, opts \\ []) do
    chunk = Keyword.get(opts, :chunk, 256)
    round3 = fn v -> Float.round(v * 1.0, 3) end

    policies =
      (s.window - 1)..(s.total - 1)//1
      |> Enum.chunk_every(chunk)
      |> Enum.flat_map(fn ts ->
        wins =
          Enum.map(ts, fn t ->
            Nx.slice_along_axis(s.embedded, t - s.window + 1, s.window, axis: 0)
          end)

        {b, mx, my, cx, cy, sh} = s.loaded.predict_fn.(s.loaded.params, Nx.stack(wins))

        b_rows = b |> Nx.sigmoid() |> Nx.to_list()
        mx_rows = mx |> batch_softmax() |> Nx.to_list()
        my_rows = my |> batch_softmax() |> Nx.to_list()
        cx_am = cx |> Nx.argmax(axis: 1) |> Nx.to_flat_list()
        cy_am = cy |> Nx.argmax(axis: 1) |> Nx.to_flat_list()
        sh_am = sh |> Nx.argmax(axis: 1) |> Nx.to_flat_list()

        Enum.zip([b_rows, mx_rows, my_rows, cx_am, cy_am, sh_am])
        |> Enum.map(fn {bb, mxx, myy, cxa, cya, sha} ->
          %{
            b: Enum.map(bb, round3),
            mx: Enum.map(mxx, round3),
            my: Enum.map(myy, round3),
            cs: [cxa, cya, sha]
          }
        end)
      end)

    policy_at = fn t ->
      if t >= s.window - 1, do: Enum.at(policies, t - (s.window - 1))
    end

    frames =
      for t <- 0..(s.total - 1) do
        f = elem(s.frames, t)
        gs = f.game_state
        opp_port = if s.port == 1, do: 2, else: 1

        %{
          f: gs.frame,
          own: compact_player(gs.players[s.port]),
          opp: compact_player(gs.players[opp_port]),
          mask: s.situations |> Enum.at(t) |> Situations.to_mask(),
          rec: compact_controller(f.controller),
          pol: policy_at.(t)
        }
      end

    stage = elem(s.frames, 0).game_state.stage
    geo = Situations.geometry(stage)

    payload = %{
      version: 1,
      window: s.window,
      port: s.port,
      total: s.total,
      stage: stage,
      labels: Enum.map(Situations.labels(), &to_string/1),
      buttons: Enum.map(@buttons, &to_string/1),
      geometry: %{
        edge: geo.edge,
        blast: geo.blast && Tuple.to_list(geo.blast),
        platforms: Enum.map(geo.platforms, &Tuple.to_list/1)
      },
      frames: frames
    }

    File.write!(path, Jason.encode!(payload))
    :ok
  end

  defp batch_softmax(logits) do
    m = Nx.reduce_max(logits, axes: [1], keep_axes: true)
    e = Nx.exp(Nx.subtract(logits, m))
    Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
  end

  defp compact_player(nil), do: nil

  defp compact_player(p) do
    [
      Float.round((p.x || 0.0) * 1.0, 2),
      Float.round((p.y || 0.0) * 1.0, 2),
      p.facing || 1,
      trunc(p.action || 0),
      Float.round((p.percent || 0.0) * 1.0, 1),
      p.stock || 0,
      if(p.on_ground, do: 1, else: 0)
    ]
  end

  defp compact_controller(nil), do: nil

  defp compact_controller(c) do
    pressed =
      [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r]
      |> Enum.with_index()
      |> Enum.filter(fn {b, _i} -> Map.get(c, b, false) end)
      |> Enum.map(&elem(&1, 1))

    ms = c.main_stick || %{x: 0.5, y: 0.5}
    [Float.round(ms.x * 1.0, 3), Float.round(ms.y * 1.0, 3), pressed]
  end

  # ==========================================================================
  # Internals
  # ==========================================================================

  defp policy_at(s, t, embedded) do
    if t < s.window - 1 do
      nil
    else
      win =
        embedded
        |> Nx.slice_along_axis(t - s.window + 1, s.window, axis: 0)
        |> Nx.new_axis(0)

      {b, mx, my, cx, cy, sh} = s.loaded.predict_fn.(s.loaded.params, win)

      button_probs =
        b |> Nx.sigmoid() |> Nx.squeeze(axes: [0]) |> Nx.to_flat_list()

      buttons = @buttons |> Enum.zip(button_probs) |> Map.new()

      axes =
        [mx, my, cx, cy, sh]
        |> Enum.map(fn logits ->
          probs = logits |> Nx.squeeze(axes: [0]) |> softmax() |> Nx.to_flat_list()

          %{
            probs: probs,
            argmax: probs |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1),
            entropy: entropy(probs)
          }
        end)

      Map.new(Enum.zip(@axis_heads, axes))
      |> Map.put(:buttons, buttons)
      |> Map.put(:buttons_entropy, bernoulli_entropy(button_probs))
      |> Map.put(:pressed, for({btn, p} <- buttons, p > 0.5, do: btn) |> Enum.sort())
    end
  end

  defp softmax(logits) do
    m = Nx.reduce_max(logits)
    e = Nx.exp(Nx.subtract(logits, m))
    Nx.divide(e, Nx.sum(e))
  end

  defp entropy(probs) do
    -Enum.reduce(probs, 0.0, fn p, acc ->
      if p > 1.0e-10, do: acc + p * :math.log(p), else: acc
    end)
  end

  defp bernoulli_entropy(probs) do
    n = length(probs)

    sum =
      Enum.reduce(probs, 0.0, fn p, acc ->
        p = min(max(p, 1.0e-10), 1.0 - 1.0e-10)
        acc - (p * :math.log(p) + (1.0 - p) * :math.log(1.0 - p))
      end)

    sum / n
  end

  defp players_summary(gs, own_port) do
    opp_port = if own_port == 1, do: 2, else: 1

    %{
      own: player_summary(gs.players[own_port]),
      opponent: player_summary(gs.players[opp_port])
    }
  end

  defp player_summary(nil), do: nil

  defp player_summary(p) do
    %{
      x: p.x,
      y: p.y,
      percent: p.percent,
      stock: p.stock,
      action: trunc(p.action || 0),
      action_frame: p.action_frame,
      facing: p.facing,
      on_ground: p.on_ground,
      jumps_left: p.jumps_left,
      shield: p.shield_strength,
      hitstun_left: p.hitstun_frames_left,
      invulnerable: p.invulnerable
    }
  end

  defp controller_summary(nil), do: nil

  defp controller_summary(c) do
    %{
      buttons: for(b <- [:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r], Map.get(c, b, false), do: b),
      main_stick: c.main_stick,
      c_stick: Map.get(c, :c_stick),
      shoulder: Map.get(c, :l_shoulder) || Map.get(c, :shoulder)
    }
  end
end
