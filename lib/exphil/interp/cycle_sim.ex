defmodule ExPhil.Interp.CycleSim do
  @moduledoc """
  Full-cycle offline multishine simulator (task #5, 2026-08-02) — the
  BasinRollout idea extended from the crouch basin to the whole shine
  cycle, without Dolphin.

  Instead of hand-coding Melee frame data, the game dynamics come from
  the fixture itself: a transition table
  `{action, af, prev_b, prev_x, b, x} -> {action', af'}` extracted from
  consecutive fixture frames (the teacher's trajectory covers the happy
  path exactly). Rolling out: embed the current synthetic frame, ask the
  policy, decode buttons, step the table. The simulator is EXACT along
  states the fixture visits and HONEST about leaving them — the first
  off-graph `(state, buttons)` pair is recorded as the break, with the
  state family as its phase (the offline analog of
  analyze_break_phases.exs).

  Fidelity caveats (v1, accepted): positions/velocities are frozen from
  the template frame (BasinRollout precedent — 93.9% offline/live parity
  on basin dynamics); the dummy is a standing Fox eating shines, which
  matches the drill/eval distribution. Validation gate: the champion
  seed must chain and the metronome seed must not (ms_open_z vs
  ms_open_zz).

  Frame-pairing note (measured 2026-08-02, see CycleMargins.events/2):
  training frames pair state_t with the controller whose effect state_t
  already shows, i.e. state_t = step(state_{t-1}, ctrl_t) — the table is
  keyed accordingly.
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Eval.ShineChain
  alias ExPhil.Interp.BasinRollout

  @window_size 16

  defmodule Table do
    @moduledoc false
    defstruct [:transitions, :grounded, :conflicts, :states, :deltas]
  end

  @doc """
  Extract the transition table from a fixture path or a training-frame
  list. Conflicting observations for one key resolve by frequency.
  Returns a `%Table{}` with `:transitions`, per-action `:grounded`
  majority, `:conflicts` count, and `:states` (distinct {action, af}).
  """
  def transition_table(fixture) when is_binary(fixture) do
    {:ok, replay} = Peppi.parse(fixture)

    replay
    |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
    |> Enum.reject(&(&1.game_state.frame < 0))
    |> transition_table()
  end

  def transition_table(frames) when is_list(frames) do
    pairs =
      frames
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [p, f] -> f.game_state.frame == p.game_state.frame + 1 end)

    # Key on button EDGES, not held pairs: Melee registers edges in the
    # shine cycle (press-B = shine, press-X = JC; holds/releases are
    # no-ops). Keying on raw held pairs made every button history the
    # teacher didn't exhibit off-graph — the champion broke at t=3
    # releasing a held B the fixture happened to keep held (v1 lesson).
    observed =
      Enum.reduce(pairs, %{}, fn [p, f], acc ->
        pp = p.game_state.players[1]
        fp = f.game_state.players[1]
        b_edge = f.controller.button_b and not p.controller.button_b
        x_edge = f.controller.button_x and not p.controller.button_x

        key = {pp.action, pp.action_frame, b_edge, x_edge}

        Map.update(acc, key, %{{fp.action, fp.action_frame} => 1}, fn dist ->
          Map.update(dist, {fp.action, fp.action_frame}, 1, &(&1 + 1))
        end)
      end)

    conflicts = Enum.count(observed, fn {_k, dist} -> map_size(dist) > 1 end)

    transitions =
      Map.new(observed, fn {k, dist} ->
        {next, _n} = Enum.max_by(dist, fn {_next, n} -> n end)
        {k, next}
      end)

    grounded =
      frames
      |> Enum.group_by(& &1.game_state.players[1].action)
      |> Map.new(fn {action, fs} ->
        on = Enum.count(fs, & &1.game_state.players[1].on_ground)
        {action, on * 2 >= length(fs)}
      end)

    states =
      frames
      |> Enum.map(&{&1.game_state.players[1].action, &1.game_state.players[1].action_frame})
      |> MapSet.new()

    # Per-{action, af} mean ABSOLUTE y + speeds (v3): in this drill both
    # are near-deterministic functions of the state, so statistical
    # reconstruction beats delta integration (drift-free) and gives the
    # policy physically-plausible speed inputs (a jump with speed_y == 0
    # never happens in training data; v1's frozen fields left z waiting
    # 35 airborne frames for a height that never came).
    deltas =
      frames
      |> Enum.group_by(fn f ->
        p = f.game_state.players[1]
        {p.action, p.action_frame}
      end)
      |> Map.new(fn {key, fs} ->
        n = length(fs)

        sums =
          Enum.reduce(fs, %{y: 0.0, sax: 0.0, sgx: 0.0, sy: 0.0}, fn f, acc ->
            p = f.game_state.players[1]

            %{
              y: acc.y + p.y,
              sax: acc.sax + (p.speed_air_x_self || 0.0),
              sgx: acc.sgx + (p.speed_ground_x_self || 0.0),
              sy: acc.sy + (p.speed_y_self || 0.0)
            }
          end)

        {key, %{y: sums.y / n, sax: sums.sax / n, sgx: sums.sgx / n, sy: sums.sy / n}}
      end)

    %Table{
      transitions: transitions,
      grounded: grounded,
      conflicts: conflicts,
      states: states,
      deltas: deltas
    }
  end

  @doc """
  Roll the policy's closed loop through the fixture graph.

  `entry` is `{window_tensor, template_frame}` (any BasinRollout entry
  builder); the rollout starts from the template frame's {action, af}.

  Returns a map:
    * `:frames` — steps simulated
    * `:actions` — the simulated action-id stream (feed to ShineChain)
    * `:chains` — ShineChain.chains/1 of that stream
    * `:break` — nil (survived `:max_frames`) or
      `%{at:, action:, af:, family:, buttons: {b, x}}` for the first
      off-graph decision

  Options: `:max_frames` (default 600); `:trace` (default false) — when
  true the result gains `:trace`, a per-step list of
  `%{t:, action:, af:, b_logit:, x_logit:, emb:}` (emb = the frame
  embedding fed that step, BinaryBackend) for offline diffing against
  live-replay embeddings at matched states (the gate diagnostic).

  `:decode_lag` (default 2): frames between decoding and application.
  The MEASURED live pipeline latency is N+2 at --frame-delay N (the
  intrinsic +2, qtrace-verified 2026-07-31): z decides its aerial B at
  jumpsquat af0 (cyclesim_cycle_dump: B logit +1.6 exactly there) and
  the press lands two frames later ON the first airborne frame — the
  shine. Lag 0/1 ask the model early and the press dies in jumpsquat.

  `:press_threshold` / `:release_threshold` (defaults 0.45 / 0.30):
  the agent path's probability hysteresis, replicated — B/X decode is
  sigmoid(logit) vs press-threshold when released, release-threshold
  when held (sticky). The raw per-frame logit>0 decode manufactured
  presses/releases the live pipeline never emits (the v4 gate lesson).
  """
  def rollout(predict_fn, params, {entry_window, template}, %Table{} = table, opts \\ []) do
    max_frames = Keyword.get(opts, :max_frames, 600)
    trace? = Keyword.get(opts, :trace, false)
    decode_lag = Keyword.get(opts, :decode_lag, 2)
    press_t = Keyword.get(opts, :press_threshold, 0.45)
    release_t = Keyword.get(opts, :release_threshold, 0.30)
    player0 = template.game_state.players[1]
    base_frame = template.game_state.frame

    # Queue/delay-id layout (2026-08-04): `:config` = the policy's export
    # config, `:delay_id` = its deploy id; omitted -> classic 288-dim
    # layout. `ring` = last K DECODED controllers (newest first, live-agent
    # semantics — same stream the prev-action slot always tracked).
    embed_config = ExPhil.Interp.Activations.embed_config_for(Keyword.get(opts, :config))
    delay_id = Keyword.get(opts, :delay_id) || 0
    embed_queue_depth = Map.get(embed_config, :queue_depth) || 1

    # decoded_prev = last DECODED controller (the prev-action channel and
    # the hysteresis held-state track the model's OWN emitted stream —
    # live-agent semantics); applied_prev = last APPLIED (edge
    # reference); queue = decoded-but-not-yet-applied, length decode_lag.
    queue0 = List.duplicate(template.controller, max(decode_lag, 0))
    ring0 = List.duplicate(template.controller, embed_queue_depth)

    init =
      {entry_window, ring0, template.controller, queue0,
       {player0.action, player0.action_frame}, [], 0, []}

    result =
      Enum.reduce_while(1..max_frames, init, fn t,
                                                {win, ring, applied_prev, queue,
                                                 {action, af}, actions, soft, trace} ->
        decoded_prev = hd(ring)
        grounded = Map.get(table.grounded, action, true)

        # Statistical state reconstruction: y + speeds from the per-state
        # table (x stays at the template's — lateral drift is not part of
        # the stand-dummy drill).
        recon = Map.get(table.deltas, {action, af}, %{})

        player = %{
          player0
          | action: action,
            action_frame: af,
            on_ground: grounded,
            y: Map.get(recon, :y, player0.y),
            speed_air_x_self: Map.get(recon, :sax, 0.0),
            speed_ground_x_self: Map.get(recon, :sgx, 0.0),
            speed_y_self: Map.get(recon, :sy, 0.0)
        }

        gs = %{
          template.game_state
          | frame: base_frame + t,
            players: Map.put(template.game_state.players, 1, player)
        }

        emb =
          ExPhil.Embeddings.Game.embed(gs, decoded_prev, 1,
            config: embed_config,
            queue_controllers: ring,
            delay_id: delay_id
          )
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Nx.reshape({1, :auto})

        win = Nx.concatenate([Nx.slice_along_axis(win, 1, @window_size - 1, axis: 0), emb], axis: 0)
        out = predict_fn.(params, Nx.new_axis(win, 0))
        ctrl = decode_hysteresis(out, decoded_prev, press_t, release_t)

        trace =
          if trace? do
            buttons = out |> elem(0) |> Nx.squeeze() |> Nx.to_flat_list()

            [
              %{t: t, action: action, af: af, b_logit: Enum.at(buttons, 1),
                x_logit: Enum.at(buttons, 2), emb: Nx.squeeze(emb)}
              | trace
            ]
          else
            trace
          end

        # Application queue: this step's transition applies the decode
        # from decode_lag steps ago; the fresh decode joins the tail.
        {applied, next_queue} =
          case queue do
            [] -> {ctrl, []}
            [head | rest] -> {head, rest ++ [ctrl]}
          end

        b_edge = applied.button_b and not applied_prev.button_b
        x_edge = applied.button_x and not applied_prev.button_x

        # Lookup ladder (all non-exact rungs count as :soft):
        #   1. exact {action, af, edges}
        #   2. af-tolerant (±4, nearest first): an aerial B is legal at
        #      ANY airborne af, but the graph only has the afs its
        #      sources visited — af-exact keying silently swallowed z's
        #      aerial shines as edge-drops (v3 lesson).
        #   3. drop an unobserved edge (game showed no reaction to it in
        #      this state anywhere, e.g. B is EATEN in jumpsquat).
        # Only a state where even the no-edge key misses is a hard break.
        af_candidates = [af | Enum.flat_map(1..4, &[af - &1, af + &1])]

        lookup =
          Enum.find_value(
            Enum.map(af_candidates, &{&1, b_edge, x_edge}) ++
              [{af, b_edge, false}, {af, false, x_edge}, {af, false, false}],
            fn {af_c, b, x} ->
              case Map.get(table.transitions, {action, af_c, b, x}) do
                nil -> nil
                next -> {next, {af_c, b, x} != {af, b_edge, x_edge}}
              end
            end
          )

        case lookup do
          nil ->
            brk = %{
              at: t,
              action: action,
              af: af,
              family: ShineChain.family(action),
              buttons: {applied.button_b, applied.button_x}
            }

            {:halt, {:break, brk, Enum.reverse(actions), soft, Enum.reverse(trace)}}

          {{next_action, next_af}, soft?} ->
            {:cont,
             {win, [ctrl | Enum.take(ring, embed_queue_depth - 1)], applied, next_queue,
              {next_action, next_af}, [next_action | actions],
              soft + ((soft? && 1) || 0), trace}}
        end
      end)

    case result do
      {:break, brk, actions, soft, trace} ->
        %{
          frames: brk.at,
          actions: actions,
          chains: ShineChain.chains(actions),
          break: brk,
          soft: soft,
          trace: trace
        }

      {_win, _decoded, _applied, _queue, _state, actions, soft, trace} ->
        actions = Enum.reverse(actions)

        %{
          frames: max_frames,
          actions: actions,
          chains: ShineChain.chains(actions),
          break: nil,
          soft: soft,
          trace: Enum.reverse(trace)
        }
    end
  end

  @doc """
  Parse a replay to positive-frame training frames (ports 1/2 — the
  convention every entry builder uses). Returns [] on parse failure so
  graph-replay globs tolerate SD-flake files.
  """
  def load_frames(path) do
    case Peppi.parse(path) do
      {:ok, replay} ->
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      _ ->
        []
    end
  end

  @doc """
  Entry at the fixture's own cycle start (first `#{@window_size}` frames)
  plus a transition table. `:graph_replays` (list of paths) merges
  additional OBSERVED dynamics into the graph — one teacher trajectory
  has the happy path but not the game's tolerance (the champion JCs at
  reflector af3 where the teacher JCs at af2; both are legal Melee).
  Frame-number continuity checks drop the seams between sources.
  """
  def from_fixture(fixture, opts \\ []) do
    frames = load_frames(fixture)

    graph_frames =
      opts
      |> Keyword.get(:graph_replays, [])
      |> Enum.flat_map(&load_frames/1)

    {BasinRollout.entry_from_frames(
       Enum.take(frames, @window_size),
       Keyword.take(opts, [:config, :delay_id])
     ), transition_table(frames ++ graph_frames)}
  end

  # Agent-path button decode: sigmoid(logit) vs press threshold when the
  # button is released, release threshold when held (sticky hysteresis,
  # agent.ex semantics). Held-state reference is the previous DECODED
  # controller. Only B/X matter to the transition table.
  defp decode_hysteresis(out, prev, press_t, release_t) do
    buttons = out |> elem(0) |> Nx.squeeze() |> Nx.to_flat_list()

    decide = fn logit, held ->
      p = 1.0 / (1.0 + :math.exp(-logit))
      if held, do: p >= release_t, else: p >= press_t
    end

    %{
      prev
      | button_b: decide.(Enum.at(buttons, 1), prev.button_b),
        button_x: decide.(Enum.at(buttons, 2), prev.button_x)
    }
  end
end
