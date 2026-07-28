defmodule ExPhil.Data.RecoverySynth do
  @moduledoc """
  Synthesises the off-trajectory frames a drill policy actually falls into,
  labelled by the scripted expert — DAgger's benefit without the rollout loop.

  ## Why

  A policy trained on one fixture trajectory only knows the states on that
  trajectory. One frame of timing slip puts it somewhere it has never seen, its
  output there is undefined, and the error compounds. Measured on the
  multishine policy (docs/planning/EXPOSURE_BIAS.md): the fixture contains
  grounded reflector 361 only at `action_frame` **1..2**, but live the policy
  sits at **af 3..28** and never escapes — 97.8% of frames in one action.

  The expert already knows what to do there. `MultishineExpert`'s recovery
  rules cover exactly these off-table states ("grounded reflector past the
  jump-cancel window -> tap jump"); they simply never appear in TRAINING data,
  because the fixture never visits them.

  This module manufactures that data. It takes each reflector segment in the
  fixture and extends it — af 3, 4, ... N — asking the expert to label every
  extended frame. No Dolphin, no rollouts, seconds to run.

  ## Sequences, not frames

  The policy is temporal (GRU over a window), so isolated frames are useless:
  the trainer slices windows out of a contiguous frame list. Each synthetic
  run is therefore emitted as a CONTIGUOUS block — real fixture frames leading
  in, then the extended tail — so the windows the trainer builds contain a
  plausible history followed by the unseen continuation.

  ## What this cannot do

  It only covers states reachable by extending a segment the fixture already
  visits. States that require genuinely different play (getting hit, ledge,
  tech) are not synthesisable this way and still need rollouts or recordings.

  ## Manufactured states — the crouch absorber (`build_crouch/2`)

  `build/2` extends what the fixture visits; the crouch absorber
  (EXPOSURE_BIAS 6-replication) is a state the teacher NEVER visits: vs an
  idle opponent the policy botches a shine, shorthops, lands holding down and
  crouches forever at high confidence. `build_crouch/2` MANUFACTURES the trap:
  it grafts a synthetic Squat -> SquatWait tail onto real post-shine frames,
  labelled by the expert's grounded fallback (start a shine, edge-alternated).
  Tails run past the training window so some windows are ENTIRELY crouch —
  the deep-basin state itself gets coverage, not just the entrance.

  Unlike `extend/4` (which labels each frame with `prev = nil`), the crouch
  tail THREADS each frame's label into the next frame's `prev`, so labels
  alternate press/release exactly as the expert would behave live — a
  constant "press B" tail would re-teach the held-button pathology the
  alternation rules exist to prevent.
  """

  alias ExPhil.Agents.LedgeExpert
  alias ExPhil.Agents.MultishineExpert
  alias ExPhil.Constants

  @doc """
  Build synthetic recovery sequences from fixture frames.

  Options:
    * `:port` — player port (default 1)
    * `:max_af` — how far to extend a held action (default 30, covering the
      af 3..28 range observed live)
    * `:lead_in` — real frames prepended to each block so the window has a
      plausible history (default 16, the trainer's window)
    * `:actions` — action ids to extend (default: the grounded and airborne
      reflector ranges, which is where the policy actually gets stuck)
    * `:ratio` — cap synthetic output at this multiple of the input frame
      count (default 1.0). Without a cap the fixture's ~560 reflector segments
      generate ~25k frames against 1.7k real ones — 94% synthetic, which both
      swamps the core loop the policy must keep and pushes an epoch past
      several minutes. Sampled evenly across segments so the cap costs
      coverage of segments, not coverage of the af range.

  Returns frames in the same shape as `Peppi.to_training_frames/2`, ready to
  concatenate before `Data.from_frames/1`.
  """
  @spec build([map()], keyword()) :: [map()]
  def build(frames, opts \\ []) do
    port = Keyword.get(opts, :port, 1)
    max_af = Keyword.get(opts, :max_af, 30)
    lead_in = Keyword.get(opts, :lead_in, 16)

    actions =
      Keyword.get(opts, :actions, [
        Constants.reflector_ground(),
        Constants.reflector_air()
      ])
      |> Enum.flat_map(&Enum.to_list/1)
      |> MapSet.new()

    expert = Keyword.get(opts, :expert) || MultishineExpert.from_fixture()

    # ONE block per reflector SEGMENT, not per frame. Emitting a lead-in for
    # every frame of a segment duplicates the same history dozens of times:
    # the naive version produced 49,678 synthetic frames from 1,679 real ones
    # (97% synthetic), which would both swamp the core loop in training and
    # push an epoch past 8 minutes. Segments give the same coverage at a
    # fraction of the size.
    ratio = Keyword.get(opts, :ratio, 1.0)
    budget = trunc(length(frames) * ratio)

    segments =
      frames
      |> Enum.with_index()
      |> segment_ends(port, actions)

    segments
    |> take_evenly(budget_segments(segments, budget, max_af, lead_in))
    |> Enum.flat_map(fn {frame, i} ->
      lead = Enum.slice(frames, max(i - lead_in + 1, 0), min(i + 1, lead_in))
      lead ++ extend(frame, port, max_af, expert)
    end)
  end

  @doc """
  Manufacture GAME-OPENING coverage: the spawn-platform descent as lead-in,
  followed by the same expert-labelled `Squat -> SquatWait` tail as
  `build_crouch/2`.

  Why (INIT_FORENSICS_OPTIONS.md, 2026-07-28): the crouch recipe's remaining
  live failure mode is the opening — seeds m/g/p/r/s absorb at frame ~104
  via the entry route (spawn-platform fall -> crouch) and never shine once.
  `build_crouch/2` grafts tails onto POST-SHINE lead-ins, so basin windows
  with entry-animation history are never covered; offline, dead seeds pass
  post-shine-entry probes and die only from their own openings.

  Lead-in sources: the opening segment (game frames `0..@opening_max_frame`)
  of the given fixture frames, plus any `:extra_sources` frame lists — pass
  the real openings harvested from dead seeds' replays for exact-history
  coverage. Multiple grafts per opening (at the landing and shortly after)
  cover slightly different descent phases.

  Options: `:port`, `:max_af`, `:lead_in`, `:extra_sources` (flat frame list,
  may contain several replays — split on frame-number resets), `:graft_frames`
  (game-frame graft points, default `[90, 105, 120]`), `:expert`.
  """
  @opening_max_frame 200

  @spec build_opening([map()], keyword()) :: [map()]
  def build_opening(frames, opts \\ []) do
    port = Keyword.get(opts, :port, 1)
    max_af = Keyword.get(opts, :max_af, 40)
    lead_in = Keyword.get(opts, :lead_in, 16)
    graft_frames = Keyword.get(opts, :graft_frames, [90, 105, 120])
    expert = Keyword.get(opts, :expert) || MultishineExpert.from_fixture()
    extra = Keyword.get(opts, :extra_sources, [])

    ([frames] ++ split_on_frame_reset(extra))
    |> Enum.flat_map(fn source ->
      opening = Enum.filter(source, &(&1.game_state.frame <= @opening_max_frame))

      Enum.flat_map(graft_frames, fn gf ->
        case Enum.find_index(opening, &(&1.game_state.frame >= gf)) do
          nil ->
            []

          i ->
            frame = Enum.at(opening, i)
            lead = Enum.slice(opening, max(i - lead_in + 1, 0), min(i + 1, lead_in))

            # RELABEL the lead with the expert, threading prev through it
            # into the tail. The recorded controllers of an extra_sources
            # lead are a DEAD POLICY'S own outputs (hold-B, crouch) —
            # keeping them as labels teaches the absorber's behavior in
            # exactly the states this synthesis exists to fix (measured
            # 2026-07-28: farm 5 seeds trained on unrelabeled dead-seed
            # openings absorbed at frame 104 via the identical route).
            {lead, prev} =
              Enum.map_reduce(lead, nil, fn f, prev ->
                case MultishineExpert.label(expert, f.game_state.players[port], prev) do
                  {:ok, c} -> {%{f | controller: c}, c}
                  :skip -> {f, f.controller}
                end
              end)

            lead ++ crouch_tail(frame, port, max_af, expert, prev)
        end
      end)
    end)
  end

  # A flat frame list may concatenate several replays; game frame numbers
  # reset at each boundary.
  defp split_on_frame_reset([]), do: []

  defp split_on_frame_reset(frames) do
    frames
    |> Enum.chunk_while(
      [],
      fn f, acc ->
        case acc do
          [] -> {:cont, [f]}
          [prev | _] when f.game_state.frame < prev.game_state.frame -> {:cont, Enum.reverse(acc), [f]}
          _ -> {:cont, [f | acc]}
        end
      end,
      fn acc -> {:cont, Enum.reverse(acc), []} end
    )
  end

  @doc """
  Manufacture crouch-absorber coverage: real post-shine lead-ins followed by
  a synthetic `Squat -> SquatWait` tail, expert-labelled with edge alternation.

  Options:
    * `:port` — player port (default 1)
    * `:max_af` — SquatWait frames per tail (default 40: past the 16-frame
      window, so fully-crouched windows exist; live absorption runs minutes,
      but af coverage past the window adds nothing the GRU can distinguish)
    * `:lead_in` — real frames prepended per block (default 16)
    * `:ratio` — cap output at this multiple of input frames (default 0.5 —
      the trap needs coverage, not dominance over the core loop)

  Source points are the ends of grounded-reflector segments (post-shine — the
  closest fixture analog of the observed entry route: botched cycle -> land ->
  crouch), sampled evenly like `build/2`.
  """
  @spec build_crouch([map()], keyword()) :: [map()]
  def build_crouch(frames, opts \\ []) do
    port = Keyword.get(opts, :port, 1)
    max_af = Keyword.get(opts, :max_af, 40)
    lead_in = Keyword.get(opts, :lead_in, 16)
    ratio = Keyword.get(opts, :ratio, 0.5)

    expert = Keyword.get(opts, :expert) || MultishineExpert.from_fixture()
    actions = Constants.reflector_ground() |> Enum.to_list() |> MapSet.new()

    budget = trunc(length(frames) * ratio)
    # Each block: lead-in + 2 Squat frames + max_af SquatWait frames.
    per_block = lead_in + 2 + max_af

    segments =
      frames
      |> Enum.with_index()
      |> segment_ends(port, actions)

    segments
    |> take_evenly(max(min(length(segments), div(budget, per_block)), 1))
    |> Enum.flat_map(fn {frame, i} ->
      lead = Enum.slice(frames, max(i - lead_in + 1, 0), min(i + 1, lead_in))
      prev = List.last(lead) |> then(&(&1 && &1.controller))
      lead ++ crouch_tail(frame, port, max_af, expert, prev)
    end)
  end

  @doc """
  Manufacture ledge coverage: real lead-ins followed by a synthetic
  `CliffCatch -> CliffWait` tail at the stage edge, labelled by
  `ExPhil.Agents.LedgeExpert` (strategy-parameterized; default `:getup`).

  Options:
    * `:port` — player port (default 1)
    * `:max_af` — CliffWait frames per tail (default 30)
    * `:lead_in` — real frames prepended per block (default 16)
    * `:ratio` — cap output at this multiple of input frames (default 0.3 —
      the ledge is a time-sink valley, not the main absorber)
    * `:strategy` — LedgeExpert strategy (default `:getup`)
    * `:edge_x` — stage edge |x| (default `Constants.fd_edge_x/0`; per-stage
      generalization pending)

  Blocks alternate left/right ledge (facing toward the stage). Player
  percents come from the sampled real frames, so both sides of the 100%
  slow/quick threshold get covered for free wherever the fixture's percents
  vary.
  """
  @spec build_ledge([map()], keyword()) :: [map()]
  def build_ledge(frames, opts \\ []) do
    port = Keyword.get(opts, :port, 1)
    max_af = Keyword.get(opts, :max_af, 30)
    lead_in = Keyword.get(opts, :lead_in, 16)
    ratio = Keyword.get(opts, :ratio, 0.3)
    edge_x = Keyword.get(opts, :edge_x, Constants.fd_edge_x())
    expert = Keyword.get(opts, :expert) || LedgeExpert.new(strategy: Keyword.get(opts, :strategy, :getup))

    budget = trunc(length(frames) * ratio)
    per_block = lead_in + 7 + max_af

    # Any real frame works as a lead-in source; sample evenly for phase
    # coverage like build/2.
    sources =
      frames
      |> Enum.with_index()
      |> Enum.filter(fn {f, _} -> f.game_state.players[port] != nil end)

    sources
    |> take_evenly(max(min(length(sources), div(budget, per_block)), 1))
    |> Enum.with_index()
    |> Enum.flat_map(fn {{frame, i}, block_idx} ->
      side = if rem(block_idx, 2) == 0, do: 1, else: -1
      lead = Enum.slice(frames, max(i - lead_in + 1, 0), min(i + 1, lead_in))
      prev = List.last(lead) |> then(&(&1 && &1.controller))
      lead ++ ledge_tail(frame, port, side, edge_x, max_af, expert, prev)
    end)
  end

  # CliffCatch af 1..7 (no control), then CliffWait af 1..max_af, at the
  # ledge coordinates, facing the stage. Labels thread prev for alternation.
  defp ledge_tail(frame, port, side, edge_x, max_af, expert, prev0) do
    player = frame.game_state.players[port]

    states =
      Enum.map(1..7, &{Constants.cliff_catch(), &1}) ++
        Enum.map(1..max_af, &{Constants.cliff_wait(), &1})

    {tail, _prev} =
      Enum.reduce(states, {[], prev0}, fn {action, af}, {acc, prev} ->
        shifted = %{
          player
          | action: action,
            action_frame: af,
            on_ground: false,
            x: side * (edge_x + 2.0),
            y: -12.0,
            facing: -side
        }

        case LedgeExpert.label(expert, shifted, prev) do
          {:ok, controller} ->
            players = Map.put(frame.game_state.players, port, shifted)

            out = %{
              frame
              | game_state: %{frame.game_state | players: players},
                controller: controller
            }

            {[out | acc], controller}

          :skip ->
            {acc, prev}
        end
      end)

    Enum.reverse(tail)
  end

  # Squat af 1..2, then SquatWait af 1..max_af. Threads each label into the
  # next frame's `prev` so the expert's press/release alternation survives
  # into the data (see moduledoc).
  #
  # Tail frames carry CONSECUTIVE frame numbers continuing from the source
  # frame: precompute_frame_embeddings only threads the prev-action channel
  # across consecutive frame numbers, and with the source number duplicated
  # (pre-2026-07-27 behavior) every tail frame embedded prev as ABSENT — the
  # release-when-prev-B signal was invisible in exactly the states this
  # synthesis exists to cover, leaving escape to init luck (measured 6/12;
  # INIT_FORENSICS_OPTIONS.md).
  defp crouch_tail(frame, port, max_af, expert, prev0) do
    player = frame.game_state.players[port]

    states =
      Enum.map(1..2, &{Constants.squat(), &1}) ++
        Enum.map(1..max_af, &{Constants.squat_wait(), &1})

    {tail, _prev, _n} =
      Enum.reduce(states, {[], prev0, 1}, fn {action, af}, {acc, prev, n} ->
        shifted = %{player | action: action, action_frame: af, on_ground: true}

        case MultishineExpert.label(expert, shifted, prev) do
          {:ok, controller} ->
            players = Map.put(frame.game_state.players, port, shifted)

            out = %{
              frame
              | game_state: %{
                  frame.game_state
                  | players: players,
                    frame: frame.game_state.frame + n
                },
                controller: controller
            }

            {[out | acc], controller, n + 1}

          :skip ->
            {acc, prev, n}
        end
      end)

    Enum.reverse(tail)
  end

  # Roughly how many segments fit the budget: each contributes a lead-in plus
  # the extended tail.
  defp budget_segments(segments, budget, max_af, lead_in) do
    per_segment = max(lead_in + div(max_af, 2), 1)
    max(min(length(segments), div(budget, per_segment)), 1)
  end

  # Evenly spaced rather than the first N — segments early in a fixture are
  # all warm-up, and taking a contiguous prefix would bias the synthetic set
  # toward one phase of the loop.
  defp take_evenly(list, n) when n >= length(list), do: list

  defp take_evenly(list, n) do
    total = length(list)
    step = total / n

    0..(n - 1)
    |> Enum.map(&Enum.at(list, trunc(&1 * step)))
    |> Enum.reject(&is_nil/1)
  end

  # The LAST frame of each contiguous run of the target actions — that is the
  # point the policy would carry on past, so it is where the extension belongs.
  defp segment_ends(indexed, port, actions) do
    indexed
    |> Enum.chunk_every(2, 1, [nil])
    |> Enum.filter(fn
      [{f, _i}, nil] -> held_action?(f, port, actions)
      [{f, _i}, {nxt, _}] -> held_action?(f, port, actions) and not same_action?(f, nxt, port)
    end)
    |> Enum.map(fn [pair | _] -> pair end)
  end

  defp same_action?(a, b, port) do
    case {a.game_state.players[port], b.game_state.players[port]} do
      {%{action: x}, %{action: y}} -> trunc(x) == trunc(y)
      _ -> false
    end
  end

  defp held_action?(frame, port, actions) do
    case frame.game_state.players[port] do
      nil -> false
      p -> MapSet.member?(actions, trunc(p.action))
    end
  end

  # Walk action_frame forward through the range the policy actually occupies
  # live, asking the expert for the correct input at each step. The expert
  # falls through its table (which has no entry past the fixture's af) into the
  # recovery rules — which is exactly the behaviour we want in training.
  defp extend(frame, port, max_af, expert) do
    player = frame.game_state.players[port]
    start_af = trunc(player.action_frame) + 1

    if start_af > max_af do
      []
    else
      Enum.flat_map(start_af..max_af, fn af ->
        shifted = %{player | action_frame: af}

        case MultishineExpert.label(expert, shifted) do
          {:ok, controller} ->
            players = Map.put(frame.game_state.players, port, shifted)

            [
              %{
                frame
                | game_state: %{frame.game_state | players: players},
                  controller: controller
              }
            ]

          :skip ->
            []
        end
      end)
    end
  end
end
