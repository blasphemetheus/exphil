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
  """

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
