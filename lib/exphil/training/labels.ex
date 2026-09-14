defmodule ExPhil.Training.Labels do
  @moduledoc """
  The ONE way a training list gets its delayed labels (INVARIANTS.md item 14,
  2026-09-12).

  Two kinds of label exist, and they have different futures:

    * `:recorded` — the controller is what a real player did next (fixture,
      human replays). Its future IS the recording, so a reaction-delay-k
      label is the recorded controller k frames later (`Data.shift_actions/2`).

    * `{:expert, module}` — the controller is `module.label/4` for the
      state (relabeled rollouts, openers, mined snippets, every redress of
      those). Its future is NOT the recording: the recording is the
      student's, and where the student broke the loop the label k frames
      later is a recovery input — the same state with two futures. The
      delayed label must come from the expert itself: `module.label_ahead/4`
      (phase-indexed on the canonical loop). Where the expert cannot
      project (off the loop) the frame is DROPPED at k > 0 — the held
      "current commitment" projection was measured wrong against the
      teacher's executed futures (18/21 off-loop mismatches at shift 4,
      RECOVERY_LABEL_CONFIRMATION 2026-09-13). Delayed recovery supervision
      comes from recorded teacher futures instead (`RecordedFrames`).

  Frames carry `:label_source`; untagged frames are `:recorded`.
  `Data.shift_actions/2` refuses expert-tagged lists, so the old path (shift
  a relabel along the recorded future — RESULTS 09-12 §9) cannot be written.
  """

  alias ExPhil.Training.Data

  @type source :: :recorded | {:expert, module()}

  @doc "Tag every frame of a list with its label source."
  @spec tag([map()], source()) :: [map()]
  def tag(frames, source) do
    validate_source!(source)
    Enum.map(frames, &Map.put(&1, :label_source, source))
  end

  @doc """
  Validate the source of every frame and return their common source.

  Untagged frames retain the legacy recorded interpretation. Mixed sources
  are rejected, including partially untagged expert lists. `require_tagged:
  true` rejects missing tags at ingestion boundaries that require provenance.
  An empty list is recorded. Complete tag loss cannot be detected in legacy mode.
  """
  @spec source([map()], keyword()) :: source()
  def source(frames, opts \\ []) do
    frames
    |> Enum.with_index()
    |> Enum.reduce(nil, fn {frame, index}, common ->
      source = Map.get(frame, :label_source)

      if source == nil and Keyword.get(opts, :require_tagged, false),
        do: raise(ArgumentError, "Missing label_source at frame index #{index}")

      source = if source == nil, do: :recorded, else: source
      validate_source!(source)

      if common != nil and source != common,
        do:
          raise(
            ArgumentError,
            "Mixed label sources at frame index #{index}: #{inspect(common)} and #{inspect(source)}; split sources before delaying"
          )

      source
    end)
    |> case do
      nil -> :recorded
      common -> common
    end
  end

  defp validate_source!(:recorded), do: :ok

  defp validate_source!({:expert, module})
       when is_atom(module) and module not in [nil, true, false],
       do: :ok

  defp validate_source!(source),
    do: raise(ArgumentError, "Invalid label_source: #{inspect(source)}")

  @doc """
  Frames relabeled for reaction delay `k`.

  Options for expert-tagged lists:
    * `:expert` — the expert struct passed to `label_ahead/4` / `label/4` (required)
    * `:player_port` — the subject's port in `game_state.players` (default 1)
    * `:off_loop` — `:drop` (default: omit frames the expert cannot project
      k ahead, i.e. `label_ahead/4` returns `:skip` or the state is off the
      loop) or `:hold` (LEGACY, measured wrong: label the frame with the
      expert's current commitment `label/4`; kept only for equal-budget A/B
      comparisons against the old g26 recipe)
    * `:require_tagged` — reject missing provenance, including at delay zero
    * `:require_projection` — reject experts without `label_ahead/4` even
      when `off_loop: :hold` asks for their current commitment. An expert
      without `label_ahead/4` always raises under `:drop`: it would drop
      every frame, and that must be a decision, not a default.
  """
  @spec at_delay([map()], non_neg_integer(), keyword()) :: [map()]
  def at_delay(frames, k, opts \\ [])

  def at_delay(frames, k, opts) when is_integer(k) and k >= 0 do
    source = source(frames, opts)
    off_loop = Keyword.get(opts, :off_loop, :drop)
    unless off_loop in [:hold, :drop], do: raise(ArgumentError, "off_loop must be :hold or :drop")
    if k == 0, do: frames, else: delayed(frames, source, k, off_loop, opts)
  end

  def at_delay(_frames, k, _opts),
    do: raise(ArgumentError, "Reaction delay must be a nonnegative integer, got #{inspect(k)}")

  defp delayed(frames, source, k, off_loop, opts) do
    case source do
      :recorded ->
        Data.shift_actions(frames, k)

      {:expert, mod} ->
        expert =
          Keyword.get(opts, :expert) ||
            raise ArgumentError,
                  "Labels.at_delay/3: expert-labeled frames (#{inspect(mod)}) need the expert struct (`expert:`)"

        port = Keyword.get(opts, :player_port, 1)
        Code.ensure_loaded!(mod)
        ahead? = function_exported?(mod, :label_ahead, 4)

        if not ahead? and (Keyword.get(opts, :require_projection, false) or off_loop == :drop),
          do:
            raise(
              ArgumentError,
              "#{inspect(mod)} cannot project delayed labels: label_ahead/4 is required " <>
                "(pass `off_loop: :hold` to use its current commitment, knowing it is unverified)"
            )

        Enum.flat_map(frames, fn f ->
          player = f.game_state.players[port]
          prev = Map.get(f, :prev_controller)

          # The expert's own projection (phase-indexed on the loop; `:skip`
          # where it cannot project, e.g. off the loop).
          projected =
            cond do
              player == nil -> :skip
              ahead? -> mod.label_ahead(expert, player, k, prev)
              true -> :skip
            end

          # Where the expert abstains, the only legacy fallback is its held
          # current commitment — opt-in, because it is measured wrong for
          # recoveries. `on_loop?/2` is not consulted here: the projection
          # itself is the authority on what it can label.
          result =
            case {projected, off_loop} do
              {:skip, :hold} when player != nil -> mod.label(expert, player, prev)
              _ -> projected
            end

          case result do
            {:ok, controller} -> [%{f | controller: controller}]
            :skip -> []
            other -> raise ArgumentError, "Invalid expert label result: #{inspect(other)}"
          end
        end)
    end
  end
end
