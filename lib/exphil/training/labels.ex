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
      (phase-indexed on the canonical loop), or, for experts without one,
      the expert's current commitment held.

  Frames carry `:label_source`; untagged frames are `:recorded`.
  `Data.shift_actions/2` refuses expert-tagged lists, so the old path (shift
  a relabel along the recorded future — RESULTS 09-12 §9) cannot be written.
  """

  alias ExPhil.Training.Data

  @type source :: :recorded | {:expert, module()}

  @doc "Tag every frame of a list with its label source."
  @spec tag([map()], source()) :: [map()]
  def tag(frames, source) when source == :recorded or (is_tuple(source) and elem(source, 0) == :expert) do
    Enum.map(frames, &Map.put(&1, :label_source, source))
  end

  @doc "The label source of a list (`:recorded` when untagged or empty)."
  @spec source([map()]) :: source()
  def source([]), do: :recorded
  def source([first | _]), do: Map.get(first, :label_source) || :recorded

  @doc """
  Frames relabeled for reaction delay `k`.

  Options for expert-tagged lists:
    * `:expert` — the expert struct passed to `label_ahead/4` / `label/4` (required)
    * `:player_port` — the subject's port in `game_state.players` (default 1)
    * `:off_loop` — `:hold` (default: the expert's current commitment) or
      `:drop` (omit frames the expert cannot project k ahead)
  """
  @spec at_delay([map()], non_neg_integer(), keyword()) :: [map()]
  def at_delay(frames, k, opts \\ [])
  def at_delay(frames, 0, _opts), do: frames

  def at_delay(frames, k, opts) when is_integer(k) and k > 0 do
    case source(frames) do
      :recorded ->
        Data.shift_actions(frames, k)

      {:expert, mod} ->
        expert =
          Keyword.get(opts, :expert) ||
            raise ArgumentError,
                  "Labels.at_delay/3: expert-labeled frames (#{inspect(mod)}) need the expert struct (`expert:`)"

        port = Keyword.get(opts, :player_port, 1)
        off_loop = Keyword.get(opts, :off_loop, :hold)
        ahead? = function_exported?(mod, :label_ahead, 4)
        on_loop? = function_exported?(mod, :on_loop?, 2)

        Enum.flat_map(frames, fn f ->
          player = f.game_state.players[port]
          prev = Map.get(f, :prev_controller)

          cond do
            player == nil ->
              []

            off_loop == :drop and (not on_loop? or not mod.on_loop?(expert, player)) ->
              []

            true ->
              result = if ahead?, do: mod.label_ahead(expert, player, k, prev), else: mod.label(expert, player, prev)

              case result do
                {:ok, controller} -> [%{f | controller: controller}]
                _ -> []
              end
          end
        end)
    end
  end
end
