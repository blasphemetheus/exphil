defmodule ExPhil.Data.LabelConvention do
  @moduledoc """
  The ONE place that knows how a training label relates to the state it is
  paired with, and how that maps onto the live `--frame-delay` flag.
  INVARIANTS.md item 1 (structural form), GOTCHA #113.

  ## The format fact

  Slippi records each frame's controller input on the frame whose
  (post-update) state it PRODUCED. The raw pair (state[t], controller[t])
  is therefore label-leaked: the state already contains the effect of the
  input it is supposed to predict.

  ## Conventions

    * `:causal` — every training frame pairs state[t] with the input
      ISSUED from it, i.e. raw controller[t+1]. `frame_delay` /
      `action_delay` are ADDITIONAL reaction delay on top: reaction delay
      k pairs state[t] with raw controller[t+1+k]. Reaction delay 0 is the
      causal pairing; the leaked pairing is not representable. This is
      what `ExPhil.Data.Peppi.to_training_frames/2` emits since
      2026-09-09; new checkpoints are stamped `label_convention: :causal`.

    * `:producing` — the legacy convention (every checkpoint without a
      stamp): the configured delay d paired state[t] with raw
      controller[t+d]. Reaction delay = d - 1; d = 0 was the leak
      (reaction delay -1).

  ## Live mapping

  On this rig the live `--frame-delay N` flag (Slippi's own input-delay
  setting, passed straight to libmelee) plays a policy trained at legacy
  delay N — every deploy card is keyed that way (ms_g19 trained {2,3} ->
  local d3; v16e trained legacy 1 -> `--frame-delay 1`). In reaction
  terms: live N <-> reaction delay N - 1. The flag is deliberately NOT
  rebased (its numbers are Dolphin's, and the records/knob docs are keyed
  on them); this module translates instead.
  """

  @type convention :: :causal | :producing

  @doc "The convention the parser emits today."
  @spec current() :: :causal
  def current, do: :causal

  @doc "The convention a checkpoint/training config was built under (unstamped = legacy)."
  @spec of(map() | keyword() | nil) :: convention()
  def of(nil), do: :producing

  def of(config) do
    case fetch(config, :label_convention) do
      :causal -> :causal
      "causal" -> :causal
      _ -> :producing
    end
  end

  @doc """
  Reaction delay of a config: frames between the observed state and the
  frame the paired input was issued on. 0 = causal pairing, -1 = the
  legacy leak. The streaming path's `frame_delay` and the standard path's
  `action_delay` are one concept. New configs store `label_delay` directly
  in reaction terms. The larger legacy alias is retained only when reading
  old artifacts without that canonical field.
  """
  @spec reaction_delay(map() | keyword() | nil) :: integer()
  def reaction_delay(config) do
    case fetch(config, :label_delay) do
      nil ->
        delay = max(int(fetch(config, :frame_delay)), int(fetch(config, :action_delay)))
        to_reaction(delay, of(config))
      delay ->
        int(delay)
    end
  end

  @doc "A delay number in the given convention, expressed as reaction delay."
  @spec to_reaction(integer(), convention()) :: integer()
  def to_reaction(d, :causal), do: d
  def to_reaction(d, :producing), do: d - 1

  @doc "The checkpoint's `train_delays` (its own numbering) as reaction delays."
  @spec train_reaction_delays(map() | keyword() | nil) :: [integer()] | nil
  def train_reaction_delays(config) do
    case fetch(config, :train_delays) do
      list when is_list(list) and list != [] ->
        conv = of(config)
        list |> Enum.map(&to_reaction(int(&1), conv)) |> Enum.sort() |> Enum.uniq()

      _ ->
        nil
    end
  end

  @doc "True when the config's labels are the legacy leaked pairing (GOTCHA #113)."
  @spec leaky?(map() | keyword() | nil) :: boolean()
  def leaky?(config), do: reaction_delay(config) < 0

  # INVARIANTS item 12 (2026-09-12): the live mapping is the HARNESS TABLE
  # (ExPhil.Eval.HarnessRung), not a constant here. Both Dolphin runners on
  # this rig apply a decision --frame-delay + 2 frames after the observed
  # state, so live N plays reaction delay N + 1 — the earlier "N - 1" law
  # was the drill line's undeclared pipeline offset (2) cancelling the
  # runner's pipeline (2). These two stay as the async-runner convenience.

  @doc "The nearest live `--frame-delay` for a trained reaction delay (0 when the exact rung is below the runner's floor)."
  @spec live_frame_delay(integer()) :: non_neg_integer()
  def live_frame_delay(reaction_delay) do
    case ExPhil.Eval.HarnessRung.knob(:async_runner, ExPhil.Eval.HarnessRung.latency(:training, max(reaction_delay, 0))) do
      {:ok, k} -> k
      {:error, _} -> 0
    end
  end

  @doc "The reaction delay a live `--frame-delay N` run actually plays at (async or sync runner)."
  @spec live_reaction_delay(non_neg_integer()) :: integer()
  def live_reaction_delay(live_frame_delay), do: ExPhil.Eval.HarnessRung.reaction_delay(:async_runner, live_frame_delay)

  @doc """
  The delay-id to feed a delay-conditioned checkpoint when deployed at live
  `--frame-delay N`, in the CHECKPOINT'S OWN numbering (the id is an
  embedding index chosen at training time: legacy checkpoints used their
  producing-convention delay, causal ones use reaction delay).
  """
  @spec delay_id(non_neg_integer(), map() | keyword() | nil) :: non_neg_integer()
  def delay_id(live_frame_delay, config),
    do: ExPhil.Eval.HarnessRung.delay_id(:async_runner, live_frame_delay, config)

  # -- internals ------------------------------------------------------------

  defp fetch(nil, _k), do: nil
  defp fetch(config, k) when is_list(config), do: Keyword.get(config, k)

  defp fetch(config, k) when is_map(config) do
    case Map.fetch(config, k) do
      {:ok, v} -> v
      :error -> Map.get(config, Atom.to_string(k))
    end
  end

  defp int(nil), do: 0
  defp int(v) when is_integer(v), do: v
  defp int(v) when is_binary(v), do: String.to_integer(v)
  defp int(_), do: 0
end
