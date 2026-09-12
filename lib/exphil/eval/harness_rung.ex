defmodule ExPhil.Eval.HarnessRung do
  @moduledoc """
  The ONE table of decision-to-application latency per harness, and the
  mapping from a harness knob to the delay-id a delay-conditioned checkpoint
  must be fed there. INVARIANTS.md item 12, GOTCHA #115.

  ## The physical quantity

  `latency` = frames from the OBSERVED state's frame to the frame the
  decision's input is applied on (the frame Slippi records it on). Training
  counts the same thing as *reaction delay* k: state[t] pairs with the input
  recorded on frame t+1+k (`ExPhil.Data.LabelConvention`), so

      latency = reaction_delay + 1

  Every harness adds its own pipeline on top of its knob (measured on THIS
  rig, nixos_slanka / RTX 5090 — the laptop harness ran one frame faster,
  see memory `project_5090_harness_delay_offset`):

  | harness | knob | latency | measured |
  |---|---|---|---|
  | `:training` | reaction delay k | k + 1 | LabelConvention (format fact) |
  | `:sync_runner` (`play_dolphin.exs`) | `--frame-delay` N | N + 2 | 09-12 pin on ep57 (eval_runs/0912_sync_rung): fd3/id2 chains 427/436, fd2/id2 chains 2, fd4/id2 chains 3/106. RETRACTS the 07-28 "sync d3 == async d2" note (laptop-era; this rig) |
  | `:async_runner` (`play_dolphin_async.exs`) | `--frame-delay` N | N + 2 | 07-31 live-queue forensics; rung law async d3 <-> drill id 2 |
  | `:scenario_suite` (`scenario_suite.exs`) | `--response-delay` N | N + 1 | 09-12 grid on ep57's own game (rd 2 <-> id 0, 6/6 == teacher) |

  ## Delay-id

  A drill checkpoint's delay-id d is a NOMINAL tag: the drill shifts labels
  by d + `--pipeline-offset` P, so id d was trained at reaction delay d + P.
  P was never stamped before 2026-09-12 (`delay_id_reaction_offset`); the
  live derivation (`id = --frame-delay - 1`) was right on both Dolphin
  runners only because their 2-frame pipeline cancelled the drill's
  undeclared +2 — and wrong by two on the scenario suite (latency 1).
  Unstamped delay-conditioned checkpoints are ASSUMED to carry the recipe's
  P = 2 (`delay_id_reaction_offset/1` says so); everything else has P = 0.

      id = reaction_delay(harness, knob) - P
      aligned knob = knob(harness, latency(:training, id + P))

  One frame FASTER than trained breaks a tight loop outright (suite rd1/id0,
  sync fd2/id2: chains 2); one frame SLOWER degrades it (sync fd4/id2: chains
  3/106 vs 427/436 aligned) — play the aligned knob exactly.
  """

  alias ExPhil.Data.LabelConvention

  @type harness :: :training | :sync_runner | :async_runner | :scenario_suite
  @harnesses [:training, :sync_runner, :async_runner, :scenario_suite]

  # Pipeline frames each harness adds on top of its knob (latency at knob 0).
  @pipeline %{training: 1, sync_runner: 2, async_runner: 2, scenario_suite: 1}

  # The drill's --pipeline-offset since 2026-07-31 (every ms_g* checkpoint).
  @assumed_drill_offset 2

  @doc "Every harness this table knows."
  @spec harnesses() :: [harness()]
  def harnesses, do: @harnesses

  @doc "Decision->application latency (frames) of a harness at its knob value."
  @spec latency(harness(), non_neg_integer()) :: pos_integer()
  def latency(harness, knob) when harness in @harnesses and is_integer(knob) and knob >= 0 do
    knob + @pipeline[harness]
  end

  @doc "The smallest latency a harness can play at (its knob at 0)."
  @spec min_latency(harness()) :: pos_integer()
  def min_latency(harness), do: latency(harness, 0)

  @doc """
  The knob value that plays a harness at `latency`, or
  `{:error, {:unreachable, min_latency}}` when the harness cannot be that fast.
  """
  @spec knob(harness(), pos_integer()) :: {:ok, non_neg_integer()} | {:error, {:unreachable, pos_integer()}}
  def knob(harness, latency) when harness in @harnesses and is_integer(latency) do
    min = min_latency(harness)
    if latency >= min, do: {:ok, latency - min}, else: {:error, {:unreachable, min}}
  end

  @doc "The reaction delay (training numbering) a harness plays at its knob."
  @spec reaction_delay(harness(), non_neg_integer()) :: non_neg_integer()
  def reaction_delay(harness, knob), do: latency(harness, knob) - 1

  @doc """
  The reaction-delay offset behind a checkpoint's delay-ids: `{offset, source}`
  where source is `:stamped` (`delay_id_reaction_offset` in the config),
  `:assumed_drill` (delay-conditioned but unstamped -> the recipe's 2), or
  `:none` (not delay-conditioned -> 0).
  """
  @spec delay_id_reaction_offset(map() | keyword() | nil) :: {non_neg_integer(), :stamped | :assumed_drill | :none}
  def delay_id_reaction_offset(config) do
    case fetch(config, :delay_id_reaction_offset) do
      v when is_integer(v) and v >= 0 -> {v, :stamped}
      v when is_binary(v) -> {String.to_integer(v), :stamped}
      _ -> if delay_conditioned?(config), do: {@assumed_drill_offset, :assumed_drill}, else: {0, :none}
    end
  end

  @doc """
  The delay-id (in the CHECKPOINT'S numbering) to feed a checkpoint played
  on `harness` at `knob`. Never negative.
  """
  @spec delay_id(harness(), non_neg_integer(), map() | keyword() | nil) :: non_neg_integer()
  def delay_id(harness, knob, config) do
    {offset, _} = delay_id_reaction_offset(config)
    nominal_reaction = reaction_delay(harness, knob) - offset

    case LabelConvention.of(config) do
      :causal -> max(nominal_reaction, 0)
      :producing -> max(nominal_reaction + 1, 0)
    end
  end

  @doc """
  The knob that plays `harness` exactly at the rung a checkpoint's delay-id
  was trained for; `{:error, {:unreachable, min_latency}}` when the harness
  cannot be that fast (then knob 0 is the nearest, slower rung).
  """
  @spec aligned_knob(harness(), non_neg_integer(), map() | keyword() | nil) ::
          {:ok, non_neg_integer()} | {:error, {:unreachable, pos_integer()}}
  def aligned_knob(harness, delay_id, config) do
    {offset, _} = delay_id_reaction_offset(config)
    knob(harness, latency(:training, trained_reaction(delay_id, config) + offset))
  end

  @doc """
  The knob that plays `harness` at a non-conditioned checkpoint's trained
  reaction delay (`LabelConvention.reaction_delay/1`), same return shape as
  `aligned_knob/3`.
  """
  @spec deploy_knob(harness(), map() | keyword() | nil) ::
          {:ok, non_neg_integer()} | {:error, {:unreachable, pos_integer()}}
  def deploy_knob(harness, config) do
    # A leaked legacy checkpoint (reaction -1, GOTCHA #113) has no valid rung;
    # treat it as reaction 0 (its nearest meaningful one).
    knob(harness, latency(:training, max(LabelConvention.reaction_delay(config), 0)))
  end

  @doc "One line for logs: what this harness+knob physically plays and which id it maps to."
  @spec describe(harness(), non_neg_integer(), map() | keyword() | nil) :: String.t()
  def describe(harness, knob, config) do
    {offset, src} = delay_id_reaction_offset(config)

    "#{harness} knob #{knob}: latency #{latency(harness, knob)} (reaction delay " <>
      "#{reaction_delay(harness, knob)}) -> delay-id #{delay_id(harness, knob, config)} " <>
      "(id reaction offset #{offset}, #{src})"
  end

  # -- internals ------------------------------------------------------------

  # The reaction delay a delay-id nominally names, in the checkpoint's numbering
  defp trained_reaction(delay_id, config), do: LabelConvention.to_reaction(delay_id, LabelConvention.of(config))

  defp delay_conditioned?(config) do
    with_id =
      fetch(config, :with_delay_id) ||
        (case fetch(config, :embed_config) do
           nil -> nil
           ec -> fetch(ec, :with_delay_id)
         end)

    delays = fetch(config, :train_delays)
    with_id == true or (is_list(delays) and length(delays) > 1)
  end

  defp fetch(nil, _k), do: nil
  defp fetch(config, k) when is_list(config), do: Keyword.get(config, k)

  defp fetch(config, k) when is_map(config) do
    case Map.fetch(config, k) do
      {:ok, v} -> v
      :error -> Map.get(config, Atom.to_string(k))
    end
  end
end
