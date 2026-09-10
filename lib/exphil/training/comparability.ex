defmodule ExPhil.Training.Comparability do
  @moduledoc """
  INVARIANTS.md item 11: a checkpoint's COMPARABILITY KEY.

  Two checkpoints' losses (val or train) mean the same thing only when
  they were trained against the same target definition. Concretely they
  must share:

    * `label_delay` — the REACTION delay (`ExPhil.Data.LabelConvention`):
      0 = causal pairing, -1 = the legacy leak. GOTCHA #113: leaked
      targets are EASIER, so a leaked val loss is optimistic by
      construction (v2 read 1.83 against v16e's 2.42 for a strictly
      better policy). Counted in reaction terms so a legacy checkpoint at
      producing-delay 1 and a causal one at 0 compare as the SAME target;
    * `embed_canary` — the input layout/values fingerprint;
    * `loss_recipe` — the per-frame/per-button weighting knobs
      (pos_weight, focal, smoothing, oversample, entropy, neutral,
      transition, offstage): each reshapes the objective;
    * `train_delays` — the delay-id set the policy was conditioned on.

  `key/1` computes the tuple from a training config (atom- or
  string-keyed, i.e. the in-memory opts or the `_config.json`);
  `Config.build_config_json/2` stamps it into every checkpoint;
  `ExPhil.Training.Registry.best/1` refuses to rank across differing
  keys unless told `allow_incomparable: true`.
  """

  @type key :: %{
          label_delay: integer(),
          embed_canary: String.t() | nil,
          loss_recipe: String.t(),
          train_delays: [integer()] | nil
        }

  alias ExPhil.Data.LabelConvention

  @loss_knobs [
    :label_smoothing,
    :button_pos_weight,
    :focal_loss,
    :focal_gamma,
    :button_weight,
    :action_oversample,
    :entropy_weight,
    :neutral_weight,
    :transition_weight,
    :offstage_weight,
    :stick_edge_weight,
    :head
  ]

  @doc "Compute the comparability key from a training config."
  @spec key(map() | keyword()) :: key()
  def key(config) do
    get = fn k -> fetch(config, k) end

    %{
      label_delay: LabelConvention.reaction_delay(config),
      embed_canary: hash_of(get.(:embed_canary)),
      loss_recipe: hash_of(Enum.map(@loss_knobs, fn k -> {k, normalize(get.(k))} end)),
      train_delays: LabelConvention.train_reaction_delays(config)
    }
  end

  @doc "True when losses from the two configs/keys may be compared."
  @spec comparable?(map() | keyword(), map() | keyword()) :: boolean()
  def comparable?(a, b), do: as_key(a) == as_key(b)

  @doc """
  Check a list of `{label, config_or_key}`; `:ok` when all share one key,
  else `{:error, groups}` where groups maps each distinct key to its labels.
  """
  @spec check([{term(), map() | keyword()}]) :: :ok | {:error, %{key() => [term()]}}
  def check(entries) do
    groups =
      entries
      |> Enum.group_by(fn {_label, c} -> as_key(c) end, fn {label, _} -> label end)

    if map_size(groups) <= 1, do: :ok, else: {:error, groups}
  end

  @doc "Human-readable one-line description of why two keys differ."
  @spec explain(key(), key()) :: String.t()
  def explain(a, b) do
    a
    |> Map.keys()
    |> Enum.filter(&(Map.get(a, &1) != Map.get(b, &1)))
    |> Enum.map(fn f -> "#{f}: #{inspect(Map.get(a, f))} vs #{inspect(Map.get(b, f))}" end)
    |> Enum.join("; ")
  end

  # -- internals ------------------------------------------------------------

  defp as_key(%{label_delay: _, loss_recipe: _} = k), do: k
  defp as_key(config), do: key(config)

  defp fetch(config, k) when is_list(config), do: Keyword.get(config, k)

  defp fetch(config, k) when is_map(config) do
    case Map.fetch(config, k) do
      {:ok, v} -> v
      :error -> Map.get(config, Atom.to_string(k))
    end
  end

  # JSON round-trips atoms/booleans to strings and tensors to lists; fold
  # them to one canonical form so in-memory and on-disk configs agree.
  defp normalize(nil), do: nil
  defp normalize("true"), do: true
  defp normalize("false"), do: false
  defp normalize("auto"), do: :auto
  defp normalize(v) when is_atom(v), do: v
  defp normalize(v) when is_binary(v), do: String.to_atom(v)
  defp normalize(v) when is_list(v), do: Enum.map(v, &normalize/1)
  defp normalize(v) when is_float(v), do: Float.round(v, 6)
  defp normalize(%Nx.Tensor{} = t), do: t |> Nx.to_flat_list() |> normalize()
  defp normalize(v), do: v

  defp hash_of(nil), do: nil

  defp hash_of(term) do
    :crypto.hash(:sha256, :erlang.term_to_binary(term)) |> Base.encode16(case: :lower) |> binary_part(0, 16)
  end
end
