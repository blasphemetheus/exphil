defmodule ExPhil.Training.Config.Inference do
  @moduledoc """
  Smart flag inference for training configuration.

  Instead of warning about conflicting flags, this module automatically
  applies sensible defaults and returns what was inferred. This reduces
  user friction and makes the CLI more ergonomic.

  ## Inferences Applied

  1. **Temporal mode for recurrent backbones**: LSTM, GRU, Mamba, Jamba require temporal
  2. **Validation split for early stopping**: Early stopping needs validation data
  3. **Disable cache with augmentation**: Augmented data invalidates cache
  4. **Disable cache in streaming mode**: Streaming doesn't use cache
  5. **Ignore MLP-only flags for other backbones**: residual, layer_norm
  6. **Auto-enable prefetch for streaming**: Streaming benefits from prefetch

  ## Examples

      iex> opts = [backbone: :lstm, temporal: false]
      iex> {new_opts, inferences} = Inference.infer_smart_defaults(opts)
      iex> new_opts[:temporal]
      true
      iex> hd(inferences)
      "Auto-enabled --temporal (required for LSTM backbone)"

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Config.Validator` - Validation rules
  """

  @recurrent_backbones [:lstm, :gru, :mamba, :jamba, :sliding_window]
  @mlp_backbones [:mlp, :sliding_window]

  @doc """
  Apply smart defaults based on flag combinations.

  Returns `{opts, inferences}` where `inferences` is a list of strings
  describing what was auto-applied.

  ## Parameters

  - `opts` - Keyword list of training options

  ## Returns

  - `{updated_opts, inferences}` - Updated options and list of inference messages

  ## Examples

      iex> opts = [backbone: :lstm, temporal: false]
      iex> {new_opts, inferences} = Inference.infer_smart_defaults(opts)
      iex> new_opts[:temporal]
      true

      iex> opts = [early_stopping: true, val_split: 0.0]
      iex> {new_opts, _} = Inference.infer_smart_defaults(opts)
      iex> new_opts[:val_split]
      0.1

  """
  @spec infer_smart_defaults(keyword()) :: {keyword(), list(String.t())}
  def infer_smart_defaults(opts) do
    {opts, []}
    |> infer_temporal_for_backbone()
    |> infer_val_split_for_early_stopping()
    |> infer_cache_disabled_for_augment()
    |> infer_cache_disabled_for_streaming()
    |> infer_ignore_mlp_flags_for_other_backbones()
    |> infer_prefetch_for_streaming()
    |> finalize_inferences()
  end

  # ============================================================================
  # Individual Inference Rules
  # ============================================================================

  # Recurrent backbones require temporal mode
  defp infer_temporal_for_backbone({opts, inferences}) do
    backbone = opts[:backbone] || :mlp
    temporal = opts[:temporal] || false

    if backbone in @recurrent_backbones and not temporal do
      backbone_name = backbone |> to_string() |> String.upcase()

      {
        Keyword.put(opts, :temporal, true),
        ["Auto-enabled --temporal (required for #{backbone_name} backbone)" | inferences]
      }
    else
      {opts, inferences}
    end
  end

  # Early stopping needs validation data
  defp infer_val_split_for_early_stopping({opts, inferences}) do
    early_stopping = opts[:early_stopping] || false
    val_split = opts[:val_split] || 0.0
    # Check if user explicitly set val_split to 0
    val_split_explicit_zero = opts[:val_split] == 0.0 and Keyword.has_key?(opts, :val_split)

    if early_stopping and val_split == 0.0 and not val_split_explicit_zero do
      {
        Keyword.put(opts, :val_split, 0.1),
        ["Auto-set --val-split 0.1 (recommended for --early-stopping)" | inferences]
      }
    else
      {opts, inferences}
    end
  end

  # Augmentation invalidates cache
  defp infer_cache_disabled_for_augment({opts, inferences}) do
    augment = opts[:augment] || false
    cache = opts[:cache_embeddings] || false
    cache_augmented = opts[:cache_augmented] || false

    if augment and cache and not cache_augmented do
      {
        Keyword.put(opts, :cache_embeddings, false),
        [
          "Auto-disabled --cache-embeddings (incompatible with --augment, use --cache-augmented instead)"
          | inferences
        ]
      }
    else
      {opts, inferences}
    end
  end

  # Streaming mode doesn't use cache
  defp infer_cache_disabled_for_streaming({opts, inferences}) do
    streaming = opts[:stream_chunk_size] != nil
    cache = opts[:cache_embeddings] || false

    if streaming and cache do
      {
        Keyword.put(opts, :cache_embeddings, false),
        ["Auto-disabled --cache-embeddings (not used in streaming mode)" | inferences]
      }
    else
      {opts, inferences}
    end
  end

  # residual and layer_norm only apply to MLP
  defp infer_ignore_mlp_flags_for_other_backbones({opts, inferences}) do
    backbone = opts[:backbone] || :mlp
    residual = opts[:residual] || false
    layer_norm = opts[:layer_norm] || false

    cond do
      backbone in @mlp_backbones ->
        {opts, inferences}

      residual and layer_norm ->
        {
          opts |> Keyword.put(:residual, false) |> Keyword.put(:layer_norm, false),
          ["Ignored --residual and --layer-norm (only apply to MLP backbone)" | inferences]
        }

      residual ->
        {
          Keyword.put(opts, :residual, false),
          ["Ignored --residual (only applies to MLP backbone)" | inferences]
        }

      layer_norm ->
        {
          Keyword.put(opts, :layer_norm, false),
          ["Ignored --layer-norm (only applies to MLP backbone)" | inferences]
        }

      true ->
        {opts, inferences}
    end
  end

  # Auto-enable prefetch in streaming mode (unless explicitly disabled)
  defp infer_prefetch_for_streaming({opts, inferences}) do
    streaming = opts[:stream_chunk_size] != nil
    no_prefetch = opts[:no_prefetch] || false

    if streaming and not no_prefetch and not (opts[:prefetch_enabled] || false) do
      {
        Keyword.put(opts, :prefetch_enabled, true),
        ["Auto-enabled prefetch (recommended for streaming mode)" | inferences]
      }
    else
      {opts, inferences}
    end
  end

  # Reverse inferences to show in order they were applied
  defp finalize_inferences({opts, inferences}) do
    {opts, Enum.reverse(inferences)}
  end
end
