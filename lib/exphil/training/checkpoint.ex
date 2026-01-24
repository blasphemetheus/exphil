defmodule ExPhil.Training.Checkpoint do
  @moduledoc """
  Checkpoint loading and validation utilities.

  Provides safe checkpoint loading with embed size validation to catch
  incompatible models early with helpful error messages.

  ## Usage

      # Load with validation
      {:ok, checkpoint} = Checkpoint.load("model.axon", current_embed_size: 1204)

      # Load policy file
      {:ok, export} = Checkpoint.load_policy("model_policy.bin", current_embed_size: 1204)

      # Validate manually
      :ok = Checkpoint.validate_embed_size(checkpoint.config, current_embed_size: 1204)

  ## Validation Warnings

  When embed sizes don't match, you'll see:

      ⚠️  Embed size mismatch!
         Checkpoint: 1991 dims
         Current:    1204 dims

         This usually means the embedding config has changed since training.
         The model will fail during inference.

         Options:
         1. Train a new model with current config
         2. Use --embed-size 1991 to match checkpoint (if supported)
         3. Export checkpoint config and retrain

  """

  alias ExPhil.Training.Output

  @type checkpoint :: %{
          policy_params: map(),
          optimizer_state: map() | nil,
          config: map(),
          step: non_neg_integer(),
          metrics: map() | nil
        }

  @type policy_export :: %{
          params: map(),
          config: map()
        }

  @doc """
  Load a full checkpoint (.axon file) with optional validation.

  ## Options

    * `:current_embed_size` - Expected embed size (validates if provided)
    * `:warn_on_mismatch` - Print warning on mismatch (default: true)
    * `:error_on_mismatch` - Return error on mismatch (default: false)

  """
  @spec load(Path.t(), keyword()) :: {:ok, checkpoint()} | {:error, term()}
  def load(path, opts \\ []) do
    case File.read(path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)
        validate_and_return(checkpoint, :checkpoint, opts)

      {:error, reason} ->
        {:error, {:file_read, reason}}
    end
  end

  @doc """
  Load a policy export (.bin file) with optional validation.

  ## Options

  Same as `load/2`.
  """
  @spec load_policy(Path.t(), keyword()) :: {:ok, policy_export()} | {:error, term()}
  def load_policy(path, opts \\ []) do
    case File.read(path) do
      {:ok, binary} ->
        export = :erlang.binary_to_term(binary)
        validate_and_return(export, :policy, opts)

      {:error, reason} ->
        {:error, {:file_read, reason}}
    end
  end

  @doc """
  Validate embed size matches between checkpoint and current config.

  Returns `:ok` if sizes match or no validation requested,
  `{:warning, message}` if mismatch but not erroring,
  `{:error, message}` if mismatch and error_on_mismatch is true.
  """
  @spec validate_embed_size(map(), keyword()) :: :ok | {:warning, String.t()} | {:error, String.t()}
  def validate_embed_size(config, opts) do
    current = Keyword.get(opts, :current_embed_size)
    error_on_mismatch = Keyword.get(opts, :error_on_mismatch, false)

    if current do
      checkpoint_size = config[:embed_size]

      if checkpoint_size && checkpoint_size != current do
        message = embed_mismatch_message(checkpoint_size, current)

        if error_on_mismatch do
          {:error, message}
        else
          {:warning, message}
        end
      else
        :ok
      end
    else
      :ok
    end
  end

  @doc """
  Print embed size mismatch warning to stderr.
  """
  @spec warn_embed_mismatch(non_neg_integer(), non_neg_integer()) :: :ok
  def warn_embed_mismatch(checkpoint_size, current_size) do
    message = embed_mismatch_message(checkpoint_size, current_size)
    Output.warning(message)
    :ok
  end

  @doc """
  Get embed size from checkpoint config, with fallback.
  """
  @spec get_embed_size(map(), non_neg_integer()) :: non_neg_integer()
  def get_embed_size(config, fallback) do
    config[:embed_size] || fallback
  end

  @doc """
  Compare checkpoint config with current config and return differences.

  Useful for debugging why a checkpoint might be incompatible.
  """
  @spec config_diff(map(), keyword()) :: [{atom(), {term(), term()}}]
  def config_diff(checkpoint_config, current_opts) do
    keys_to_check = [:embed_size, :hidden_sizes, :axis_buckets, :shoulder_buckets,
                     :temporal, :backbone, :window_size, :num_layers]

    Enum.flat_map(keys_to_check, fn key ->
      checkpoint_val = Map.get(checkpoint_config, key)
      current_val = Keyword.get(current_opts, key)

      if checkpoint_val != nil && current_val != nil && checkpoint_val != current_val do
        [{key, {checkpoint_val, current_val}}]
      else
        []
      end
    end)
  end

  @doc """
  Print config differences in a readable format.
  """
  @spec print_config_diff(map(), keyword()) :: :ok
  def print_config_diff(checkpoint_config, current_opts) do
    diffs = config_diff(checkpoint_config, current_opts)

    if diffs != [] do
      Output.puts("")
      Output.warning("Config differences between checkpoint and current:")
      for {key, {checkpoint_val, current_val}} <- diffs do
        Output.puts("  #{key}: checkpoint=#{inspect(checkpoint_val)}, current=#{inspect(current_val)}")
      end
      Output.puts("")
    end

    :ok
  end

  # Private helpers

  defp validate_and_return(data, type, opts) do
    config = case type do
      :checkpoint -> data.config || %{}
      :policy -> data.config || %{}
    end

    warn_on_mismatch = Keyword.get(opts, :warn_on_mismatch, true)

    case validate_embed_size(config, opts) do
      :ok ->
        {:ok, data}

      {:warning, message} ->
        if warn_on_mismatch do
          Output.warning(message)
        end
        {:ok, data}

      {:error, message} ->
        {:error, {:embed_mismatch, message}}
    end
  end

  defp embed_mismatch_message(checkpoint_size, current_size) do
    """
    Embed size mismatch!
       Checkpoint: #{checkpoint_size} dims
       Current:    #{current_size} dims

       This usually means the embedding config has changed since training.
       The model will fail during inference.

       Options:
       1. Train a new model with current config
       2. Match embedding config to checkpoint (action_mode, nana_mode, etc.)
       3. Check docs/ARCHITECTURE.md for embedding dimension breakdown
    """
  end
end
