defmodule ExPhil.Training.Registry do
  @moduledoc """
  Model registry for tracking trained checkpoints with metadata.

  Stores model entries in a JSON file with training config, metrics, tags, and lineage.
  Enables reproducibility by tracking exactly how each model was trained.

  ## Registry File Structure

  The registry is stored at `checkpoints/registry.json`:

      {
        "version": "1.0",
        "models": [
          {
            "id": "abc123",
            "name": "wavedashing_falcon",
            "checkpoint_path": "checkpoints/model.axon",
            "policy_path": "checkpoints/model_policy.bin",
            "created_at": "2026-01-19T12:00:00Z",
            "training_config": { ... },
            "metrics": { "final_loss": 1.234, "epochs": 10 },
            "tags": ["mewtwo", "production"],
            "parent_id": null
          }
        ]
      }

  ## Usage

      # Register a model after training
      Registry.register(%{
        checkpoint_path: "checkpoints/model.axon",
        training_config: opts,
        metrics: %{final_loss: 1.234}
      })

      # List all models
      Registry.list()

      # Filter by tags
      Registry.list(tags: ["mewtwo"])

      # Add tags
      Registry.tag("abc123", ["production", "v1"])

      # Get a specific model
      Registry.get("abc123")

  """

  alias ExPhil.Training.Naming
  alias ExPhil.Error.RegistryError

  @default_registry_file "checkpoints/registry.json"
  @version "1.0"

  # Get registry file path (allows override for testing)
  defp registry_file do
    Application.get_env(:exphil, :registry_path, @default_registry_file)
  end

  @type model_entry :: %{
          id: String.t(),
          name: String.t(),
          checkpoint_path: String.t(),
          policy_path: String.t() | nil,
          config_path: String.t() | nil,
          created_at: String.t(),
          training_config: map(),
          metrics: map(),
          tags: [String.t()],
          parent_id: String.t() | nil
        }

  @type registry :: %{
          version: String.t(),
          models: [model_entry()]
        }

  @doc """
  Register a new model in the registry.

  ## Options

    * `:checkpoint_path` - Required. Path to the .axon checkpoint
    * `:policy_path` - Optional. Path to the _policy.bin file
    * `:config_path` - Optional. Path to the _config.json file
    * `:training_config` - Required. Training configuration map
    * `:metrics` - Optional. Training metrics (loss, epochs, etc.)
    * `:tags` - Optional. List of tags for categorization
    * `:parent_id` - Optional. ID of parent model if fine-tuning
    * `:name` - Optional. Override the auto-generated name

  Returns `{:ok, model_entry}` on success.
  """
  @spec register(map()) :: {:ok, model_entry()} | {:error, term()}
  def register(opts) when is_map(opts) do
    with {:ok, checkpoint_path} <- Map.fetch(opts, :checkpoint_path),
         {:ok, training_config} <- Map.fetch(opts, :training_config),
         {:ok, registry} <- load_or_create() do
      entry = %{
        id: generate_id(),
        name: Map.get(opts, :name) || Naming.generate(),
        checkpoint_path: checkpoint_path,
        policy_path: Map.get(opts, :policy_path),
        config_path: Map.get(opts, :config_path),
        created_at: DateTime.utc_now() |> DateTime.to_iso8601(),
        training_config: sanitize_config(training_config),
        metrics: Map.get(opts, :metrics, %{}),
        tags: Map.get(opts, :tags, []),
        parent_id: Map.get(opts, :parent_id)
      }

      updated = %{registry | models: [entry | registry.models]}

      case save(updated) do
        :ok -> {:ok, entry}
        error -> error
      end
    else
      :error -> {:error, RegistryError.new(:missing_required_field, context: %{field: :checkpoint_path})}
      error -> error
    end
  end

  @doc """
  List all models in the registry.

  ## Options

    * `:tags` - Filter by tags (models must have ALL specified tags)
    * `:backbone` - Filter by backbone type (mlp, mamba, lstm, etc.)
    * `:limit` - Maximum number of models to return
    * `:sort` - Sort by :created_at (default) or :loss

  """
  @spec list(keyword()) :: {:ok, [model_entry()]} | {:error, term()}
  def list(opts \\ []) do
    with {:ok, registry} <- load_or_create() do
      models =
        registry.models
        |> filter_by_tags(Keyword.get(opts, :tags))
        |> filter_by_backbone(Keyword.get(opts, :backbone))
        |> sort_models(Keyword.get(opts, :sort, :created_at))
        |> limit_models(Keyword.get(opts, :limit))

      {:ok, models}
    end
  end

  @doc """
  Get a specific model by ID or name.
  """
  @spec get(String.t()) :: {:ok, model_entry()} | {:error, RegistryError.t()}
  def get(id_or_name) do
    with {:ok, registry} <- load_or_create() do
      model =
        Enum.find(registry.models, fn m ->
          m.id == id_or_name || m.name == id_or_name
        end)

      case model do
        nil -> {:error, RegistryError.new(:not_found, model_id: id_or_name)}
        m -> {:ok, m}
      end
    end
  end

  @doc """
  Add tags to a model.
  """
  @spec tag(String.t(), [String.t()]) :: :ok | {:error, term()}
  def tag(id_or_name, new_tags) when is_list(new_tags) do
    with {:ok, registry} <- load_or_create() do
      {models, found} =
        Enum.map_reduce(registry.models, false, fn model, acc ->
          if model.id == id_or_name || model.name == id_or_name do
            updated = %{model | tags: Enum.uniq(model.tags ++ new_tags)}
            {updated, true}
          else
            {model, acc}
          end
        end)

      if found do
        save(%{registry | models: models})
      else
        {:error, RegistryError.new(:not_found, model_id: id_or_name)}
      end
    end
  end

  @doc """
  Remove tags from a model.
  """
  @spec untag(String.t(), [String.t()]) :: :ok | {:error, term()}
  def untag(id_or_name, tags_to_remove) when is_list(tags_to_remove) do
    with {:ok, registry} <- load_or_create() do
      {models, found} =
        Enum.map_reduce(registry.models, false, fn model, acc ->
          if model.id == id_or_name || model.name == id_or_name do
            updated = %{model | tags: model.tags -- tags_to_remove}
            {updated, true}
          else
            {model, acc}
          end
        end)

      if found do
        save(%{registry | models: models})
      else
        {:error, RegistryError.new(:not_found, model_id: id_or_name)}
      end
    end
  end

  @doc """
  Delete a model entry from the registry.

  Note: This only removes the registry entry. The checkpoint files remain on disk.
  Use `delete/2` with `delete_files: true` to also delete the files.
  """
  @spec delete(String.t(), keyword()) :: :ok | {:error, term()}
  def delete(id_or_name, opts \\ []) do
    with {:ok, registry} <- load_or_create() do
      {deleted, remaining} =
        Enum.split_with(registry.models, fn m ->
          m.id == id_or_name || m.name == id_or_name
        end)

      case deleted do
        [] ->
          {:error, RegistryError.new(:not_found, model_id: id_or_name)}

        [model] ->
          if Keyword.get(opts, :delete_files, false) do
            delete_model_files(model)
          end

          save(%{registry | models: remaining})
      end
    end
  end

  @doc """
  Get the best model by a metric.

  ## Options

    * `:metric` - The metric to compare (default: :final_loss)
    * `:minimize` - Whether to minimize the metric (default: true for loss)
    * `:tags` - Filter by tags first

  """
  @spec best(keyword()) :: {:ok, model_entry()} | {:error, term()}
  def best(opts \\ []) do
    metric_key = Keyword.get(opts, :metric, :final_loss)
    minimize = Keyword.get(opts, :minimize, true)

    with {:ok, models} <- list(opts) do
      models_with_metric =
        Enum.filter(models, fn m ->
          get_in(m, [:metrics, metric_key]) != nil
        end)

      case models_with_metric do
        [] ->
          {:error, RegistryError.new(:no_models_with_metric, context: %{metric: metric_key})}

        _ ->
          best =
            if minimize do
              Enum.min_by(models_with_metric, &get_in(&1, [:metrics, metric_key]))
            else
              Enum.max_by(models_with_metric, &get_in(&1, [:metrics, metric_key]))
            end

          {:ok, best}
      end
    end
  end

  @doc """
  Get models with the same lineage (parent chain).
  """
  @spec lineage(String.t()) :: {:ok, [model_entry()]} | {:error, term()}
  def lineage(id_or_name) do
    with {:ok, model} <- get(id_or_name),
         {:ok, registry} <- load_or_create() do
      # Find ancestors
      ancestors = find_ancestors(model, registry.models, [])
      # Find descendants
      descendants = find_descendants(model.id, registry.models)

      {:ok, ancestors ++ [model] ++ descendants}
    end
  end

  @doc """
  Count models in the registry.
  """
  @spec count(keyword()) :: {:ok, non_neg_integer()} | {:error, term()}
  def count(opts \\ []) do
    with {:ok, models} <- list(opts) do
      {:ok, length(models)}
    end
  end

  @doc """
  Get the path to the registry file.
  """
  @spec registry_path() :: String.t()
  def registry_path, do: registry_file()

  @doc """
  Check if a model exists by ID or name.
  """
  @spec exists?(String.t()) :: boolean()
  def exists?(id_or_name) do
    case get(id_or_name) do
      {:ok, _} -> true
      _ -> false
    end
  end

  # Private functions

  defp load_or_create do
    path = registry_file()

    if File.exists?(path) do
      case File.read(path) do
        {:ok, content} ->
          case Jason.decode(content, keys: :atoms) do
            {:ok, data} -> {:ok, normalize_registry(data)}
            error -> error
          end

        error ->
          error
      end
    else
      {:ok, %{version: @version, models: []}}
    end
  end

  defp save(registry) do
    path = registry_file()

    # Ensure directory exists
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    content = Jason.encode!(registry, pretty: true)

    case File.write(path, content) do
      :ok -> :ok
      error -> error
    end
  end

  defp normalize_registry(data) do
    models =
      (data[:models] || [])
      |> Enum.map(fn m ->
        %{
          id: m[:id],
          name: m[:name],
          checkpoint_path: m[:checkpoint_path],
          policy_path: m[:policy_path],
          config_path: m[:config_path],
          created_at: m[:created_at],
          training_config: m[:training_config] || %{},
          metrics: m[:metrics] || %{},
          tags: m[:tags] || [],
          parent_id: m[:parent_id]
        }
      end)

    %{version: data[:version] || @version, models: models}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.url_encode64(padding: false)
  end

  defp sanitize_config(config) when is_list(config) do
    # Handle keyword lists by converting to map first
    sanitize_config(Map.new(config))
  end

  defp sanitize_config(config) when is_map(config) do
    # Remove function values that can't be serialized
    config
    |> Enum.reject(fn {_k, v} -> is_function(v) end)
    |> Enum.reject(fn {_k, v} -> is_pid(v) end)
    |> Enum.into(%{})
  end

  defp filter_by_tags(models, nil), do: models
  defp filter_by_tags(models, []), do: models

  defp filter_by_tags(models, tags) do
    Enum.filter(models, fn m ->
      Enum.all?(tags, &(&1 in m.tags))
    end)
  end

  defp filter_by_backbone(models, nil), do: models

  defp filter_by_backbone(models, backbone) do
    backbone_str = to_string(backbone)

    Enum.filter(models, fn m ->
      config_backbone = get_in(m, [:training_config, :backbone])
      to_string(config_backbone) == backbone_str
    end)
  end

  defp sort_models(models, :created_at) do
    Enum.sort_by(models, & &1.created_at, :desc)
  end

  defp sort_models(models, :loss) do
    Enum.sort_by(models, &get_in(&1, [:metrics, :final_loss]))
  end

  defp sort_models(models, _), do: models

  defp limit_models(models, nil), do: models
  defp limit_models(models, n) when is_integer(n), do: Enum.take(models, n)

  defp find_ancestors(_model, _all, ancestors) when length(ancestors) > 100 do
    # Prevent infinite loops
    ancestors
  end

  defp find_ancestors(%{parent_id: nil}, _all, ancestors) do
    # No more parents - list is already in root-first order
    # (older ancestors were prepended first)
    ancestors
  end

  defp find_ancestors(%{parent_id: parent_id}, all, ancestors) do
    case Enum.find(all, &(&1.id == parent_id)) do
      nil ->
        # Parent not found - return what we have
        ancestors

      parent ->
        # Found parent - prepend it (older ancestors go to front)
        find_ancestors(parent, all, [parent | ancestors])
    end
  end

  defp find_descendants(id, all) do
    direct = Enum.filter(all, &(&1.parent_id == id))
    direct ++ Enum.flat_map(direct, &find_descendants(&1.id, all))
  end

  defp delete_model_files(model) do
    paths = [
      model.checkpoint_path,
      model.policy_path,
      model.config_path
    ]

    Enum.each(paths, fn path ->
      if path && File.exists?(path) do
        File.rm(path)
      end
    end)
  end
end
