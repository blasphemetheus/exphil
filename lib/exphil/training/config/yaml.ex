defmodule ExPhil.Training.Config.Yaml do
  @moduledoc """
  YAML configuration file loading and saving for training.

  Provides utilities for:
  - Loading training configuration from YAML files
  - Parsing YAML content into keyword options
  - Saving configuration back to YAML format
  - Safe atom conversion for YAML values

  ## Usage

      # Load from file
      {:ok, opts} = Yaml.load("config.yaml", context)

      # Parse YAML string
      {:ok, opts} = Yaml.parse(yaml_content, context)

      # Save to file
      :ok = Yaml.save(opts, "config.yaml")

  ## Context

  The context map provides allowlists for safe atom conversion:

      context = %{
        valid_backbones: [:lstm, :gru, :mamba],
        valid_optimizers: [:adam, :adamw],
        valid_lr_schedules: [:constant, :cosine],
        valid_precision_modes: [:f32, :bf16],
        valid_presets: [:quick, :standard],
        valid_characters: [:fox, :falco, :marth],
        valid_stages: [:battlefield, :final_destination]
      }

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Config.AtomSafety` - Safe atom conversion
  """

  alias ExPhil.Training.Config.AtomSafety

  @type yaml_context :: %{
          valid_backbones: [atom()],
          valid_optimizers: [atom()],
          valid_lr_schedules: [atom()],
          valid_precision_modes: [atom()],
          valid_presets: [atom()],
          valid_characters: [atom()],
          valid_stages: [atom()]
        }

  @doc """
  Load training configuration from a YAML file.

  Returns `{:ok, opts}` on success, or `{:error, reason}` on failure.

  ## Parameters

  - `path` - Path to YAML file
  - `context` - Map with allowlists for safe atom conversion

  ## File Format

  The YAML file should contain training options as key-value pairs.
  Keys can use either snake_case or kebab-case.

  ## Example config.yaml

      # Basic training settings
      epochs: 20
      batch_size: 128
      hidden_sizes: [256, 256]

      # Model architecture
      temporal: true
      backbone: mamba
      window_size: 60

  ## Examples

      iex> Yaml.load("missing.yaml", %{})
      {:error, :file_not_found}

  """
  @spec load(String.t(), yaml_context()) :: {:ok, keyword()} | {:error, atom() | String.t()}
  def load(path, context) do
    case File.read(path) do
      {:ok, content} ->
        parse(content, context)

      {:error, :enoent} ->
        {:error, :file_not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Parse YAML content into training options.

  ## Parameters

  - `content` - YAML string content
  - `context` - Map with allowlists for safe atom conversion

  ## Examples

      iex> Yaml.parse("epochs: 10", %{})
      {:ok, [epochs: 10]}

  """
  @spec parse(String.t(), yaml_context()) :: {:ok, keyword()} | {:error, any()}
  def parse(content, context) do
    case YamlElixir.read_from_string(content) do
      {:ok, map} when is_map(map) ->
        opts = convert_yaml_map(map, context)
        {:ok, opts}

      {:ok, _other} ->
        {:error, :invalid_yaml_format}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Save configuration to a YAML file.

  Useful for saving the effective configuration after a training run.

  ## Parameters

  - `opts` - Keyword list of options
  - `path` - Output file path

  ## Examples

      iex> Yaml.save([epochs: 10, batch_size: 64], "/tmp/test.yaml")
      :ok

  """
  @spec save(keyword(), String.t()) :: :ok | {:error, any()}
  def save(opts, path) do
    yaml = opts_to_yaml(opts)
    File.write(path, yaml)
  end

  @doc """
  Convert options to YAML string format.

  ## Examples

      iex> Yaml.to_string([epochs: 10, batch_size: 64])
      "epochs: 10\\nbatch-size: 64"

  """
  @spec to_string(keyword()) :: String.t()
  def to_string(opts) do
    opts_to_yaml(opts)
  end

  # ============================================================================
  # Private Helpers - YAML Parsing
  # ============================================================================

  # Convert YAML map to keyword list with proper atom/type conversion
  defp convert_yaml_map(map, context) do
    map
    |> Enum.map(fn {key, value} ->
      atom_key = normalize_key(key)
      converted_value = convert_value(atom_key, value, context)
      {atom_key, converted_value}
    end)
    |> Keyword.new()
  end

  # Normalize key from string (handles kebab-case and snake_case)
  # Uses safe_to_existing_atom to prevent atom table exhaustion from untrusted YAML
  defp normalize_key(key) when is_binary(key) do
    normalized = String.replace(key, "-", "_")

    case AtomSafety.safe_to_existing_atom(normalized) do
      {:ok, atom} -> atom
      # Fall back to known config keys or raise for unknown keys
      {:error, :not_existing} ->
        raise ArgumentError, "Unknown config key: #{inspect(key)}"
    end
  end

  defp normalize_key(key) when is_atom(key), do: key

  # Convert values based on expected types using safe atom conversion
  defp convert_value(:backbone, value, context) when is_binary(value) do
    valid = (context[:valid_backbones] || []) ++ [:mlp]
    AtomSafety.safe_to_atom!(value, valid)
  end

  defp convert_value(:lr_schedule, value, context) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, context[:valid_lr_schedules] || [])
  end

  defp convert_value(:optimizer, value, context) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, context[:valid_optimizers] || [])
  end

  defp convert_value(:precision, value, context) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, context[:valid_precision_modes] || [])
  end

  defp convert_value(:preset, value, context) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, context[:valid_presets] || [])
  end

  defp convert_value(:character, value, context) when is_binary(value) do
    AtomSafety.safe_to_atom!(value, context[:valid_characters] || [])
  end

  defp convert_value(:characters, values, context) when is_list(values) do
    valid_chars = context[:valid_characters] || []
    Enum.map(values, &AtomSafety.safe_to_atom!(&1, valid_chars))
  end

  defp convert_value(:stages, values, context) when is_list(values) do
    valid_stages = context[:valid_stages] || []
    Enum.map(values, &AtomSafety.safe_to_atom!(&1, valid_stages))
  end

  defp convert_value(:hidden_sizes, values, _context) when is_list(values), do: values
  defp convert_value(_key, value, _context), do: value

  # ============================================================================
  # Private Helpers - YAML Generation
  # ============================================================================

  # Convert opts to YAML string
  defp opts_to_yaml(opts) do
    opts
    |> Enum.map(fn {key, value} ->
      yaml_key = key |> Kernel.to_string() |> String.replace("_", "-")
      yaml_value = format_yaml_value(value)
      "#{yaml_key}: #{yaml_value}"
    end)
    |> Enum.join("\n")
  end

  defp format_yaml_value(value) when is_list(value) do
    items = Enum.map(value, &Kernel.to_string/1) |> Enum.join(", ")
    "[#{items}]"
  end

  defp format_yaml_value(value) when is_atom(value), do: Kernel.to_string(value)
  defp format_yaml_value(value) when is_binary(value), do: "\"#{value}\""
  defp format_yaml_value(value), do: Kernel.to_string(value)
end
