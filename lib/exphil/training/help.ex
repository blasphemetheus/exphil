defmodule ExPhil.Training.Help do
  @moduledoc """
  Contextual help links for training errors and warnings.

  Provides relevant documentation links based on the type of error or warning.
  """

  @doc_base "docs/"

  # Topic -> {doc_file, anchor}
  @help_topics %{
    # Architecture
    temporal: {"TRAINING.md", "temporal-training"},
    backbone: {"TRAINING.md", "temporal-options"},
    mamba: {"TRAINING.md", "mamba-specific-options"},
    window_size: {"TRAINING.md", "temporal-options"},
    hidden_sizes: {"TRAINING.md", "core-options"},

    # Training
    batch_size: {"TRAINING.md", "core-options"},
    learning_rate: {"TRAINING.md", "learning-rate-options"},
    lr_schedule: {"TRAINING.md", "learning-rate-options"},
    epochs: {"TRAINING.md", "core-options"},
    accumulation: {"TRAINING.md", "gradient-accumulation"},
    precision: {"TRAINING.md", "precision-options"},

    # Data
    replays: {"TRAINING.md", "core-options"},
    streaming: {"TRAINING.md", "streaming-data-loading"},
    augmentation: {"TRAINING.md", "data-augmentation"},
    frame_delay: {"TRAINING.md", "training-with-frame-delay-for-online-play"},

    # Regularization
    label_smoothing: {"TRAINING.md", "training-features"},
    focal_loss: {"TRAINING.md", "training-features"},
    early_stopping: {"TRAINING.md", "early-stopping-options"},

    # Checkpointing
    checkpoint: {"TRAINING.md", "checkpointing"},
    resume: {"TRAINING.md", "resume-training"},
    checkpoint_safety: {"TRAINING.md", "checkpoint-safety"},
    batch_checkpoint: {"TRAINING.md", "batch-interval-checkpointing"},
    save_every_batches: {"TRAINING.md", "batch-interval-checkpointing"},

    # Monitoring
    wandb: {"TRAINING.md", "monitoring"},
    verbosity: {"TRAINING.md", "verbosity-control"},
    seed: {"TRAINING.md", "reproducibility"},

    # Embedding
    stage_mode: {"TRAINING.md", "embedding-options"},
    player_names: {"TRAINING.md", "embedding-options"},

    # Discretization
    kmeans: {"TRAINING.md", "k-means-stick-discretization"},
    stick_discretization: {"TRAINING.md", "k-means-stick-discretization"},

    # Presets
    presets: {"TRAINING.md", "presets"},
    setup_wizard: {"TRAINING.md", "interactive-setup-wizard"},

    # Environment
    env_vars: {"TRAINING.md", "environment-variables"},

    # Architecture docs
    architecture: {"ARCHITECTURE.md", nil},
    gotchas: {"GOTCHAS.md", nil},

    # Testing
    testing: {"TESTING.md", nil}
  }

  @doc """
  Get a help link for a topic.

  ## Examples

      iex> Help.link(:temporal)
      "See docs/TRAINING.md#temporal-training"

      iex> Help.link(:unknown)
      nil
  """
  @spec link(atom()) :: String.t() | nil
  def link(topic) do
    case Map.get(@help_topics, topic) do
      {file, nil} -> "See #{@doc_base}#{file}"
      {file, anchor} -> "See #{@doc_base}#{file}##{anchor}"
      nil -> nil
    end
  end

  @doc """
  Get a help link or empty string (for easy interpolation).
  """
  @spec link!(atom()) :: String.t()
  def link!(topic) do
    link(topic) || ""
  end

  @doc """
  Format an error message with a contextual help link.

  ## Examples

      iex> Help.with_link("Invalid backbone", :backbone)
      "Invalid backbone (See docs/TRAINING.md#temporal-options)"
  """
  @spec with_link(String.t(), atom()) :: String.t()
  def with_link(message, topic) do
    case link(topic) do
      nil -> message
      help -> "#{message} (#{help})"
    end
  end

  @doc """
  Format a warning with help link appended on new line.
  """
  @spec warning_with_help(String.t(), atom()) :: String.t()
  def warning_with_help(message, topic) do
    case link(topic) do
      nil -> message
      help -> "#{message}\n       #{help}"
    end
  end

  @doc """
  Get all available help topics.
  """
  @spec topics() :: [atom()]
  def topics, do: Map.keys(@help_topics)

  @doc """
  Suggest a topic based on a keyword in an error message.
  """
  @spec suggest_topic(String.t()) :: atom() | nil
  def suggest_topic(message) do
    message_lower = String.downcase(message)

    cond do
      message_lower =~ "temporal" -> :temporal
      message_lower =~ "backbone" -> :backbone
      message_lower =~ "mamba" -> :mamba
      message_lower =~ "window" -> :window_size
      message_lower =~ "batch" and message_lower =~ "checkpoint" -> :batch_checkpoint
      message_lower =~ "save" and message_lower =~ "batch" -> :batch_checkpoint
      message_lower =~ "batch" -> :batch_size
      message_lower =~ "learning rate" or message_lower =~ "lr" -> :learning_rate
      message_lower =~ "epoch" -> :epochs
      message_lower =~ "replay" -> :replays
      message_lower =~ "checkpoint" -> :checkpoint
      message_lower =~ "wandb" or message_lower =~ "w&b" -> :wandb
      message_lower =~ "augment" -> :augmentation
      message_lower =~ "frame delay" or message_lower =~ "online" -> :frame_delay
      message_lower =~ "precision" or message_lower =~ "bf16" or message_lower =~ "f32" -> :precision
      message_lower =~ "hidden" -> :hidden_sizes
      message_lower =~ "accumulation" -> :accumulation
      message_lower =~ "label smooth" -> :label_smoothing
      message_lower =~ "focal" -> :focal_loss
      message_lower =~ "early stop" -> :early_stopping
      message_lower =~ "stream" -> :streaming
      message_lower =~ "stage" -> :stage_mode
      message_lower =~ "preset" -> :presets
      message_lower =~ "env" or message_lower =~ "environment" -> :env_vars
      message_lower =~ "seed" -> :seed
      message_lower =~ "verbose" or message_lower =~ "quiet" -> :verbosity
      message_lower =~ "backup" or message_lower =~ "overwrite" -> :checkpoint_safety
      message_lower =~ "kmeans" or message_lower =~ "k-means" or message_lower =~ "discretiz" -> :kmeans
      true -> nil
    end
  end

  @doc """
  Auto-add help link to a message based on keyword detection.
  """
  @spec auto_help(String.t()) :: String.t()
  def auto_help(message) do
    case suggest_topic(message) do
      nil -> message
      topic -> with_link(message, topic)
    end
  end
end
