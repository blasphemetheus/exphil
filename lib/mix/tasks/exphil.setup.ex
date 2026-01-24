defmodule Mix.Tasks.Exphil.Setup do
  @moduledoc """
  Interactive training configuration wizard.

  Walks through questions to build a training command with optimal settings
  for your hardware and goals.

  ## Usage

      mix exphil.setup

  ## What it configures

  1. **Goal** - Quick experiment, character training, or production model
  2. **Character** - Specific character or general training
  3. **Hardware** - GPU/CPU, available memory
  4. **Data** - Replay directory, filtering options
  5. **Advanced** - Backbone, augmentation, etc.

  At the end, outputs a ready-to-run training command.

  ## Examples

      $ mix exphil.setup

      ╔════════════════════════════════════════════════════════════════╗
      ║                ExPhil Training Setup Wizard                    ║
      ╚════════════════════════════════════════════════════════════════╝

      What would you like to do?
        [1] Quick experiment (test setup, ~5 minutes)
        [2] Train a character-specific model
        [3] Train a general model
        [4] Fine-tune an existing model

      Choice [1]:

  """
  use Mix.Task

  alias ExPhil.Training.{Config, GPUUtils, Output}

  @shortdoc "Interactive training configuration wizard"

  # Character options with descriptions
  @characters [
    {:mewtwo, "Mewtwo", "Floaty, teleport recovery, tail hitboxes"},
    {:ganondorf, "Ganondorf", "Heavy, powerful, spacing-focused"},
    {:link, "Link", "Projectile-heavy, bomb recovery"},
    {:gameandwatch, "Game & Watch", "Light, no L-cancel, bucket"},
    {:zelda, "Zelda", "Transform mechanics, kicks"},
    {:ice_climbers, "Ice Climbers", "Desync, wobbling, Nana AI"}
  ]

  # GPU memory tiers
  @gpu_tiers [
    {4, "4GB or less", [batch_size: 32, gradient_checkpoint: true]},
    {8, "8GB", [batch_size: 64]},
    {12, "12GB", [batch_size: 128]},
    {24, "24GB+", [batch_size: 256, hidden_sizes: [512, 512]]}
  ]

  @impl Mix.Task
  def run(_args) do
    # Start required apps for GPU detection
    {:ok, _} = Application.ensure_all_started(:exla)

    Output.puts_raw("""

    ╔════════════════════════════════════════════════════════════════╗
    ║                ExPhil Training Setup Wizard                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Detect hardware
    gpu_info = detect_gpu()
    show_hardware_info(gpu_info)

    # Step 1: What's your goal?
    goal = ask_goal()

    # Step 2: Character selection (if applicable)
    character = if goal in [:character, :production] do
      ask_character()
    else
      nil
    end

    # Step 3: Hardware configuration
    hw_opts = ask_hardware(gpu_info)

    # Step 4: Data configuration
    data_opts = ask_data()

    # Step 5: Advanced options (optional)
    advanced_opts = ask_advanced(goal)

    # Step 6: Fine-tuning (if applicable)
    resume_opts = if goal == :finetune do
      ask_finetune()
    else
      []
    end

    # Build the command
    opts = build_opts(goal, character, hw_opts, data_opts, advanced_opts, resume_opts)
    command = build_command(opts)

    # Show summary
    show_summary(opts, command)

    # Ask to run
    if ask_yes_no("\nRun this command now?", false) do
      Output.puts_raw("\nStarting training...\n")
      System.cmd("mix", ["run", "scripts/train_from_replays.exs" | command_to_args(opts)],
        into: IO.stream(:stdio, :line))
    else
      Output.puts_raw("\nCommand saved. Run it anytime with:\n")
      Output.puts_raw("  #{command}\n")
    end
  end

  # ============================================================================
  # Hardware Detection
  # ============================================================================

  defp detect_gpu do
    case GPUUtils.device_name() do
      {:ok, name} ->
        memory = case GPUUtils.get_memory_info() do
          {:ok, %{total_mb: total}} -> total
          _ -> nil
        end
        %{available: true, name: name, memory_mb: memory}

      {:error, _} ->
        %{available: false, name: nil, memory_mb: nil}
    end
  end

  defp show_hardware_info(gpu_info) do
    Output.puts_raw("Detected Hardware:")
    if gpu_info.available do
      mem_str = if gpu_info.memory_mb, do: " (#{div(gpu_info.memory_mb, 1024)} GB)", else: ""
      Output.puts_raw("  GPU: #{gpu_info.name}#{mem_str}")
    else
      Output.puts_raw("  GPU: None detected (will use CPU)")
    end
    Output.puts_raw("")
  end

  # ============================================================================
  # Questions
  # ============================================================================

  defp ask_goal do
    Output.puts_raw("""
    What would you like to do?

      [1] Quick experiment (test setup, ~5 minutes)
      [2] Train a character-specific model
      [3] Train a general-purpose model
      [4] Fine-tune an existing model
    """)

    choice = ask_choice("Choice", 1, 1..4)

    case choice do
      1 -> :quick
      2 -> :character
      3 -> :production
      4 -> :finetune
    end
  end

  defp ask_character do
    Output.puts_raw("\nWhich character?\n")

    @characters
    |> Enum.with_index(1)
    |> Enum.each(fn {{_atom, name, desc}, idx} ->
      Output.puts_raw("  [#{idx}] #{name} - #{desc}")
    end)

    Output.puts_raw("  [7] Other/General\n")

    choice = ask_choice("Choice", 1, 1..7)

    if choice == 7 do
      nil
    else
      {atom, _, _} = Enum.at(@characters, choice - 1)
      atom
    end
  end

  defp ask_hardware(gpu_info) do
    if gpu_info.available do
      # Auto-detect tier based on memory
      tier = find_gpu_tier(gpu_info.memory_mb)
      {_, tier_name, tier_opts} = tier

      Output.puts_raw("\nGPU Memory Tier: #{tier_name}")
      Output.puts_raw("  Recommended batch size: #{tier_opts[:batch_size]}")

      if ask_yes_no("Use recommended settings?", true) do
        tier_opts
      else
        ask_custom_hardware()
      end
    else
      Output.puts_raw("\nNo GPU detected. Using CPU-optimized settings.")
      [batch_size: 32, temporal: false]
    end
  end

  defp find_gpu_tier(nil), do: Enum.at(@gpu_tiers, 1)  # Default 8GB
  defp find_gpu_tier(memory_mb) do
    memory_gb = div(memory_mb, 1024)
    Enum.find(@gpu_tiers, Enum.at(@gpu_tiers, 1), fn {threshold, _, _} ->
      memory_gb <= threshold
    end)
  end

  defp ask_custom_hardware do
    batch_size = ask_int("Batch size", 64)
    gradient_checkpoint = ask_yes_no("Enable gradient checkpointing (saves memory)?", false)
    [batch_size: batch_size, gradient_checkpoint: gradient_checkpoint]
  end

  defp ask_data do
    Output.puts_raw("\nData Configuration:\n")

    default_replays = System.get_env("EXPHIL_REPLAYS_DIR") || "./replays"
    replays = ask_string("Replay directory", default_replays)

    max_files = if ask_yes_no("Limit number of replay files?", false) do
      ask_int("Max files", 100)
    else
      nil
    end

    [replays: replays, max_files: max_files]
  end

  defp ask_advanced(goal) do
    # Quick mode skips advanced options
    if goal == :quick do
      []
    else
      Output.puts_raw("\nAdvanced Options:\n")

      if ask_yes_no("Configure advanced options?", false) do
        # Architecture options
        temporal = ask_yes_no("Enable temporal training (better quality, slower)?", true)

        backbone = if temporal do
          Output.puts_raw("""

          Backbone architecture:
            [1] Mamba (recommended - fast, good quality)
            [2] Jamba (Mamba + Attention hybrid)
            [3] LSTM (classic recurrent)
            [4] Sliding Window Attention
          """)
          case ask_choice("Choice", 1, 1..4) do
            1 -> :mamba
            2 -> :jamba
            3 -> :lstm
            4 -> :sliding_window
          end
        else
          :mlp
        end

        # Data options
        augment = ask_yes_no("Enable data augmentation (better generalization)?", true)
        dual_port = ask_yes_no("Train on both players per replay (2x data)?", false)

        # Training options
        Output.puts_raw("\nTraining options:\n")

        lr = ask_float("Learning rate", 1.0e-4)

        early_stopping = ask_yes_no("Enable early stopping (stop when validation loss plateaus)?", true)
        val_split = if early_stopping do
          ask_float("Validation split (0.0-0.5)", 0.1)
        else
          0.0
        end
        patience = if early_stopping, do: ask_int("Early stopping patience (epochs)", 5), else: nil

        # Precision
        Output.puts_raw("""

        Precision:
          [1] bf16 (recommended - 2x faster, minimal accuracy loss)
          [2] f32 (full precision, slower but more stable)
        """)
        precision = case ask_choice("Choice", 1, 1..2) do
          1 -> :bf16
          2 -> :f32
        end

        # Online play
        online_robust = ask_yes_no("Train for online play (adds frame delay augmentation)?", false)

        # Monitoring
        wandb = ask_yes_no("Enable Weights & Biases logging?", false)

        # K-means
        kmeans = ask_kmeans()

        # Reproducibility
        seed = if ask_yes_no("Set random seed for reproducibility?", false) do
          ask_int("Seed", :rand.uniform(1_000_000_000))
        else
          nil
        end

        # Build options list
        opts = [
          temporal: temporal,
          backbone: backbone,
          augment: augment,
          dual_port: dual_port,
          learning_rate: lr,
          early_stopping: early_stopping,
          val_split: val_split,
          patience: patience,
          precision: precision,
          online_robust: online_robust,
          wandb: wandb,
          seed: seed
        ]
        |> Enum.reject(fn {_k, v} -> v == nil or v == false or v == 0.0 end)

        if kmeans, do: [{:kmeans_centers, kmeans} | opts], else: opts
      else
        # Sensible defaults for non-quick
        [temporal: true, backbone: :mamba, augment: true, early_stopping: true, val_split: 0.1]
      end
    end
  end

  defp ask_float(prompt, default) do
    result = ask_string(prompt, to_string(default))
    case Float.parse(result) do
      {float, ""} -> float
      {float, _} -> float
      :error -> default
    end
  end

  defp ask_kmeans do
    # Check if K-means centers file exists
    default_path = "priv/kmeans_centers.nx"
    has_centers = File.exists?(default_path)

    if has_centers do
      Output.puts_raw("\n  Found K-means centers at #{default_path}")
      if ask_yes_no("Use K-means stick discretization (~5% better on precision inputs)?", true) do
        default_path
      else
        nil
      end
    else
      if ask_yes_no("Use K-means stick discretization (requires training centers first)?", false) do
        Output.puts_raw("""

          To train K-means centers, run:
            mix run scripts/train_kmeans.exs --replays ./replays --k 21

        """)
        path = ask_string("Path to K-means centers file", default_path)
        if File.exists?(path), do: path, else: nil
      else
        nil
      end
    end
  end

  defp ask_finetune do
    Output.puts_raw("\nFine-tuning Configuration:\n")

    # List available checkpoints
    checkpoints = Path.wildcard("checkpoints/**/*.axon") ++ Path.wildcard("checkpoints/**/*.bin")

    if checkpoints == [] do
      Output.puts_raw("  No checkpoints found in ./checkpoints")
      resume = ask_string("Path to checkpoint to fine-tune", "")
      if resume != "", do: [resume: resume], else: []
    else
      Output.puts_raw("  Available checkpoints:\n")
      checkpoints
      |> Enum.with_index(1)
      |> Enum.each(fn {path, idx} ->
        Output.puts_raw("    [#{idx}] #{Path.basename(path)}")
      end)

      Output.puts_raw("    [0] Enter custom path\n")

      choice = ask_int("Choice", 1)

      resume = if choice == 0 do
        ask_string("Path to checkpoint", "")
      else
        Enum.at(checkpoints, choice - 1)
      end

      if resume && resume != "", do: [resume: resume], else: []
    end
  end

  # ============================================================================
  # Build Configuration
  # ============================================================================

  defp build_opts(goal, character, hw_opts, data_opts, advanced_opts, resume_opts) do
    base_opts = case goal do
      :quick -> [epochs: 1, max_files: 5]
      :character -> [epochs: 20]
      :production -> [epochs: 50, ema: true]
      :finetune -> [epochs: 10]
    end

    character_opts = if character do
      # Use character preset for window size tuning
      preset_opts = Config.preset(character)
      [character: character, window_size: preset_opts[:window_size] || 60]
    else
      []
    end

    # Merge all options (later overrides earlier)
    base_opts
    |> Keyword.merge(character_opts)
    |> Keyword.merge(hw_opts)
    |> Keyword.merge(data_opts)
    |> Keyword.merge(advanced_opts)
    |> Keyword.merge(resume_opts)
    |> Enum.reject(fn {_k, v} -> v == nil end)
  end

  defp build_command(opts) do
    args = command_to_args(opts)
    "mix run scripts/train_from_replays.exs #{Enum.join(args, " ")}"
  end

  defp command_to_args(opts) do
    opts
    |> Enum.flat_map(fn
      # Core
      {:replays, v} -> ["--replays", v]
      {:epochs, v} -> ["--epochs", to_string(v)]
      {:batch_size, v} -> ["--batch-size", to_string(v)]
      {:max_files, v} when not is_nil(v) -> ["--max-files", to_string(v)]
      {:character, v} -> ["--train-character", to_string(v)]
      {:dual_port, true} -> ["--dual-port"]
      # Architecture
      {:temporal, true} -> ["--temporal"]
      {:temporal, false} -> []
      {:backbone, v} when v != :mlp -> ["--backbone", to_string(v)]
      {:window_size, v} -> ["--window-size", to_string(v)]
      {:hidden_sizes, v} -> ["--hidden-sizes", Enum.join(v, ",")]
      # Training
      {:learning_rate, v} when v != 1.0e-4 -> ["--lr", to_string(v)]
      {:early_stopping, true} -> ["--early-stopping"]
      {:val_split, v} when v > 0 -> ["--val-split", to_string(v)]
      {:patience, v} when not is_nil(v) -> ["--patience", to_string(v)]
      {:precision, :f32} -> ["--precision", "f32"]
      {:precision, :bf16} -> []  # default
      {:online_robust, true} -> ["--online-robust"]
      # Augmentation
      {:augment, true} -> ["--augment"]
      {:augment, false} -> []
      # Regularization
      {:label_smoothing, v} when v > 0 -> ["--label-smoothing", to_string(v)]
      {:ema, true} -> ["--ema"]
      {:ema, false} -> []
      # Memory
      {:gradient_checkpoint, true} -> ["--gradient-checkpoint"]
      {:gradient_checkpoint, false} -> []
      # Monitoring
      {:wandb, true} -> ["--wandb"]
      {:wandb, false} -> []
      # Checkpointing
      {:resume, v} -> ["--resume", v]
      {:save_best, true} -> ["--save-best"]
      # Discretization
      {:kmeans_centers, v} when not is_nil(v) -> ["--kmeans-centers", v]
      # Reproducibility
      {:seed, v} when not is_nil(v) -> ["--seed", to_string(v)]
      _ -> []
    end)
  end

  # ============================================================================
  # Summary
  # ============================================================================

  defp show_summary(opts, command) do
    Output.puts_raw("""

    ════════════════════════════════════════════════════════════════
                          Configuration Summary
    ════════════════════════════════════════════════════════════════
    """)

    # Show key settings
    show_opt(opts, :character, "Character")
    show_opt(opts, :epochs, "Epochs")
    show_opt(opts, :batch_size, "Batch Size")
    show_opt(opts, :max_files, "Max Files")
    show_opt(opts, :dual_port, "Dual Port")
    show_opt(opts, :temporal, "Temporal")
    show_opt(opts, :backbone, "Backbone")
    show_opt(opts, :augment, "Augmentation")
    show_opt(opts, :learning_rate, "Learning Rate")
    show_opt(opts, :early_stopping, "Early Stop")
    show_opt(opts, :val_split, "Val Split")
    show_opt(opts, :precision, "Precision")
    show_opt(opts, :online_robust, "Online Mode")
    show_opt(opts, :wandb, "W&B Logging")
    show_opt(opts, :kmeans_centers, "K-means")
    show_opt(opts, :seed, "Seed")
    show_opt(opts, :resume, "Resume From")

    Output.puts_raw("""

    ════════════════════════════════════════════════════════════════
                              Command
    ════════════════════════════════════════════════════════════════

      #{command}
    """)
  end

  defp show_opt(opts, key, label) do
    case Keyword.get(opts, key) do
      nil -> :ok
      true -> Output.puts_raw("  #{String.pad_trailing(label, 15)}: enabled")
      false -> Output.puts_raw("  #{String.pad_trailing(label, 15)}: disabled")
      value -> Output.puts_raw("  #{String.pad_trailing(label, 15)}: #{value}")
    end
  end

  # ============================================================================
  # Input Helpers
  # ============================================================================

  defp ask_string(prompt, default) do
    default_str = if default && default != "", do: " [#{default}]", else: ""
    result = Mix.shell().prompt("#{prompt}#{default_str}:") |> String.trim()
    if result == "", do: default, else: result
  end

  defp ask_int(prompt, default) do
    result = ask_string(prompt, to_string(default))
    case Integer.parse(result) do
      {int, ""} -> int
      _ -> default
    end
  end

  defp ask_choice(prompt, default, range) do
    result = ask_int(prompt, default)
    if result in range, do: result, else: default
  end

  defp ask_yes_no(prompt, default) do
    default_str = if default, do: "Y/n", else: "y/N"
    result = Mix.shell().prompt("#{prompt} [#{default_str}]:") |> String.trim() |> String.downcase()

    case result do
      "" -> default
      "y" -> true
      "yes" -> true
      "n" -> false
      "no" -> false
      _ -> default
    end
  end
end
