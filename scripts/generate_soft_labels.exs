#!/usr/bin/env elixir
# Generate soft labels from a trained teacher model for knowledge distillation.
#
# This script runs a trained policy (LSTM, Mamba, etc.) on replay frames and
# saves the probability distributions (soft labels) for training a smaller
# student model (MLP).
#
# Usage:
#   mix run scripts/generate_soft_labels.exs \
#     --teacher checkpoints/mamba_policy.bin \
#     --replays /path/to/replays \
#     --output soft_labels.bin \
#     --temperature 2.0 \
#     --max-files 50
#
# The temperature parameter softens the probability distributions:
#   - T=1.0: Original (sharp) distributions
#   - T=2.0: Softer distributions (recommended for distillation)
#   - T>3.0: Very soft, may lose too much information
#
# Output format:
#   Binary file containing:
#   - config: %{embed_size, num_frames, temperature, ...}
#   - frames: list of %{state: tensor, soft_labels: %{buttons: ..., main_x: ..., ...}}

Mix.install([])

defmodule SoftLabelGenerator do
  @moduledoc """
  Generate soft labels from a teacher model for knowledge distillation.
  """

  alias ExPhil.Training
  alias ExPhil.Training.Output
  alias ExPhil.Data.ReplayParser
  alias ExPhil.Embeddings.Game, as: GameEmbed

  @default_temperature 2.0
  @default_batch_size 64

  def run(args) do
    opts = parse_args(args)

    Output.banner("Soft Label Generation for Distillation")

    Output.config([
      {"Teacher", opts.teacher},
      {"Replays", opts.replays},
      {"Output", opts.output},
      {"Temperature", opts.temperature},
      {"Max files", opts.max_files}
    ])

    # Load teacher policy
    Output.step(1, 6, "Loading teacher policy")
    {:ok, policy} = Training.load_policy(opts.teacher)
    Output.puts("  Loaded: #{opts.teacher}")
    Output.puts("  Config: #{inspect(policy.config, pretty: true, limit: 5)}")

    # Check if teacher is temporal
    is_temporal = policy.config[:temporal] || false
    window_size = policy.config[:window_size] || 60

    if is_temporal do
      Output.puts("  Type: Temporal (#{policy.config[:backbone]}, window=#{window_size})")
    else
      Output.puts("  Type: Single-frame MLP")
    end

    # Find replay files
    Output.step(2, 6, "Finding replay files")
    replay_files = find_replays(opts.replays, opts.max_files)
    Output.puts("  Found #{length(replay_files)} replay files")

    # Parse replays and extract frames
    Output.step(3, 6, "Parsing replays and extracting frames")
    frames = parse_all_replays(replay_files, opts.player_port)
    Output.puts("  Total frames: #{length(frames)}")

    # Build predict function
    Output.step(4, 6, "Building inference function")
    {_init_fn, predict_fn} = Axon.build(policy.model, mode: :inference)
    Output.puts("  Model compiled")

    # Generate soft labels
    Output.step(5, 6, "Generating soft labels (temperature=#{opts.temperature})")

    soft_labels =
      generate_labels(
        frames,
        policy,
        predict_fn,
        is_temporal,
        window_size,
        opts.temperature,
        opts.batch_size
      )

    # Save to file
    Output.step(6, 6, "Saving soft labels")

    save_soft_labels(soft_labels, opts.output, %{
      teacher_path: opts.teacher,
      teacher_config: policy.config,
      temperature: opts.temperature,
      num_frames: length(soft_labels),
      embed_size: policy.config[:embed_size] || GameEmbed.embedding_size()
    })

    Output.divider()
    Output.section("Complete!")
    Output.puts("Output: #{opts.output}")
    Output.puts("Frames: #{length(soft_labels)}")
    Output.puts("Size: #{File.stat!(opts.output).size |> format_bytes()}")
    Output.puts("")
    Output.puts("Next step:")
    Output.puts("  mix run scripts/train_distillation.exs \\")
    Output.puts("    --soft-labels #{opts.output} \\")
    Output.puts("    --hidden 64,64 \\")
    Output.puts("    --epochs 10")
  end

  defp parse_args(args) do
    {parsed, _, _} =
      OptionParser.parse(args,
        strict: [
          teacher: :string,
          replays: :string,
          output: :string,
          temperature: :float,
          batch_size: :integer,
          max_files: :integer,
          player_port: :integer
        ]
      )

    %{
      teacher: Keyword.fetch!(parsed, :teacher),
      replays: Keyword.get(parsed, :replays, "replays"),
      output: Keyword.get(parsed, :output, "soft_labels.bin"),
      temperature: Keyword.get(parsed, :temperature, @default_temperature),
      batch_size: Keyword.get(parsed, :batch_size, @default_batch_size),
      max_files: Keyword.get(parsed, :max_files, 100),
      player_port: Keyword.get(parsed, :player_port, 1)
    }
  end

  defp find_replays(dir, max_files) do
    Path.wildcard(Path.join(dir, "**/*.slp"))
    |> Enum.take(max_files)
  end

  defp parse_all_replays(files, player_port) do
    files
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {file, idx} ->
      IO.write("\r  Parsing file #{idx}/#{length(files)}...\e[K")

      case ReplayParser.parse_replay(file) do
        {:ok, game} ->
          ReplayParser.extract_training_frames(game, player_port)

        {:error, _} ->
          []
      end
    end)
    |> tap(fn _ -> IO.puts("") end)
  end

  defp generate_labels(
         frames,
         policy,
         predict_fn,
         is_temporal,
         window_size,
         temperature,
         batch_size
       ) do
    total = length(frames)

    if is_temporal do
      generate_temporal_labels(
        frames,
        policy,
        predict_fn,
        window_size,
        temperature,
        batch_size,
        total
      )
    else
      generate_single_frame_labels(frames, policy, predict_fn, temperature, batch_size, total)
    end
  end

  defp generate_single_frame_labels(frames, policy, predict_fn, temperature, batch_size, total) do
    frames
    |> Enum.chunk_every(batch_size)
    |> Enum.with_index()
    |> Enum.flat_map(fn {batch, batch_idx} ->
      progress = min(100, round((batch_idx + 1) * batch_size / total * 100))
      IO.write("\r  Progress: #{progress}%\e[K")

      # Embed all frames in batch
      states =
        batch
        |> Enum.map(fn %{game_state: gs} -> GameEmbed.embed(gs) end)
        |> Nx.stack()

      # Run teacher model
      logits = predict_fn.(policy.params, states)

      # Apply temperature and convert to probabilities
      soft_labels = apply_temperature(logits, temperature)

      # Zip with original frames
      batch
      |> Enum.zip(unbatch_labels(soft_labels, length(batch)))
      |> Enum.map(fn {frame, labels} ->
        %{
          game_state: frame.game_state,
          controller: frame.controller,
          soft_labels: labels
        }
      end)
    end)
    |> tap(fn _ -> IO.puts("") end)
  end

  defp generate_temporal_labels(
         frames,
         policy,
         predict_fn,
         window_size,
         temperature,
         batch_size,
         total
       ) do
    # For temporal models, we need to create sequences
    # Generate labels only for the last frame of each sequence
    sequences = create_sequences(frames, window_size)
    seq_total = length(sequences)

    sequences
    |> Enum.chunk_every(batch_size)
    |> Enum.with_index()
    |> Enum.flat_map(fn {batch, batch_idx} ->
      progress = min(100, round((batch_idx + 1) * batch_size / seq_total * 100))
      IO.write("\r  Progress: #{progress}%\e[K")

      # Embed sequences [batch, window, embed]
      states =
        batch
        |> Enum.map(fn seq ->
          seq
          |> Enum.map(fn %{game_state: gs} -> GameEmbed.embed(gs) end)
          |> Nx.stack()
        end)
        |> Nx.stack()

      # Run teacher model
      logits = predict_fn.(policy.params, states)

      # Apply temperature
      soft_labels = apply_temperature(logits, temperature)

      # Each sequence produces one label (for the last frame)
      batch
      |> Enum.zip(unbatch_labels(soft_labels, length(batch)))
      |> Enum.map(fn {seq, labels} ->
        last_frame = List.last(seq)

        %{
          game_state: last_frame.game_state,
          controller: last_frame.controller,
          soft_labels: labels,
          # For temporal student, also save the embedded sequence
          sequence_states: Enum.map(seq, fn f -> GameEmbed.embed(f.game_state) end)
        }
      end)
    end)
    |> tap(fn _ -> IO.puts("") end)
  end

  defp create_sequences(frames, window_size) do
    if length(frames) < window_size do
      []
    else
      frames
      |> Enum.chunk_every(window_size, 1, :discard)
    end
  end

  defp apply_temperature({buttons, main_x, main_y, c_x, c_y, shoulder}, temperature) do
    # For buttons (Bernoulli), apply sigmoid with temperature
    # sigmoid(logits/T) gives softer probabilities
    buttons_soft = Nx.sigmoid(Nx.divide(buttons, temperature))

    # For categorical heads, apply softmax with temperature
    # softmax(logits/T) gives softer distributions
    main_x_soft = Axon.Activations.softmax(Nx.divide(main_x, temperature), axis: -1)
    main_y_soft = Axon.Activations.softmax(Nx.divide(main_y, temperature), axis: -1)
    c_x_soft = Axon.Activations.softmax(Nx.divide(c_x, temperature), axis: -1)
    c_y_soft = Axon.Activations.softmax(Nx.divide(c_y, temperature), axis: -1)
    shoulder_soft = Axon.Activations.softmax(Nx.divide(shoulder, temperature), axis: -1)

    %{
      buttons: buttons_soft,
      main_x: main_x_soft,
      main_y: main_y_soft,
      c_x: c_x_soft,
      c_y: c_y_soft,
      shoulder: shoulder_soft
    }
  end

  defp unbatch_labels(labels, batch_size) do
    # Split batched tensors into individual frames
    Enum.map(0..(batch_size - 1), fn i ->
      %{
        buttons: Nx.slice(labels.buttons, [i, 0], [1, 8]) |> Nx.squeeze(axes: [0]),
        main_x: Nx.slice(labels.main_x, [i, 0], [1, 17]) |> Nx.squeeze(axes: [0]),
        main_y: Nx.slice(labels.main_y, [i, 0], [1, 17]) |> Nx.squeeze(axes: [0]),
        c_x: Nx.slice(labels.c_x, [i, 0], [1, 17]) |> Nx.squeeze(axes: [0]),
        c_y: Nx.slice(labels.c_y, [i, 0], [1, 17]) |> Nx.squeeze(axes: [0]),
        shoulder: Nx.slice(labels.shoulder, [i, 0], [1, 5]) |> Nx.squeeze(axes: [0])
      }
    end)
  end

  defp save_soft_labels(labels, path, config) do
    # Convert tensors to binary backend for serialization
    labels_binary =
      Enum.map(labels, fn frame ->
        %{
          frame
          | soft_labels:
              Map.new(frame.soft_labels, fn {k, v} ->
                {k, Nx.backend_copy(v, Nx.BinaryBackend)}
              end)
        }
        |> Map.update(:sequence_states, nil, fn
          nil -> nil
          states -> Enum.map(states, &Nx.backend_copy(&1, Nx.BinaryBackend))
        end)
      end)

    data = %{
      config: config,
      labels: labels_binary
    }

    File.write!(path, :erlang.term_to_binary(data, [:compressed]))
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / 1024 / 1024, 1)} MB"
end

# Run the generator
SoftLabelGenerator.run(System.argv())
