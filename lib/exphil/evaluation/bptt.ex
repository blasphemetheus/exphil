defmodule ExPhil.Evaluation.BPTT do
  @moduledoc """
  Ordered, per-frame evaluation of BPTT policies over complete replay segments.

  Each replay is embedded independently. Carry resets at replay boundaries and
  frame gaps, never at ordinary chunk edges. Short segments and final partial
  chunks are retained. All frames are scored once with plain teacher-forced
  cross entropy; this is not the weighted training objective or live sampling.
  """

  alias ExPhil.Data.{LabelConvention, Peppi, SubjectResolver}
  alias ExPhil.Embeddings
  alias ExPhil.Evaluation.{Forward, Metrics}
  alias ExPhil.Networks.Policy
  alias ExPhil.Training.{Data, Output}

  @heads [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

  def batches(dataset, chunk_size) when is_integer(chunk_size) and chunk_size > 0 do
    dataset.frames
    |> Enum.with_index()
    |> Enum.chunk_while(
      [],
      fn {frame, index}, segment ->
        case segment do
          [{previous, _} | _] when frame.game_state.frame != previous.game_state.frame + 1 ->
            {:cont, Enum.reverse(segment), [{frame, index}]}

          _ ->
            {:cont, [{frame, index} | segment]}
        end
      end,
      fn
        [] -> {:cont, []}
        segment -> {:cont, Enum.reverse(segment), []}
      end
    )
    |> Stream.flat_map(fn segment ->
      segment
      |> Enum.chunk_every(chunk_size)
      |> Enum.with_index()
      |> Stream.map(fn {chunk, chunk_index} ->
        {_first, start} = hd(chunk)

        actions =
          chunk
          |> Enum.map(fn {frame, _} -> Data.frame_action(frame) end)
          |> Data.actions_to_tensors()

        %{
          states:
            dataset.embedded_frames
            |> Nx.slice_along_axis(start, length(chunk), axis: 0)
            |> Nx.new_axis(0),
          actions: Map.new(actions, fn {key, tensor} -> {key, Nx.new_axis(tensor, 0)} end),
          is_resetting: Nx.tensor([if(chunk_index == 0, do: 1, else: 0)], type: :u8)
        }
      end)
    end)
  end

  def evaluate(evaluator, batches) do
    zero = Map.new(@heads, &{&1, 0.0})

    totals =
      evaluator
      |> Forward.stream(batches)
      |> Enum.reduce(%{loss: 0.0, frames: 0, accuracy: zero}, fn {logits, actions}, totals ->
        logits = Map.new(Enum.zip(@heads, Tuple.to_list(logits)))
        count = Nx.axis_size(logits.buttons, 0)

        loss =
          Policy.imitation_loss(logits, actions, button_weight: 1.0, label_smoothing: 0.0)
          |> Nx.to_number()

        if not is_number(loss),
          do: raise(ArgumentError, "non-finite BPTT evaluation loss: #{inspect(loss)}")

        accuracy =
          Map.new(@heads, fn head ->
            metric =
              if head == :buttons,
                do: Metrics.button_accuracy(logits[head], actions[head]),
                else: Metrics.accuracy(logits[head], actions[head])

            {head, totals.accuracy[head] + metric * count}
          end)

        %{loss: totals.loss + loss * count, frames: totals.frames + count, accuracy: accuracy}
      end)

    if totals.frames == 0, do: raise(ArgumentError, "No frames available for BPTT evaluation")

    %{
      protocol: "bptt_teacher_forced_plain_ce_v1",
      frames: totals.frames,
      loss: totals.loss / totals.frames,
      accuracy: Map.new(totals.accuracy, fn {head, total} -> {head, total / totals.frames} end)
    }
  end

  def run(artifacts, opts) do
    unless Enum.all?(artifacts, & &1.config[:bptt]),
      do:
        raise(
          ArgumentError,
          "Evaluate BPTT and windowed checkpoints separately; their metrics differ"
        )

    for key <- [:export_csv, :export_sequences] do
      if opts[key], do: raise(ArgumentError, "#{key} is not supported by BPTT evaluation")
    end

    files =
      Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
      |> Enum.sort()
      |> Enum.take(opts[:max_files])

    if files == [], do: raise(ArgumentError, "No replay files found for BPTT evaluation")

    Enum.map(artifacts, fn artifact ->
      config = artifact.config

      if (config[:axis_buckets] || 16) != 16 or (config[:shoulder_buckets] || 4) != 4 or
           config[:kmeans_centers],
         do:
           raise(
             ArgumentError,
             "Replay BPTT evaluation currently requires the standard 16/4 discretization"
           )

      if config[:with_delay_id] || (config[:queue_depth] || 1) > 1,
        do:
          raise(
            ArgumentError,
            "BPTT evaluation does not yet construct queued-action/delay-id inputs"
          )

      embed_config = Embeddings.config(Map.to_list(config))

      if Embeddings.embedding_size(embed_config) != config[:embed_size],
        do:
          raise(
            ArgumentError,
            "Checkpoint embedding layout does not match its trained input width"
          )

      reaction_delay = LabelConvention.reaction_delay(config)

      if reaction_delay < 0,
        do:
          raise(
            ArgumentError,
            "Legacy leaked-label checkpoints cannot be evaluated as causal BPTT policies"
          )

      evaluator = Forward.new(artifact.params, config)

      stream =
        Stream.flat_map(files, fn path ->
          {:ok, metadata} = Peppi.metadata(path)

          selector =
            if opts[:character],
              do: [subject_character: opts[:character]],
              else: [subject_port: opts[:player_port] || 1]

          {:ok, subject} = SubjectResolver.resolve(metadata.players, selector)
          {:ok, replay} = Peppi.parse(path, player_port: subject.subject_port)

          frames =
            Peppi.to_training_frames(replay,
              player_port: subject.subject_port,
              opponent_port: subject.opponent_port,
              remap_ports: true,
              frame_delay: reaction_delay
            )

          if frames == [] do
            []
          else
            dataset =
              frames
              |> Data.from_frames(embed_config: embed_config)
              |> Data.precompute_frame_embeddings(
                show_progress: false,
                use_prev_action: config[:use_prev_action] || false
              )

            batches(dataset, config[:unroll] || 80)
          end
        end)

      result = evaluate(evaluator, stream)
      Output.puts("BPTT evaluation: #{artifact.path}")

      Output.puts(
        "  #{result.frames} frames; teacher-forced plain CE=#{Float.round(result.loss, 6)}"
      )

      Output.puts("  Per-head accuracy: #{inspect(result.accuracy)}")

      Output.puts(
        "  Full replay carry; not directly comparable to weighted training loss or windowed evaluation"
      )

      result
    end)
  end
end
