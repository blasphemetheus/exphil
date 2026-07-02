defmodule ExPhil.Training.Callbacks.Curriculum do
  @moduledoc """
  Curriculum learning: gradually increase dataset size during training.

  Starts training on a subset of data, then expands to the full dataset
  after a specified number of epochs. This helps the model learn basic
  features before being overwhelmed by the full data distribution.

  ## Usage

      {Curriculum, [
        stages: [
          {100, 3},   # 100 files for first 3 epochs
          {200, 30}   # 200 files for remaining epochs
        ],
        replays: "./replays/huggingface"
      ]}

  ## Why

  The model collapses on 200 files but works on 100. Curriculum learning
  lets it establish representations on the easier subset first, then
  fine-tune on the full dataset.
  """

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Pipeline, Output}

  require Logger

  @impl true
  def init(opts) do
    stages = Keyword.get(opts, :stages, [])
    %{
      stages: stages,
      current_stage: 0,
      stage_opts: Keyword.delete(opts, :stages)
    }
  end

  @impl true
  def on_epoch_begin(state, cb) do
    # Check if we need to advance to the next stage
    case find_stage(cb.stages, state.epoch) do
      {stage_idx, {max_files, _until_epoch}} when stage_idx != cb.current_stage ->
        Logger.info("Curriculum: advancing to stage #{stage_idx + 1} (#{max_files} files)")
        Output.puts("  Curriculum: expanding to #{max_files} files")

        # Rebuild pipeline with new max_files
        new_opts = Keyword.merge(state.opts, [max_files: max_files])
        case Pipeline.setup(new_opts) do
          {:ok, new_pipeline} ->
            state = %{state | pipeline: new_pipeline, opts: new_opts}
            {:cont, state, %{cb | current_stage: stage_idx}}

          {:error, reason} ->
            Output.error("Curriculum stage change failed: #{inspect(reason)}")
            {:cont, state, cb}
        end

      _ ->
        {:cont, state, cb}
    end
  end

  defp find_stage(stages, epoch) do
    stages
    |> Enum.with_index()
    |> Enum.find(fn {{_files, until_epoch}, _idx} ->
      epoch <= until_epoch
    end)
    |> case do
      {{files, until}, idx} -> {idx, {files, until}}
      nil ->
        # Past all stages — use the last one
        case List.last(stages) do
          {files, until} -> {length(stages) - 1, {files, until}}
          nil -> {0, {nil, nil}}
        end
    end
  end
end
