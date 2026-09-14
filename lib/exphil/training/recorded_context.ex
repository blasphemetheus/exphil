defmodule ExPhil.Training.RecordedContext do
  @moduledoc "Input-only recorded prefix for lazy temporal supervision. Label delay is applied after slicing."

  def slice(frames, handoff, response_frames, context_frames)
      when response_frames > 0 and context_frames >= 0 do
    first = handoff - context_frames
    last = handoff + response_frames - 1
    selected = Enum.filter(frames, &(&1.game_state.frame in first..last))

    unless Enum.map(selected, & &1.game_state.frame) == Enum.to_list(first..last),
      do: raise(ArgumentError, "missing or noncontiguous recorded context/response")

    Enum.map(selected, fn frame ->
      frame
      |> Map.put(:label_source, :recorded)
      |> Map.put(:input_only, frame.game_state.frame < handoff)
    end)
  end
end
