defmodule ExPhil.Training.RecordedFrames do
  @moduledoc "Loads unshifted causal recordings without expert relabeling."
  alias ExPhil.Training.Labels

  def envelope(lists, report) do
    %{
      label_convention: :causal,
      label_delay: 0,
      frame_lists: lists,
      teacher_validation_json: Jason.encode!(report)
    }
  end

  def load!(path) do
    path |> File.read!() |> :erlang.binary_to_term([:safe]) |> validate!()
  end

  def validate!(%{label_convention: :causal, label_delay: 0, frame_lists: lists})
      when is_list(lists) and lists != [] do
    Enum.each(lists, fn frames ->
      if frames == [] or Labels.source(frames, require_tagged: true) != :recorded,
        do: raise(ArgumentError, "expected nonempty explicitly recorded frame lists")

      if Enum.any?(frames, &Map.has_key?(&1, :prev_controller)),
        do: raise(ArgumentError, "recorded frames must not override committed-action history")

      unless Enum.all?(frames, &(&1[:input_only] in [nil, false, true])),
        do: raise(ArgumentError, "input_only must be boolean")

      {_prefix, targets} = Enum.split_while(frames, &(&1[:input_only] == true))

      if targets == [] or Enum.any?(targets, &(&1[:input_only] == true)),
        do: raise(ArgumentError, "input-only context must precede nonempty targets")

      contiguous =
        frames
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.all?(fn [previous, current] ->
          current.game_state.frame == previous.game_state.frame + 1
        end)

      unless contiguous, do: raise(ArgumentError, "recorded list crosses a gap or reset")
    end)

    lists
  end

  def validate!(_payload),
    do: raise(ArgumentError, "expected unshifted causal recorded-frame envelope")
end
