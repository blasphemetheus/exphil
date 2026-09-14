defmodule ExPhil.Training.RecordedPrefixSampling do
  @moduledoc "Repeats early recorded-teacher windows without truncating clips or changing targets."

  def mark(lists, source) do
    Enum.with_index(lists)
    |> Enum.map(fn {frames, clip} ->
      Enum.map_reduce(frames, 0, fn frame, index ->
        if frame[:input_only] == true do
          {Map.delete(frame, :recorded_prefix), index}
        else
          {Map.put(frame, :recorded_prefix, {source, clip, index}), index + 1}
        end
      end)
      |> elem(0)
    end)
  end

  def frame_weights(lists, weight, prefix_frames)
      when is_integer(weight) and weight >= 1 and is_integer(prefix_frames) and prefix_frames > 0 do
    frames = List.flatten(lists)

    weights =
      Enum.map(frames, fn frame ->
        case {frame[:input_only], frame[:recorded_prefix]} do
          {true, _} -> 0.0
          {_, {_, _, index}} when index < prefix_frames -> weight * 1.0
          _ -> 1.0
        end
      end)

    marked = Enum.count(frames, &match?({_, _, _}, &1[:recorded_prefix]))
    if marked == 0, do: raise(ArgumentError, "no marked recorded teacher targets")

    early =
      Enum.count(frames, fn frame ->
        case frame[:recorded_prefix] do
          {_, _, index} -> index < prefix_frames
          _ -> false
        end
      end)

    {weights,
     %{
       targets: Enum.count(frames, &(&1[:input_only] != true)),
       early_targets: early,
       teacher_targets: marked,
       draws: trunc(Enum.sum(weights)),
       early_draws: early * weight,
       prefix_frames: prefix_frames,
       copies: weight
     }}
  end

  def frame_weights(_, _, _),
    do: raise(ArgumentError, "prefix copies and prefix frames must be positive integers")
end
