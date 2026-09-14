defmodule ExPhil.Training.RecordedFramesTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Labels, RecordedFrames}

  defp payload do
    frames = Enum.map(10..16, &%{game_state: %{frame: &1}, controller: &1})
    %{label_convention: :causal, label_delay: 0, frame_lists: [Labels.tag(frames, :recorded)]}
  end

  test "preserves recorded futures and boundaries, rather than retagging expert labels" do
    original = payload()
    lists = RecordedFrames.validate!(original)
    assert lists == original.frame_lists
    assert Enum.map(Labels.at_delay(hd(lists), 2), & &1.controller) == Enum.to_list(12..16)
    assert RecordedFrames.validate!(%{original | frame_lists: lists ++ lists}) == lists ++ lists
  end

  test "rejects ambiguous provenance, pre-shifted targets, and action-history overrides" do
    original = payload()
    frames = hd(original.frame_lists)

    for invalid <- [
          Map.delete(original, :label_convention),
          %{original | label_delay: 2},
          %{original | frame_lists: []},
          %{original | frame_lists: [Enum.map(frames, &Map.delete(&1, :label_source))]},
          %{original | frame_lists: [Labels.tag(frames, {:expert, __MODULE__})]},
          %{original | frame_lists: [Enum.map(frames, &Map.put(&1, :prev_controller, 0))]},
          %{original | frame_lists: [List.delete_at(frames, 2)]},
          %{original | frame_lists: [Enum.reverse(frames)]}
        ] do
      assert_raise ArgumentError, fn -> RecordedFrames.validate!(invalid) end
    end
  end

  test "safe file decoding preserves boundaries and keeps validation metadata as JSON" do
    lists = payload().frame_lists
    envelope = RecordedFrames.envelope(lists, %{valid: true, external_annotation: "verified"})
    assert Jason.decode!(envelope.teacher_validation_json)["valid"]
    path = Path.join(System.tmp_dir!(), "recorded_frames_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm(path) end)
    File.write!(path, :erlang.term_to_binary(envelope, [:compressed]))
    assert RecordedFrames.load!(path) == lists
  end
end
