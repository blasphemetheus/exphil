defmodule ExPhil.Training.MixFrames do
  @moduledoc """
  Curriculum mixing: load expert-relabeled drill frames (exported by
  `scripts/export_drill_frames.exs`) into a corpus training run.

  Drills teach execution the corpus can't (on-policy mistakes and their
  corrections: recovery, frame-perfect tech); the corpus teaches decisions.
  `train.exs --mix-frames "drills/*.frames"` concatenates both.

  File format (term_to_binary, :compressed): %{
    expert: String.t(),
    exported_at: iso8601,
    action_delay: integer,      # what the :prev_controller indexing assumed
    frame_lists: [[frame]]      # one list per source replay — the delay
  }                             # shift must not cross replay boundaries

  Frames carry the expert's corrected :controller and the policy's actual
  press in :prev_controller (see the DAgger protocol lessons in
  docs/planning/HANDOFF_2026-07-08.md).
  """

  require Logger

  alias ExPhil.Training.Data

  @doc """
  Load and delay-shift drill frames from a comma-separated list of paths or
  globs. Returns `{frames, stats}` — frames are shifted per source segment
  with `opts[:action_delay]` (must match training).
  """
  @spec load(String.t(), keyword()) :: {[map()], [map()]}
  def load(spec, opts \\ []) do
    delay = Keyword.get(opts, :action_delay, 0)

    paths =
      spec
      |> String.split(",", trim: true)
      |> Enum.map(&Path.expand/1)
      |> Enum.flat_map(&Path.wildcard/1)
      |> Enum.uniq()

    results =
      Enum.map(paths, fn path ->
        case load_file(path, delay) do
          {:ok, frames, meta} ->
            %{path: path, expert: meta[:expert], frames: length(frames), data: frames}

          {:error, reason} ->
            Logger.warning("[MixFrames] Skipping #{path}: #{inspect(reason)}")
            %{path: path, expert: nil, frames: 0, data: []}
        end
      end)

    frames = Enum.flat_map(results, & &1.data)
    stats = Enum.map(results, &Map.delete(&1, :data))
    {frames, stats}
  end

  defp load_file(path, delay) do
    # Plain binary_to_term (no :safe): .frames files are self-generated
    # artifacts from scripts/export_drill_frames.exs, and :safe rejects
    # their struct-laden frames. Never point --mix-frames at untrusted files.
    with {:ok, binary} <- File.read(path),
         %{frame_lists: frame_lists} = payload <-
           :erlang.binary_to_term(binary) do
      export_delay = payload[:action_delay] || 0

      if export_delay != delay do
        Logger.warning(
          "[MixFrames] #{Path.basename(path)} exported at action_delay=#{export_delay} " <>
            "but training uses #{delay} — the :prev_controller channel is misaligned " <>
            "by #{abs(export_delay - delay)} frame(s); re-export with --action-delay #{delay}"
        )
      end

      frames =
        frame_lists
        |> Enum.flat_map(&Data.shift_actions(&1, delay))

      {:ok, frames, expert: payload[:expert]}
    else
      {:error, reason} -> {:error, reason}
      other -> {:error, {:bad_format, other}}
    end
  rescue
    e -> {:error, e}
  end
end
