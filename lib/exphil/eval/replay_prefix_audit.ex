defmodule ExPhil.Eval.ReplayPrefixAudit do
  @moduledoc """
  Exact comparison of the inputs and player state exposed by Peppi over an
  inclusive frame interval. Missing frames/players fail the audit. This is
  an observable-state check, not a proof that all emulator memory is equal.
  """

  @doc "Compare parsed frame lists, retaining the first mismatch per port and category."
  def compare(source, rerun, first, last, ports \\ [1, 2])
      when is_integer(first) and is_integer(last) and first <= last and ports != [] do
    source = index!(source)
    rerun = index!(rerun)

    results =
      for port <- ports do
        Enum.reduce(
          first..last,
          %{port: port, compared: 0, missing: nil, input: nil, state: nil},
          fn frame, acc ->
            a = player(source, frame, port)
            b = player(rerun, frame, port)

            if is_nil(a) or is_nil(b) do
              missing = %{frame: frame, source_present: a != nil, rerun_present: b != nil}
              %{acc | missing: acc.missing || missing}
            else
              %{
                acc
                | compared: acc.compared + 1,
                  input: acc.input || mismatch(a.controller, b.controller, frame),
                  state:
                    acc.state ||
                      mismatch(Map.delete(a, :controller), Map.delete(b, :controller), frame)
              }
            end
          end
        )
      end

    %{
      first: first,
      last: last,
      expected_frames: last - first + 1,
      ports: results,
      valid: Enum.all?(results, &(&1.missing == nil and &1.input == nil and &1.state == nil))
    }
  end

  @doc "Audit a mixed replay: raw stick bytes and RNG seeds are exact on selected float ports."
  def compare_mixed(source, rerun, first, last, float_ports) do
    # The existing byte path reconstructs sticks from processed axes. Melee's
    # circular clamp can map different raw byte pairs to the same processed
    # pair. Byte ports also do not restore RNG. Still check every other
    # input field and all exposed player state on every port.
    project = fn frames ->
      for frame <- frames do
        players =
          Map.new(frame.players, fn {port, player} ->
            controller = player.controller

            controller =
              if port not in float_ports and is_map(controller.processed) do
                processed =
                  Map.drop(controller.processed, [
                    :raw_main_x,
                    :raw_main_y,
                    :raw_c_x,
                    :raw_c_y,
                    :rng_seed
                  ])

                %{controller | processed: processed}
              else
                controller
              end

            {port, %{player | controller: controller}}
          end)

        %{frame | players: players}
      end
    end

    compare(project.(source), project.(rerun), first, last)
    |> Map.put(:exact_raw_input_ports, float_ports)
  end

  @doc "Wait for Dolphin's asynchronous replay writer, then audit the closed run."
  def verify_directory(source, directory, first, last, float_ports, attempts \\ 20) do
    with {:ok, source} <- ExPhil.Data.Peppi.parse(source),
         {:ok, rerun} <- read_recording(directory, attempts) do
      compare_mixed(source.frames, rerun.frames, first, last, float_ports)
    else
      error -> %{valid: false, error: inspect(error)}
    end
  end

  defp read_recording(directory, attempts) do
    result =
      case Path.wildcard(Path.join(directory, "*.slp")) do
        [path] -> ExPhil.Data.Peppi.parse(path)
        [] -> {:error, :replay_not_written_yet}
        paths -> {:ambiguous, paths}
      end

    case result do
      {:error, _} when attempts > 0 ->
        Process.sleep(50)
        read_recording(directory, attempts - 1)

      result ->
        result
    end
  end

  defp player(frames, frame, port) do
    case frames[frame] do
      nil -> nil
      f -> f.players[port]
    end
  end

  defp index!(frames) do
    Enum.reduce(frames, %{}, fn frame, acc ->
      if Map.has_key?(acc, frame.frame_number),
        do:
          raise(
            ArgumentError,
            "duplicate frame #{frame.frame_number}; audit finalized frames only"
          )

      Map.put(acc, frame.frame_number, frame)
    end)
  end

  defp mismatch(a, b, frame) do
    if exact(a) == exact(b),
      do: nil,
      else: %{frame: frame, source: report_value(a), rerun: report_value(b)}
  end

  defp report_value(value) when is_map(value) do
    value |> Map.delete(:__struct__) |> Map.new(fn {k, v} -> {k, report_value(v)} end)
  end

  defp report_value(value), do: value

  # Numeric equality treats +0.0 and -0.0 as equal. Their representations
  # differ; f32 -> f64 promotion preserves the sign bit.
  defp exact(value) when is_float(value), do: {:float, <<value::float-64>>}

  defp exact(value) when is_map(value),
    do: Map.new(Map.to_list(value), fn {k, v} -> {k, exact(v)} end)

  defp exact(value), do: value
end
