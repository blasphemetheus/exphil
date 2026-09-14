defmodule ExPhil.Eval.ScenarioInputTiming do
  @moduledoc """
  Checks scenario decisions and sends against the same run's Slippi recording.
  A causal replay frame at t contains the input recorded at t+1. Unsent tail
  decisions are excluded; missing or mismatching recorded evidence fails closed.

  The default pipe_v2 contract includes quantization, radius-80 stick clamp,
  axis deadzones and independent digital trigger clicks. Historical pipe_v1
  recordings require an explicit profile; profiles are never inferred from fit.
  """

  alias ExPhil.Bridge.ControllerState

  def verify_directory(directory, trace, response_delay, attempts \\ 20, opts \\ []) do
    result =
      case Path.wildcard(Path.join(directory, "*.slp")) do
        [path] ->
          case ExPhil.Data.Peppi.parse(path) do
            {:ok, replay} ->
              inputs =
                replay
                |> ExPhil.Data.Peppi.to_training_frames(player_port: 1, opponent_port: 2)
                |> Map.new(&{&1.game_state.frame, &1.controller})

              {:ok, verify(trace, inputs, response_delay, opts)}

            error ->
              {:retry, "unreadable replay #{path}: #{inspect(error)}"}
          end

        [] ->
          {:retry, "no replay in #{directory}"}

        paths ->
          {:ok, %{valid: false, reason: "ambiguous timing replays: #{inspect(paths)}"}}
      end

    case result do
      {:ok, timing} ->
        timing

      {:retry, _reason} when attempts > 0 ->
        Process.sleep(50)
        verify_directory(directory, trace, response_delay, attempts - 1, opts)

      {:retry, reason} ->
        %{valid: false, reason: reason}
    end
  end

  def verify(trace, causal_inputs, response_delay, opts \\ [])
      when is_integer(response_delay) and response_delay >= 0 do
    frames = MapSet.new(trace, & &1.frame)
    issued = Enum.filter(trace, &MapSet.member?(frames, &1.frame + response_delay))
    profile = Keyword.get(opts, :profile, :pipe_v2)
    sent = count(trace, causal_inputs, :sent, 0, profile)
    decisions = count(issued, causal_inputs, :issued, response_delay, profile)

    contiguous =
      trace
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.all?(fn [previous, current] -> current.frame == previous.frame + 1 end)

    %{
      verifier: :recorded_components_v2,
      profile: profile,
      valid:
        contiguous and MapSet.size(frames) == length(trace) and sent.count > 0 and
          decisions.count > 0 and sent.matches == sent.count and
          decisions.matches == decisions.count,
      contiguous: contiguous,
      expected_send_latency: 1,
      expected_decision_latency: response_delay + 1,
      unsent_tail_decisions: length(trace) - length(issued),
      sent: sent,
      decisions: decisions
    }
  end

  defp count(trace, inputs, field, shift, profile) do
    comparisons =
      Enum.map(trace, fn sample ->
        differences =
          case inputs[sample.frame + shift] do
            nil ->
              [%{component: :missing_frame}]

            recorded ->
              differences(expected_recording(Map.fetch!(sample, field), profile), recorded)
          end

        %{frame: sample.frame, differences: differences}
      end)

    failures = Enum.reject(comparisons, &(&1.differences == []))

    %{
      count: length(trace),
      matches: length(trace) - length(failures),
      mismatches: Enum.take(failures, 5)
    }
  end

  @doc "Expected Melee readback for quantized standard-pipe commands; legacy is explicit."
  def expected_recording(input, profile \\ :pipe_v2) when profile in [:pipe_v1, :pipe_v2] do
    controller = ControllerState.from_input(input)
    raw = round_even(controller.l_shoulder * 140)
    left = if profile == :pipe_v1, do: max(2 * raw - 255, 0), else: raw

    %{
      controller
      | main_stick: recorded_stick(controller.main_stick),
        c_stick: recorded_stick(controller.c_stick),
        l_shoulder: if(controller.button_l, do: 1.0, else: min(left, 140) / 140),
        r_shoulder: if(controller.button_r, do: 1.0, else: 0.0)
    }
  end

  defp recorded_stick(%{x: horizontal, y: vertical}) do
    horizontal = round_even((horizontal - 0.5) * 160)
    vertical = round_even((vertical - 0.5) * 160)
    radius = :math.sqrt(horizontal * horizontal + vertical * vertical)

    {horizontal, vertical} =
      if radius > 80,
        do: {trunc(horizontal * 80 / radius), trunc(vertical * 80 / radius)},
        else: {horizontal, vertical}

    normalize = fn axis -> if abs(axis) < 23, do: 0.5, else: axis / 160 + 0.5 end
    %{x: normalize.(horizontal), y: normalize.(vertical)}
  end

  defp round_even(value) do
    lower = floor(value)
    fraction = value - lower

    cond do
      fraction < 0.5 -> lower
      fraction > 0.5 -> lower + 1
      rem(lower, 2) == 0 -> lower
      true -> lower + 1
    end
  end

  defp differences(left, right) do
    buttons = [
      :button_a,
      :button_b,
      :button_x,
      :button_y,
      :button_z,
      :button_l,
      :button_r,
      :button_d_up
    ]

    analog = fn controller ->
      [
        controller.main_stick.x,
        controller.main_stick.y,
        controller.c_stick.x,
        controller.c_stick.y,
        controller.l_shoulder,
        controller.r_shoulder
      ]
    end

    digital_differences =
      for button <- buttons,
          Map.fetch!(left, button) != Map.fetch!(right, button),
          do: %{
            component: button,
            expected: Map.fetch!(left, button),
            actual: Map.fetch!(right, button)
          }

    names = [:main_x, :main_y, :c_x, :c_y, :left_trigger, :right_trigger]

    analog_differences =
      for {name, expected, actual} <- Enum.zip([names, analog.(left), analog.(right)]),
          abs(expected - actual) > 0.00001,
          do: %{component: name, expected: expected, actual: actual}

    digital_differences ++ analog_differences
  end
end
