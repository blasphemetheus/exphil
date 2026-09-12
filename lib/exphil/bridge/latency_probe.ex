defmodule ExPhil.Bridge.LatencyProbe do
  @moduledoc """
  Measures a harness's decision->application latency at game start instead
  of trusting a declared constant (INVARIANTS.md item 12, structural form).

  During the pre-GO countdown (frames -123..-1 the game polls pads but the
  characters cannot act) the probe sends ONE marker input — main stick held
  hard right for a single frame — and watches the frame on which the game
  reports that stick position back in the subject's `controller_state`.

      latency = frame the marker is reported on - frame it was sent on

  That number is exactly what training calls reaction delay + 1
  (`ExPhil.Eval.HarnessRung.latency(:training, k)`), so a runner asked for
  `--reaction-delay k` expects `k + 1` and refuses (or warns) on anything
  else. Pure state machine: the caller feeds frames and sends what `step/3`
  says; nothing here touches the bridge.

  ## Usage

      probe = LatencyProbe.new(expected: k + 1)
      {what, probe} = LatencyProbe.step(probe, game_state, player_port)
      # what :: :pass | {:send, input} | {:done, {:ok, latency} | {:mismatch, latency} | :unmeasured}

  `:pass` means "not my frame, send your own input"; `{:send, input}` means
  "send this instead"; `{:done, result}` fires once, on the frame the marker
  is seen (or the window expires), after which the probe is inert.
  """

  @marker_x 0.95
  @marker_threshold 0.8
  # Send the marker no earlier than this countdown frame (pads are polled
  # from -123; the first few frames can be dropped while the bridge settles)
  @default_first_frame -110
  # Give up if the marker has not appeared this many frames after the send
  @default_max_wait 20

  @type result :: {:ok, pos_integer()} | {:mismatch, pos_integer()} | :unmeasured
  @type t :: %__MODULE__{
          expected: pos_integer() | nil,
          first_frame: integer(),
          max_wait: pos_integer(),
          sent_at: integer() | nil,
          result: result() | nil
        }

  defstruct expected: nil, first_frame: @default_first_frame, max_wait: @default_max_wait, sent_at: nil, result: nil

  @doc """
  New probe. Options: `:expected` (latency the caller is counting on; nil =
  measure only), `:first_frame` (earliest countdown frame to send the marker
  on), `:max_wait` (frames to wait for it).
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      expected: Keyword.get(opts, :expected),
      first_frame: Keyword.get(opts, :first_frame, @default_first_frame),
      max_wait: Keyword.get(opts, :max_wait, @default_max_wait)
    }
  end

  @doc "The marker input (bridge input map) — main stick hard right, nothing else."
  @spec marker_input() :: map()
  def marker_input do
    %{main_stick: %{x: @marker_x, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, shoulder: 0.0, buttons: %{}}
  end

  @doc "Neutral input the probe sends on the frames after the marker while waiting."
  @spec neutral_input() :: map()
  def neutral_input do
    %{main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, shoulder: 0.0, buttons: %{}}
  end

  @doc "True once the probe has produced its result (it then always returns :pass)."
  @spec done?(t()) :: boolean()
  def done?(%__MODULE__{result: r}), do: r != nil

  @doc "The measured latency, or nil."
  @spec latency(t()) :: pos_integer() | nil
  def latency(%__MODULE__{result: {_, l}}), do: l
  def latency(_), do: nil

  @doc "Advance the probe with one game frame. See moduledoc for the return."
  @spec step(t(), map(), pos_integer()) :: {:pass | {:send, map()} | {:done, result()}, t()}
  def step(%__MODULE__{result: r} = probe, _gs, _port) when r != nil, do: {:pass, probe}

  def step(%__MODULE__{sent_at: nil} = probe, %{frame: frame}, _port) when is_integer(frame) do
    cond do
      # Too early, or the countdown is already over (GO at 0): nothing to do
      # yet / this game cannot be probed — leave the policy in control.
      frame < probe.first_frame ->
        {:pass, probe}

      frame >= -1 ->
        {{:done, :unmeasured}, %{probe | result: :unmeasured}}

      true ->
        {{:send, marker_input()}, %{probe | sent_at: frame}}
    end
  end

  def step(%__MODULE__{sent_at: sent} = probe, %{frame: frame} = gs, port) when is_integer(frame) do
    seen? = marker_seen?(gs, port)

    cond do
      seen? ->
        latency = frame - sent
        result = if probe.expected == nil or probe.expected == latency, do: {:ok, latency}, else: {:mismatch, latency}
        {{:done, result}, %{probe | result: result}}

      frame - sent > probe.max_wait ->
        {{:done, :unmeasured}, %{probe | result: :unmeasured}}

      true ->
        {{:send, neutral_input()}, probe}
    end
  end

  def step(probe, _gs, _port), do: {:pass, probe}

  @doc "One line for logs."
  @spec describe(t()) :: String.t()
  def describe(%__MODULE__{result: {:ok, l}, expected: e}),
    do: "latency #{l} frames measured (expected #{e || "any"}) — aligned"

  def describe(%__MODULE__{result: {:mismatch, l}, expected: e}),
    do: "latency #{l} frames measured but #{e} expected: this harness is #{if l < e, do: "FASTER", else: "slower"} than the policy's rung by #{abs(l - e)}"

  def describe(%__MODULE__{result: :unmeasured}), do: "latency not measured (marker never reported in the countdown window)"
  def describe(_), do: "latency probe pending"

  defp marker_seen?(%{players: players}, port) when is_map(players) do
    case players[port] do
      %{controller_state: %{main_stick: %{x: x}}} when is_number(x) -> x >= @marker_threshold
      _ -> false
    end
  end

  defp marker_seen?(_, _), do: false
end
