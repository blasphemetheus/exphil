defmodule ExPhil.Agents.Decode do
  @moduledoc """
  The ONE source of live decode configuration (INVARIANTS.md item 7).

  Phase A (2026-09-09) collapsed four hand-built `sample_opts` lists
  (windowed / stateful-step / incremental-SSM / single-frame, plus warmup)
  into one builder. Phase B makes it a typed struct held on the agent
  state (`state.decode`), rebuilt ONLY by `from_state/1` at init and
  reconfigure, and turned into sampler options by `opts/3` at the
  decision. No path reads decode fields off the agent state directly any
  more — the struct is the contract, `opts/3` the only exit.

  History: the incremental path passed only deterministic + temperature
  (silently dropping mode-of-N, the critic selector, and hysteresis), and
  `reconfigure(style_tag:)` was a no-op because init and reconfigure
  resolved tags differently.
  """

  require Logger

  @enforce_keys [:deterministic, :temperature, :deterministic_buttons]
  defstruct deterministic: false,
            temperature: 1.0,
            deterministic_buttons: false,
            mode_of_n: nil,
            press_threshold: nil,
            release_threshold: nil

  @type t :: %__MODULE__{
          deterministic: boolean(),
          temperature: number() | map(),
          deterministic_buttons: boolean(),
          mode_of_n: pos_integer() | nil,
          press_threshold: number() | nil,
          release_threshold: number() | nil
        }

  @doc """
  Build the struct from the agent's configured fields. Called at init and
  after every reconfigure — the only constructor.
  """
  @spec from_state(map()) :: t()
  def from_state(state) do
    %__MODULE__{
      deterministic: state.deterministic || false,
      temperature: state.temperature || 1.0,
      deterministic_buttons: state.deterministic_buttons || false,
      mode_of_n: state.mode_of_n,
      press_threshold: state.press_threshold,
      release_threshold: state.release_threshold
    }
    |> validate!()
  end

  @doc "Sampler options for one decision: struct + per-call overrides + hysteresis memory."
  @spec opts(t(), map(), keyword()) :: keyword()
  def opts(%__MODULE__{} = d, state, per_call \\ []) do
    [
      deterministic: Keyword.get(per_call, :deterministic, d.deterministic),
      temperature: Keyword.get(per_call, :temperature, d.temperature),
      deterministic_buttons: Keyword.get(per_call, :deterministic_buttons, d.deterministic_buttons),
      mode_of_n: Keyword.get(per_call, :mode_of_n, d.mode_of_n),
      select_n: Keyword.get(per_call, :select_n),
      select_fn: Keyword.get(per_call, :select_fn),
      press_threshold: d.press_threshold,
      release_threshold: d.release_threshold,
      prev_buttons: state.last_action && state.last_action[:buttons]
    ]
  end

  @doc """
  Decode options for a decision. Uses `state.decode` when the agent carries
  one (the phase-B path); falls back to building from fields for callers
  that construct bare state maps (tests, probes).
  """
  @spec sample_opts(map(), keyword()) :: keyword()
  def sample_opts(state, per_call \\ []) do
    decode =
      case Map.get(state, :decode) do
        %__MODULE__{} = d -> d
        _ -> from_state(state)
      end

    opts(decode, state, per_call)
  end

  @doc """
  Resolve `:style_id` / `:style_tag` opts to a name id. An explicit
  integer `:style_id` wins; a `:style_tag` resolves through the
  `:player_registry` JSON (training's --player-registry vocab); anything
  unresolvable is id 0 (the unconditioned bucket) with a warning.
  Used by BOTH `Agent.init/1` and `reconfigure/2`.
  """
  @spec resolve_style_id(keyword(), String.t() | nil) :: non_neg_integer()
  def resolve_style_id(opts, registry_path \\ nil) do
    path = Keyword.get(opts, :player_registry) || registry_path

    case {Keyword.get(opts, :style_id), Keyword.get(opts, :style_tag)} do
      {id, _} when is_integer(id) ->
        id

      {nil, tag} when is_binary(tag) ->
        with p when is_binary(p) <- path,
             {:ok, registry} <- ExPhil.Training.PlayerRegistry.from_json(p),
             id when is_integer(id) <- ExPhil.Training.PlayerRegistry.get_id(registry, tag) do
          id
        else
          _ ->
            Logger.warning(
              "[agent] style_tag #{inspect(tag)} not resolvable " <>
                "(missing/invalid :player_registry or unknown tag) — using style_id 0"
            )

            0
        end

      _ ->
        0
    end
  end

  # Loud at construction, not at the 60th frame.
  defp validate!(%__MODULE__{} = d) do
    temp_ok? =
      case d.temperature do
        t when is_number(t) -> t > 0
        %{} -> true
        _ -> false
      end

    unless temp_ok?, do: raise(ArgumentError, "Decode: temperature must be > 0 or a per-head map, got #{inspect(d.temperature)}")

    if is_number(d.press_threshold) and is_number(d.release_threshold) and d.release_threshold > d.press_threshold do
      raise ArgumentError,
            "Decode: release_threshold (#{d.release_threshold}) must be <= press_threshold (#{d.press_threshold})"
    end

    if d.mode_of_n != nil and (not is_integer(d.mode_of_n) or d.mode_of_n < 1) do
      raise ArgumentError, "Decode: mode_of_n must be a positive integer or nil, got #{inspect(d.mode_of_n)}"
    end

    d
  end
end
