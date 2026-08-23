defmodule ExPhil.Training.Utils do
  @moduledoc """
  Utility functions for training.
  """

  @doc """
  Ensures params are wrapped in Axon.ModelState.

  Axon's inference methods now require ModelState instead of plain maps.
  This helper provides backwards compatibility with existing checkpoints
  that store params as maps.

  ## Examples

      # Already ModelState - returned as-is
      iex> state = %Axon.ModelState{data: %{}, state: %{}}
      iex> Utils.ensure_model_state(state)
      %Axon.ModelState{data: %{}, state: %{}}

      # Plain map - wrapped in ModelState
      iex> Utils.ensure_model_state(%{"layer" => %{"kernel" => tensor}})
      %Axon.ModelState{data: %{"layer" => %{"kernel" => tensor}}, state: %{}}
  """
  @spec ensure_model_state(%Axon.ModelState{} | map()) :: %Axon.ModelState{}
  def ensure_model_state(%Axon.ModelState{} = state), do: state

  def ensure_model_state(map) when is_map(map) do
    %Axon.ModelState{data: map, state: %{}}
  end

  @doc """
  Build an Axon model with EXLA graph compilation when available.

  Without `compiler: EXLA`, each `pred_fn.(params, input)` call re-traces
  all Axon layer callbacks (~700-2800ms for typical models). With graph
  compilation, XLA compiles the graph once and caches it (~2-4ms per call).

  Falls back gracefully to uncompiled build when EXLA is not loaded.

  ## Examples

      {init_fn, pred_fn} = Utils.build_compiled(model)
      {init_fn, pred_fn} = Utils.build_compiled(model, mode: :inference)
  """
  @spec build_compiled(Axon.t(), keyword()) :: {function(), function()}
  def build_compiled(model, opts \\ []) do
    opts =
      if Code.ensure_loaded?(EXLA) do
        Keyword.put_new(opts, :compiler, EXLA)
      else
        opts
      end

    Axon.build(model, opts)
  end

  @doc """
  Persistent XLA executable cache options (JIT_WARMUP.md step 1):
  EXLA's `cache: path` serializes a compiled executable to disk, so a
  fresh process deserializes instead of recompiling. One file per
  jit'd function; weights are runtime arguments, so keying is
  architecture+shapes — and EXLA's own disk key (client, arg shapes,
  options) auto-invalidates with a warning on any mismatch, making a
  stale file cost at most the old compile-every-boot behavior.

  DEFAULT OFF (2026-08-24): the first live-game outing hung the
  inference process mid-game (counter frozen, inputs latched on
  down-B) with cached executables; the identical session with
  `EXPHIL_XLA_EXEC_CACHE=0` played normally — the deserialized-
  executable path is unsafe on this stack (xla 0.10 / exla 0.13 /
  5090) until bisected. `EXPHIL_XLA_EXEC_CACHE`: unset/`0` =
  disabled, `1` = `~/.cache/exphil/xla_exec`, anything else = the
  cache directory.
  """
  @spec xla_exec_cache(String.t(), term()) :: keyword()
  def xla_exec_cache(name, key_parts) do
    case System.get_env("EXPHIL_XLA_EXEC_CACHE", "0") do
      "0" ->
        []

      value ->
        base =
          if value == "1",
            do: Path.join(System.user_home!(), ".cache/exphil/xla_exec"),
            else: value

        if exec_cache_enabled_for?(name) do
          [cache: Path.join(base, "#{name}-#{:erlang.phash2(key_parts)}.exlaexec")]
        else
          []
        end
    end
  end

  # Bisect control (the cached-executable hang, JIT_WARMUP.md):
  # EXPHIL_XLA_EXEC_CACHE_ONLY="predict,heads" caches only the named
  # functions (prefix match, so "sampling" covers every fused
  # sampler); unset = all four sites. Function names: predict,
  # trunk_step, heads, sampling_*.
  defp exec_cache_enabled_for?(name) do
    case System.get_env("EXPHIL_XLA_EXEC_CACHE_ONLY") do
      nil ->
        true

      only ->
        only
        |> String.split(",")
        |> Enum.any?(&String.starts_with?(name, String.trim(&1)))
    end
  end
end
