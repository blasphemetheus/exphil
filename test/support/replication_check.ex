defmodule ExPhil.Test.ReplicationCheck do
  @moduledoc """
  Configurable-strictness comparison of an expected vs. an emitted controller
  sequence, for overfit-replication correctness tests (see
  `docs/planning/HANDOFF.md` and `ExPhil.Test.ReplayFixtures.tech_fixture/2`).

  The idea (per Vlad Firoiu's advice): after a model is trained to *memorize* a
  short behavior like a multishine, decode its per-frame predictions back into
  `%ControllerState{}` values and compare that emitted sequence against the
  ground-truth sequence it was trained on. A memorized behavior it cannot
  reproduce means a **pipeline bug** (embedding / discretization / decode /
  sampling), not underfitting.

  This module is deliberately **pure** — it takes two `%ControllerState{}` lists
  and knows nothing about the model — so it is fully unit-testable without a GPU
  (see `test/exphil/replication_check_test.exs`).

  ## Strictness levels (`:strictness` option)

    * `:exact`    — every frame's emitted controller must match the expected one
      (buttons bitwise-equal; sticks equal to the same discretization bucket).
      Strictest; requires deterministic (argmax) sampling or legitimate stick
      stochasticity will fail it.
    * `:periodic` *(default)* — the shine *events* must recur at the expected
      period ± `:period_tolerance`, and the shine *count* must match ±
      `:count_tolerance`. Robust to a stray frame; still catches phase, timing,
      and decode bugs.
    * `:loose`    — the shine *rate* over the clip must be within
      `:count_tolerance`. Weakest; catches gross failure ("emits zero shines").

  ## Options

    * `:strictness`       — `:exact | :periodic | :loose` (default `:periodic`)
    * `:period_tolerance` — frames of slack when aligning shine events (default 1)
    * `:count_tolerance`  — allowed difference in shine count (default 1)
    * `:axis_buckets`     — stick discretization used for `:exact` equality
      (default 16, matching `ExPhil.Embeddings.Controller` default)

  ## Return

  `{:ok, diagnostic}` or `{:error, diagnostic}` where `diagnostic` is a map with
  at least `:level`, `:pass`, and `:message`, plus level-specific detail
  (expected/actual shine indices, inferred period, first mismatching frame, …).
  A rich diagnostic is the whole point: a bare boolean tells you nothing, but
  "expected shine every 8f, got every 8f offset by 3, first divergence at frame
  41" tells you it is a phase bug, not a decode bug.
  """

  alias ExPhil.Bridge.ControllerState

  @default_axis_buckets 16

  @type diagnostic :: %{required(:level) => atom(), required(:pass) => boolean(), optional(atom()) => term()}

  @doc "Run the configured strictness check over two controller sequences."
  @spec check([ControllerState.t()], [ControllerState.t()], keyword()) ::
          {:ok, diagnostic()} | {:error, diagnostic()}
  def check(expected, actual, opts \\ []) when is_list(expected) and is_list(actual) do
    case Keyword.get(opts, :strictness, :periodic) do
      :exact -> check_exact(expected, actual, opts)
      :periodic -> check_periodic(expected, actual, opts)
      :loose -> check_loose(expected, actual, opts)
      other -> {:error, %{level: other, pass: false, message: "unknown strictness #{inspect(other)}"}}
    end
  end

  @doc """
  Is this frame a shine input? Defined as **B held while the main stick is pushed
  down** (`main_stick.y < 0.5`). Matches how `tech_fixture(:multishine)` encodes
  the shine, so fixture and checker agree by construction.
  """
  @spec shine_frame?(ControllerState.t()) :: boolean()
  def shine_frame?(%ControllerState{button_b: true, main_stick: %{y: y}}) when y < 0.5, do: true
  def shine_frame?(_), do: false

  @doc "Indices (0-based) of the frames that are shine inputs."
  @spec shine_indices([ControllerState.t()]) :: [non_neg_integer()]
  def shine_indices(controllers) do
    controllers
    |> Enum.with_index()
    |> Enum.filter(fn {c, _i} -> shine_frame?(c) end)
    |> Enum.map(fn {_c, i} -> i end)
  end

  # --- :exact -------------------------------------------------------------

  defp check_exact(expected, actual, opts) do
    buckets = Keyword.get(opts, :axis_buckets, @default_axis_buckets)
    len_e = length(expected)
    len_a = length(actual)

    first_mismatch =
      Enum.zip(expected, actual)
      |> Enum.with_index()
      |> Enum.find(fn {{e, a}, _i} -> not controllers_equal?(e, a, buckets) end)

    cond do
      len_e != len_a ->
        {:error,
         %{
           level: :exact,
           pass: false,
           message: "length mismatch: expected #{len_e} frames, got #{len_a}",
           expected_frames: len_e,
           actual_frames: len_a
         }}

      first_mismatch != nil ->
        {{_e, _a}, idx} = first_mismatch
        {:error,
         %{
           level: :exact,
           pass: false,
           message: "first controller mismatch at frame #{idx}",
           first_mismatch_frame: idx
         }}

      true ->
        {:ok, %{level: :exact, pass: true, message: "all #{len_e} frames match exactly", frames: len_e}}
    end
  end

  # --- :periodic ----------------------------------------------------------

  defp check_periodic(expected, actual, opts) do
    period_tol = Keyword.get(opts, :period_tolerance, 1)
    count_tol = Keyword.get(opts, :count_tolerance, 1)

    exp_idx = shine_indices(expected)
    act_idx = shine_indices(actual)
    period = infer_period(exp_idx)

    # Each expected shine should have an actual shine within period_tolerance.
    unmatched =
      Enum.reject(exp_idx, fn e ->
        Enum.any?(act_idx, fn a -> abs(a - e) <= period_tol end)
      end)

    count_ok = abs(length(act_idx) - length(exp_idx)) <= count_tol
    align_ok = length(unmatched) <= count_tol

    base = %{
      level: :periodic,
      expected_period: period,
      expected_shines: exp_idx,
      actual_shines: act_idx,
      unmatched_expected: unmatched
    }

    if count_ok and align_ok do
      {:ok, Map.merge(base, %{pass: true, message: "shine pattern reproduced (period ~#{period}f, #{length(act_idx)}/#{length(exp_idx)} shines)"})}
    else
      msg =
        cond do
          not count_ok -> "shine count off: expected #{length(exp_idx)}, got #{length(act_idx)}"
          true -> "#{length(unmatched)} expected shine(s) had no actual shine within ±#{period_tol}f (first unmatched frame: #{List.first(unmatched)})"
        end

      {:error, Map.merge(base, %{pass: false, message: msg})}
    end
  end

  # --- :loose -------------------------------------------------------------

  defp check_loose(expected, actual, opts) do
    count_tol = Keyword.get(opts, :count_tolerance, 1)
    exp_n = length(shine_indices(expected))
    act_n = length(shine_indices(actual))
    diff = abs(act_n - exp_n)

    base = %{level: :loose, expected_shine_count: exp_n, actual_shine_count: act_n}

    if diff <= count_tol do
      {:ok, Map.merge(base, %{pass: true, message: "shine rate within tolerance (#{act_n} vs #{exp_n}, ±#{count_tol})"})}
    else
      {:error, Map.merge(base, %{pass: false, message: "shine rate out of tolerance: #{act_n} vs #{exp_n} (±#{count_tol})"})}
    end
  end

  # --- helpers ------------------------------------------------------------

  @doc false
  def controllers_equal?(%ControllerState{} = e, %ControllerState{} = a, buckets) do
    buttons_equal?(e, a) and
      bucket(e.main_stick.x, buckets) == bucket(a.main_stick.x, buckets) and
      bucket(e.main_stick.y, buckets) == bucket(a.main_stick.y, buckets) and
      bucket(e.c_stick.x, buckets) == bucket(a.c_stick.x, buckets) and
      bucket(e.c_stick.y, buckets) == bucket(a.c_stick.y, buckets)
  end

  defp buttons_equal?(e, a) do
    e.button_a == a.button_a and e.button_b == a.button_b and
      e.button_x == a.button_x and e.button_y == a.button_y and
      e.button_z == a.button_z and e.button_l == a.button_l and
      e.button_r == a.button_r and e.button_d_up == a.button_d_up
  end

  # Map a 0..1 stick value to a discrete bucket index (matches the uniform
  # discretization used for training targets in ExPhil.Embeddings.Controller).
  defp bucket(v, buckets) when is_number(v), do: round(v * buckets)

  # Infer the shine period as the median gap between consecutive shine frames.
  @doc false
  def infer_period([]), do: nil
  def infer_period([_]), do: nil

  def infer_period(indices) do
    gaps =
      indices
      |> Enum.sort()
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] -> b - a end)

    case Enum.sort(gaps) do
      [] -> nil
      sorted -> Enum.at(sorted, div(length(sorted), 2))
    end
  end
end
