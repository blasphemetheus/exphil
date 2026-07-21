defmodule ExPhil.Interp.DeficitReport do
  @moduledoc """
  P5 deficit report: per-checkpoint scorecard of probe accuracies across
  the ground-truth feature dictionary. The lowest features, judged
  AGAINST THE INPUT FLOOR, are missing knowledge — each deficit maps to a
  rollout recipe (weak tech-direction -> more tech_random games; no
  offstage representation -> edgeguard scenarios; ...). This is the
  harness that lets representation gaps CHOOSE what to farm.

  Baked-in P1 hygiene (the saturation postmortem):
  - **Input floor**: every feature is also probed on the raw last-frame
    embedding (`Activations.input_trunk/1`). A trained trunk BELOW the
    floor has compressed the feature away; a trunk at ceiling WITH the
    floor at ceiling means the target leaks from the input — not
    knowledge.
  - **Random-init control**: same architecture, fresh params — what the
    windowed input + a random projection give for free.
  - **Shuffled-label control** (Hewitt-Liang) on one feature per run:
    above-chance means the probe, not the representation, does the work.

  Also emits persona-vector-style **direction monitoring** (Anthropic
  Aug 2025, adopted 2026-07-20): mean/p95 projection of trunk
  activations onto known bad-habit directions (e.g. the shield-lock
  steer vector) — the relapse dashboard across DAgger rounds.

  Cross-checkpoint (or cross-ARCHITECTURE — GRU vs Mamba trunks over the
  same replays) comparison falls out of running this on several policies
  and diffing the scorecards.
  """

  alias ExPhil.Interp.{Activations, Probe, Steering}
  alias ExPhil.Training.Output

  @doc """
  Run the report for one policy over a replay set.

  Options:
  - `:eval_fraction` - fraction of replays (by position, from the end)
    held out for probe eval (default 1/3, min 1)
  - `:directions` - list of `{name, steer_vector_path}` for projection
    monitoring (default: `[]`)
  - `:controls` - include random-init + shuffled controls (default true;
    the input floor always runs — it is the interpretive baseline)

  Returns `%{policy, suite, floor_suite, random_suite, shuffled_control,
  projections, deficits}`.
  """
  def run(policy_path, replay_paths, opts \\ []) do
    directions = Keyword.get(opts, :directions, [])
    controls? = Keyword.get(opts, :controls, true)

    trunk = Activations.load_trunk(policy_path)
    capture = Activations.capture(trunk, replay_paths)
    split = split(capture, replay_paths, opts)

    Output.puts("  #{Path.basename(policy_path)}: #{Nx.axis_size(capture.activations, 0)} rows")

    suite = run_suites(split)

    floor_trunk =
      Activations.input_trunk(
        window: trunk.window,
        embed_size: embed_size_of(trunk),
        use_prev_action: Map.get(trunk.config, :use_prev_action, true)
      )

    floor_capture = Activations.capture(floor_trunk, replay_paths)
    floor_suite = run_suites(split(floor_capture, replay_paths, opts))

    {random_suite, shuffled} =
      if controls? do
        random_trunk = Activations.load_trunk(policy_path, init: :random)
        random_capture = Activations.capture(random_trunk, replay_paths)
        random_suite = run_suites(split(random_capture, replay_paths, opts))

        # One shuffled control on a mid-difficulty memory target
        shuffled = Probe.shuffled_control(split, :opp_damaged_recent, 2)
        {random_suite, %{feature: :opp_damaged_recent, result: drop_params(shuffled)}}
      else
        {nil, nil}
      end

    projections =
      Map.new(directions, fn {name, path} ->
        v = Steering.load!(path).v
        proj = capture.activations |> Nx.as_type(:f32) |> Nx.dot(v)
        abs_proj = Nx.abs(proj)

        {name,
         %{
           mean: proj |> Nx.mean() |> Nx.to_number() |> Float.round(4),
           mean_abs: abs_proj |> Nx.mean() |> Nx.to_number() |> Float.round(4),
           p95_abs: percentile(abs_proj, 0.95)
         }}
      end)

    %{
      policy: Path.basename(policy_path),
      suite: suite,
      floor_suite: floor_suite,
      random_suite: random_suite,
      shuffled_control: shuffled,
      projections: projections,
      deficits: deficits(suite, floor_suite)
    }
  end

  @doc """
  Deficits: features ranked by trunk-minus-floor balanced accuracy
  (ascending). Negative delta = the trunk KNOWS LESS than its own input —
  training compressed the feature away (the P1 finding, now per-feature
  and actionable). Saturated features (floor >= 0.95) are excluded: they
  leak from the input and say nothing about the trunk.
  """
  def deficits(suite, floor_suite) do
    suite
    |> Enum.flat_map(fn {feature, r} ->
      floor = floor_suite[feature]

      cond do
        is_nil(r.balanced_accuracy) or is_nil(floor) or is_nil(floor.balanced_accuracy) -> []
        floor.balanced_accuracy >= 0.95 -> []
        true -> [%{feature: feature, trunk: r.balanced_accuracy, floor: floor.balanced_accuracy,
                   delta: Float.round(r.balanced_accuracy - floor.balanced_accuracy, 4)}]
      end
    end)
    |> Enum.sort_by(& &1.delta)
  end

  @doc "Format one report as an aligned text table."
  def format(report) do
    header = "  feature                      trunk   floor   delta"

    rows =
      report.deficits
      |> Enum.map(fn d ->
        "  #{String.pad_trailing(to_string(d.feature), 27)}#{fmt(d.trunk)}   #{fmt(d.floor)}   #{fmt(d.delta)}"
      end)

    proj_rows =
      Enum.map(report.projections, fn {name, p} ->
        "  direction #{name}: mean=#{p.mean} mean|.|=#{p.mean_abs} p95|.|=#{p.p95_abs}"
      end)

    Enum.join([report.policy, header] ++ rows ++ proj_rows, "\n")
  end

  # -- internals -------------------------------------------------------------

  defp run_suites(split) do
    v1 = Probe.suite(split)
    v2 = Probe.suite_v2(split)

    Map.merge(v1, v2) |> Map.new(fn {k, r} -> {k, drop_params(r)} end)
  end

  defp split(capture, replay_paths, opts) do
    frac = Keyword.get(opts, :eval_fraction, 1 / 3)
    n = length(replay_paths)
    n_eval = max(trunc(n * frac), 1)
    eval_positions = Enum.to_list((n - n_eval)..(n - 1))

    Probe.split_by_replay(capture, eval_positions)
  end

  defp drop_params(result), do: Map.delete(result, :params)

  defp embed_size_of(trunk) do
    # input_trunk needs the raw embed width, not the hidden width
    Map.get(trunk.config, :embed_size, 288)
  end

  defp percentile(t, p) do
    sorted = Nx.sort(t)
    n = Nx.axis_size(sorted, 0)
    idx = min(trunc(p * n), n - 1)
    sorted |> Nx.slice_along_axis(idx, 1) |> Nx.squeeze() |> Nx.to_number() |> Float.round(4)
  end

  defp fmt(nil), do: "  nil"
  defp fmt(x), do: x |> Float.round(3) |> :erlang.float_to_binary(decimals: 3) |> String.pad_leading(6)
end
