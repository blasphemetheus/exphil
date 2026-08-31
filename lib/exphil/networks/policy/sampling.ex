defmodule ExPhil.Networks.Policy.Sampling do
  @moduledoc """
  Action sampling functions for policy networks.

  Provides sampling methods for autoregressive action generation:
  - Button sampling (independent Bernoulli)
  - Stick/shoulder sampling (categorical with temperature)
  - Confidence scoring for action predictions

  ## Usage

      # Sample from policy output
      actions = Sampling.sample(params, predict_fn, state,
        temperature: 1.0,
        deterministic: false
      )

      # Get confidence scores
      confidence = Sampling.compute_confidence(actions)

  ## Sampling Methods

  - **Buttons**: Independent Bernoulli sampling from sigmoid probabilities
  - **Sticks/Shoulder**: Categorical sampling using Gumbel-max trick
  - **Temperature**: Controls exploration (higher = more random)
  - **Deterministic**: Uses argmax instead of sampling

  ## See Also

  - `ExPhil.Networks.Policy` - Main policy module
  - `ExPhil.Networks.Policy.Loss` - Loss computation
  """

  alias ExPhil.Training.Utils

  import Nx.Defn

  @doc """
  Sample actions from the policy.

  This performs autoregressive sampling, where each controller component
  is sampled conditioned on previously sampled components.

  ## Options
    - `:temperature` - Softmax temperature for exploration (default: 1.0). A
      number applies to every categorical head (buttons stay raw, as before);
      a map applies per head — `%{buttons: t, main_x: t, main_y: t, c_x: t,
      c_y: t, shoulder: t}` with `:main`/`:c` group shorthands for x+y.
    - `:deterministic` - If true, use argmax instead of sampling (default: false)
    - `:axis_buckets` - Number of stick buckets (default: 16)
    - `:shoulder_buckets` - Number of shoulder buckets (default: 4)
    - `:press_threshold` / `:release_threshold` - Button hysteresis for the
      argmax button modes (`:deterministic` or `:deterministic_buttons`).
      A button turns ON above `press_threshold` and OFF below
      `release_threshold`; in between it keeps its previous state (from
      `:prev_buttons`). Melee registers inputs on press EDGES, so a flat 0.5
      cut both drops borderline presses and locks held buttons (a held X is
      not a jump). Typical: press 0.6, release 0.4.
    - `:prev_buttons` - previously emitted button tensor (shape [1, buttons]);
      nil (e.g. first frame) applies `press_threshold` everywhere.
    - `:mode_of_n` - integer N >= 2: draw N joint samples from the same
      logits (one forward) and return the most frequent one (ties -> first).
      Critic-free Best-of-N; ignored when `:deterministic`.
  """
  @spec sample(map(), function(), Nx.Tensor.t(), keyword()) :: map()
  def sample(params, predict_fn, state, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)

    # Forward pass to get all logits
    {buttons_logits, main_x_logits, main_y_logits, c_x_logits, c_y_logits, shoulder_logits} =
      logits_tuple = predict_fn.(Utils.ensure_model_state(params), state)

    # Sample ALL heads (+ confidence) in ONE compiled program. Doing this with
    # per-head eager Nx ops costs >100ms per decision (dozens of separate XLA
    # dispatches); fused it is ~1ms. See scripts/profile_agent_inference.exs.
    deterministic_buttons = Keyword.get(opts, :deterministic_buttons, false)

    # Mode-of-N (2026-08-29, eval_runs/0829_critic/RESULTS.md): draw N joint
    # samples from the SAME logits and play the most frequent one. Offline it
    # recovers ~28% of the sampling->oracle gap on both corpora (pass@1
    # 14.9 -> 22.9 in-distribution) with no critic and no retrain, and beat
    # the learned linear selector. One trunk/heads forward; the N draws are
    # a single fused kernel on batch-tiled logits (new JIT shape per N).
    mode_of_n = Keyword.get(opts, :mode_of_n)

    {buttons, main_x, main_y, c_x, c_y, shoulder, conf} =
      cond do
        deterministic ->
          jitted(:fused_det, &fused_sample_deterministic/1).(logits_tuple)

        is_integer(mode_of_n) and mode_of_n > 1 ->
          key = Nx.Random.key(:erlang.unique_integer([:positive]))

          tiled =
            logits_tuple
            |> Tuple.to_list()
            |> Enum.map(&Nx.tile(&1, [mode_of_n | List.duplicate(1, Nx.rank(&1) - 1)]))
            |> List.to_tuple()

          {b, mx, my, cx, cy, sh, conf} =
            jitted(:fused_stoch, &fused_sample_stochastic/3).(
              tiled,
              key,
              temperature_tuple(temperature)
            )

          i = mode_index(b, mx, my, cx, cy, sh)
          row = &Nx.slice_along_axis(&1, i, 1, axis: 0)
          {row.(b), row.(mx), row.(my), row.(cx), row.(cy), row.(sh), conf}

        true ->
          key = Nx.Random.key(:erlang.unique_integer([:positive]))

          jitted(:fused_stoch, &fused_sample_stochastic/3).(
            logits_tuple,
            key,
            temperature_tuple(temperature)
          )
      end

    # Mixed decode: argmax the buttons while sticks keep sampling — kills
    # stray rare-button rolls (taunts) without argmax's modal lock on
    # movement. One tiny eager op on a [1,8] tensor.
    buttons =
      if deterministic_buttons and not deterministic do
        Nx.greater(Nx.sigmoid(buttons_logits), 0.5)
      else
        buttons
      end

    buttons =
      apply_hysteresis(buttons, buttons_logits, deterministic or deterministic_buttons, opts)

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder,
      # Precomputed confidence scalars (tensors) — see compute_confidence/1
      confidence_raw: conf,
      # Also include logits for loss computation
      logits: %{
        buttons: buttons_logits,
        main_x: main_x_logits,
        main_y: main_y_logits,
        c_x: c_x_logits,
        c_y: c_y_logits,
        shoulder: shoulder_logits
      }
    }
  end

  @doc """
  Sample from a policy with the TRUE autoregressive head
  (AUTOREGRESSIVE_HEAD_PLAN §3): components are drawn sequentially, each
  conditioned on the samples before it via the residual stream.

  `trunk_predict_fn` maps `(params, state)` to `[batch, hidden]` trunk
  features (built from `Policy.build_temporal_trunk/1` — the AR head math
  is replayed here from the exported `ar_*` params, mirroring
  `Heads.build_autoregressive_head/2` layer names exactly).

  Supports the same opts as `sample/4` (per-head `:temperature`,
  `:deterministic`, `:deterministic_buttons`, hysteresis thresholds,
  `:mode_of_n`). Button decode modifications (argmax / hysteresis) are
  applied BEFORE conditioning, so downstream components see the buttons
  actually sent to the game.
  """
  @spec sample_autoregressive(map(), function(), Nx.Tensor.t(), keyword()) :: map()
  def sample_autoregressive(params, trunk_predict_fn, state, opts \\ []) do
    features = trunk_predict_fn.(Utils.ensure_model_state(params), state)
    sample_autoregressive_from_features(params, features, opts)
  end

  @doc """
  Autoregressive sampling from precomputed trunk features `[batch, hidden]`
  (the stateful-step path's entry point). See `sample_autoregressive/4`.
  """
  @spec sample_autoregressive_from_features(map(), Nx.Tensor.t(), keyword()) :: map()
  def sample_autoregressive_from_features(params, features, opts \\ []) do
    temperature = Keyword.get(opts, :temperature, 1.0)
    deterministic = Keyword.get(opts, :deterministic, false)
    deterministic_buttons = Keyword.get(opts, :deterministic_buttons, false)
    mode_of_n = Keyword.get(opts, :mode_of_n)

    head = ar_head_params(params)
    argmax_buttons? = deterministic or deterministic_buttons
    n = if is_integer(mode_of_n) and mode_of_n > 1 and not deterministic, do: mode_of_n, else: 1
    temps = temperature_tuple(temperature)

    # Hysteresis config as tensors (applied BEFORE conditioning, inside the
    # fused kernel — press == release == 0.5 degrades to the plain 0.5 cut).
    press = Keyword.get(opts, :press_threshold)
    release = Keyword.get(opts, :release_threshold)
    prev = Keyword.get(opts, :prev_buttons)
    use_hyst? = argmax_buttons? and is_number(press) and is_number(release)
    press_t = Nx.tensor(if(use_hyst?, do: press * 1.0, else: 0.5), type: :f32)
    release_t = Nx.tensor(if(use_hyst?, do: release * 1.0, else: 0.5), type: :f32)

    {prev_t, has_prev} =
      case {use_hyst?, prev} do
        {true, %Nx.Tensor{} = p} -> {Nx.as_type(p, :u8), Nx.tensor(1, type: :u8)}
        _ -> {Nx.broadcast(Nx.tensor(0, type: :u8), {1, 8}), Nx.tensor(0, type: :u8)}
      end

    {buttons, {mx, my, cx, cy, sh, mx_l, my_l, cx_l, cy_l, sh_l, conf}, b_l} =
      cond do
        n > 1 ->
          # Mode-of-N (instrument only): two-stage path, batch-tiled.
          {r0, b_l} = jitted(:ar_stage1, &ar_stage1/2).(head, features)
          {r0n, b_ln} = {Nx.tile(r0, [n, 1]), Nx.tile(b_l, [n, 1])}
          {t_b, _, _, _, _, _} = temps
          key = Nx.Random.key(:erlang.unique_integer([:positive]))
          {u, _} = Nx.Random.uniform(key, shape: Nx.shape(b_ln))
          buttons = Nx.less(u, Nx.sigmoid(Nx.divide(b_ln, t_b)))
          key = Nx.Random.key(:erlang.unique_integer([:positive]))

          rest =
            jitted(:ar_stage2, &ar_stage2_stochastic/6).(
              head,
              r0n,
              Nx.as_type(buttons, :f32),
              b_ln,
              key,
              temps
            )

          {buttons, rest, b_l}

        deterministic ->
          # Fully fused: one XLA program per decision (50% staleness at 60Hz
          # with the two-stage + eager-glue path, 2026-08-30 AR bracket).
          {b, rest, b_l} =
            jitted(:ar_full_det, &ar_full_deterministic/5).(
              head,
              features,
              press_t,
              release_t,
              {prev_t, has_prev}
            )

          {b, rest, b_l}

        argmax_buttons? ->
          key = Nx.Random.key(:erlang.unique_integer([:positive]))

          jitted(:ar_full_mixed, &ar_full_mixed/7).(
            head,
            features,
            key,
            temps,
            press_t,
            release_t,
            {prev_t, has_prev}
          )

        true ->
          key = Nx.Random.key(:erlang.unique_integer([:positive]))
          jitted(:ar_full_stoch, &ar_full_stochastic/4).(head, features, key, temps)
      end

    # Mode-of-N vote on the joint action (instrument only — disqualified
    # for play, same as the independent path)
    {buttons, mx, my, cx, cy, sh, mx_l, my_l, cx_l, cy_l, sh_l} =
      if n > 1 do
        i = mode_index(buttons, mx, my, cx, cy, sh)
        row = &Nx.slice_along_axis(&1, i, 1, axis: 0)

        {row.(buttons), row.(mx), row.(my), row.(cx), row.(cy), row.(sh), row.(mx_l), row.(my_l),
         row.(cx_l), row.(cy_l), row.(sh_l)}
      else
        {buttons, mx, my, cx, cy, sh, mx_l, my_l, cx_l, cy_l, sh_l}
      end

    %{
      buttons: buttons,
      main_x: mx,
      main_y: my,
      c_x: cx,
      c_y: cy,
      shoulder: sh,
      confidence_raw: conf,
      # NOTE: categorical logits are CONDITIONAL on the sampled prefix
      # (document in interp readers; B3 entropies become conditional
      # entropies, which is the right thing)
      logits: %{
        buttons: b_l,
        main_x: mx_l,
        main_y: my_l,
        c_x: cx_l,
        c_y: cy_l,
        shoulder: sh_l
      }
    }
  end

  # Extract the AR head parameter subtree ("ar_*" layers) from a params
  # map or Axon.ModelState — mirrors Heads.build_autoregressive_head names.
  defp ar_head_params(params) do
    data =
      case params do
        %Axon.ModelState{data: d} -> d
        %{data: d} when is_map(d) -> d
        m when is_map(m) -> m
      end

    head = Map.filter(data, fn {k, _v} -> is_binary(k) and String.starts_with?(k, "ar_") end)

    if map_size(head) == 0 do
      raise ArgumentError,
            "no ar_* head params found — sample_autoregressive needs a checkpoint " <>
              "trained with head: :autoregressive"
    end

    head
  end

  # --- AR head math (defn mirrors of Heads.build_autoregressive_head) ---

  defnp ar_dense(x, layer) do
    Nx.dot(x, layer["kernel"]) |> Nx.add(layer["bias"])
  end

  # NOTE: layer maps (not a name string) — defn args must be tensors/containers
  defnp ar_component(r, hidden_layer, logits_layer) do
    r
    |> ar_dense(hidden_layer)
    |> Nx.max(0)
    |> ar_dense(logits_layer)
  end

  defnp ar_stage1(head, features) do
    r0 = ar_dense(features, head["ar_residual_proj"])
    b_l = ar_component(r0, head["ar_buttons_hidden"], head["ar_buttons_logits"])
    {r0, b_l}
  end

  defnp ar_stage2_stochastic(head, r0, buttons_f32, b_l, key, {_t_b, t_mx, t_my, t_cx, t_cy, t_sh}) do
    r1 = r0 + Nx.dot(buttons_f32, head["ar_buttons_embed"]["kernel"])

    mx_l = ar_component(r1, head["ar_main_x_hidden"], head["ar_main_x_logits"])
    {mx, key} = gumbel_argmax(mx_l, key, t_mx)
    r2 = r1 + Nx.take(head["ar_main_x_embed"]["kernel"], mx)

    my_l = ar_component(r2, head["ar_main_y_hidden"], head["ar_main_y_logits"])
    {my, key} = gumbel_argmax(my_l, key, t_my)
    r3 = r2 + Nx.take(head["ar_main_y_embed"]["kernel"], my)

    cx_l = ar_component(r3, head["ar_c_x_hidden"], head["ar_c_x_logits"])
    {cx, key} = gumbel_argmax(cx_l, key, t_cx)
    r4 = r3 + Nx.take(head["ar_c_x_embed"]["kernel"], cx)

    cy_l = ar_component(r4, head["ar_c_y_hidden"], head["ar_c_y_logits"])
    {cy, key} = gumbel_argmax(cy_l, key, t_cy)
    r5 = r4 + Nx.take(head["ar_c_y_embed"]["kernel"], cy)

    sh_l = ar_component(r5, head["ar_shoulder_hidden"], head["ar_shoulder_logits"])
    {sh, _key} = gumbel_argmax(sh_l, key, t_sh)

    {mx, my, cx, cy, sh, mx_l, my_l, cx_l, cy_l, sh_l,
     confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  defnp ar_stage2_deterministic(head, r0, buttons_f32, b_l) do
    r1 = r0 + Nx.dot(buttons_f32, head["ar_buttons_embed"]["kernel"])

    mx_l = ar_component(r1, head["ar_main_x_hidden"], head["ar_main_x_logits"])
    mx = Nx.argmax(mx_l, axis: -1)
    r2 = r1 + Nx.take(head["ar_main_x_embed"]["kernel"], mx)

    my_l = ar_component(r2, head["ar_main_y_hidden"], head["ar_main_y_logits"])
    my = Nx.argmax(my_l, axis: -1)
    r3 = r2 + Nx.take(head["ar_main_y_embed"]["kernel"], my)

    cx_l = ar_component(r3, head["ar_c_x_hidden"], head["ar_c_x_logits"])
    cx = Nx.argmax(cx_l, axis: -1)
    r4 = r3 + Nx.take(head["ar_c_x_embed"]["kernel"], cx)

    cy_l = ar_component(r4, head["ar_c_y_hidden"], head["ar_c_y_logits"])
    cy = Nx.argmax(cy_l, axis: -1)
    r5 = r4 + Nx.take(head["ar_c_y_embed"]["kernel"], cy)

    sh_l = ar_component(r5, head["ar_shoulder_hidden"], head["ar_shoulder_logits"])
    sh = Nx.argmax(sh_l, axis: -1)

    {mx, my, cx, cy, sh, mx_l, my_l, cx_l, cy_l, sh_l,
     confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  # --- fully fused single-decision variants (buttons decode in-kernel) ---

  # Argmax buttons with hysteresis: threshold = release while held, press
  # while up; no prev (first frame) -> press everywhere. press==release==0.5
  # is the plain deterministic cut.
  defnp ar_buttons_argmax(b_l, press, release, {prev, has_prev}) do
    probs = Nx.sigmoid(b_l)
    held = Nx.greater(prev, 0)
    thr = Nx.select(Nx.greater(has_prev, 0), Nx.select(held, release, press), press)
    Nx.greater(probs, thr)
  end

  defnp ar_full_deterministic(head, features, press, release, prev_pair) do
    {r0, b_l} = ar_stage1(head, features)
    buttons = ar_buttons_argmax(b_l, press, release, prev_pair)
    rest = ar_stage2_deterministic(head, r0, Nx.as_type(buttons, :f32), b_l)
    {buttons, rest, b_l}
  end

  defnp ar_full_mixed(head, features, key, temps, press, release, prev_pair) do
    {r0, b_l} = ar_stage1(head, features)
    buttons = ar_buttons_argmax(b_l, press, release, prev_pair)
    rest = ar_stage2_stochastic(head, r0, Nx.as_type(buttons, :f32), b_l, key, temps)
    {buttons, rest, b_l}
  end

  defnp ar_full_stochastic(head, features, key, temps) do
    {r0, b_l} = ar_stage1(head, features)
    {t_b, _, _, _, _, _} = temps
    probs = Nx.sigmoid(b_l / t_b)
    {u, key} = Nx.Random.uniform(key, shape: Nx.shape(probs))
    buttons = Nx.less(u, probs)
    rest = ar_stage2_stochastic(head, r0, Nx.as_type(buttons, :f32), b_l, key, temps)
    {buttons, rest, b_l}
  end

  # Index of the most frequent JOINT action among N draws (buttons [N, 8],
  # heads [N]); ties break to the first occurrence, matching the offline
  # mode-of-N in scripts/interp_bestofn.exs. Small host-side work on N rows.
  @doc false
  def mode_index(buttons, mx, my, cx, cy, sh) do
    keys =
      Enum.zip([
        Nx.to_list(Nx.as_type(buttons, :u8)),
        Nx.to_list(mx),
        Nx.to_list(my),
        Nx.to_list(cx),
        Nx.to_list(cy),
        Nx.to_list(sh)
      ])

    counts = Enum.frequencies(keys)

    {_key, idx} =
      keys
      |> Enum.with_index()
      |> Enum.max_by(fn {k, _} -> counts[k] end)

    idx
  end

  @doc """
  Button hysteresis: per-button threshold depends on the button's previous
  state — `:release_threshold` while held, `:press_threshold` while up.

  Applied only to argmax button modes (stochastic sampling has its own
  dynamics); returns `buttons` unchanged unless both thresholds are set.
  Eager ops on a [1, buttons] tensor; cost is negligible next to inference.
  """
  @spec apply_hysteresis(Nx.Tensor.t(), Nx.Tensor.t(), boolean(), keyword()) :: Nx.Tensor.t()
  def apply_hysteresis(buttons, buttons_logits, argmax_buttons?, opts) do
    press = Keyword.get(opts, :press_threshold)
    release = Keyword.get(opts, :release_threshold)
    prev = Keyword.get(opts, :prev_buttons)

    if argmax_buttons? and is_number(press) and is_number(release) do
      probs = Nx.sigmoid(buttons_logits)

      case prev do
        nil ->
          Nx.greater(probs, press)

        prev ->
          held = Nx.greater(Nx.as_type(prev, :u8), 0)
          thresholds = Nx.select(held, release, press)
          Nx.greater(probs, thresholds)
      end
    else
      buttons
    end
  end

  # --- Per-head temperature resolution ---

  # Heads in the fused stochastic kernel, in tuple order. Buttons were
  # historically sampled raw (no temperature); the map form can now temper
  # them too (INTERP_GEN_V1 G1: buttons are the highest-entropy head and want
  # the coldest T).
  @temp_heads [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

  defp temperature_tuple(temperature) do
    temps = resolve_temperatures(temperature)
    f = fn v -> Nx.tensor(v, type: :f32) end

    {f.(temps.buttons), f.(temps.main_x), f.(temps.main_y), f.(temps.c_x), f.(temps.c_y),
     f.(temps.shoulder)}
  end

  # Scalar (backward-compatible): buttons stay raw (T=1.0), every categorical
  # head shares the scalar — bit-identical to the pre-per-head fused path.
  @doc false
  def resolve_temperatures(temperature) when is_number(temperature) do
    %{buttons: 1.0, main_x: temperature, main_y: temperature, c_x: temperature,
      c_y: temperature, shoulder: temperature}
  end

  # Map: group shorthands :main (x+y) and :c (x+y) expand first; explicit
  # per-head keys win; any head left unspecified defaults to 1.0 (raw).
  @doc false
  def resolve_temperatures(map) when is_map(map) do
    base = %{buttons: 1.0, main_x: 1.0, main_y: 1.0, c_x: 1.0, c_y: 1.0, shoulder: 1.0}

    with_groups =
      base
      |> put_group(:main, map[:main])
      |> put_group(:c, map[:c])

    Enum.reduce(@temp_heads, with_groups, fn head, acc ->
      case map[head] do
        t when is_number(t) -> Map.put(acc, head, t)
        _ -> acc
      end
    end)
  end

  defp put_group(acc, :main, t) when is_number(t), do: %{acc | main_x: t, main_y: t}
  defp put_group(acc, :c, t) when is_number(t), do: %{acc | c_x: t, c_y: t}
  defp put_group(acc, _group, _t), do: acc

  import Nx.Defn

  # Nx's default defn options are EMPTY, so a bare defn call runs on the
  # pure-Elixir evaluator (~120ms here). Jit explicitly with EXLA and cache
  # the compiled closure. (Same gotcha as scripts/test_fused_kernels.exs.)
  defp jitted(name, fun) do
    pt_key = {__MODULE__, name}

    case :persistent_term.get(pt_key, nil) do
      nil ->
        compiled =
          if Code.ensure_loaded?(EXLA) do
            # Disk-cached executable (JIT_WARMUP.md step 1): the fused
            # samplers were the dominant residual compile after the
            # predict/trunk/heads caches landed.
            Nx.Defn.jit(
              fun,
              [compiler: EXLA] ++
                ExPhil.Training.Utils.xla_exec_cache("sampling_#{name}", {__MODULE__, name})
            )
          else
            Nx.Defn.jit(fun)
          end

        :persistent_term.put(pt_key, compiled)
        compiled

      compiled ->
        compiled
    end
  end

  # --- Fused sampling: one XLA program for all six heads + confidence ---

  # temperatures: {t_buttons, t_main_x, t_main_y, t_c_x, t_c_y, t_shoulder},
  # one f32 scalar per head. Buttons are now temperature-scaled too (sigmoid
  # of logits/t) so a cold button temperature can tame the 69%-of-uniform
  # button tail (INTERP_GEN_V1 G1); at t_buttons=1.0 this is bit-identical to
  # the pre-per-head raw-Bernoulli path.
  defnp fused_sample_stochastic(
         {b_l, mx_l, my_l, cx_l, cy_l, sh_l},
         key,
         {t_b, t_mx, t_my, t_cx, t_cy, t_sh}
       ) do
    probs = Nx.sigmoid(b_l / t_b)
    {u, key} = Nx.Random.uniform(key, shape: Nx.shape(probs))
    buttons = Nx.less(u, probs)

    {mx, key} = gumbel_argmax(mx_l, key, t_mx)
    {my, key} = gumbel_argmax(my_l, key, t_my)
    {cx, key} = gumbel_argmax(cx_l, key, t_cx)
    {cy, key} = gumbel_argmax(cy_l, key, t_cy)
    {sh, _key} = gumbel_argmax(sh_l, key, t_sh)

    {buttons, mx, my, cx, cy, sh, confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  defnp fused_sample_deterministic({b_l, mx_l, my_l, cx_l, cy_l, sh_l}) do
    buttons = Nx.greater(Nx.sigmoid(b_l), 0.5)
    mx = Nx.argmax(mx_l, axis: -1)
    my = Nx.argmax(my_l, axis: -1)
    cx = Nx.argmax(cx_l, axis: -1)
    cy = Nx.argmax(cy_l, axis: -1)
    sh = Nx.argmax(sh_l, axis: -1)

    {buttons, mx, my, cx, cy, sh, confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l)}
  end

  defnp gumbel_argmax(logits, key, temperature) do
    scaled = logits / temperature
    {u, key} = Nx.Random.uniform(key, shape: Nx.shape(scaled))
    gumbel = -Nx.log(-Nx.log(u + 1.0e-10))
    {Nx.argmax(scaled + gumbel, axis: -1), key}
  end

  defnp confidence_scalars(b_l, mx_l, my_l, cx_l, cy_l, sh_l) do
    buttons = Nx.mean(Nx.abs(Nx.sigmoid(b_l) - 0.5) * 2)
    main = (max_softmax(mx_l) + max_softmax(my_l)) / 2
    c = (max_softmax(cx_l) + max_softmax(cy_l)) / 2
    shoulder = max_softmax(sh_l)
    overall = buttons * 0.4 + main * 0.3 + c * 0.15 + shoulder * 0.15
    %{buttons: buttons, main: main, c: c, shoulder: shoulder, overall: overall}
  end

  defnp max_softmax(logits) do
    Nx.exp(logits - Nx.logsumexp(logits, axes: [-1], keep_axes: true))
    |> Nx.reduce_max(axes: [-1])
    |> Nx.mean()
  end

  @doc """
  Sample buttons from logits (independent Bernoulli).
  """
  @spec sample_buttons(Nx.Tensor.t(), boolean()) :: Nx.Tensor.t()
  def sample_buttons(logits, deterministic \\ false) do
    probs = Nx.sigmoid(logits)

    if deterministic do
      Nx.greater(probs, 0.5)
    else
      # Sample from Bernoulli using Nx.Random
      key = Nx.Random.key(System.system_time())
      {random, _new_key} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      Nx.less(random, probs)
    end
  end

  @doc """
  Sample from categorical distribution with temperature.
  """
  @spec sample_categorical(Nx.Tensor.t(), float(), boolean()) :: Nx.Tensor.t()
  def sample_categorical(logits, temperature \\ 1.0, deterministic \\ false) do
    if deterministic do
      Nx.argmax(logits, axis: -1)
    else
      # Apply temperature
      scaled_logits = Nx.divide(logits, temperature)

      # Gumbel-max trick for sampling
      key = Nx.Random.key(System.system_time())
      {gumbel_noise, _new_key} = Nx.Random.uniform(key, shape: Nx.shape(scaled_logits))
      gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(gumbel_noise, 1.0e-10)))))

      Nx.argmax(Nx.add(scaled_logits, gumbel), axis: -1)
    end
  end

  @doc """
  Compute confidence scores from action logits.

  Returns a map with confidence scores (0-1 scale) for each component:
  - `:buttons` - Average button confidence (how far from 0.5 the sigmoid probs are)
  - `:main` - Main stick confidence (max softmax probability)
  - `:c` - C-stick confidence (max softmax probability)
  - `:shoulder` - Shoulder confidence (max softmax probability)
  - `:overall` - Weighted average of all components

  Higher values = more confident predictions.
  """
  @spec compute_confidence(map()) :: map()
  def compute_confidence(%{confidence_raw: raw}) do
    # Precomputed inside the fused sampling program — just read the scalars.
    %{
      buttons: raw.buttons |> Nx.to_number() |> Float.round(3),
      main: raw.main |> Nx.to_number() |> Float.round(3),
      c: raw.c |> Nx.to_number() |> Float.round(3),
      shoulder: raw.shoulder |> Nx.to_number() |> Float.round(3),
      overall: raw.overall |> Nx.to_number() |> Float.round(3)
    }
  end

  def compute_confidence(%{logits: logits}) do
    compute_confidence(logits)
  end

  def compute_confidence(%{
        buttons: buttons_logits,
        main_x: main_x_logits,
        main_y: main_y_logits,
        c_x: c_x_logits,
        c_y: c_y_logits,
        shoulder: shoulder_logits
      }) do
    # Button confidence: how far from 0.5 (uncertain) the probabilities are
    # Confidence = mean(|sigmoid(logit) - 0.5| * 2)
    button_probs = Nx.sigmoid(buttons_logits)

    button_confidence =
      button_probs
      |> Nx.subtract(0.5)
      |> Nx.abs()
      |> Nx.multiply(2)
      |> Nx.mean()
      |> Nx.to_number()

    # Categorical confidence: max softmax probability
    main_x_conf = max_softmax_prob(main_x_logits)
    main_y_conf = max_softmax_prob(main_y_logits)
    main_confidence = (main_x_conf + main_y_conf) / 2

    c_x_conf = max_softmax_prob(c_x_logits)
    c_y_conf = max_softmax_prob(c_y_logits)
    c_confidence = (c_x_conf + c_y_conf) / 2

    shoulder_confidence = max_softmax_prob(shoulder_logits)

    # Overall: weighted average (buttons are most important for gameplay)
    overall =
      button_confidence * 0.4 + main_confidence * 0.3 +
        c_confidence * 0.15 + shoulder_confidence * 0.15

    %{
      buttons: Float.round(button_confidence, 3),
      main: Float.round(main_confidence, 3),
      c: Float.round(c_confidence, 3),
      shoulder: Float.round(shoulder_confidence, 3),
      overall: Float.round(overall, 3)
    }
  end

  def compute_confidence(_), do: %{overall: 0.0, buttons: 0.0, main: 0.0, c: 0.0, shoulder: 0.0}

  # Helper: compute max probability from softmax of logits
  defp max_softmax_prob(logits) do
    # Softmax then max
    probs = Axon.Activations.softmax(logits, axis: -1)

    probs
    |> Nx.reduce_max(axes: [-1])
    # Average across batch if present
    |> Nx.mean()
    |> Nx.to_number()
  end
end
