defmodule ExPhil.Networks.Policy.AutoregressiveHeadTest do
  @moduledoc """
  Tests for the TRUE autoregressive controller head
  (docs/planning/AUTOREGRESSIVE_HEAD_PLAN.md work item 7):

  1. the head builds and produces the standard 6-logit shape contract
  2. zero-init: at init the head IS an independent factorization
  3. the teacher-forced graph and the sequential sampler agree given the
     same prefix (the training/inference consistency invariant)
  4. trained on a synthetic distribution with a within-frame dependency
     (main_y = up iff B pressed), the AR head recovers the conditional
     that an independent head cannot represent
  """
  use ExUnit.Case, async: true
  @moduletag :backbone

  alias ExPhil.Networks.Policy.{Heads, Sampling}
  alias ExPhil.Training.Utils

  @hidden 32
  @batch 4
  @axis_buckets 16
  @shoulder_buckets 4

  defp build_head_model do
    input = Axon.input("trunk", shape: {nil, @hidden})

    Heads.build_autoregressive_head(input,
      axis_buckets: @axis_buckets,
      shoulder_buckets: @shoulder_buckets
    )
  end

  defp init_model(model) do
    {init_fn, predict_fn} = Axon.build(model)

    template =
      Map.merge(%{"trunk" => Nx.template({1, @hidden}, :f32)}, Heads.tf_templates(1))

    params = init_fn.(template, Axon.ModelState.empty())
    {params, predict_fn}
  end

  defp tf_input_map(trunk, actions) do
    Map.put(Heads.tf_inputs(actions), "trunk", trunk)
  end

  defp random_actions(key, batch) do
    {b, key} = Nx.Random.randint(key, 0, 2, shape: {batch, 8}, type: :s64)
    {mx, key} = Nx.Random.randint(key, 0, @axis_buckets + 1, shape: {batch}, type: :s64)
    {my, key} = Nx.Random.randint(key, 0, @axis_buckets + 1, shape: {batch}, type: :s64)
    {cx, key} = Nx.Random.randint(key, 0, @axis_buckets + 1, shape: {batch}, type: :s64)
    {cy, _key} = Nx.Random.randint(key, 0, @axis_buckets + 1, shape: {batch}, type: :s64)

    %{buttons: b, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: Nx.broadcast(0, {batch})}
  end

  # Give the zero-initialised embeddings real weight so conditioning is
  # active (otherwise test 3 would pass trivially).
  defp randomize_embeds(%Axon.ModelState{data: data} = params, seed) do
    key = Nx.Random.key(seed)

    {data, _key} =
      Enum.reduce(
        ["ar_buttons_embed", "ar_main_x_embed", "ar_main_y_embed", "ar_c_x_embed", "ar_c_y_embed"],
        {data, key},
        fn layer, {d, k} ->
          kernel = d[layer]["kernel"]
          {noise, k} = Nx.Random.normal(k, 0.0, 0.5, shape: Nx.shape(kernel), type: :f32)
          {put_in(d, [layer, "kernel"], noise), k}
        end
      )

    %{params | data: data}
  end

  test "builds and produces the 6-head shape contract" do
    model = build_head_model()
    {params, predict_fn} = init_model(model)

    key = Nx.Random.key(1)
    {trunk, key} = Nx.Random.normal(key, shape: {@batch, @hidden})
    actions = random_actions(key, @batch)

    {b, mx, my, cx, cy, sh} = predict_fn.(params, tf_input_map(trunk, actions))

    assert Nx.shape(b) == {@batch, 8}
    assert Nx.shape(mx) == {@batch, @axis_buckets + 1}
    assert Nx.shape(my) == {@batch, @axis_buckets + 1}
    assert Nx.shape(cx) == {@batch, @axis_buckets + 1}
    assert Nx.shape(cy) == {@batch, @axis_buckets + 1}
    assert Nx.shape(sh) == {@batch, @shoulder_buckets + 1}
  end

  test "zero-initialised embeddings: teacher-forced inputs do not move logits at init" do
    model = build_head_model()
    {params, predict_fn} = init_model(model)

    key = Nx.Random.key(2)
    {trunk, key} = Nx.Random.normal(key, shape: {@batch, @hidden})

    a1 = random_actions(Nx.Random.key(3), @batch)
    a2 = random_actions(Nx.Random.key(4), @batch)

    out1 = predict_fn.(params, tf_input_map(trunk, a1))
    out2 = predict_fn.(params, tf_input_map(trunk, a2))
    _ = key

    for i <- 0..5 do
      assert Nx.all_close(elem(out1, i), elem(out2, i)) |> Nx.to_number() == 1
    end
  end

  test "teacher-forced logits equal sequential-sampler logits given the same prefix" do
    model = build_head_model()
    {params, predict_fn} = init_model(model)
    params = randomize_embeds(params, 42)

    {trunk, _} = Nx.Random.normal(Nx.Random.key(5), shape: {1, @hidden})

    # Sequential deterministic replay from the ar_* params
    sampled = Sampling.sample_autoregressive_from_features(params, trunk, deterministic: true)

    # Teacher-force the graph with exactly the sampled prefix
    actions = %{
      buttons: Nx.as_type(sampled.buttons, :s64),
      main_x: sampled.main_x,
      main_y: sampled.main_y,
      c_x: sampled.c_x,
      c_y: sampled.c_y,
      shoulder: sampled.shoulder
    }

    {b, mx, my, cx, cy, sh} =
      predict_fn.(Utils.ensure_model_state(params), tf_input_map(trunk, actions))

    for {graph_logits, sampler_logits} <- [
          {b, sampled.logits.buttons},
          {mx, sampled.logits.main_x},
          {my, sampled.logits.main_y},
          {cx, sampled.logits.c_x},
          {cy, sampled.logits.c_y},
          {sh, sampled.logits.shoulder}
        ] do
      assert Nx.all_close(graph_logits, sampler_logits, atol: 1.0e-5) |> Nx.to_number() == 1
    end

    # And the deterministic re-sample of the teacher-forced logits matches
    assert Nx.to_flat_list(Nx.argmax(mx, axis: -1)) == Nx.to_flat_list(sampled.main_x)
  end

  test "sample_autoregressive_n: n coherent samples, key-reproducible, actually varied" do
    model = build_head_model()
    {params, predict_fn} = init_model(model)
    params = randomize_embeds(params, 42)
    _ = predict_fn

    {trunk, _} = Nx.Random.normal(Nx.Random.key(6), shape: {1, @hidden})

    n = 16
    samples = Sampling.sample_autoregressive_n(params, trunk, n, key: Nx.Random.key(9))

    assert length(samples) == n

    for s <- samples do
      assert Nx.shape(s.buttons) == {8}
      assert Nx.shape(s.main_x) == {}
      assert Nx.shape(s.shoulder) == {}
    end

    # Same key => identical draw (the --seed contract in interp_passk)
    again = Sampling.sample_autoregressive_n(params, trunk, n, key: Nx.Random.key(9))

    assert Enum.zip(samples, again)
           |> Enum.all?(fn {a, b} ->
             Nx.to_flat_list(a.buttons) == Nx.to_flat_list(b.buttons) and
               Nx.to_number(a.main_y) == Nx.to_number(b.main_y)
           end)

    # Independent draws at T=1 from a random head must not be n copies of
    # one sample (the 08-28 Leg S run-1 bug signature).
    distinct =
      samples
      |> Enum.map(&{Nx.to_flat_list(&1.buttons), Nx.to_number(&1.main_x), Nx.to_number(&1.main_y)})
      |> Enum.uniq()
      |> length()

    assert distinct > 1, "all #{n} samples identical — not independent draws"
  end

  test "sample_autoregressive_kn: k samples per feature row, batched shapes" do
    model = build_head_model()
    {params, _} = init_model(model)
    params = randomize_embeds(params, 42)

    {feats, _} = Nx.Random.normal(Nx.Random.key(11), shape: {5, @hidden})

    {b, mx, my, cx, cy, sh} =
      Sampling.sample_autoregressive_kn(params, feats, 7, key: Nx.Random.key(12))

    assert Nx.shape(b) == {7, 5, 8}
    assert Nx.shape(mx) == {7, 5}
    assert Nx.shape(my) == {7, 5}
    assert Nx.shape(cx) == {7, 5}
    assert Nx.shape(cy) == {7, 5}
    assert Nx.shape(sh) == {7, 5}

    # Same key reproduces; distinct rows/samples vary at T=1
    {b2, mx2, _, _, _, _} =
      Sampling.sample_autoregressive_kn(params, feats, 7, key: Nx.Random.key(12))

    assert Nx.to_flat_list(b) == Nx.to_flat_list(b2)
    assert Nx.to_flat_list(mx) == Nx.to_flat_list(mx2)
    assert length(Enum.uniq(Nx.to_flat_list(mx))) > 1
  end

  @tag :slow
  @tag timeout: 120_000
  test "AR head learns P(up | B) on a synthetic dependent distribution" do
    # Within-frame dependency the independent factorization cannot express:
    # B ~ Bernoulli(0.5); main_y = 12 (up) iff B else 4. Features carry NO
    # information (constant), so only the conditioning wire can learn it.
    n = 512
    key = Nx.Random.key(7)
    trunk = Nx.broadcast(Nx.tensor(1.0, type: :f32), {n, @hidden})

    {b_col, _key} = Nx.Random.randint(key, 0, 2, shape: {n, 1}, type: :s64)
    buttons = Nx.concatenate([Nx.broadcast(0, {n, 1}), b_col, Nx.broadcast(0, {n, 6})], axis: 1)
    main_y = Nx.select(Nx.squeeze(b_col, axes: [1]) |> Nx.equal(1), 12, 4)

    actions = %{
      buttons: buttons,
      main_x: Nx.broadcast(8, {n}),
      main_y: main_y,
      c_x: Nx.broadcast(8, {n}),
      c_y: Nx.broadcast(8, {n}),
      shoulder: Nx.broadcast(0, {n})
    }

    model = build_head_model()
    {params, predict_fn} = init_model(model)

    inputs = tf_input_map(trunk, actions)

    # Tensors flow as ARGUMENTS through the jit boundary (closure-captured
    # EXLA tensors break value_and_grad — GOTCHA #3, same pattern as
    # Imitation.Loss.build_loss_and_grad_fn).
    train_step =
      Nx.Defn.jit(
        fn data, inputs, actions ->
          loss_fn = fn d ->
            {b_l, mx_l, my_l, cx_l, cy_l, sh_l} =
              predict_fn.(Utils.ensure_model_state(d), inputs)

            ExPhil.Networks.Policy.imitation_loss(
              %{buttons: b_l, main_x: mx_l, main_y: my_l, c_x: cx_l, c_y: cy_l, shoulder: sh_l},
              actions
            )
          end

          {_loss, grads} = Nx.Defn.value_and_grad(loss_fn).(data)

          deep_update = fn du, d, g ->
            Map.new(d, fn {k, v} ->
              gv = g[k]

              cond do
                is_map(v) and not is_struct(v) -> {k, du.(du, v, gv)}
                # LR 0.05: 0.5 overshoots into logit saturation (loss
                # freezes at ~4.18 with near-zero grads — verified 08-30)
                true -> {k, Nx.subtract(v, Nx.multiply(0.05, gv))}
              end
            end)
          end

          deep_update.(deep_update, data, grads)
        end,
        on_conflict: :reuse
      )

    data =
      Enum.reduce(1..500, params.data, fn _, d -> train_step.(d, inputs, actions) end)

    trained = %{params | data: data}

    # Sample many frames stochastically and measure the joint
    feats = Nx.broadcast(Nx.tensor(1.0, type: :f32), {2048, @hidden})
    out = Sampling.sample_autoregressive_from_features(trained, feats, temperature: 1.0)

    b_pressed = out.buttons |> Nx.slice_along_axis(1, 1, axis: 1) |> Nx.squeeze(axes: [1])
    up = Nx.equal(out.main_y, 12)

    n_b = Nx.sum(b_pressed) |> Nx.to_number()
    p_up_given_b = Nx.sum(Nx.multiply(b_pressed, up)) |> Nx.to_number() |> Kernel./(max(n_b, 1))

    not_b = Nx.subtract(1, b_pressed)
    n_not_b = Nx.sum(not_b) |> Nx.to_number()

    p_up_given_not_b =
      Nx.sum(Nx.multiply(not_b, up)) |> Nx.to_number() |> Kernel./(max(n_not_b, 1))

    # The expert's rule is deterministic; the AR head should get most of it.
    # An independent head would give p_up_given_b == p_up_given_not_b.
    assert p_up_given_b > 0.7,
           "P(up|B)=#{p_up_given_b} — conditioning not learned"

    assert p_up_given_b - p_up_given_not_b > 0.4,
           "P(up|B)=#{p_up_given_b} vs P(up|!B)=#{p_up_given_not_b} — no dependency recovered"
  end
end
