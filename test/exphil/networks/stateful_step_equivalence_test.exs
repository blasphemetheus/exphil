defmodule ExPhil.Networks.StatefulStepEquivalenceTest do
  @moduledoc """
  Golden equivalence pin for the Edifice.Stateful step path (task #16 P1).

  The agent's `--stateful-step` mode replaces the windowed full-model forward
  (O(window) per frame) with per-frame trunk stepping via
  `Edifice.Recurrent.step/3` plus a heads-only predict over the resulting
  features. That decomposition is only valid if:

    windowed_full_model(frames)[last] == heads(step*(init_state, frames))

  for the SAME exported params — this test pins exactly that, for GRU and
  LSTM, along with the state-reset and snapshot/deserialize (rollback)
  properties the netplay path relies on.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy
  alias ExPhil.Networks.Policy.Heads
  alias ExPhil.Training.Utils

  @embed_size 16
  @hidden_size 16
  @num_layers 2
  @window 6
  @axis_buckets 16
  @shoulder_buckets 4

  # Full windowed forward and stepwise trunk+heads must agree within 1e-4
  # (f32; both run on the test backend).
  @tol 1.0e-4

  defp build_full_policy(cell_type) do
    model =
      Policy.build_temporal(
        embed_size: @embed_size,
        backbone: cell_type,
        hidden_size: @hidden_size,
        num_layers: @num_layers,
        window_size: @window,
        dropout: 0.0,
        axis_buckets: @axis_buckets,
        shoulder_buckets: @shoulder_buckets
      )

    {init_fn, predict_fn} = Utils.build_compiled(model)

    params =
      init_fn.(
        Nx.template({1, @window, @embed_size}, :f32),
        Axon.ModelState.empty()
      )

    {params, predict_fn}
  end

  defp build_heads_predict do
    input = Axon.input("features", shape: {nil, @hidden_size})
    model = Heads.build_controller_head(input, @axis_buckets, @shoulder_buckets)
    {_init_fn, predict_fn} = Utils.build_compiled(model)
    predict_fn
  end

  defp random_frames(seed) do
    key = Nx.Random.key(seed)
    {frames, _} = Nx.Random.normal(key, shape: {1, @window, @embed_size}, type: :f32)
    frames
  end

  defp raw(%Axon.ModelState{data: data}), do: data

  defp init_state(params, cell_type) do
    Edifice.Recurrent.init_state(raw(params),
      batch_size: 1,
      hidden_size: @hidden_size,
      num_layers: @num_layers,
      cell_type: cell_type
    )
  end

  defp step_all(params, state, frames) do
    0..(Nx.axis_size(frames, 1) - 1)
    |> Enum.reduce({nil, state}, fn t, {_out, st} ->
      frame = frames |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
      Edifice.Recurrent.step(raw(params), st, frame)
    end)
  end

  defp max_delta(a, b), do: a |> Nx.subtract(b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  defp assert_logits_close(tuple_a, tuple_b) do
    names = [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

    deltas =
      Enum.zip([names, Tuple.to_list(tuple_a), Tuple.to_list(tuple_b)])
      |> Enum.map(fn {name, a, b} -> {name, max_delta(a, b)} end)

    for {name, delta} <- deltas do
      assert delta < @tol,
             "#{name} logits diverge between windowed and step path: " <>
               "max |delta| = #{delta} (tol #{@tol})"
    end

    deltas
  end

  for cell_type <- [:gru, :lstm] do
    @cell_type cell_type

    describe "#{cell_type} step path" do
      test "final-frame head logits match the windowed full-model forward" do
        {params, full_predict} = build_full_policy(@cell_type)
        heads_predict = build_heads_predict()
        frames = random_frames(42)

        # (a) Windowed: full model over the whole window
        windowed_logits = full_predict.(params, frames)

        # (b) Stepwise: trunk step per frame, then heads-only predict on the
        #     final features — with the SAME full-policy params
        state = init_state(params, @cell_type)
        {features, _state} = step_all(params, state, frames)
        assert Nx.shape(features) == {1, @hidden_size}

        step_logits = heads_predict.(params, features)

        assert_logits_close(windowed_logits, step_logits)
      end

      test "state reset reproduces a fresh run exactly" do
        {params, _} = build_full_policy(@cell_type)
        frames = random_frames(7)

        # Pollute a state with some frames, then re-init (= agent reset_buffer)
        polluted = init_state(params, @cell_type)
        {_out, _polluted} = step_all(params, polluted, frames)

        fresh = init_state(params, @cell_type)
        reinit = init_state(params, @cell_type)

        {out_fresh, _} = step_all(params, fresh, frames)
        {out_reinit, _} = step_all(params, reinit, frames)

        assert max_delta(out_fresh, out_reinit) == 0.0
      end

      test "snapshot/serialize/deserialize round-trip resumes identically (rollback pin)" do
        {params, _} = build_full_policy(@cell_type)
        frames = random_frames(1234)

        head = Nx.slice_along_axis(frames, 0, 3, axis: 1)
        tail = Nx.slice_along_axis(frames, 3, @window - 3, axis: 1)

        state = init_state(params, @cell_type)
        {_out, mid_state} = step_all(params, state, head)

        # Rollback: serialize at frame 3, keep playing, then restore + replay
        blob = Edifice.Stateful.serialize(mid_state)
        {out_original, _} = step_all(params, mid_state, tail)

        restored = Edifice.Stateful.deserialize(blob)
        {out_replayed, _} = step_all(params, restored, tail)

        assert max_delta(out_original, out_replayed) == 0.0
      end
    end
  end

  test "reports max logit deltas (informational)" do
    for cell_type <- [:gru, :lstm] do
      {params, full_predict} = build_full_policy(cell_type)
      heads_predict = build_heads_predict()
      frames = random_frames(99)

      windowed = full_predict.(params, frames)
      state = init_state(params, cell_type)
      {features, _} = step_all(params, state, frames)
      step_logits = heads_predict.(params, features)

      deltas = assert_logits_close(windowed, step_logits)
      worst = deltas |> Enum.map(&elem(&1, 1)) |> Enum.max()
      IO.puts("[stateful-step] #{cell_type} max head-logit delta: #{worst}")
    end
  end
end
