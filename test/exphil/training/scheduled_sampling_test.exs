defmodule ExPhil.Training.ScheduledSamplingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Interp.Attribution
  alias ExPhil.Training.Imitation
  alias ExPhil.Training.ScheduledSampling

  describe "Attribution.prev_action_dim_range/1" do
    test "finds a contiguous 13-dim slice in the default embedding" do
      assert [offset, 13] = Attribution.prev_action_dim_range()
      assert is_integer(offset) and offset > 0
    end
  end

  describe "build/2 splice mechanics (fake predict_fn)" do
    # A fake predict_fn lets the decode/splice path be tested exactly,
    # without building a real model: logits are constructed so the decoded
    # controller is fully known.
    #   buttons: logits [+1,-1,+1,-1,+1,-1,+1,-1] -> [1,0,1,0,1,0,1,0]
    #   sticks:  argmax at bucket 16 -> 16/16 = 1.0 -> (1.0-0.5)*2 = 1.0
    #   shoulder: argmax at bucket 4 -> 4/4 = 1.0
    defp fake_predict_fn(batch) do
      btn = Nx.tile(Nx.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]), [batch, 1])
      axis = Nx.tile(Nx.new_axis(Nx.iota({17}, type: :f32), 0), [batch, 1])
      sh = Nx.tile(Nx.new_axis(Nx.iota({5}, type: :f32), 0), [batch, 1])
      fn _params, _states -> {btn, axis, axis, axis, axis, sh} end
    end

    test "mask=1 rows get the decoded prediction, mask=0 rows keep ground truth" do
      batch = 2
      window = 3
      offset = 4
      embed = 24

      ss_fn =
        ScheduledSampling.build(fake_predict_fn(batch), %{
          ss_prev_dims: [offset, 13],
          axis_buckets: 16,
          shoulder_buckets: 4,
          kmeans_centers: nil
        })

      states = Nx.broadcast(0.25, {batch, window, embed})
      mask = Nx.tensor([[1.0], [0.0]])

      out = ss_fn.(%{}, states, mask)

      expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

      spliced = out[0][window - 1][offset..(offset + 12)] |> Nx.to_flat_list()
      kept = out[1][window - 1][offset..(offset + 12)] |> Nx.to_flat_list()

      assert spliced == expected
      assert Enum.all?(kept, &(&1 == 0.25))

      # everything outside the slice, and every earlier position, is untouched
      assert out[0][0] |> Nx.to_flat_list() |> Enum.all?(&(&1 == 0.25))
      assert Nx.to_number(out[0][window - 1][offset - 1]) == 0.25
      assert Nx.to_number(out[0][window - 1][offset + 13]) == 0.25
    end

    test "raises without ss_prev_dims" do
      assert_raise ArgumentError, ~r/ss_prev_dims/, fn ->
        ScheduledSampling.build(fn _, _ -> nil end, %{axis_buckets: 16})
      end
    end
  end

  describe "end-to-end through Imitation" do
    @tag timeout: 120_000
    test "train_step with scheduled_sampling produces a finite loss" do
      trainer =
        Imitation.new(
          temporal: true,
          use_prev_action: true,
          scheduled_sampling: 0.5,
          backbone: :gru,
          window_size: 4,
          hidden_size: 16,
          num_layers: 1,
          learning_rate: 1.0e-3
        )

      assert is_function(trainer.ss_fn)

      embed_size = trainer.config.embed_size
      batch = 4

      key = Nx.Random.key(42)
      {states, _} = Nx.Random.uniform(key, shape: {batch, 4, embed_size})

      actions = %{
        buttons: Nx.broadcast(0, {batch, 8}),
        main_x: Nx.broadcast(8, {batch}),
        main_y: Nx.broadcast(8, {batch}),
        c_x: Nx.broadcast(8, {batch}),
        c_y: Nx.broadcast(8, {batch}),
        shoulder: Nx.broadcast(0, {batch})
      }

      {_predict_fn, loss_fn} = Imitation.build_loss_fn(trainer.policy_model)

      {trained, metrics} =
        Imitation.train_step(trainer, %{states: states, actions: actions}, loss_fn)

      assert trained.step == 1
      loss = Nx.to_number(metrics.loss)
      assert is_number(loss) and loss > 0.0
    end

    test "scheduled_sampling without use_prev_action raises" do
      assert_raise ArgumentError, ~r/use_prev_action/, fn ->
        Imitation.new(
          temporal: true,
          scheduled_sampling: 0.5,
          backbone: :gru,
          window_size: 4,
          hidden_size: 16
        )
      end
    end

    test "scheduled_sampling without temporal raises" do
      assert_raise ArgumentError, ~r/temporal/, fn ->
        Imitation.new(
          scheduled_sampling: 0.5,
          use_prev_action: true,
          hidden_sizes: [16]
        )
      end
    end
  end
end
