defmodule ExPhil.Training.ScheduledSamplingTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Embeddings.Game.Config
  alias ExPhil.Interp.Attribution
  alias ExPhil.Training.Imitation
  alias ExPhil.Training.ScheduledSampling

  describe "Attribution.prev_action_dim_range/1" do
    test "finds a contiguous 13-dim slice in the default embedding" do
      assert [offset, 13] = Attribution.prev_action_dim_range()
      assert is_integer(offset) and offset > 0
    end

    test "queue layouts resolve to the same block start as the classic channel" do
      queue_cfg = %{Config.default() | queue_depth: 4, with_delay_id: true}
      assert Attribution.prev_action_dim_range(config: queue_cfg) ==
               Attribution.prev_action_dim_range()
    end
  end

  describe "build/2 splice mechanics (fake predict_fn)" do
    # A fake predict_fn lets the decode/splice path be tested exactly,
    # without building a real model. All tensors are created INSIDE the
    # call so they trace as Nx.Defn.Expr (closing over concrete EXLA
    # tensors raises IncompatibleBackendsError under jit — the former
    # version of this fake did exactly that and the test was broken).
    #
    # Logits are constructed so the decoded controller is fully known,
    # and the stick/shoulder argmax bucket equals the TRUNCATED window
    # length — so each queue slot k (predicted from a window truncated by
    # k) decodes to a distinct, predictable value:
    #   buttons:  alternating +1/-1 -> [1,0,1,0,1,0,1,0]
    #   sticks:   argmax at bucket seq -> (seq/16 - 0.5) * 2
    #   shoulder: argmax at min(seq, 4) -> min(seq, 4) / 4
    defp fake_predict_fn do
      fn _params, states ->
        batch = Nx.axis_size(states, 0)
        seq = Nx.axis_size(states, 1)

        btn =
          Nx.iota({batch, 8}, axis: 1)
          |> Nx.remainder(2)
          |> Nx.multiply(-2)
          |> Nx.add(1)
          |> Nx.as_type(:f32)

        axis = Nx.equal(Nx.iota({batch, 17}, axis: 1), seq) |> Nx.as_type(:f32)
        sh = Nx.equal(Nx.iota({batch, 5}, axis: 1), min(seq, 4)) |> Nx.as_type(:f32)
        {btn, axis, axis, axis, axis, sh}
      end
    end

    defp expected_slot(seq) do
      stick = (seq / 16 - 0.5) * 2
      [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] ++ List.duplicate(stick, 4) ++ [min(seq, 4) / 4]
    end

    test "mask=1 rows get the decoded prediction, mask=0 rows keep ground truth" do
      batch = 2
      window = 3
      offset = 4
      embed = 24

      ss_fn =
        ScheduledSampling.build(fake_predict_fn(), %{
          ss_prev_dims: [offset, 13],
          axis_buckets: 16,
          shoulder_buckets: 4,
          kmeans_centers: nil
        })

      states = Nx.broadcast(0.25, {batch, window, embed})
      mask = Nx.tensor([[1.0], [0.0]])

      out = ss_fn.(%{}, states, mask)

      spliced = out[0][window - 1][offset..(offset + 12)] |> Nx.to_flat_list()
      kept = out[1][window - 1][offset..(offset + 12)] |> Nx.to_flat_list()

      assert spliced == expected_slot(window - 1)
      assert Enum.all?(kept, &(&1 == 0.25))

      # everything outside the slice, and every earlier position, is untouched
      assert out[0][0] |> Nx.to_flat_list() |> Enum.all?(&(&1 == 0.25))
      assert Nx.to_number(out[0][window - 1][offset - 1]) == 0.25
      assert Nx.to_number(out[0][window - 1][offset + 13]) == 0.25
    end

    test "ss_queue_depth 3: slot k gets the prediction from truncation depth k" do
      batch = 2
      window = 8
      offset = 4
      embed = 4 + 3 * 13 + 4

      ss_fn =
        ScheduledSampling.build(fake_predict_fn(), %{
          ss_prev_dims: [offset, 13],
          ss_queue_depth: 3,
          axis_buckets: 16,
          shoulder_buckets: 4,
          kmeans_centers: nil
        })

      states = Nx.broadcast(0.25, {batch, window, embed})
      mask = Nx.tensor([[1.0], [0.0]])

      out = ss_fn.(%{}, states, mask)

      for k <- 1..3 do
        slot_off = offset + (k - 1) * 13
        spliced = out[0][window - 1][slot_off..(slot_off + 12)] |> Nx.to_flat_list()
        kept = out[1][window - 1][slot_off..(slot_off + 12)] |> Nx.to_flat_list()

        assert spliced == expected_slot(window - k),
               "slot #{k} should decode from the window truncated by #{k}"

        assert Enum.all?(kept, &(&1 == 0.25)), "mask=0 row leaked at slot #{k}"
      end

      # outside the queue block, and every earlier position, untouched
      assert out[0][0] |> Nx.to_flat_list() |> Enum.all?(&(&1 == 0.25))
      assert Nx.to_number(out[0][window - 1][offset - 1]) == 0.25
      assert Nx.to_number(out[0][window - 1][offset + 3 * 13]) == 0.25
    end

    test "raises when window is too short for the queue depth" do
      ss_fn =
        ScheduledSampling.build(fake_predict_fn(), %{
          ss_prev_dims: [4, 13],
          ss_queue_depth: 3,
          axis_buckets: 16,
          shoulder_buckets: 4,
          kmeans_centers: nil
        })

      states = Nx.broadcast(0.25, {2, 3, 24})
      mask = Nx.tensor([[1.0], [0.0]])

      assert_raise ArgumentError, ~r/needs window > 3/, fn ->
        ss_fn.(%{}, states, mask)
      end
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
      assert trainer.config[:ss_queue_depth] == 1

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

    @tag timeout: 120_000
    test "queue embed config flows into ss_queue_depth and trains to a finite loss" do
      embed_config = %{Config.default() | queue_depth: 3, with_delay_id: true}

      trainer =
        Imitation.new(
          embed_config: embed_config,
          temporal: true,
          use_prev_action: true,
          scheduled_sampling: 0.5,
          backbone: :gru,
          window_size: 6,
          hidden_size: 16,
          num_layers: 1,
          learning_rate: 1.0e-3
        )

      assert is_function(trainer.ss_fn)
      assert trainer.config[:ss_queue_depth] == 3

      embed_size = trainer.config.embed_size
      batch = 4

      key = Nx.Random.key(43)
      {states, _} = Nx.Random.uniform(key, shape: {batch, 6, embed_size})

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

    test "scheduled_sampling with window too short for queue depth raises" do
      embed_config = %{Config.default() | queue_depth: 4}

      assert_raise ArgumentError, ~r/window_size > 4/, fn ->
        Imitation.new(
          embed_config: embed_config,
          temporal: true,
          use_prev_action: true,
          scheduled_sampling: 0.5,
          backbone: :gru,
          window_size: 4,
          hidden_size: 16
        )
      end
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
