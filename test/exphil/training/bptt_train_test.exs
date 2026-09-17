defmodule ExPhil.Training.BpttTrainTest do
  use ExUnit.Case, async: false

  alias ExPhil.Training.{Data, Imitation, TrajectoryCursors}
  alias ExPhil.Training.Imitation.TrainLoop

  @embed 32
  @hidden 24
  @layers 2
  @batch 2
  @unroll 10

  defp frame(counter, id) do
    %{
      game_state: %{frame: counter},
      action: %{
        buttons: %{
          a: rem(id, 3) == 0,
          b: false,
          x: false,
          y: false,
          z: false,
          l: false,
          r: false,
          d_up: false
        },
        main_x: rem(id, 17),
        main_y: 8,
        c_x: 8,
        c_y: 8,
        shoulder: 0
      }
    }
  end

  defp dataset(game_lengths) do
    frames =
      Enum.flat_map(game_lengths, fn len -> Enum.map(0..(len - 1), &frame(&1, &1)) end)

    n = length(frames)

    embedded =
      Nx.iota({n, @embed}, axis: 0)
      |> Nx.as_type(:f32)
      |> Nx.divide(n)

    %Data{frames: frames, embedded_frames: embedded, size: n}
  end

  defp new_bptt_trainer(head, opts \\ []) do
    Imitation.new(
      Keyword.merge(
        [
          embed_size: @embed,
          temporal: true,
          bptt: true,
          backbone: :gru,
          head: head,
          unroll: @unroll,
          hidden_size: @hidden,
          num_layers: @layers,
          precision: :f32,
          batch_size: @batch
        ],
        opts
      )
    )
  end

  describe "Imitation.new bptt mode" do
    test "dropout state advances and checkpoint resume reproduces the next update" do
      t = new_bptt_trainer(:autoregressive, seed: 905, dropout: 0.5)

      [batch | _] =
        TrajectoryCursors.batch_stream(dataset([30, 30]),
          batch_size: @batch,
          unroll: @unroll,
          gpu: false
        )
        |> Enum.to_list()

      carry = Nx.broadcast(0.0, {@batch, @layers, @hidden})
      {next, metrics, _} = TrainLoop.train_step_bptt(t, batch, carry)
      key = "recurrent_dropout_1"
      refute Nx.serialize(t.policy_params.data[key]) == Nx.serialize(next.policy_params.data[key])

      {eval_loss, _} =
        t.eval_loss_fn.(t.policy_params, batch.states, batch.actions, batch.frame_weights, carry)

      refute Nx.to_number(metrics.loss) == Nx.to_number(eval_loss)

      {again, _} =
        t.eval_loss_fn.(t.policy_params, batch.states, batch.actions, batch.frame_weights, carry)

      assert Nx.to_number(again) == Nx.to_number(eval_loss)

      path =
        Path.join(System.tmp_dir!(), "bptt_dropout_#{System.unique_integer([:positive])}.axon")

      on_exit(fn -> File.rm(path) end)
      :ok = Imitation.save_checkpoint(next, path)

      {:ok, loaded} =
        Imitation.load_checkpoint(
          new_bptt_trainer(:autoregressive, seed: 906, dropout: 0.5),
          path
        )

      {continued, cm, _} = TrainLoop.train_step_bptt(next, batch, carry)
      {resumed, rm, _} = TrainLoop.train_step_bptt(loaded, batch, carry)
      assert Nx.to_number(cm.loss) == Nx.to_number(rm.loss)

      assert Nx.serialize(continued.policy_params.data) ==
               Nx.serialize(resumed.policy_params.data)

      assert resumed.step == 2
    end

    @tag :seed_contract
    test "an explicit seed controls initial weights, not only data ordering" do
      first = new_bptt_trainer(:autoregressive, seed: 905)
      repeated = new_bptt_trainer(:autoregressive, seed: 905)
      different = new_bptt_trainer(:autoregressive, seed: 906)
      fingerprint = fn trainer -> Nx.serialize(trainer.policy_params.data) end
      assert fingerprint.(first) == fingerprint.(repeated)
      refute fingerprint.(first) == fingerprint.(different)
    end

    test "builds a trainer with the bptt loss fn and the carry-threaded eval fn" do
      trainer = new_bptt_trainer(:autoregressive)
      assert trainer.config[:bptt] == true
      assert is_function(trainer.loss_and_grad_fn)
      assert is_function(trainer.eval_loss_fn)
    end

    test "rejects non-gru backbones" do
      assert_raise ArgumentError, ~r/gru only/, fn ->
        Imitation.new(embed_size: @embed, temporal: true, bptt: true, backbone: :mamba)
      end
    end

    test "rejects non-temporal" do
      assert_raise ArgumentError, ~r/temporal/, fn ->
        Imitation.new(embed_size: @embed, temporal: false, bptt: true, backbone: :gru)
      end
    end

    test "rejects accumulation before the trainer can choose the windowed gradient path" do
      assert_raise ArgumentError, ~r/bptt does not support gradient accumulation/, fn ->
        new_bptt_trainer(:autoregressive, accumulation_steps: 2)
      end
    end
  end

  describe "end-to-end: cursors -> train_step_bptt" do
    test "masked future padding and inactive rows cannot change the optimizer update" do
      trainer = new_bptt_trainer(:autoregressive, seed: 905, dropout: 0.0)

      [batch] =
        Enum.to_list(
          TrajectoryCursors.batch_stream(dataset([3]),
            batch_size: @batch,
            unroll: @unroll,
            gpu: false,
            neutral_weight: 1.0
          )
        )

      # Change every padded input and teacher-forcing target, including the
      # completely inactive row. Only the three real frames may affect loss.
      mask = Nx.new_axis(batch.valid_mask, -1)

      altered = %{
        batch
        | states:
            Nx.select(
              Nx.broadcast(mask, Nx.shape(batch.states)),
              batch.states,
              Nx.broadcast(7.0, Nx.shape(batch.states))
            ),
          actions:
            Map.new(batch.actions, fn {head, targets} ->
              target_mask = if Nx.rank(targets) == 3, do: mask, else: batch.valid_mask

              {head,
               Nx.select(
                 Nx.broadcast(target_mask, Nx.shape(targets)),
                 targets,
                 Nx.broadcast(0, Nx.shape(targets))
               )}
            end)
      }

      carry = Nx.broadcast(0.0, {@batch, @layers, @hidden})
      {original, m1, _} = TrainLoop.train_step_bptt(trainer, batch, carry)
      {changed, m2, _} = TrainLoop.train_step_bptt(trainer, altered, carry)
      assert_in_delta Nx.to_number(m1.loss), Nx.to_number(m2.loss), 1.0e-6

      assert_params_close(original.policy_params.data, changed.policy_params.data)
    end

    for head <- [:autoregressive, :independent] do
      test "loss finite, carry flows and updates params (#{head} head)" do
        head = unquote(head)
        trainer = new_bptt_trainer(head)
        ds = dataset([60, 60])

        batches =
          TrajectoryCursors.batch_stream(ds,
            batch_size: @batch,
            unroll: @unroll,
            overlap: 0,
            gpu: false
          )
          |> Enum.take(3)

        assert length(batches) == 3

        carry = Nx.broadcast(0.0, {@batch, @layers, @hidden})

        {_final_trainer, losses, carries} =
          Enum.reduce(batches, {trainer, [], []}, fn batch, {tr, ls, cs} ->
            {tr, metrics, new_carry} = TrainLoop.train_step_bptt(tr, batch, carry_last(cs, carry))
            {tr, [Nx.to_number(metrics.loss) | ls], [new_carry | cs]}
          end)

        # Losses are finite numbers
        assert Enum.all?(losses, &(is_number(&1) and &1 == &1 and &1 != :infinity))

        # The carry is non-zero after a step (state actually flows out)
        last_carry = hd(carries)
        assert Nx.shape(last_carry) == {@batch, @layers, @hidden}
        assert Nx.to_number(Nx.reduce_max(Nx.abs(last_carry))) > 0.0
      end
    end

    test "is_resetting zeroes the carry rows before the step" do
      trainer = new_bptt_trainer(:independent)
      ds = dataset([60, 60])

      [batch | _] =
        TrajectoryCursors.batch_stream(ds,
          batch_size: @batch,
          unroll: @unroll,
          overlap: 0,
          gpu: false
        )
        |> Enum.take(1)

      # First batch: both rows resetting. A garbage carry must give the
      # SAME loss as a zero carry (the reset masks it out).
      garbage = Nx.broadcast(7.5, {@batch, @layers, @hidden})
      zeros = Nx.broadcast(0.0, {@batch, @layers, @hidden})

      {_t1, m_garbage, _} = TrainLoop.train_step_bptt(trainer, batch, garbage)
      {_t2, m_zeros, _} = TrainLoop.train_step_bptt(trainer, batch, zeros)

      assert_in_delta Nx.to_number(m_garbage.loss), Nx.to_number(m_zeros.loss), 1.0e-5
    end
  end

  defp carry_last([], initial), do: initial
  defp carry_last([c | _], _), do: c

  defp assert_params_close(%Nx.Tensor{} = a, %Nx.Tensor{} = b) do
    assert Nx.to_number(Nx.all_close(a, b, atol: 1.0e-6, rtol: 1.0e-5)) == 1
  end

  defp assert_params_close(a, b) when is_map(a) and is_map(b) do
    assert Map.keys(a) == Map.keys(b)
    for {key, value} <- a, do: assert_params_close(value, Map.fetch!(b, key))
  end
end

defmodule ExPhil.Training.BpttValTest do
  use ExUnit.Case, async: false

  alias ExPhil.Training.{Data, Imitation, TrajectoryCursors}
  alias ExPhil.Training.Imitation.Validation

  @embed 32
  @hidden 24
  @layers 2
  @unroll 10

  defp frame(counter, id) do
    %{
      game_state: %{frame: counter},
      action: %{
        buttons: %{
          a: rem(id, 3) == 0,
          b: false,
          x: false,
          y: false,
          z: false,
          l: false,
          r: false,
          d_up: false
        },
        main_x: rem(id, 17),
        main_y: 8,
        c_x: 8,
        c_y: 8,
        shoulder: 0
      }
    }
  end

  defp dataset(game_lengths) do
    frames = Enum.flat_map(game_lengths, fn len -> Enum.map(0..(len - 1), &frame(&1, &1)) end)
    n = length(frames)
    embedded = Nx.iota({n, @embed}, axis: 0) |> Nx.as_type(:f32) |> Nx.divide(n)
    %Data{frames: frames, embedded_frames: embedded, size: n}
  end

  defp val_batches(seed) do
    dataset([60, 60, 60])
    |> TrajectoryCursors.batch_stream(
      batch_size: 2,
      unroll: @unroll,
      overlap: 0,
      seed: seed,
      gpu: false
    )
    |> Enum.to_list()
  end

  defp trainer(opts \\ []) do
    Imitation.new(
      embed_size: @embed,
      temporal: true,
      bptt: true,
      backbone: :gru,
      head: :autoregressive,
      unroll: @unroll,
      hidden_size: @hidden,
      num_layers: @layers,
      precision: :f32,
      batch_size: 2,
      dropout: Keyword.get(opts, :dropout, 0.1)
    )
  end

  test "evaluate_bptt is deterministic (same batches -> identical loss)" do
    t = trainer()
    batches = val_batches(7)

    %{loss: l1, num_batches: n} = Validation.evaluate_bptt(t, batches)
    %{loss: l2, num_batches: ^n} = Validation.evaluate_bptt(t, batches)

    assert n > 1
    assert l1 == l2
  end

  test "padded loader validation equals complete independent replay forwards" do
    t = trainer()
    ds = dataset([13, 60, 21])

    chunks =
      TrajectoryCursors.batch_stream(ds,
        batch_size: 2,
        unroll: @unroll,
        gpu: false,
        neutral_weight: 1.0
      )

    result = Validation.evaluate_bptt(t, chunks)

    expected =
      ExPhil.Evaluation.BPTT.batches(ds, 100)
      |> Enum.map(fn b ->
        n = Nx.axis_size(b.states, 1)

        {loss, _} =
          t.eval_loss_fn.(
            t.policy_params,
            b.states,
            b.actions,
            Nx.broadcast(1.0, {1, n}),
            Nx.broadcast(0.0, {1, @layers, @hidden})
          )

        Nx.to_number(loss) * n
      end)
      |> Enum.sum()
      |> Kernel./(94)

    assert result.weight == 94
    assert_in_delta result.loss, expected, 1.0e-5
  end

  test "the carry is load-bearing: correct threading differs from zeroed-every-batch" do
    t = trainer()
    batches = val_batches(7)

    %{loss: threaded} = Validation.evaluate_bptt(t, batches)

    # Force a reset on EVERY batch (carry never survives) — a different
    # (wrong) protocol must give a different number.
    all_reset =
      Enum.map(batches, fn b ->
        %{b | is_resetting: Nx.broadcast(Nx.tensor(1, type: :u8), Nx.shape(b.is_resetting))}
      end)

    %{loss: zeroed} = Validation.evaluate_bptt(t, all_reset)

    refute threaded == zeroed
  end

  test "eval and train losses agree when dropout is disabled" do
    t = trainer(dropout: 0.0)
    [batch | _] = val_batches(7)
    carry = Nx.broadcast(0.0, {2, @layers, @hidden})

    {eval_loss, _h} =
      t.eval_loss_fn.(t.policy_params, batch.states, batch.actions, batch.frame_weights, carry)

    {{train_loss, _h2}, _grads} =
      t.loss_and_grad_fn.(
        t.policy_params,
        batch.states,
        batch.actions,
        batch.frame_weights,
        carry
      )

    assert_in_delta Nx.to_number(eval_loss), Nx.to_number(train_loss), 1.0e-5
  end
end
