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

  defp new_bptt_trainer(head) do
    Imitation.new(
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
    )
  end

  describe "Imitation.new bptt mode" do
    test "builds a trainer with the bptt loss fn and no eval fn" do
      trainer = new_bptt_trainer(:autoregressive)
      assert trainer.config[:bptt] == true
      assert is_function(trainer.loss_and_grad_fn)
      assert trainer.eval_loss_fn == nil
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
  end

  describe "end-to-end: cursors -> train_step_bptt" do
    for head <- [:autoregressive, :independent] do
      test "loss finite, carry flows and updates params (#{head} head)" do
        head = unquote(head)
        trainer = new_bptt_trainer(head)
        ds = dataset([60, 60])

        batches =
          TrajectoryCursors.batch_stream(ds,
            batch_size: @batch,
            unroll: @unroll,
            overlap: 1,
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
          overlap: 1,
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
end
