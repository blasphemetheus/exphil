defmodule ExPhil.Networks.RecurrentTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Recurrent

  @embed_size 64
  @hidden_size 32
  @seq_len 10
  @batch_size 4

  describe "build/1" do
    test "builds LSTM model with correct output shape" do
      model =
        Recurrent.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          cell_type: :lstm
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, hidden_size] (last timestep)
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "builds GRU model with correct output shape" do
      model =
        Recurrent.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          cell_type: :gru
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "return_sequences returns all timesteps" do
      model =
        Recurrent.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          return_sequences: true
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be [batch, seq_len, hidden_size]
      assert Nx.shape(output) == {@batch_size, @seq_len, @hidden_size}
    end

    test "stacked layers work correctly" do
      model =
        Recurrent.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          num_layers: 3,
          cell_type: :lstm
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_backbone/2" do
    test "integrates with existing input layer" do
      input = Axon.input("test_input", shape: {nil, nil, @embed_size})

      backbone =
        Recurrent.build_backbone(input,
          hidden_size: @hidden_size,
          cell_type: :gru
        )

      {init_fn, predict_fn} = Axon.build(backbone)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input_data = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_stateful/1" do
    test "processes single frames" do
      model =
        Recurrent.build_stateful(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          cell_type: :lstm
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 1, @embed_size}, :f32), Axon.ModelState.empty())

      # Single frame as sequence of 1
      frame = Nx.broadcast(0.5, {@batch_size, 1, @embed_size})
      output = predict_fn.(params, frame)

      # Output should be [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end

    test "works with GRU" do
      model =
        Recurrent.build_stateful(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          cell_type: :gru
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 1, @embed_size}, :f32), Axon.ModelState.empty())

      frame = Nx.broadcast(0.5, {@batch_size, 1, @embed_size})
      output = predict_fn.(params, frame)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end

  describe "build_hybrid/1" do
    test "combines recurrent and MLP layers" do
      model =
        Recurrent.build_hybrid(
          embed_size: @embed_size,
          recurrent_size: @hidden_size,
          mlp_sizes: [64, 32],
          cell_type: :lstm
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch_size, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      # Output should be last MLP layer size
      assert Nx.shape(output) == {@batch_size, 32}
    end
  end

  describe "initial_hidden/2" do
    test "creates LSTM hidden state" do
      {h, c} =
        Recurrent.initial_hidden(@batch_size,
          hidden_size: @hidden_size,
          cell_type: :lstm
        )

      assert Nx.shape(h) == {@batch_size, @hidden_size}
      assert Nx.shape(c) == {@batch_size, @hidden_size}
      assert Nx.to_number(Nx.sum(h)) == 0.0
      assert Nx.to_number(Nx.sum(c)) == 0.0
    end

    test "creates GRU hidden state" do
      hidden =
        Recurrent.initial_hidden(@batch_size,
          hidden_size: @hidden_size,
          cell_type: :gru
        )

      assert Nx.shape(hidden) == {@batch_size, @hidden_size}
      assert Nx.to_number(Nx.sum(hidden)) == 0.0
    end
  end

  describe "frames_to_sequence/1" do
    test "stacks frames into sequence" do
      frames =
        for _ <- 1..5 do
          Nx.broadcast(1.0, {@batch_size, @embed_size})
        end

      sequence = Recurrent.frames_to_sequence(frames)

      assert Nx.shape(sequence) == {@batch_size, 5, @embed_size}
    end

    test "handles unbatched frames" do
      frames =
        for i <- 1..3 do
          Nx.broadcast(i / 1.0, {@embed_size})
        end

      sequence = Recurrent.frames_to_sequence(frames)

      assert Nx.shape(sequence) == {1, 3, @embed_size}
    end
  end

  describe "pad_sequence/3" do
    test "pads shorter sequences" do
      sequence = Nx.broadcast(1.0, {@batch_size, 5, @embed_size})
      padded = Recurrent.pad_sequence(sequence, 10)

      assert Nx.shape(padded) == {@batch_size, 10, @embed_size}

      # First 5 frames should be zeros (padding at start)
      first_frame = Nx.slice(padded, [0, 0, 0], [@batch_size, 1, @embed_size])
      assert Nx.to_number(Nx.sum(first_frame)) == 0.0
    end

    test "truncates longer sequences" do
      sequence = Nx.broadcast(1.0, {@batch_size, 20, @embed_size})
      truncated = Recurrent.pad_sequence(sequence, 10)

      assert Nx.shape(truncated) == {@batch_size, 10, @embed_size}
    end

    test "returns unchanged if same length" do
      sequence = Nx.broadcast(1.0, {@batch_size, 10, @embed_size})
      result = Recurrent.pad_sequence(sequence, 10)

      assert Nx.shape(result) == {@batch_size, 10, @embed_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden size" do
      assert Recurrent.output_size(hidden_size: 128) == 128
      # default
      assert Recurrent.output_size() == 256
    end
  end

  describe "cell_types/0" do
    test "returns supported cell types" do
      assert Recurrent.cell_types() == [:lstm, :gru]
    end
  end

  describe "concrete seq_len regression test" do
    # Regression test for: dynamic sequence length causes slow JIT compilation
    # When seq_len is nil (dynamic), XLA compiles generic kernels which is slow.
    # The fix: default seq_len to window_size for efficient compilation.

    test "build/1 uses concrete seq_len by default (not nil)" do
      # This test verifies the model uses window_size as concrete seq_len
      window_size = 10

      model =
        Recurrent.build(
          embed_size: @embed_size,
          hidden_size: @hidden_size,
          cell_type: :lstm,
          window_size: window_size
          # NOTE: Not passing seq_len - should default to window_size
        )

      # Model should build and compile quickly (would be slow with dynamic shapes)
      {init_fn, predict_fn} = Axon.build(model)

      # Template must use window_size as seq_len
      params =
        init_fn.(
          Nx.template({@batch_size, window_size, @embed_size}, :f32),
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch_size, window_size, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @hidden_size}
    end
  end
end
