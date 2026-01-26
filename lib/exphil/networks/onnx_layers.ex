defmodule ExPhil.Networks.OnnxLayers do
  @moduledoc """
  ONNX-compatible layer implementations for use in models that need to be exported.

  These layers use `op_name` to tell axon_onnx how to serialize them to ONNX format.
  Unlike `Axon.nx`, which uses arbitrary functions that can't be serialized, these
  layers have explicit ONNX operator mappings.

  ## Usage

      # Instead of:
      Axon.nx(output_seq, fn tensor ->
        seq_len = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq_len - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end)

      # Use:
      OnnxLayers.sequence_last(output_seq, name: "last_timestep")

  ## Available Layers

  - `sequence_last/2` - Extract last timestep from sequence [batch, seq, hidden] -> [batch, hidden]
  """

  @doc """
  Extract the last timestep from a sequence tensor.

  Takes a 3D tensor of shape `[batch, seq_len, hidden]` and returns
  a 2D tensor of shape `[batch, hidden]` containing the last timestep.

  This is ONNX-serializable (uses Slice + Squeeze operators).

  ## Options

  - `:name` - Layer name (default: "sequence_last")

  ## Examples

      # LSTM output sequence -> last hidden state
      input = Axon.input("input", shape: {1, 60, 256})
      {output_seq, _states} = Axon.lstm(input, 128, name: "lstm")
      last_hidden = OnnxLayers.sequence_last(output_seq, name: "last")
      output = Axon.dense(last_hidden, 64, name: "output")

      # GRU with sequence_last for classification
      {output_seq, _} = Axon.gru(input, 128, name: "gru")
      OnnxLayers.sequence_last(output_seq)
      |> Axon.dense(num_classes, name: "classifier")
      |> Axon.softmax()
  """
  @spec sequence_last(Axon.t(), keyword()) :: Axon.t()
  def sequence_last(input, opts \\ []) do
    name = Keyword.get(opts, :name, "sequence_last")

    # Use Axon.layer/3 with signature: layer(op, inputs, opts)
    # The op_name: :sequence_last tells axon_onnx how to serialize it
    Axon.layer(
      fn inputs, _opts ->
        # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
        seq_len = Nx.axis_size(inputs, 1)

        Nx.slice_along_axis(inputs, seq_len - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      [input],
      name: name,
      op_name: :sequence_last
    )
  end
end
