defmodule ExPhil.Training.FatalBatchCaptureTest do
  use ExUnit.Case, async: false
  alias ExPhil.Training.{Imitation, Trainer}

  # CLAUDE.md "capture the crime scene": on a non-finite BPTT loss the trainer
  # writes fatal_batch_<step>.bin with the batch, carry and PRE-update
  # params/optimizer state, so the step can be replayed offline.
  @tag :tmp_dir
  test "capture writes a replayable payload next to the checkpoint", %{tmp_dir: dir} do
    trainer = Imitation.new(embed_size: 16, temporal: true, bptt: true, backbone: :gru, head: :autoregressive, unroll: 4, hidden_size: 8, num_layers: 1, precision: :f32, batch_size: 2, dropout: 0.0)

    batch = %{
      states: Nx.broadcast(0.5, {2, 4, 16}),
      actions: %{buttons: Nx.broadcast(0, {2, 4, 8}), main_x: Nx.broadcast(8, {2, 4}), main_y: Nx.broadcast(8, {2, 4}), c_x: Nx.broadcast(8, {2, 4}), c_y: Nx.broadcast(8, {2, 4}), shoulder: Nx.broadcast(0, {2, 4})},
      frame_weights: Nx.broadcast(1.0, {2, 4}),
      is_resetting: Nx.tensor([1, 1], type: :u8)
    }

    carry = Nx.broadcast(0.0, {2, 1, 8})
    st = %{trainer: trainer, opts: [checkpoint: Path.join(dir, "model.axon")], step: 4242, epoch: 3}

    Trainer.capture_fatal_batch(st, batch, carry, :nan, 41)

    path = Path.join(dir, "fatal_batch_4242.bin")
    assert File.exists?(path)
    payload = path |> File.read!() |> :erlang.binary_to_term()
    assert payload.loss == :nan and payload.step == 4242 and payload.batch_idx == 41 and payload.epoch == 3

    # tensors round-trip through Nx.serialize
    states = Nx.deserialize(payload.batch.states)
    assert Nx.shape(states) == {2, 4, 16}
    assert Nx.shape(Nx.deserialize(payload.carry)) == {2, 1, 8}
    assert map_size(payload.policy_params) > 0
    assert payload.config.bptt == true
  end
end
