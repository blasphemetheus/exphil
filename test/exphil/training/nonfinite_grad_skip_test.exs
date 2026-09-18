defmodule ExPhil.Training.NonFiniteGradSkipTest do
  use ExUnit.Case, async: false
  alias ExPhil.Training.Imitation
  alias ExPhil.Training.Imitation.TrainLoop

  # V3 2026-09-18: one non-finite gradient element made clip_by_global_norm's
  # scale NaN and Adam wrote NaN into every weight while the loss was still
  # finite. The BPTT step must skip such an update and leave params intact.
  defp trainer, do: Imitation.new(embed_size: 16, temporal: true, bptt: true, backbone: :gru, head: :autoregressive, unroll: 4, hidden_size: 8, num_layers: 1, precision: :f32, batch_size: 2, dropout: 0.0, learning_rate: 1.0e-3, warmup_steps: 0)

  defp batch(states) do
    %{
      states: states,
      actions: %{buttons: Nx.broadcast(0, {2, 4, 8}), main_x: Nx.broadcast(8, {2, 4}), main_y: Nx.broadcast(8, {2, 4}), c_x: Nx.broadcast(8, {2, 4}), c_y: Nx.broadcast(8, {2, 4}), shoulder: Nx.broadcast(0, {2, 4})},
      frame_weights: Nx.broadcast(1.0, {2, 4}),
      is_resetting: Nx.tensor([1, 1], type: :u8)
    }
  end

  test "grads_finite?/1 distinguishes finite from NaN/Inf gradient trees" do
    assert TrainLoop.grads_finite?(%{"a" => %{"k" => Nx.tensor([1.0, 2.0])}})
    refute TrainLoop.grads_finite?(%{"a" => %{"k" => Nx.tensor([1.0, :nan])}})
    refute TrainLoop.grads_finite?(%{"a" => %{"k" => Nx.tensor([:infinity, 2.0])}})
  end

  test "a finite step updates params; a step with non-finite gradients is skipped and params survive" do
    t0 = trainer()
    carry = Nx.broadcast(0.0, {2, 1, 8})
    walk = fn walk, v -> cond do
      is_struct(v, Nx.Tensor) -> Nx.to_flat_list(v)
      is_map(v) and not is_struct(v) -> Enum.flat_map(v, fn {_, x} -> walk.(walk, x) end)
      true -> []
    end end
    flat = fn tr -> tr.policy_params |> ExPhil.Training.Utils.ensure_model_state() |> Map.get(:data) |> then(&walk.(walk, &1)) end

    {t1, m1, _} = TrainLoop.train_step_bptt(t0, batch(Nx.broadcast(0.5, {2, 4, 16})), carry)
    assert m1.skipped == false
    assert flat.(t1) != flat.(t0)
    assert Enum.all?(flat.(t1), &is_float/1)

    # Inf in the inputs -> non-finite gradients somewhere -> skip
    {t2, m2, _} = TrainLoop.train_step_bptt(t1, batch(Nx.tensor(:infinity) |> Nx.broadcast({2, 4, 16})), carry)
    assert m2.skipped == true
    assert flat.(t2) == flat.(t1)
    assert t2.step == t1.step + 1
  end
end
