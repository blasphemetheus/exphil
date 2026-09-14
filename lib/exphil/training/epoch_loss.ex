defmodule ExPhil.Training.EpochLoss do
  @moduledoc "Denominator-weighted running batch objectives, not a frozen-checkpoint evaluation."

  def new, do: %{weighted_sum: 0.0, mass: 0.0, batches: 0, invalid: nil}

  def add(accumulator, loss, batch, policy_type \\ :autoregressive) do
    loss = scalar(loss)
    mass = denominator(batch, policy_type)
    accumulator = %{accumulator | batches: accumulator.batches + 1}

    cond do
      accumulator.invalid != nil ->
        accumulator

      not is_number(loss) ->
        %{accumulator | invalid: :nonfinite_batch_loss}

      not is_number(mass) or mass <= 0 ->
        %{accumulator | invalid: :invalid_batch_mass}

      true ->
        %{
          accumulator
          | weighted_sum: accumulator.weighted_sum + loss * mass,
            mass: accumulator.mass + mass
        }
    end
  rescue
    ArithmeticError ->
      %{accumulator | invalid: :nonfinite_aggregate, batches: accumulator.batches + 1}
  end

  def mean(%{invalid: reason}) when reason != nil, do: reason
  def mean(%{mass: mass}) when mass <= 0, do: :empty_epoch
  def mean(%{weighted_sum: total, mass: mass}), do: total / mass

  defp denominator(%{frame_weights: %Nx.Tensor{} = weights}, :autoregressive),
    do: weights |> Nx.sum() |> Nx.to_number()

  defp denominator(%{states: %Nx.Batch{size: size}}, _policy_type), do: size
  defp denominator(%{states: states}, _policy_type), do: Nx.axis_size(states, 0)

  defp scalar(%Nx.Tensor{} = tensor), do: Nx.to_number(tensor)
  defp scalar(value), do: value
end
