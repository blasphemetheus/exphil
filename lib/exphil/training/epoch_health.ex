defmodule ExPhil.Training.EpochHealth do
  @moduledoc "Numerical acceptance of an epoch, not a behavioral-quality gate."

  def validate(loss, parameters) do
    cond do
      not is_number(loss) -> {:error, :nonfinite_loss}
      true -> validate_parameters(tensors(parameters))
    end
  end

  defp tensors(%Nx.Tensor{} = tensor), do: [tensor]
  defp tensors(value) when is_map(value), do: value |> Map.values() |> Enum.flat_map(&tensors/1)
  defp tensors(value) when is_tuple(value), do: value |> Tuple.to_list() |> tensors()
  defp tensors(value) when is_list(value), do: Enum.flat_map(value, &tensors/1)
  defp tensors(_value), do: []

  defp validate_parameters([]), do: {:error, :missing_parameters}

  defp validate_parameters(tensors) do
    if Enum.all?(tensors, fn tensor -> Enum.all?(Nx.to_flat_list(tensor), &is_number/1) end),
      do: :ok,
      else: {:error, :nonfinite_parameters}
  end
end
