defmodule ExPhil.Networks.Policy.ExecutionContract do
  @moduledoc "Versioned windowed recurrent-state and arithmetic contracts. Unstamped checkpoints remain legacy."

  def training(config) do
    state = Map.get(config, :recurrent_state, :legacy_random)
    precision = Map.fetch!(config, :precision)

    case state do
      :zeros ->
        unless config[:temporal] == true and config[:backbone] == :gru and
                 config[:bptt] in [nil, false] and precision == :f32 and
                 config[:mixed_precision] in [nil, false],
               do:
                 raise(
                   ArgumentError,
                   "zero-state v1 requires windowed temporal GRU, F32, and no mixed precision"
                 )

        %{
          execution_contract: :windowed_gru_f32_v1,
          recurrent_state: :zeros,
          training_precision: :f32,
          inference_precision: :f32
        }

      :legacy_random ->
        %{
          execution_contract: :legacy,
          recurrent_state: :legacy_random,
          training_precision: precision,
          inference_precision: :f32
        }

      _ ->
        raise ArgumentError, "unsupported recurrent state: #{inspect(state)}"
    end
  end

  def load(config) do
    case config[:execution_contract] do
      version when version in [:windowed_gru_f32_v1, "windowed_gru_f32_v1"] ->
        unless config[:recurrent_state] in [:zeros, "zeros"] and
                 config[:training_precision] in [:f32, "f32"] and
                 config[:inference_precision] in [:f32, "f32"] and
                 config[:backbone] in [:gru, "gru"] and config[:temporal] == true and
                 config[:bptt] in [nil, false] and config[:mixed_precision] in [nil, false] and
                 config[:precision] in [nil, :f32, "f32"],
               do: raise(ArgumentError, "inconsistent windowed GRU execution contract")

        %{
          execution_contract: :windowed_gru_f32_v1,
          recurrent_state: :zeros,
          training_precision: :f32,
          inference_precision: :f32
        }

      version when version in [nil, :legacy, "legacy"] ->
        unless config[:recurrent_state] in [nil, :legacy_random, "legacy_random"] and
                 config[:inference_precision] in [nil, :f32, "f32"],
               do: raise(ArgumentError, "unsupported unstamped or legacy execution contract")

        %{
          execution_contract: :legacy,
          recurrent_state: :legacy_random,
          training_precision: config[:training_precision] || :unknown,
          inference_precision: :f32
        }

      other ->
        raise ArgumentError, "unsupported execution contract: #{inspect(other)}"
    end
  end

  def verify_report!(config, report) do
    contract = load(config)
    expected = contract |> Jason.encode!() |> Jason.decode!()
    actual = report["execution_contract"]

    unless actual == expected or (actual == nil and contract.execution_contract == :legacy),
      do: raise(ArgumentError, "teacher report execution contract mismatch")

    :ok
  end
end
