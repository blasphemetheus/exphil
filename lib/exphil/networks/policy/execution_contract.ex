defmodule ExPhil.Networks.Policy.ExecutionContract do
  @moduledoc "Versioned recurrent-state and arithmetic contracts. Unstamped checkpoints remain legacy."

  def training(%{bptt: true} = config) do
    unless config[:temporal] == true,
      do: raise(ArgumentError, "BPTT requires temporal training")

    unless config[:backbone] == :gru,
      do: raise(ArgumentError, "BPTT supports gru only")

    unless config[:temporal] == true and config[:backbone] == :gru and
             config[:precision] == :f32 and config[:mixed_precision] in [nil, false] and
             config[:recurrent_state] in [nil, :carried_zero],
           do:
             raise(ArgumentError, "BPTT v1 requires F32 GRU with explicit zero-initialized carry")

    %{
      execution_contract: :bptt_gru_f32_v1,
      recurrent_state: :carried_zero,
      training_precision: :f32,
      inference_precision: :f32
    }
  end

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
      version when version in [:bptt_gru_f32_v1, "bptt_gru_f32_v1"] ->
        unless config[:recurrent_state] in [:carried_zero, "carried_zero"] and
                 config[:training_precision] in [:f32, "f32"] and
                 config[:inference_precision] in [:f32, "f32"] and
                 config[:precision] in [:f32, "f32"] and
                 config[:backbone] in [:gru, "gru"] and config[:temporal] == true and
                 config[:bptt] == true and config[:mixed_precision] in [nil, false],
               do: raise(ArgumentError, "inconsistent BPTT GRU execution contract")

        %{
          execution_contract: :bptt_gru_f32_v1,
          recurrent_state: :carried_zero,
          training_precision: :f32,
          inference_precision: :f32
        }

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

  def validate_inference!(config, stateful_step) do
    execution = load(config)

    if execution.execution_contract == :windowed_gru_f32_v1 and stateful_step,
      do: raise(ArgumentError, "windowed GRU v1 cannot use stateful-step inference")

    if execution.execution_contract == :bptt_gru_f32_v1 and not stateful_step,
      do: raise(ArgumentError, "BPTT GRU v1 requires stateful-step inference")

    execution
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
