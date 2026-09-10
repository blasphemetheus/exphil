defmodule ExPhil.Training.Callbacks.PolicyExport do
  @moduledoc "Export policy .bin file and config JSON on training completion."

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Imitation, Output, Config}

  @impl true
  def init(opts), do: %{checkpoint_path: Keyword.get(opts, :checkpoint_path)}

  @impl true
  def on_train_end(state, cb) do
    checkpoint_path = cb.checkpoint_path || state.opts[:checkpoint]

    if checkpoint_path do
      # Export policy binary
      policy_path = Config.derive_policy_path(checkpoint_path)
      case Imitation.export_policy(state.trainer, policy_path) do
        :ok -> Output.puts("  Policy exported to #{policy_path}")
        {:error, reason} -> Output.puts("  Policy export failed: #{inspect(reason)}")
      end

      # Save config JSON (filter non-serializable values).
      #
      # 2026-09-09: this callback used to dump RAW opts, so every stamp that
      # `Config.build_config_json/2` adds (comparability_key,
      # label_convention, with_projectiles, provided_channels — INVARIANTS
      # items 1/4/11) was missing from the sidecar of every train.exs run
      # (the legacy train_from_replays.exs was the only caller). Two writers
      # of one artifact; the stamped map now wins over the raw opts.
      config_path = Config.derive_config_path(checkpoint_path)
      try do
        stamped =
          Config.build_config_json(state.opts, %{embed_size: state.trainer.config[:embed_size]})
          |> Enum.reject(fn {_k, v} -> is_nil(v) end)

        serializable = (state.opts ++ stamped)
          |> Enum.reject(fn {_k, v} -> is_function(v) or is_pid(v) or is_reference(v) end)
          |> Enum.map(fn
            {k, %Nx.Tensor{} = t} -> {k, Nx.to_flat_list(t)}
            {k, v} when is_struct(v) -> {k, inspect(v)}
            {k, v} when is_atom(v) -> {k, to_string(v)}
            {k, v} when is_tuple(v) -> {k, Tuple.to_list(v)}
            other -> other
          end)
          |> Enum.into(%{})
          |> Jason.encode!(pretty: true)

        File.write!(config_path, serializable)
        Output.puts("  Config saved to #{config_path}")
      rescue
        e -> Output.puts("  Config save failed: #{Exception.message(e)}")
      end
    end

    {:cont, state, cb}
  end
end
