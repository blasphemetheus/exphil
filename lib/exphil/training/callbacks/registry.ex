defmodule ExPhil.Training.Callbacks.Registry do
  @moduledoc "Register trained model in ExPhil model registry on training completion."

  use ExPhil.Training.Callback

  alias ExPhil.Training.{Output, Registry}

  @impl true
  def init(opts), do: %{enabled: Keyword.get(opts, :enabled, true)}

  @impl true
  def on_train_end(state, %{enabled: false} = cb), do: {:cont, state, cb}

  def on_train_end(state, cb) do
    opts = state.opts

    entry = %{
      checkpoint_path: opts[:checkpoint],
      training_config: opts,
      metrics: %{
        final_loss: state.train_loss,
        val_loss: state.val_loss,
        epochs_completed: state.epoch,
        total_steps: state.step
      },
      tags: build_tags(opts)
    }

    case Registry.register(entry) do
      {:ok, registered} ->
        Output.puts("  Registered as '#{registered.name}' (#{registered.id})")
      {:error, reason} ->
        Output.puts("  Registry failed: #{inspect(reason)}")
    end

    {:cont, state, cb}
  end

  defp build_tags(opts) do
    tags = [to_string(opts[:backbone] || :mlp)]
    tags = if opts[:temporal], do: ["temporal" | tags], else: tags
    tags = if opts[:train_character], do: [to_string(opts[:train_character]) | tags], else: tags
    tags
  end
end
