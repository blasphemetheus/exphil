defmodule ExPhil.Training.Callback do
  @moduledoc """
  Behaviour for training callbacks.

  Callbacks hook into the training loop at defined points, enabling modular
  features like progress bars, checkpointing, early stopping, and diagnostics
  without coupling them to the training loop itself.

  ## Hook Points

  Hooks are called in this order during training:

      on_train_begin
      │
      ├─ on_epoch_begin
      │  ├─ on_batch_end (per batch)
      │  └─ on_epoch_end (after validation)
      │
      └─ on_train_end

  Plus `on_interrupt` for graceful shutdown (Ctrl+C / SIGTERM).

  ## Implementing a Callback

      defmodule MyCallback do
        @behaviour ExPhil.Training.Callback

        @impl true
        def init(opts), do: %{my_state: opts[:some_option]}

        @impl true
        def on_epoch_end(state, callback_state) do
          IO.puts("Epoch \#{state.epoch} done!")
          {:cont, state, callback_state}
        end
      end

  ## Return Values

  All hooks return one of:
  - `{:cont, state, callback_state}` — continue training with (possibly modified) state
  - `{:halt_epoch, state, callback_state}` — skip remaining batches, proceed to epoch end hooks
  - `{:halt, state, callback_state}` — stop training entirely (e.g., early stopping)

  Hooks that are not implemented default to `{:cont, state, callback_state}`.

  ## Callback State

  Each callback has its own private state (returned from `init/1`), separate from
  the shared `TrainingState`. Use callback state for internal bookkeeping (counters,
  best loss tracker, etc.) and `TrainingState` for data that other callbacks need.
  """

  alias ExPhil.Training.TrainingState

  @type callback_state :: term()
  @type hook_result ::
          {:cont, TrainingState.t(), callback_state()}
          | {:halt_epoch, TrainingState.t(), callback_state()}
          | {:halt, TrainingState.t(), callback_state()}

  @doc "Initialize callback-private state from options."
  @callback init(opts :: keyword()) :: callback_state()

  @doc "Called once before training begins."
  @callback on_train_begin(TrainingState.t(), callback_state()) :: hook_result()

  @doc "Called at the start of each epoch."
  @callback on_epoch_begin(TrainingState.t(), callback_state()) :: hook_result()

  @doc "Called after each training batch."
  @callback on_batch_end(TrainingState.t(), callback_state()) :: hook_result()

  @doc "Called after each epoch (after validation if applicable)."
  @callback on_epoch_end(TrainingState.t(), callback_state()) :: hook_result()

  @doc "Called once after training completes (or is halted)."
  @callback on_train_end(TrainingState.t(), callback_state()) :: hook_result()

  @doc "Called on interrupt signal (Ctrl+C / SIGTERM)."
  @callback on_interrupt(TrainingState.t(), callback_state()) :: hook_result()

  @optional_callbacks [
    on_train_begin: 2,
    on_epoch_begin: 2,
    on_batch_end: 2,
    on_epoch_end: 2,
    on_train_end: 2,
    on_interrupt: 2
  ]

  @doc false
  defmacro __using__(_opts) do
    quote do
      @behaviour ExPhil.Training.Callback

      @impl true
      def on_train_begin(state, cb), do: {:cont, state, cb}
      @impl true
      def on_epoch_begin(state, cb), do: {:cont, state, cb}
      @impl true
      def on_batch_end(state, cb), do: {:cont, state, cb}
      @impl true
      def on_epoch_end(state, cb), do: {:cont, state, cb}
      @impl true
      def on_train_end(state, cb), do: {:cont, state, cb}
      @impl true
      def on_interrupt(state, cb), do: {:cont, state, cb}

      defoverridable [
        on_train_begin: 2,
        on_epoch_begin: 2,
        on_batch_end: 2,
        on_epoch_end: 2,
        on_train_end: 2,
        on_interrupt: 2
      ]
    end
  end

  # ============================================================================
  # Callback Runner
  # ============================================================================

  @doc """
  Run a hook across all callbacks in order.

  Returns `{status, state, updated_callbacks}` where status is:
  - `:cont` — continue normally
  - `:halt_epoch` — skip remaining batches, proceed to epoch end
  - `:halt` — stop training entirely

  All callbacks run even after one returns `:halt`/`:halt_epoch` (for cleanup).
  The most severe status wins (`:halt` > `:halt_epoch` > `:cont`).
  """
  def run(callbacks, hook, state) do
    {final_state, updated_callbacks, status} =
      Enum.reduce(callbacks, {state, [], :cont}, fn {mod, cb_state}, {st, acc, current_status} ->
        if function_exported?(mod, hook, 2) do
          case apply(mod, hook, [st, cb_state]) do
            {:halt, new_st, new_cb} ->
              {new_st, [{mod, new_cb} | acc], :halt}
            {:halt_epoch, new_st, new_cb} ->
              new_status = if current_status == :halt, do: :halt, else: :halt_epoch
              {new_st, [{mod, new_cb} | acc], new_status}
            {:cont, new_st, new_cb} ->
              {new_st, [{mod, new_cb} | acc], current_status}
          end
        else
          {st, [{mod, cb_state} | acc], current_status}
        end
      end)

    {status, final_state, Enum.reverse(updated_callbacks)}
  end

  @doc """
  Initialize all callbacks from a list of `{Module, opts}` tuples.

  Returns `[{Module, callback_state}]`.
  """
  def init_all(callback_specs) do
    Enum.map(callback_specs, fn
      {mod, opts} -> {mod, mod.init(opts)}
      mod when is_atom(mod) -> {mod, mod.init([])}
    end)
  end

  @doc """
  Increment event count for a hook in TrainingState.

  Used by Trainer.fit to track how many times each hook has fired,
  enabling filter predicates like `{:every, 10}`.
  """
  def increment_event_count(state, hook) do
    counts = Map.update(state.event_counts, hook, 1, &(&1 + 1))
    %{state | event_counts: counts}
  end

  @doc """
  Check if a filter predicate matches the current event count.

  ## Filters
  - `:always` — always matches (default)
  - `{:every, n}` — matches every n-th firing
  - `{:once}` — matches only the first firing
  - `{:after, n}` — matches after n firings
  """
  def filter_matches?(filter, state, hook) do
    count = Map.get(state.event_counts, hook, 0)

    case filter do
      :always -> true
      {:every, n} -> rem(count, n) == 0
      :once -> count == 1
      {:after, n} -> count > n
      fun when is_function(fun, 2) -> fun.(state, count)
      _ -> true
    end
  end
end
