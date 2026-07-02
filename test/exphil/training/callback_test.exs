defmodule ExPhil.Training.CallbackTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.{Callback, TrainingState}

  defmodule CounterCallback do
    use ExPhil.Training.Callback

    @impl true
    def init(opts), do: %{count: 0, label: Keyword.get(opts, :label, "test")}

    @impl true
    def on_epoch_end(state, cb) do
      {:cont, state, %{cb | count: cb.count + 1}}
    end
  end

  defmodule HaltCallback do
    use ExPhil.Training.Callback

    @impl true
    def init(opts), do: %{halt_at: Keyword.get(opts, :halt_at, 2)}

    @impl true
    def on_epoch_end(state, cb) do
      if state.epoch >= cb.halt_at do
        {:halt, %{state | halt: true}, cb}
      else
        {:cont, state, cb}
      end
    end
  end

  defmodule MetaCallback do
    use ExPhil.Training.Callback

    @impl true
    def init(_), do: %{}

    @impl true
    def on_train_begin(state, cb) do
      {:cont, TrainingState.put_meta(state, :started, true), cb}
    end

    @impl true
    def on_train_end(state, cb) do
      {:cont, TrainingState.put_meta(state, :finished, true), cb}
    end
  end

  describe "init_all/1" do
    test "initializes callbacks from {module, opts} tuples" do
      callbacks = Callback.init_all([
        {CounterCallback, [label: "a"]},
        {CounterCallback, [label: "b"]}
      ])

      assert length(callbacks) == 2
      [{mod1, state1}, {mod2, state2}] = callbacks
      assert mod1 == CounterCallback
      assert state1.label == "a"
      assert mod2 == CounterCallback
      assert state2.label == "b"
    end

    test "initializes bare module atoms" do
      [{mod, state}] = Callback.init_all([MetaCallback])
      assert mod == MetaCallback
      assert state == %{}
    end
  end

  describe "run/3" do
    test "dispatches hook to all callbacks and returns updated state" do
      callbacks = Callback.init_all([
        {CounterCallback, [label: "a"]},
        {CounterCallback, [label: "b"]}
      ])

      state = %TrainingState{epoch: 1}
      {:cont, _state, updated} = Callback.run(callbacks, :on_epoch_end, state)

      [{_, cb1}, {_, cb2}] = updated
      assert cb1.count == 1
      assert cb2.count == 1
    end

    test "preserves callback state across multiple calls" do
      callbacks = Callback.init_all([{CounterCallback, []}])
      state = %TrainingState{epoch: 1}

      {:cont, state, callbacks} = Callback.run(callbacks, :on_epoch_end, state)
      {:cont, _state, callbacks} = Callback.run(callbacks, :on_epoch_end, state)

      [{_, cb}] = callbacks
      assert cb.count == 2
    end

    test "halt propagates but all callbacks still run" do
      callbacks = Callback.init_all([
        {HaltCallback, [halt_at: 1]},
        {CounterCallback, []}
      ])

      state = %TrainingState{epoch: 1}
      {result, _state, updated} = Callback.run(callbacks, :on_epoch_end, state)

      assert result == :halt
      # CounterCallback still ran even though HaltCallback halted
      [{_, _halt_cb}, {_, counter_cb}] = updated
      assert counter_cb.count == 1
    end

    test "cont when no callback halts" do
      callbacks = Callback.init_all([{CounterCallback, []}])
      state = %TrainingState{epoch: 1}

      {result, _state, _callbacks} = Callback.run(callbacks, :on_epoch_end, state)
      assert result == :cont
    end

    test "handles hooks with no implementation (default no-op)" do
      callbacks = Callback.init_all([{CounterCallback, []}])
      state = %TrainingState{}

      # CounterCallback only implements on_epoch_end, not on_train_begin
      {result, _state, _callbacks} = Callback.run(callbacks, :on_train_begin, state)
      assert result == :cont
    end
  end

  describe "TrainingState" do
    test "put_meta and get_meta" do
      state = %TrainingState{}
      state = TrainingState.put_meta(state, :key, "value")
      assert TrainingState.get_meta(state, :key) == "value"
      assert TrainingState.get_meta(state, :missing, "default") == "default"
    end

    test "update_meta" do
      state = %TrainingState{}
      state = TrainingState.put_meta(state, :count, 0)
      state = TrainingState.update_meta(state, :count, 0, &(&1 + 1))
      assert TrainingState.get_meta(state, :count) == 1
    end

    test "callbacks can modify meta through hooks" do
      callbacks = Callback.init_all([MetaCallback])
      state = %TrainingState{}

      {:cont, state, callbacks} = Callback.run(callbacks, :on_train_begin, state)
      assert TrainingState.get_meta(state, :started) == true

      {:cont, state, _} = Callback.run(callbacks, :on_train_end, state)
      assert TrainingState.get_meta(state, :finished) == true
    end
  end
end
