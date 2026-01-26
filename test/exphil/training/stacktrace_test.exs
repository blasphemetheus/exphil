defmodule ExPhil.Training.StacktraceTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Stacktrace

  describe "keep_frame?/1" do
    test "keeps ExPhil module frames" do
      frame =
        {ExPhil.Training.Imitation, :train_step, 3,
         [file: "lib/exphil/training/imitation.ex", line: 42]}

      assert Stacktrace.keep_frame?(frame)
    end

    test "keeps user script frames" do
      frame = {:"Elixir.Mix.Tasks.Train", :run, 1, [file: "lib/mix/tasks/train.ex", line: 10]}
      assert Stacktrace.keep_frame?(frame)
    end

    test "filters Nx.Defn frames" do
      frame =
        {Nx.Defn.Compiler, :__compile__, 4, [file: "deps/nx/lib/nx/defn/compiler.ex", line: 100]}

      refute Stacktrace.keep_frame?(frame)
    end

    test "filters EXLA frames" do
      frame = {EXLA.Defn, :__jit__, 5, [file: "deps/exla/lib/exla/defn.ex", line: 50]}
      refute Stacktrace.keep_frame?(frame)
    end

    test "filters Axon frames" do
      frame = {Axon.Compiler, :build, 3, [file: "deps/axon/lib/axon/compiler.ex", line: 200]}
      refute Stacktrace.keep_frame?(frame)
    end

    test "filters Erlang OTP frames" do
      frame = {:gen_server, :call, 3, [file: ~c"gen_server.erl", line: 234]}
      refute Stacktrace.keep_frame?(frame)
    end

    test "filters based on file path" do
      frame = {SomeModule, :func, 1, [file: "_build/dev/lib/deps/nx/lib/nx.ex", line: 10]}
      refute Stacktrace.keep_frame?(frame)
    end
  end

  describe "simplify_stacktrace/1" do
    test "filters internal frames" do
      stacktrace = [
        {ExPhil.Training.Imitation, :train_step, 3,
         [file: "lib/exphil/training/imitation.ex", line: 42]},
        {Nx.Defn.Compiler, :__compile__, 4, [file: "deps/nx/lib/nx/defn/compiler.ex", line: 100]},
        {EXLA.Defn, :__jit__, 5, [file: "deps/exla/lib/exla/defn.ex", line: 50]},
        {ExPhil.Networks.Policy, :forward, 2, [file: "lib/exphil/networks/policy.ex", line: 100]}
      ]

      simplified = Stacktrace.simplify_stacktrace(stacktrace)

      assert length(simplified) == 2
      assert {ExPhil.Training.Imitation, :train_step, 3, _} = hd(simplified)
    end

    test "limits to 10 frames" do
      # Create 15 ExPhil frames
      stacktrace =
        for i <- 1..15 do
          {ExPhil.Test, :func, 1, [file: "lib/exphil/test.ex", line: i]}
        end

      simplified = Stacktrace.simplify_stacktrace(stacktrace)

      assert length(simplified) == 10
    end

    test "handles empty stacktrace" do
      assert Stacktrace.simplify_stacktrace([]) == []
    end
  end

  describe "format_exception/2" do
    test "formats exception with simplified trace" do
      exception = %RuntimeError{message: "test error"}

      stacktrace = [
        {ExPhil.Training.Imitation, :train_step, 3,
         [file: "lib/exphil/training/imitation.ex", line: 42]},
        {Nx.Defn.Compiler, :__compile__, 4, [file: "deps/nx/lib/nx/defn/compiler.ex", line: 100]}
      ]

      result = Stacktrace.format_exception(exception, stacktrace)

      assert result =~ "test error"
      assert result =~ "ExPhil.Training.Imitation"
      refute result =~ "Nx.Defn.Compiler"
    end
  end

  describe "print_exception/2" do
    import ExUnit.CaptureIO

    test "prints formatted exception" do
      exception = %RuntimeError{message: "something went wrong"}

      stacktrace = [
        {ExPhil.Training.Imitation, :train_step, 3,
         [file: "lib/exphil/training/imitation.ex", line: 42]}
      ]

      output =
        capture_io(:stderr, fn ->
          Stacktrace.print_exception(exception, stacktrace)
        end)

      assert output =~ "Error: something went wrong"
      assert output =~ "Stacktrace"
      assert output =~ "ExPhil.Training.Imitation"
    end

    test "shows hidden frame count" do
      exception = %RuntimeError{message: "error"}

      stacktrace = [
        {ExPhil.Test, :func, 1, [file: "lib/exphil/test.ex", line: 1]},
        {Nx.Defn.Compiler, :compile, 2, [file: "deps/nx/lib/nx/defn/compiler.ex", line: 50]},
        {EXLA.Defn, :jit, 3, [file: "deps/exla/lib/exla/defn.ex", line: 100]},
        {Axon.Compiler, :build, 4, [file: "deps/axon/lib/axon/compiler.ex", line: 200]}
      ]

      output =
        capture_io(:stderr, fn ->
          Stacktrace.print_exception(exception, stacktrace)
        end)

      assert output =~ "internal frames hidden"
      assert output =~ "EXPHIL_FULL_STACKTRACE"
    end
  end

  describe "show_full_trace?/0" do
    test "returns false by default" do
      # Clear env var if set
      System.delete_env("EXPHIL_FULL_STACKTRACE")
      refute Stacktrace.show_full_trace?()
    end

    test "returns true when env var is set" do
      System.put_env("EXPHIL_FULL_STACKTRACE", "1")
      assert Stacktrace.show_full_trace?()
      System.delete_env("EXPHIL_FULL_STACKTRACE")
    end
  end
end
