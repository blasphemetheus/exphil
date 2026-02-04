defmodule ExPhil.ScriptTemplateTest do
  use ExUnit.Case, async: true

  alias ExPhil.ScriptTemplate

  describe "setup_environment/1" do
    test "sets XLA multi-threading flag" do
      # Clear any existing XLA_FLAGS
      original = System.get_env("XLA_FLAGS")
      System.delete_env("XLA_FLAGS")

      ScriptTemplate.setup_environment()

      flags = System.get_env("XLA_FLAGS")
      assert String.contains?(flags, "xla_cpu_multi_thread_eigen=true")

      # Restore original
      if original, do: System.put_env("XLA_FLAGS", original)
    end

    test "preserves existing XLA flags" do
      original = System.get_env("XLA_FLAGS")
      System.put_env("XLA_FLAGS", "--existing_flag=true")

      ScriptTemplate.setup_environment()

      flags = System.get_env("XLA_FLAGS")
      assert String.contains?(flags, "--existing_flag=true")
      assert String.contains?(flags, "xla_cpu_multi_thread_eigen=true")

      # Restore original
      if original do
        System.put_env("XLA_FLAGS", original)
      else
        System.delete_env("XLA_FLAGS")
      end
    end

    test "suppresses XLA logs by default" do
      ScriptTemplate.setup_environment()

      # Should set TF_CPP_MIN_LOG_LEVEL
      assert System.get_env("TF_CPP_MIN_LOG_LEVEL") != nil
    end

    test "does not suppress logs when verbose: true" do
      original = System.get_env("TF_CPP_MIN_LOG_LEVEL")
      System.delete_env("TF_CPP_MIN_LOG_LEVEL")

      ScriptTemplate.setup_environment(verbose: true)

      # TF_CPP_MIN_LOG_LEVEL should not be set
      assert System.get_env("TF_CPP_MIN_LOG_LEVEL") == nil

      # Restore
      if original, do: System.put_env("TF_CPP_MIN_LOG_LEVEL", original)
    end
  end

  describe "validate_dir!/2" do
    test "returns :ok for existing directory" do
      assert :ok = ScriptTemplate.validate_dir!(".", "Test directory")
    end

    test "returns :ok for /tmp" do
      assert :ok = ScriptTemplate.validate_dir!("/tmp", "Temp directory")
    end
  end

  describe "validate_file!/2" do
    test "returns :ok for existing file" do
      # mix.exs always exists in the project
      assert :ok = ScriptTemplate.validate_file!("mix.exs", "Mix file")
    end
  end

  describe "validate_required!/2" do
    test "returns :ok for non-nil values" do
      assert :ok = ScriptTemplate.validate_required!("value", "--flag")
      assert :ok = ScriptTemplate.validate_required!(0, "--count")
      assert :ok = ScriptTemplate.validate_required!(false, "--enabled")
    end
  end

  describe "validate_positive!/2" do
    test "returns :ok for positive numbers" do
      assert :ok = ScriptTemplate.validate_positive!(1, "--count")
      assert :ok = ScriptTemplate.validate_positive!(0.5, "--rate")
      assert :ok = ScriptTemplate.validate_positive!(100, "--epochs")
    end
  end

  describe "validate_range!/4" do
    test "returns :ok for values in range" do
      assert :ok = ScriptTemplate.validate_range!(5, 0, 10, "--value")
      assert :ok = ScriptTemplate.validate_range!(0, 0, 10, "--value")
      assert :ok = ScriptTemplate.validate_range!(10, 0, 10, "--value")
      assert :ok = ScriptTemplate.validate_range!(0.5, 0.0, 1.0, "--rate")
    end
  end

  describe "print_startup/3" do
    test "prints banner and config" do
      # Capture output - this is mostly a smoke test
      import ExUnit.CaptureIO

      output = capture_io(:stderr, fn ->
        ScriptTemplate.print_startup("Test Script", [
          {"Key1", "value1"},
          {"Key2", 42}
        ], show_gpu: false)
      end)

      assert output =~ "Test Script"
      assert output =~ "Key1"
      assert output =~ "value1"
      assert output =~ "Key2"
      assert output =~ "42"
    end
  end

  describe "step/3" do
    test "prints step indicator" do
      import ExUnit.CaptureIO

      output = capture_io(:stderr, fn ->
        ScriptTemplate.step(1, 3, "First step")
      end)

      assert output =~ "Step 1/3"
      assert output =~ "First step"
    end
  end

  describe "success/1" do
    test "prints success message" do
      import ExUnit.CaptureIO

      output = capture_io(:stderr, fn ->
        ScriptTemplate.success("Test passed!")
      end)

      assert output =~ "Test passed!"
      assert output =~ "✓"
    end
  end

  describe "error/1" do
    test "prints error message" do
      import ExUnit.CaptureIO

      output = capture_io(:stderr, fn ->
        ScriptTemplate.error("Something failed")
      end)

      assert output =~ "Something failed"
      assert output =~ "❌"
    end
  end
end
