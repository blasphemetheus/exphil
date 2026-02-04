defmodule ExPhil.Training.ReplayValidationTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.ReplayValidation
  alias ExPhil.Error.ValidationError

  @test_dir Path.join(System.tmp_dir!(), "replay_validation_test_#{:erlang.unique_integer()}")

  setup do
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      File.rm_rf!(@test_dir)
    end)

    :ok
  end

  describe "validate_file/1" do
    test "returns error for non-existent file" do
      path = Path.join(@test_dir, "missing.slp")
      {^path, {:error, %ValidationError{reason: :not_found}}} = ReplayValidation.validate_file(path)
    end

    test "returns error for file too small" do
      path = Path.join(@test_dir, "tiny.slp")
      # Write a file that's too small
      File.write!(path, "small")

      {^path, {:error, %ValidationError{reason: :file_too_small}}} = ReplayValidation.validate_file(path)
    end

    test "returns error for invalid format" do
      path = Path.join(@test_dir, "invalid.slp")
      # Write a file with wrong magic bytes but correct size
      content = String.duplicate("X", 5000)
      File.write!(path, content)

      {^path, {:error, %ValidationError{reason: :invalid_format}}} = ReplayValidation.validate_file(path)
    end

    test "returns ok for valid SLP format" do
      path = Path.join(@test_dir, "valid.slp")
      # Write a file with correct magic bytes and size
      # SLP files start with {U (0x7B 0x55)
      header = <<0x7B, 0x55, 0x03, 0x36>>
      padding = String.duplicate(<<0>>, 5000)
      File.write!(path, header <> padding)

      {^path, :ok} = ReplayValidation.validate_file(path)
    end
  end

  describe "validate/2" do
    test "returns all valid paths and stats for empty list" do
      {:ok, paths, stats} = ReplayValidation.validate([], show_progress: false)

      assert paths == []
      assert stats.total == 0
      assert stats.valid == 0
      assert stats.invalid == 0
    end

    test "filters out invalid files and returns stats" do
      # Create one valid and one invalid file
      valid_path = Path.join(@test_dir, "valid.slp")
      invalid_path = Path.join(@test_dir, "invalid.slp")

      header = <<0x7B, 0x55, 0x03, 0x36>>
      padding = String.duplicate(<<0>>, 5000)
      File.write!(valid_path, header <> padding)
      File.write!(invalid_path, "too small")

      {:ok, paths, stats} =
        ReplayValidation.validate(
          [valid_path, invalid_path],
          show_progress: false
        )

      assert paths == [valid_path]
      assert stats.total == 2
      assert stats.valid == 1
      assert stats.invalid == 1
      assert length(stats.errors) == 1
    end

    test "handles non-existent files" do
      missing_path = Path.join(@test_dir, "missing.slp")

      {:ok, paths, stats} =
        ReplayValidation.validate(
          [missing_path],
          show_progress: false
        )

      assert paths == []
      assert stats.invalid == 1
      assert [{^missing_path, %ValidationError{reason: :not_found}}] = stats.errors
    end

    test "runs in parallel by default" do
      # Create multiple valid files
      paths =
        for i <- 1..10 do
          path = Path.join(@test_dir, "valid_#{i}.slp")
          header = <<0x7B, 0x55, 0x03, 0x36>>
          padding = String.duplicate(<<0>>, 5000)
          File.write!(path, header <> padding)
          path
        end

      {:ok, valid_paths, stats} =
        ReplayValidation.validate(
          paths,
          show_progress: false,
          parallel: true
        )

      assert length(valid_paths) == 10
      assert stats.valid == 10
    end

    test "sequential validation works" do
      path = Path.join(@test_dir, "valid.slp")
      header = <<0x7B, 0x55, 0x03, 0x36>>
      padding = String.duplicate(<<0>>, 5000)
      File.write!(path, header <> padding)

      {:ok, [^path], stats} =
        ReplayValidation.validate(
          [path],
          show_progress: false,
          parallel: false
        )

      assert stats.valid == 1
    end
  end

  describe "quick_validate/1" do
    test "separates existing from non-existing files" do
      existing_path = Path.join(@test_dir, "exists.slp")
      missing_path = Path.join(@test_dir, "missing.slp")

      File.write!(existing_path, "content")

      {:ok, valid, invalid} = ReplayValidation.quick_validate([existing_path, missing_path])

      assert valid == [existing_path]
      assert invalid == [missing_path]
    end

    test "handles empty list" do
      {:ok, [], []} = ReplayValidation.quick_validate([])
    end
  end
end
