defmodule ExPhilTest do
  use ExUnit.Case, async: true

  # Run doctests for the main module
  doctest ExPhil

  describe "version/0" do
    test "returns a version string" do
      version = ExPhil.version()
      assert is_binary(version)
      assert version =~ ~r/^\d+\.\d+\.\d+/
    end
  end

  describe "default_embed_config/0" do
    test "returns a config struct with expected keys" do
      config = ExPhil.default_embed_config()
      assert is_struct(config)
      assert Map.has_key?(config, :stage_mode)
      assert Map.has_key?(config, :player)
    end
  end

  describe "embedding_size/0" do
    test "returns default embedding size" do
      size = ExPhil.embedding_size()
      assert is_integer(size)
      # Default learned embeddings should be ~288 dims
      assert size > 200 and size < 400
    end
  end
end
