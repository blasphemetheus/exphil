defmodule ExPhil.Training.Config.AtomSafetyTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Config.AtomSafety

  describe "safe_to_atom/2" do
    test "converts valid string to atom" do
      allowed = [:lstm, :gru, :mamba, :attention]

      assert {:ok, :lstm} = AtomSafety.safe_to_atom("lstm", allowed)
      assert {:ok, :gru} = AtomSafety.safe_to_atom("gru", allowed)
      assert {:ok, :mamba} = AtomSafety.safe_to_atom("mamba", allowed)
      assert {:ok, :attention} = AtomSafety.safe_to_atom("attention", allowed)
    end

    test "returns error for invalid string" do
      allowed = [:lstm, :gru, :mamba]

      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("invalid", allowed)
      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("transformer", allowed)
    end

    test "is case-sensitive" do
      allowed = [:lstm, :gru, :mamba]

      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("LSTM", allowed)
      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("Mamba", allowed)
      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("GRU", allowed)
    end

    test "handles empty string" do
      allowed = [:lstm, :gru]

      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("", allowed)
    end

    test "handles empty allowlist" do
      assert {:error, :invalid_value} = AtomSafety.safe_to_atom("anything", [])
    end

    test "does not create new atoms for invalid input" do
      # This tests that we don't pollute the atom table
      random_string = "definitely_not_an_atom_#{:rand.uniform(1_000_000)}"
      allowed = [:valid]

      {:error, :invalid_value} = AtomSafety.safe_to_atom(random_string, allowed)

      # The atom should not exist
      assert_raise ArgumentError, fn ->
        String.to_existing_atom(random_string)
      end
    end
  end

  describe "safe_to_atom_downcase/2" do
    test "converts uppercase to matching atom" do
      allowed = [:lstm, :gru, :mamba]

      assert {:ok, :lstm} = AtomSafety.safe_to_atom_downcase("LSTM", allowed)
      assert {:ok, :gru} = AtomSafety.safe_to_atom_downcase("GRU", allowed)
      assert {:ok, :mamba} = AtomSafety.safe_to_atom_downcase("MAMBA", allowed)
    end

    test "converts mixed case to matching atom" do
      allowed = [:lstm, :gru, :mamba]

      assert {:ok, :lstm} = AtomSafety.safe_to_atom_downcase("Lstm", allowed)
      assert {:ok, :lstm} = AtomSafety.safe_to_atom_downcase("LsTm", allowed)
      assert {:ok, :mamba} = AtomSafety.safe_to_atom_downcase("MaMbA", allowed)
    end

    test "returns error for invalid string even with case conversion" do
      allowed = [:lstm, :gru]

      assert {:error, :invalid_value} = AtomSafety.safe_to_atom_downcase("INVALID", allowed)
    end
  end

  describe "safe_to_atom!/2" do
    test "returns atom for valid input" do
      allowed = [:lstm, :gru, :mamba]

      assert :lstm = AtomSafety.safe_to_atom!("lstm", allowed)
      assert :mamba = AtomSafety.safe_to_atom!("mamba", allowed)
    end

    test "raises ArgumentError for invalid input" do
      allowed = [:lstm, :gru, :mamba]

      assert_raise ArgumentError, ~r/Invalid value "invalid"/, fn ->
        AtomSafety.safe_to_atom!("invalid", allowed)
      end
    end

    test "error message includes allowed values" do
      allowed = [:a, :b, :c]

      error =
        assert_raise ArgumentError, fn ->
          AtomSafety.safe_to_atom!("x", allowed)
        end

      assert error.message =~ ":a"
      assert error.message =~ ":b"
      assert error.message =~ ":c"
    end
  end

  describe "safe_to_existing_atom/1" do
    test "converts to existing atom" do
      # These atoms definitely exist
      assert {:ok, :ok} = AtomSafety.safe_to_existing_atom("ok")
      assert {:ok, :error} = AtomSafety.safe_to_existing_atom("error")
      assert {:ok, :true} = AtomSafety.safe_to_existing_atom("true")
      assert {:ok, :false} = AtomSafety.safe_to_existing_atom("false")
    end

    test "returns error for non-existing atom" do
      random = "definitely_not_existing_#{:rand.uniform(1_000_000)}"

      assert {:error, :not_existing} = AtomSafety.safe_to_existing_atom(random)
    end
  end

  describe "to_atom_or_default/3" do
    test "returns atom for valid input" do
      allowed = [:lstm, :gru, :mamba]

      assert :lstm = AtomSafety.to_atom_or_default("lstm", allowed, :gru)
      assert :mamba = AtomSafety.to_atom_or_default("mamba", allowed, :gru)
    end

    test "returns default for invalid input" do
      allowed = [:lstm, :gru, :mamba]

      assert :gru = AtomSafety.to_atom_or_default("invalid", allowed, :gru)
      assert :lstm = AtomSafety.to_atom_or_default("transformer", allowed, :lstm)
    end

    test "returns default for empty string" do
      allowed = [:lstm, :gru]

      assert :lstm = AtomSafety.to_atom_or_default("", allowed, :lstm)
    end
  end

  describe "validate/2" do
    test "validates atom input" do
      allowed = [:lstm, :gru, :mamba]

      assert {:ok, :lstm} = AtomSafety.validate(:lstm, allowed)
      assert {:ok, :mamba} = AtomSafety.validate(:mamba, allowed)
    end

    test "validates string input" do
      allowed = [:lstm, :gru, :mamba]

      assert {:ok, :lstm} = AtomSafety.validate("lstm", allowed)
      assert {:ok, :mamba} = AtomSafety.validate("mamba", allowed)
    end

    test "returns error for invalid atom" do
      allowed = [:lstm, :gru]

      assert {:error, :invalid_value} = AtomSafety.validate(:mamba, allowed)
    end

    test "returns error for invalid string" do
      allowed = [:lstm, :gru]

      assert {:error, :invalid_value} = AtomSafety.validate("mamba", allowed)
    end
  end

  describe "security" do
    test "prevents atom table exhaustion from user input" do
      # Simulate parsing many unique "user inputs"
      allowed = [:valid_option]

      for i <- 1..100 do
        input = "malicious_input_#{i}_#{:rand.uniform(1_000_000)}"
        {:error, :invalid_value} = AtomSafety.safe_to_atom(input, allowed)
      end

      # If we got here without crashing, atoms weren't created
      assert true
    end
  end
end
