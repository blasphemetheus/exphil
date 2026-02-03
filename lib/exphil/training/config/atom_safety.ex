defmodule ExPhil.Training.Config.AtomSafety do
  @moduledoc """
  Safe atom conversion utilities for config parsing.

  Elixir atoms are not garbage collected, so creating atoms from untrusted
  user input can lead to atom table exhaustion (denial of service).

  This module provides safe alternatives to `String.to_atom/1` that either:
  1. Validate input against a known allowlist before conversion
  2. Use `String.to_existing_atom/1` with proper error handling

  ## Usage

      alias ExPhil.Training.Config.AtomSafety

      # With allowlist (preferred for known values)
      case AtomSafety.safe_to_atom("mamba", [:lstm, :gru, :mamba, :attention]) do
        {:ok, :mamba} -> # use the atom
        {:error, :invalid_value} -> # handle error
      end

      # For existing atoms only (config keys, etc.)
      case AtomSafety.safe_to_existing_atom("epochs") do
        {:ok, :epochs} -> # use the atom
        {:error, :not_existing} -> # handle error
      end

  """

  @doc """
  Safely convert a string to an atom using an allowlist.

  Returns `{:ok, atom}` if the string matches an allowed value,
  or `{:error, :invalid_value}` if not in the allowlist.

  ## Parameters

  - `string` - The string to convert
  - `allowed` - List of allowed atom values

  ## Examples

      iex> AtomSafety.safe_to_atom("mamba", [:lstm, :gru, :mamba])
      {:ok, :mamba}

      iex> AtomSafety.safe_to_atom("unknown", [:lstm, :gru, :mamba])
      {:error, :invalid_value}

      iex> AtomSafety.safe_to_atom("LSTM", [:lstm, :gru, :mamba])
      {:error, :invalid_value}

  """
  @spec safe_to_atom(String.t(), [atom()]) :: {:ok, atom()} | {:error, :invalid_value}
  def safe_to_atom(string, allowed) when is_binary(string) and is_list(allowed) do
    # Build a map of string -> atom for O(1) lookup
    allowed_map = Map.new(allowed, fn atom -> {Atom.to_string(atom), atom} end)

    case Map.fetch(allowed_map, string) do
      {:ok, atom} -> {:ok, atom}
      :error -> {:error, :invalid_value}
    end
  end

  @doc """
  Safely convert a string to an atom, case-insensitive.

  Converts the input to lowercase before matching against the allowlist.

  ## Examples

      iex> AtomSafety.safe_to_atom_downcase("MAMBA", [:lstm, :gru, :mamba])
      {:ok, :mamba}

      iex> AtomSafety.safe_to_atom_downcase("Lstm", [:lstm, :gru, :mamba])
      {:ok, :lstm}

  """
  @spec safe_to_atom_downcase(String.t(), [atom()]) :: {:ok, atom()} | {:error, :invalid_value}
  def safe_to_atom_downcase(string, allowed) when is_binary(string) and is_list(allowed) do
    safe_to_atom(String.downcase(string), allowed)
  end

  @doc """
  Safely convert a string to an atom, raising on invalid input.

  ## Examples

      iex> AtomSafety.safe_to_atom!("mamba", [:lstm, :gru, :mamba])
      :mamba

  Invalid values raise ArgumentError:

      AtomSafety.safe_to_atom!("unknown", [:lstm, :gru, :mamba])
      # => raises ArgumentError

  """
  @spec safe_to_atom!(String.t(), [atom()]) :: atom()
  def safe_to_atom!(string, allowed) when is_binary(string) and is_list(allowed) do
    case safe_to_atom(string, allowed) do
      {:ok, atom} ->
        atom

      {:error, :invalid_value} ->
        allowed_str = allowed |> Enum.map(&inspect/1) |> Enum.join(", ")

        raise ArgumentError,
              "Invalid value #{inspect(string)}. Must be one of: #{allowed_str}"
    end
  end

  @doc """
  Safely convert a string to an existing atom.

  Uses `String.to_existing_atom/1` internally, which only succeeds if
  the atom already exists in the atom table.

  Returns `{:ok, atom}` on success, `{:error, :not_existing}` if the atom
  doesn't exist.

  ## Examples

      iex> AtomSafety.safe_to_existing_atom("ok")
      {:ok, :ok}

      iex> AtomSafety.safe_to_existing_atom("definitely_not_an_existing_atom_xyz123")
      {:error, :not_existing}

  """
  @spec safe_to_existing_atom(String.t()) :: {:ok, atom()} | {:error, :not_existing}
  def safe_to_existing_atom(string) when is_binary(string) do
    {:ok, String.to_existing_atom(string)}
  rescue
    ArgumentError -> {:error, :not_existing}
  end

  @doc """
  Convert a string to an atom, falling back to a default on failure.

  Tries the allowlist first, returns default if not found.

  ## Examples

      iex> AtomSafety.to_atom_or_default("mamba", [:lstm, :gru, :mamba], :lstm)
      :mamba

      iex> AtomSafety.to_atom_or_default("unknown", [:lstm, :gru, :mamba], :lstm)
      :lstm

  """
  @spec to_atom_or_default(String.t(), [atom()], atom()) :: atom()
  def to_atom_or_default(string, allowed, default)
      when is_binary(string) and is_list(allowed) and is_atom(default) do
    case safe_to_atom(string, allowed) do
      {:ok, atom} -> atom
      {:error, _} -> default
    end
  end

  @doc """
  Validate that a value is one of the allowed atoms.

  Works with both atoms and strings.

  ## Examples

      iex> AtomSafety.validate(:mamba, [:lstm, :gru, :mamba])
      {:ok, :mamba}

      iex> AtomSafety.validate("mamba", [:lstm, :gru, :mamba])
      {:ok, :mamba}

      iex> AtomSafety.validate(:unknown, [:lstm, :gru, :mamba])
      {:error, :invalid_value}

  """
  @spec validate(atom() | String.t(), [atom()]) :: {:ok, atom()} | {:error, :invalid_value}
  def validate(value, allowed) when is_atom(value) and is_list(allowed) do
    if value in allowed do
      {:ok, value}
    else
      {:error, :invalid_value}
    end
  end

  def validate(value, allowed) when is_binary(value) and is_list(allowed) do
    safe_to_atom(value, allowed)
  end
end
