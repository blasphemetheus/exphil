defmodule ExPhil.Bridge.SimRows do
  @moduledoc """
  Binary row codec for the sim boundary (SIM_INTEGRATION.md step 10a, the
  "binary rows over the existing Port" tier the NIF must beat).

  The worker sends its numpy dtypes as recursive layout descriptors
  (`kind: scalar|struct|array`, `fmt`, `offset`, `shape`, `itemsize`) on
  `init`/`ping`; this module compiles a descriptor into a decoder
  (`binary -> nested map with the SAME string keys as the JSON rows`, so
  `ExPhil.Bridge.SimState` reads both) and an encoder (nested map ->
  binary) for controller input rows. `_pad*` fields are skipped on decode
  and zero-filled on encode. Little-endian numpy formats only, which is
  what the sim emits on this box.
  """

  @type layout :: map()

  @doc "Row size in bytes for a layout."
  def itemsize(%{"itemsize" => n}), do: n

  @doc "Decode `count` consecutive rows."
  def decode_rows(layout, bin, count) do
    size = itemsize(layout)
    for i <- 0..(count - 1)//1, do: decode(layout, binary_part(bin, i * size, size))
  end

  @doc "Decode one row."
  def decode(%{"kind" => "struct", "fields" => fields}, bin) do
    Enum.reduce(fields, %{}, fn %{"name" => name, "offset" => off} = f, acc ->
      if String.starts_with?(name, "_pad"), do: acc, else: Map.put(acc, name, decode_at(f, bin, off))
    end)
  end

  defp decode_at(%{"kind" => "scalar", "fmt" => fmt, "itemsize" => n}, bin, off), do: scalar(fmt, binary_part(bin, off, n))

  # Absent entries (items with exists = 0, slots with present = 0) decode to
  # nil instead of a full struct: 15 item slots x 20 fields were most of the
  # per-row decode cost (bench 2026-09-21), and SimState skips nil entries.
  defp decode_at(%{"kind" => "struct", "itemsize" => n, "fields" => [%{"name" => flag, "offset" => 0, "fmt" => "|u1"} | _]} = l, bin, off)
       when flag in ["exists", "present"] do
    case binary_part(bin, off, 1) do
      <<0>> -> nil
      _ -> decode(l, binary_part(bin, off, n))
    end
  end

  defp decode_at(%{"kind" => "struct", "itemsize" => n} = l, bin, off), do: decode(l, binary_part(bin, off, n))

  defp decode_at(%{"kind" => "array", "shape" => [count], "item" => item}, bin, off) do
    size = item["itemsize"]
    for i <- 0..(count - 1)//1, do: decode_at(item, bin, off + i * size)
  end

  defp scalar("<f4", <<v::float-32-little>>), do: v
  defp scalar("<i4", <<v::signed-32-little>>), do: v
  defp scalar("<u4", <<v::unsigned-32-little>>), do: v
  defp scalar("<i2", <<v::signed-16-little>>), do: v
  defp scalar("<u2", <<v::unsigned-16-little>>), do: v
  defp scalar("|u1", <<v::unsigned-8>>), do: v
  defp scalar("|i1", <<v::signed-8>>), do: v
  defp scalar(fmt, _), do: raise(ArgumentError, "unsupported numpy format #{inspect(fmt)}")

  @doc """
  Encode one row from a nested map (atom or string keys; missing fields and
  `_pad*` are zero). Arrays take lists (shorter lists are zero-padded).
  """
  def encode(%{"kind" => "struct", "itemsize" => size, "fields" => fields}, map) do
    fields
    |> Enum.reduce({<<>>, 0}, fn %{"name" => name, "offset" => off} = f, {acc, pos} ->
      pad = off - pos
      value = if String.starts_with?(name, "_pad"), do: nil, else: get(map, name)
      chunk = encode_at(f, value)
      {acc <> :binary.copy(<<0>>, pad) <> chunk, off + byte_size(chunk)}
    end)
    |> then(fn {acc, pos} -> acc <> :binary.copy(<<0>>, size - pos) end)
  end

  defp encode_at(%{"kind" => "scalar", "fmt" => fmt}, v), do: scalar_bin(fmt, v)
  defp encode_at(%{"kind" => "struct"} = l, v), do: encode(l, v || %{})

  defp encode_at(%{"kind" => "array", "shape" => [count], "item" => item}, v) do
    list = v || []
    for i <- 0..(count - 1)//1, into: <<>>, do: encode_at(item, Enum.at(list, i))
  end

  defp scalar_bin("<f4", v), do: <<(num(v) * 1.0)::float-32-little>>
  defp scalar_bin("<i4", v), do: <<int(v)::signed-32-little>>
  defp scalar_bin("<u4", v), do: <<int(v)::unsigned-32-little>>
  defp scalar_bin("<i2", v), do: <<int(v)::signed-16-little>>
  defp scalar_bin("<u2", v), do: <<int(v)::unsigned-16-little>>
  defp scalar_bin("|u1", v), do: <<int(v)::unsigned-8>>
  defp scalar_bin("|i1", v), do: <<int(v)::signed-8>>

  defp num(nil), do: 0
  defp num(true), do: 1
  defp num(false), do: 0
  defp num(v) when is_number(v), do: v

  defp int(v), do: trunc(num(v))

  defp get(map, name) when is_map(map) do
    case Map.fetch(map, name) do
      {:ok, v} -> v
      :error -> Map.get(map, String.to_atom(name))
    end
  end

  defp get(_, _), do: nil
end
