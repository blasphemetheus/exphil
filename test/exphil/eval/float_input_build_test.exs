defmodule ExPhil.Eval.FloatInputBuildTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.FloatInputBuild

  test "requires a matching executable, code file and protocol" do
    dir = Path.join(System.tmp_dir!(), "exphil-float-build-#{System.unique_integer([:positive])}")
    File.mkdir_p!(Path.join(dir, "Sys/GameSettings"))
    on_exit(fn -> File.rm_rf!(dir) end)
    exe = Path.join(dir, "dolphin-emu-headless")
    ini = Path.join(dir, "Sys/GameSettings/GALE01r2.ini")
    manifest = Path.join(dir, "float-input-v2.json")
    File.write!(exe, "test executable")
    File.write!(ini, "test gecko code")

    hash = fn p ->
      p |> File.read!() |> then(&:crypto.hash(:sha256, &1)) |> Base.encode16(case: :lower)
    end

    data = %{protocol: 2, binary_sha256: hash.(exe), gecko_sha256: hash.(ini)}
    File.write!(manifest, Jason.encode!(data))
    assert :ok = FloatInputBuild.verify!(exe)
    assert_raise RuntimeError, ~r/does not support/, fn -> FloatInputBuild.verify!(exe, true) end
    File.write!(manifest, Jason.encode!(Map.put(data, :accurate_nmsub, true)))
    assert :ok = FloatInputBuild.verify!(exe, true)
    File.write!(exe, "changed executable")
    assert_raise RuntimeError, ~r/hash mismatch/, fn -> FloatInputBuild.verify!(exe) end
    File.write!(exe, "test executable")
    File.write!(ini, "changed code")
    assert_raise RuntimeError, ~r/hash mismatch/, fn -> FloatInputBuild.verify!(exe) end
    File.write!(manifest, Jason.encode!(%{data | protocol: 999}))
    assert_raise RuntimeError, ~r/unsupported/, fn -> FloatInputBuild.verify!(exe) end
  end
end
