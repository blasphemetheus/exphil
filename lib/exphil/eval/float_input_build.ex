defmodule ExPhil.Eval.FloatInputBuild do
  @moduledoc "Checks a separately installed float-input build before sending protocol v2 messages."

  def verify!(executable, accurate_nmsub \\ false) do
    dir = Path.dirname(executable)
    manifest = dir |> Path.join("float-input-v2.json") |> File.read!() |> Jason.decode!()
    unless manifest["protocol"] == 2, do: raise("unsupported float input protocol")
    if accurate_nmsub and manifest["accurate_nmsub"] != true,
      do: raise("float input build does not support --accurate-nmsub")

    for {path, key} <- [
          {executable, "binary_sha256"},
          {Path.join(dir, "Sys/GameSettings/GALE01r2.ini"), "gecko_sha256"}
        ] do
      digest =
        path |> File.read!() |> then(&:crypto.hash(:sha256, &1)) |> Base.encode16(case: :lower)

      unless digest == manifest[key], do: raise("float input build hash mismatch: #{path}")
    end

    :ok
  end
end
