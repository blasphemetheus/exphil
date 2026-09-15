#!/usr/bin/env elixir
# Install into a new directory; preserve every existing installation.
{opts, [], []} =
  OptionParser.parse(System.argv(), strict: [dolphin: :string, asm: :string, out: :string])

dolphin = Keyword.fetch!(opts, :dolphin)
asm = Keyword.fetch!(opts, :asm)
out = Keyword.fetch!(opts, :out)
binary = Path.join(dolphin, "build/Binaries/dolphin-emu-nogui")
hook = Path.join(asm, "Output/float-inputs.txt")
source = Path.join(asm, "AI/OverwriteInputs/OverwriteProcessedInputs.asm")

for path <- [binary, hook, source],
    do: File.regular?(path) || raise("missing build artifact: #{path}")

code = File.read!(hook)
label = "$Optional: Allow Bot Processed Input Overrides"

unless String.starts_with?(code, label) and String.contains?(code, "C206B0DC ") and
         String.contains?(code, "Direct protocol v2"),
       do: raise("unexpected processed-input hook artifact")

unless :binary.match(File.read!(binary), "SLIPPI_PROCESSED_INPUT_V2") != :nomatch,
  do: raise("Dolphin binary lacks the processed-input v2 capability marker")

File.mkdir_p!(Path.dirname(out))
# mkdir is exclusive, including when out is a symlink.
case File.mkdir(out) do
  :ok -> :ok
  error -> raise "refusing to overwrite installation #{out}: #{inspect(error)}"
end

File.cp_r!(Path.join(dolphin, "Data/Sys"), Path.join(out, "Sys"))
installed = Path.join(out, "dolphin-emu-headless")
File.cp!(binary, installed)
File.chmod!(installed, File.stat!(binary).mode)
File.touch!(Path.join(out, "portable.txt"))
ini = Path.join(out, "Sys/GameSettings/GALE01r2.ini")
# Remove only this named Gecko block, retaining subsequent codes/sections.
pattern = ~r/^\$Optional: Allow Bot Processed Input Overrides[^\r\n]*\r?\n.*?(?=^\$|^\[|\z)/ms
text = Regex.replace(pattern, File.read!(ini), "")
File.write!(ini, String.trim_trailing(text) <> "\n\n" <> code)
sha = fn path -> :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower) end

head = fn path ->
  {value, 0} = System.cmd("git", ["-C", path, "rev-parse", "HEAD"])
  String.trim(value)
end

manifest = %{
  protocol: 2,
  accurate_nmsub: :binary.match(File.read!(binary), "AccurateNmsub") != :nomatch,
  binary_sha256: sha.(installed),
  gecko_sha256: sha.(ini),
  asm_sha256: sha.(source),
  hook_sha256: sha.(hook),
  dolphin_base: head.(dolphin),
  asm_base: head.(asm)
}

File.write!(Path.join(out, "float-input-v2.json"), JSON.encode!(manifest) <> "\n")
IO.puts(installed)
