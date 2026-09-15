# Diagnostic launcher helper; change only the isolated run's CPU engine.
args = System.argv()
user_index = Enum.find_index(args, &(&1 in ["-u", "--user"])) || raise "missing Dolphin user directory"
home = Enum.at(args, user_index + 1)
path = Path.join(home, "Config/Dolphin.ini")
ini = File.read!(path)
ini = Regex.replace(~r/^CPUCore\s*=.*\r?\n/m, ini, "")
ini = String.replace(ini, "[Core]\n", "[Core]\nCPUCore = 0\n", global: false)
File.write!(path, ini)
IO.puts("Human replay diagnostic: interpreter CPUCore=0 in #{path}")
