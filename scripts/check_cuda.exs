# Quick CUDA diagnostic check
# Usage: mix run scripts/check_cuda.exs

alias ExPhil.Training.GPUUtils

case GPUUtils.diagnose_cuda() do
  :ok -> System.halt(0)
  {:error, _} -> System.halt(1)
end
