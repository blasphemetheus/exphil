path = Path.expand("../libmelee_ex/lib/melee/console.ex")
source = File.read!(path)
old = """
  def handle_call(:step, from, state) do
    state = flush_controllers(state)

    case pop_frame(state) do
"""
replacement = """
  def handle_call(:step, from, state) do
    state = if :queue.is_empty(state.frames), do: flush_controllers(state), else: state

    case pop_frame(state) do
"""
source =
  cond do
    length(String.split(source, old)) == 2 ->
      String.replace(source, old, replacement, global: false)

    length(String.split(source, replacement)) == 2 ->
      source

    true ->
      raise("console experiment source does not match; inspect before applying")
  end

Code.put_compiler_option(:ignore_module_conflict, true)
Code.compile_string(source, path)
IO.puts("EXPERIMENT: skip controller flush when a completed frame is already queued")
