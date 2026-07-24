# Inspect the MultishineExpert table built from a fixture — the cheap
# (no-Dolphin) half of Track B's gate 2.
#
#   mix run scripts/inspect_multishine_table.exs [--fixture PATH] [--port N]
#
# Answers three questions a live demo answers slowly and expensively:
#
#  1. Does the table COVER the 9-frame loop, and what does it press on each
#     key? (A multishine needs B on exactly one jumpsquat frame; if the modal
#     input smeared that away, the technique is gone at the table layer.)
#
#  2. LABEL fidelity — does E(state[N]) match controller[N]? This is the
#     replay convention the trainer expects (`Data.shift_actions/2` applies
#     the offset), and is what `MultishineExpert.button_agreement/2` reports.
#
#  3. LIVE fidelity — does E(state[N]) match controller[N+1]? An input
#     decided at frame N lands at N+1, so to reproduce the fixture live the
#     expert must emit what landed one frame LATER.
#
#     ⚠️ TREAT THIS NUMBER AS A HINT, NOT A VERDICT (GOTCHAS #81). It is
#     computed entirely in PARSED-replay space, and Peppi's action_frame
#     does not agree with libmelee's live action_frame (parsed jumpsquat af
#     0,1,2 vs live 1,2,3, etc). So the keys this simulation hits are not
#     the keys a live lookup hits. Measured 2026-07-24: this reported 50.4%
#     live fidelity for a table whose live teacher then sustained a
#     103-shine chain — the live misses fall through to recovery rules that
#     emit the right input anyway. `scripts/demo_expert.exs` is the
#     authoritative live gate; use this script for COVERAGE and to see what
#     each key presses.
#
# Exit code 1 if the loop keys are missing, so a runbook can gate on it.

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.MultishineExpert
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [fixture: :string, port: :integer, keys: :integer])

fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed.slp"
port = opts[:port] || 1

{:ok, replay} = ExPhil.Data.Peppi.parse(Path.expand(fixture))

frames =
  replay
  |> ExPhil.Data.Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
  |> Enum.reject(&(&1.game_state.frame < 0))

expert = MultishineExpert.from_frames(frames, player_port: port)

Output.banner("MultishineExpert table — #{Path.basename(fixture)}")

btn = fn c ->
  [
    if(c.button_b, do: "B", else: "-"),
    if(c.button_x, do: "X", else: "-"),
    if(c.button_y, do: "Y", else: "-"),
    if(c.button_a, do: "A", else: "-")
  ]
  |> Enum.join()
end

fmt = fn c -> "#{btn.(c)} stick=(#{Float.round(c.main_stick.x, 2)},#{Float.round(c.main_stick.y, 2)})" end

buttons = fn c ->
  {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r}
end

# ---- 1. Coverage of the loop states, in cycle order ------------------------
Output.puts("")
Output.puts("Table keys {action, action_frame, on_ground} present (loop states):")

loop_actions = [24, 25, 360, 361, 362, 363, 365, 366, 367, 368]

table_keys =
  expert.table
  |> Map.keys()
  |> Enum.filter(fn {a, _af, _g} -> a in loop_actions end)
  |> Enum.sort()

for {a, af, g} = key <- table_keys do
  label =
    case a do
      24 -> "jumpsquat"
      25 -> "jump/airborne"
      x when x in 360..363 -> "GROUND reflector"
      x when x in 365..368 -> "AIR reflector"
    end

  Output.puts(
    "  {#{a}, af=#{af}, gnd=#{g}}  #{String.pad_trailing(label, 17)} -> #{fmt.(expert.table[key])}"
  )
end

Output.puts("  (total table keys: #{map_size(expert.table)}, loop keys: #{length(table_keys)})")

# The single most important cell: which jumpsquat frame presses B? That press
# is what lands on airborne frame 1 and makes the shine stall the jump.
js_keys = Enum.filter(table_keys, fn {a, _, _} -> a == 24 end)
js_with_b = Enum.filter(js_keys, fn k -> expert.table[k].button_b end)

Output.puts("")

if js_with_b == [] do
  Output.warning(
    "NO jumpsquat key presses B. Under the LANDING convention this is expected " <>
      "(the B that lands on airborne frame 1 is recorded on the AIR-shine frame, " <>
      "not on jumpsquat) — check live fidelity below."
  )
else
  Output.success("jumpsquat keys pressing B: #{inspect(js_with_b)}")
end

# ---- 1b. STATE PURITY (is the fixture learnable at all?) -------------------
# closed_loop's premise: every frame's input is a pure function of the
# observable state. If one {action, af, on_ground} key carries more than one
# input, no state-conditioned policy can fit it — the excess is entropy the
# loss can never remove, and BC will floor well above its target while
# looking "almost converged". Caught exactly that on 2026-07-24: a fastfall
# stick-flick keyed on wall-clock parity over an odd-length cycle gave
# identical states different sticks on alternate cycles, flooring BC at loss
# 0.121 vs a 2e-3 target. Check this BEFORE spending a training run.
ambiguous =
  frames
  |> Enum.reject(fn f ->
    c = f.controller
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)
  |> Enum.group_by(
    fn f ->
      p = f.game_state.players[port]
      {trunc(p.action), trunc(p.action_frame), p.on_ground}
    end,
    fn f -> {buttons.(f.controller), round(f.controller.main_stick.y * 20)} end
  )
  |> Enum.map(fn {k, vs} -> {k, vs |> Enum.uniq() |> length()} end)
  |> Enum.filter(fn {_k, n} -> n > 1 end)
  |> Enum.sort_by(fn {_k, n} -> -n end)

Output.puts("")

if ambiguous == [] do
  Output.success("State purity: every state key maps to exactly ONE input — fully learnable.")
else
  Output.warning(
    "State purity VIOLATED: #{length(ambiguous)} keys carry multiple inputs. " <>
      "BC cannot fit these; expect the loss to floor. Something in the recorder " <>
      "is keyed on the wall clock rather than on observable state."
  )

  for {{a, af, g}, n} <- Enum.take(ambiguous, 8) do
    Output.puts("  {#{a}, af=#{af}, gnd=#{g}} -> #{n} distinct inputs")
  end
end

# ---- 2/3. Label vs live fidelity ------------------------------------------
labeled =
  frames
  |> Enum.map(fn f ->
    case MultishineExpert.label(expert, f.game_state.players[port]) do
      {:ok, c} -> {:ok, c, f.controller, f.game_state.players[port]}
      :skip -> :skip
    end
  end)

pairs = Enum.filter(labeled, &match?({:ok, _, _, _}, &1))

label_matches =
  Enum.count(pairs, fn {:ok, e, actual, _p} -> buttons.(e) == buttons.(actual) end)

# live: expert's output at frame N vs what actually landed at frame N+1
live_pairs =
  pairs
  |> Enum.zip(tl(pairs) ++ [nil])
  |> Enum.reject(fn {_a, b} -> is_nil(b) end)

live_matches =
  Enum.count(live_pairs, fn {{:ok, e, _, _}, {:ok, _, next_actual, _}} ->
    buttons.(e) == buttons.(next_actual)
  end)

label_fid = if pairs == [], do: 0.0, else: label_matches / length(pairs)
live_fid = if live_pairs == [], do: 0.0, else: live_matches / length(live_pairs)

Output.puts("")
Output.divider()
Output.puts("LABEL fidelity (E(state_N) vs controller_N, the trainer convention): #{Float.round(label_fid * 100, 1)}%")
Output.puts("LIVE  fidelity (E(state_N) vs controller_N+1, what a rollout needs): #{Float.round(live_fid * 100, 1)}%")

cond do
  label_fid > 0.9 and live_fid > 0.9 ->
    Output.success("Both conventions agree — the loop is mostly self-similar; live should reproduce.")

  label_fid > 0.9 and live_fid < 0.75 ->
    Output.warning(
      "Table is FAITHFUL for training but ONE FRAME LATE for live control. " <>
        "A live table-driven teacher will miss the airborne-frame-1 shine. " <>
        "Fix: build the live table from SHIFTED pairs (state_N -> controller_N+1)."
    )

  true ->
    Output.warning("Neither convention reproduces cleanly — inspect the per-key dump above.")
end

# ---- Where the live mismatches concentrate --------------------------------
mismatch_by_state =
  live_pairs
  |> Enum.reject(fn {{:ok, e, _, _}, {:ok, _, next_actual, _}} ->
    buttons.(e) == buttons.(next_actual)
  end)
  |> Enum.frequencies_by(fn {{:ok, _e, _, p}, _} -> {trunc(p.action), trunc(p.action_frame)} end)
  |> Enum.sort_by(fn {_k, v} -> -v end)
  |> Enum.take(8)

if mismatch_by_state != [] do
  Output.puts("")
  Output.puts("Top LIVE mismatches by {action, af}:")

  for {{a, af}, n} <- mismatch_by_state do
    Output.puts("  {#{a}, af=#{af}} x#{n}")
  end
end

if table_keys == [] do
  Output.error("Table has NO loop keys — fixture is not a multishine.")
  System.halt(1)
end
