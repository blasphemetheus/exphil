defmodule ExPhil.Agents.JumpDebounceTest do
  use ExUnit.Case, async: true

  alias ExPhil.Agents.Agent

  # Buttons tensor layout: 8 buttons, jump = X (col 2) and Y (col 3).
  defp buttons(x, y) do
    Nx.tensor([[0, 0, x, y, 0, 0, 0, 0]], type: :f32)
  end

  defp action(x, y), do: %{buttons: buttons(x, y)}

  defp state(opts) do
    %{
      jump_debounce: Keyword.get(opts, :n, 10),
      jump_cooldown: Keyword.get(opts, :cooldown, 0),
      last_action: Keyword.get(opts, :last_action)
    }
  end

  defp jump?(%{buttons: b}) do
    b |> Nx.slice_along_axis(2, 2, axis: 1) |> Nx.reduce_max() |> Nx.to_number() > 0
  end

  defp other_buttons_intact?(%{buttons: b}) do
    # Suppression must only zero the jump columns, never e.g. A/B/L
    Nx.to_flat_list(b) |> Enum.take(2) == [1.0, 1.0]
  end

  test "disabled debounce (nil or 0) passes actions through untouched" do
    for st <- [%{jump_debounce: nil}, %{jump_debounce: 0}, %{}] do
      a = action(1, 0)
      assert {^a, ^st} = Agent.apply_jump_debounce(a, st, true)
    end
  end

  test "airborne new press edge during cooldown is suppressed" do
    st = state(cooldown: 5, last_action: action(0, 0))
    {out, new_st} = Agent.apply_jump_debounce(action(1, 0), st, true)

    refute jump?(out)
    # cooldown decrements by one per frame
    assert new_st.jump_cooldown == 4
  end

  test "suppression only zeroes the jump columns" do
    st = state(cooldown: 5, last_action: %{buttons: Nx.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], type: :f32)})
    pressed = %{buttons: Nx.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], type: :f32)}
    {out, _} = Agent.apply_jump_debounce(pressed, st, true)

    refute jump?(out)
    assert other_buttons_intact?(out)
  end

  test "GROUNDED new press edge during cooldown passes (OOS jump case)" do
    st = state(cooldown: 5, last_action: action(0, 0))
    {out, _} = Agent.apply_jump_debounce(action(1, 0), st, false)
    assert jump?(out)
  end

  test "airborne press with expired cooldown passes" do
    st = state(cooldown: 0, last_action: action(0, 0))
    {out, _} = Agent.apply_jump_debounce(action(0, 1), st, true)
    assert jump?(out)
  end

  test "held jump is never cut, even airborne during cooldown" do
    st = state(cooldown: 5, last_action: action(1, 0))
    {out, new_st} = Agent.apply_jump_debounce(action(1, 0), st, true)

    assert jump?(out)
    assert new_st.jump_cooldown == 4
  end

  test "release edge arms the cooldown to n" do
    st = state(n: 10, cooldown: 0, last_action: action(1, 0))
    {out, new_st} = Agent.apply_jump_debounce(action(0, 0), st, true)

    refute jump?(out)
    assert new_st.jump_cooldown == 10
  end

  test "first frame ever (no last_action) with a press passes" do
    st = state(cooldown: 0, last_action: nil)
    {out, _} = Agent.apply_jump_debounce(action(1, 0), st, true)
    assert jump?(out)
  end

  test "full liftoff-DJ scenario: 1-frame metronome blocked, patient DJ passes" do
    n = 10

    # Frame 1: grounded jump press (passes; policy leaves the ground)
    st = state(n: n, cooldown: 0, last_action: action(0, 0))
    {a1, st} = Agent.apply_jump_debounce(action(1, 0), st, false)
    assert jump?(a1)
    st = %{st | last_action: a1}

    # Frame 2: release in the air -> cooldown armed
    {a2, st} = Agent.apply_jump_debounce(action(0, 0), st, true)
    st = %{st | last_action: a2}
    assert st.jump_cooldown == n

    # Frame 3: the pathological instant re-press (the r8 metronome) -> blocked
    {a3, st} = Agent.apply_jump_debounce(action(0, 1), st, true)
    refute jump?(a3)
    st = %{st | last_action: a3}

    # Frames 4..n+2: idle out the cooldown
    st =
      Enum.reduce(1..n, st, fn _, acc ->
        {a, acc} = Agent.apply_jump_debounce(action(0, 0), acc, true)
        %{acc | last_action: a}
      end)

    assert st.jump_cooldown == 0

    # A data-like patient DJ now passes
    {dj, _} = Agent.apply_jump_debounce(action(0, 1), st, true)
    assert jump?(dj)
  end

  test "X release while Y still held is not a release edge" do
    # Both slices reduce over cols 2-3 jointly: Y held keeps prev_jump true,
    # want_jump true -> falls to passthrough, cooldown just decrements
    st = state(cooldown: 3, last_action: action(1, 1))
    {out, new_st} = Agent.apply_jump_debounce(action(0, 1), st, true)

    assert jump?(out)
    assert new_st.jump_cooldown == 2
  end
end
