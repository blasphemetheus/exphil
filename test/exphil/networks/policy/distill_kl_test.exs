defmodule ExPhil.Networks.Policy.DistillKlTest do
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy.Loss

  # Standard head layout: buttons 8, four sticks 17, shoulder 5 = 81
  @sizes [buttons: 8, main_x: 17, main_y: 17, c_x: 17, c_y: 17, shoulder: 5]

  defp student(fun) do
    Map.new(@sizes, fn {k, n} ->
      {k, Nx.tensor(for b <- 0..1, do: for(i <- 0..(n - 1), do: fun.(k, b, i)))}
    end)
  end

  defp concat(logit_map) do
    Nx.concatenate(Enum.map(Keyword.keys(@sizes), &logit_map[&1]), axis: 1)
  end

  test "teacher == student gives ~0" do
    s = student(fn _, b, i -> 0.3 * i - 0.5 * b end)
    kl = Loss.distill_kl(s, concat(s), Nx.tensor([1.0, 1.0]))
    assert Nx.to_number(kl) < 1.0e-6
  end

  test "all-zero mask gives exactly 0 even with disagreement" do
    s = student(fn _, _, _ -> 0.0 end)
    t = student(fn _, _, i -> 2.0 * i end)
    kl = Loss.distill_kl(s, concat(t), Nx.tensor([0.0, 0.0]))
    assert Nx.to_number(kl) == 0.0
  end

  test "disagreement under an active mask is positive" do
    s = student(fn _, _, _ -> 0.0 end)
    t = student(fn _, _, i -> 2.0 * i end)
    kl = Loss.distill_kl(s, concat(t), Nx.tensor([1.0, 1.0]))
    assert Nx.to_number(kl) > 0.1
  end

  test "mask selects rows: masking out the disagreeing row zeroes the loss" do
    # Row 0 agrees, row 1 disagrees
    s = student(fn _, _, _ -> 0.0 end)
    t = student(fn _, b, i -> if b == 1, do: 3.0 * i, else: 0.0 end)

    only_agreeing = Loss.distill_kl(s, concat(t), Nx.tensor([1.0, 0.0]))
    only_disagreeing = Loss.distill_kl(s, concat(t), Nx.tensor([0.0, 1.0]))

    assert Nx.to_number(only_agreeing) < 1.0e-6
    assert Nx.to_number(only_disagreeing) > 0.1
  end

  test "extreme logits stay finite (clamp)" do
    s = student(fn _, _, _ -> -1.0e9 end)
    t = student(fn _, _, _ -> 1.0e9 end)
    kl = Loss.distill_kl(s, concat(t), Nx.tensor([1.0, 1.0]))
    n = Nx.to_number(kl)
    assert is_number(n)
    refute n != n
  end

  test "tau softens the categorical term" do
    s = student(fn _, _, _ -> 0.0 end)
    t = student(fn k, _, i -> if k == :buttons, do: 0.0, else: 1.0 * i end)

    sharp = Loss.distill_kl(s, concat(t), Nx.tensor([1.0, 1.0]), tau: 1.0)
    soft = Loss.distill_kl(s, concat(t), Nx.tensor([1.0, 1.0]), tau: 4.0)

    assert Nx.to_number(soft) < Nx.to_number(sharp)
  end
end
