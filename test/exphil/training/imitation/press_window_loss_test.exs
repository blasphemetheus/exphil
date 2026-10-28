defmodule ExPhil.Training.Imitation.PressWindowLossTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.Imitation.Loss

  # 2 samples, 8 buttons, K = 2 future heads. Sample 0: button A pressed at t+1 only
  # (a press onset inside the window). Sample 1: nothing pressed anywhere.
  defp actions do
    zeros = Nx.broadcast(0, {2, 8})
    fut = fn j -> Nx.put_slice(Nx.broadcast(0, {2, 2, 8}), [0, j, 0], Nx.tensor([[[1, 0, 0, 0, 0, 0, 0, 0]]])) end
    %{buttons: zeros, future_buttons: fut.(1), future_mask: Nx.broadcast(1.0, {2, 2})}
  end

  defp heads(logit_main, logit_f0, logit_f1) do
    mk = fn l -> {Nx.broadcast(l, {2, 8}), nil, nil, nil, nil, nil} end
    {mk.(logit_main), {mk.(logit_f0), mk.(logit_f1)}}
  end

  test "committing on one frame of the window beats spreading 0.1 everywhere" do
    w = Nx.tensor([1.0, 1.0])
    {m, f} = heads(-2.2, -2.2, -2.2)   # p ~ 0.1 on every frame
    spread = Loss.press_window_loss(m, f, actions(), w) |> Nx.to_number()
    # commit: button A high at t+1 for sample 0 only, everything else very low
    low = Nx.broadcast(-6.0, {2, 8})
    hi = Nx.put_slice(low, [0, 0], Nx.tensor([[4.0]]))
    mk = fn l -> {l, nil, nil, nil, nil, nil} end
    commit = Loss.press_window_loss(mk.(low), {mk.(low), mk.(hi)}, actions(), w) |> Nx.to_number()
    assert commit < spread
    assert commit < 0.05
  end

  test "jump window: committing on X OR Y inside the window beats spreading over both" do
    w = Nx.tensor([1.0, 1.0])
    # sample 0: Y pressed at t+1 (jump family); sample 1: nothing
    zeros = Nx.broadcast(0, {2, 8})
    fut1 = Nx.put_slice(Nx.broadcast(0, {2, 2, 8}), [0, 1, 0], Nx.tensor([[[0, 0, 0, 1, 0, 0, 0, 0]]]))
    a = %{buttons: zeros, future_buttons: fut1, future_mask: Nx.broadcast(1.0, {2, 2})}
    mk = fn l -> {l, nil, nil, nil, nil, nil} end
    spread = Nx.broadcast(-2.2, {2, 8})
    s = Loss.jump_window_loss(mk.(spread), {mk.(spread), mk.(spread)}, a, w) |> Nx.to_number()
    low = Nx.broadcast(-6.0, {2, 8})
    # commit with X (not Y): the family term is satisfied either way
    hi_x = Nx.put_slice(low, [0, 2], Nx.tensor([[4.0]]))
    c = Loss.jump_window_loss(mk.(low), {mk.(low), mk.(hi_x)}, a, w) |> Nx.to_number()
    assert c < s
    assert c < 0.05
  end

  test "frames past the game's end do not count" do
    w = Nx.tensor([1.0, 1.0])
    a = %{actions() | future_mask: Nx.tensor([[1.0, 0.0], [1.0, 0.0]])}
    # the only press is at t+1, which is masked out: the window target is 0 everywhere, and low
    # logits everywhere should be near-perfect
    low = Nx.broadcast(-6.0, {2, 8})
    mk = fn l -> {l, nil, nil, nil, nil, nil} end
    loss = Loss.press_window_loss(mk.(low), {mk.(low), mk.(Nx.broadcast(4.0, {2, 8}))}, a, w) |> Nx.to_number()
    assert loss < 0.02
  end
end
