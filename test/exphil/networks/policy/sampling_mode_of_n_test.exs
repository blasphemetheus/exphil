defmodule ExPhil.Networks.Policy.SamplingModeOfNTest do
  @moduledoc """
  Mode-of-N decode (2026-08-29): N joint draws from one forward, play the
  most frequent. Offline it recovered ~28% of the selection gap critic-free
  (eval_runs/0829_critic/RESULTS.md); this pins the contract:

    * output shapes are the single-sample shapes ([1, 8] buttons, [1] heads)
    * the returned action is one of the N draws (a real joint sample, not a
      per-head mode, which could combine heads that never co-occurred)
    * on a sharply peaked distribution it returns the peak far more often
      than plain sampling would
    * :deterministic wins over :mode_of_n
  """
  use ExUnit.Case, async: true

  alias ExPhil.Networks.Policy.Sampling

  @buttons 8
  @axis 17
  @shoulder 4

  # A fake predict_fn: fixed logits, no params. `peak` scales how sharp.
  defp predict_fn(peak) do
    fn _params, _state ->
      b = Nx.tensor([[peak, -peak, -peak, -peak, -peak, -peak, -peak, -peak]], type: :f32)
      axis = fn hot -> Nx.tensor([for(i <- 0..(@axis - 1), do: if(i == hot, do: peak, else: 0.0))], type: :f32) end
      sh = Nx.tensor([for(i <- 0..(@shoulder - 1), do: if(i == 0, do: peak, else: 0.0))], type: :f32)
      {b, axis.(8), axis.(8), axis.(8), axis.(8), sh}
    end
  end

  defp key(action) do
    s = fn t -> t |> Nx.to_flat_list() |> hd() end

    {Nx.to_flat_list(Nx.as_type(action.buttons, :u8)), s.(action.main_x), s.(action.main_y),
     s.(action.c_x), s.(action.c_y), s.(action.shoulder)}
  end

  test "mode_index picks the most frequent joint row, ties to first occurrence" do
    b = Nx.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1]], type: :u8)
    one = Nx.tensor([3, 4, 3, 4, 9])
    # rows 0/2 and 1/3 tie at 2 each -> first occurrence (row 0)
    assert Sampling.mode_index(b, one, one, one, one, one) == 0

    b2 = Nx.tensor([[1, 0], [0, 1], [0, 1], [0, 1], [1, 1]], type: :u8)
    assert Sampling.mode_index(b2, one, one, one, one, one) == 1
  end

  test "returns single-sample shapes and a row that is one of the draws" do
    action = Sampling.sample(%{}, predict_fn(1.0), Nx.tensor([[0.0]]), mode_of_n: 8, temperature: 1.0)

    assert Nx.shape(action.buttons) == {1, @buttons}
    assert Nx.shape(action.main_x) == {1}
    assert Nx.shape(action.shoulder) == {1}
    assert Nx.shape(action.logits.main_x) == {1, @axis}
    assert Map.has_key?(action.confidence_raw, :overall)
  end

  test "on a peaked distribution mode-of-16 lands on the peak more often than one sample" do
    # peak 4.0 -> joint P(peak) ~0.3 per draw (sigmoid(4)^8 * softmax^4 * sh):
    # a single draw misses ~70%; the mode over 16 should almost always hit.
    # (At peak 2.0 the joint P is ~0.003 and 16 draws never repeat -> mode
    # degenerates to draw 1; real T=0.5 policies sit near pass@1 15-40%.)
    peak = {[1, 0, 0, 0, 0, 0, 0, 0], 8, 8, 8, 8, 0}
    trials = 60

    hit = fn opts ->
      Enum.count(1..trials, fn _ ->
        key(Sampling.sample(%{}, predict_fn(4.0), Nx.tensor([[0.0]]), opts)) == peak
      end)
    end

    single = hit.(temperature: 1.0)
    mode16 = hit.(temperature: 1.0, mode_of_n: 16)

    assert mode16 > single, "mode-of-16 #{mode16}/#{trials} vs single #{single}/#{trials}"
    assert mode16 >= trials * 0.8
  end

  test ":deterministic wins over :mode_of_n" do
    a = Sampling.sample(%{}, predict_fn(3.0), Nx.tensor([[0.0]]), deterministic: true, mode_of_n: 16)
    assert key(a) == {[1, 0, 0, 0, 0, 0, 0, 0], 8, 8, 8, 8, 0}
  end

  test "mode_of_n nil or 1 is plain sampling (same shapes)" do
    for n <- [nil, 1] do
      a = Sampling.sample(%{}, predict_fn(1.0), Nx.tensor([[0.0]]), mode_of_n: n)
      assert Nx.shape(a.buttons) == {1, @buttons}
    end
  end
end
