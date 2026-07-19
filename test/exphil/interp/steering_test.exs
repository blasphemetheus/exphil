defmodule ExPhil.Interp.SteeringTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.Steering

  @hidden 8

  defp unit_v do
    # Fixed non-axis-aligned direction, normalized
    raw = Nx.tensor([1.0, -2.0, 0.5, 3.0, 0.0, -1.5, 2.5, 0.25])
    Nx.divide(raw, Nx.LinAlg.norm(raw))
  end

  defp batch_x do
    Nx.tensor([
      [0.3, -1.2, 4.0, 0.9, -2.2, 1.1, 0.0, 5.5],
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      [-3.0, 0.4, 0.0, 2.0, 7.0, -0.5, 1.5, -1.0]
    ])
  end

  test "alpha=0 is the identity" do
    x = batch_x()
    steered = Steering.steer(x, unit_v(), 0.0)
    assert Nx.all_close(steered, x, atol: 0.0) |> Nx.to_number() == 1
  end

  test "alpha=1 removes exactly the component along v (feature delta = proj * v)" do
    x = batch_x()
    v = unit_v()
    steered = Steering.steer(x, v, 1.0)

    # Residual projection onto v is ~0
    residual = Nx.dot(steered, v)
    assert Nx.all_close(residual, Nx.broadcast(0.0, {3}), atol: 1.0e-5) |> Nx.to_number() == 1

    # The delta is exactly (x·v) v per row
    proj = Nx.dot(x, v)
    expected_delta = Nx.multiply(Nx.new_axis(proj, -1), v)
    delta = Nx.subtract(x, steered)
    assert Nx.all_close(delta, expected_delta, atol: 1.0e-5) |> Nx.to_number() == 1

    # And it actually changed something (x has nonzero component along v)
    assert Nx.to_number(Nx.reduce_max(Nx.abs(delta))) > 1.0e-3
  end

  test "alpha=0.5 removes half the projection" do
    x = batch_x()
    v = unit_v()
    steered = Steering.steer(x, v, 0.5)

    proj_before = Nx.dot(x, v)
    proj_after = Nx.dot(steered, v)
    assert Nx.all_close(proj_after, Nx.multiply(proj_before, 0.5), atol: 1.0e-5)
           |> Nx.to_number() == 1
  end

  test "orthogonal directions are untouched at any alpha" do
    v = Nx.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = Nx.tensor([[0.0, 2.0, -1.0, 3.0, 0.5, 0.0, 1.0, -2.0]])
    steered = Steering.steer(x, v, 1.0)
    assert Nx.all_close(steered, x, atol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "load!/1 round-trips and re-normalizes" do
    dir = System.tmp_dir!()
    path = Path.join(dir, "steer_test_#{System.unique_integer([:positive])}.bin")

    # Deliberately NOT unit norm — load! must normalize
    v = Nx.tensor([3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0], backend: Nx.BinaryBackend)
    File.write!(path, :erlang.term_to_binary(%{v: v, meta: %{source: :test}}))

    on_exit(fn -> File.rm(path) end)

    %{v: loaded, meta: meta} = Steering.load!(path)
    assert meta.source == :test
    assert_in_delta Nx.to_number(Nx.LinAlg.norm(loaded)), 1.0, 1.0e-6
    assert_in_delta Nx.to_number(loaded[0]), 0.6, 1.0e-6
    assert_in_delta Nx.to_number(loaded[2]), 0.8, 1.0e-6
  end

  test "load!/1 rejects zero vectors" do
    dir = System.tmp_dir!()
    path = Path.join(dir, "steer_zero_#{System.unique_integer([:positive])}.bin")
    v = Nx.broadcast(Nx.tensor(0.0, backend: Nx.BinaryBackend), {@hidden})
    File.write!(path, :erlang.term_to_binary(%{v: v, meta: %{}}))
    on_exit(fn -> File.rm(path) end)

    assert_raise ArgumentError, ~r/zero norm/, fn -> Steering.load!(path) end
  end
end
