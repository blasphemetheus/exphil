defmodule ExPhil.Native.SelectiveScanTest do
  use ExUnit.Case, async: true

  alias ExPhil.Native.SelectiveScan

  @moduletag :nif
  @moduletag :cuda

  describe "availability" do
    test "available? returns boolean" do
      result = SelectiveScan.available?()
      assert is_boolean(result)
    end

    test "cuda_available? returns boolean" do
      result = SelectiveScan.cuda_available?()
      assert is_boolean(result)
    end
  end

  describe "forward scan" do
    @tag :requires_nif
    test "scan produces correct output shape" do
      skip_unless_nif_available()

      batch = 2
      seq_len = 4
      hidden = 8
      state = 4

      x = Nx.iota({batch, seq_len, hidden}, type: :f32) |> Nx.divide(100)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.iota({hidden, state}, type: :f32) |> Nx.negate() |> Nx.subtract(1)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      result = SelectiveScan.scan(x, dt, a, b, c)

      assert Nx.shape(result) == {batch, seq_len, hidden}
      assert Nx.type(result) == {:f, 32}
    end

    @tag :requires_nif
    test "scan produces non-zero output" do
      skip_unless_nif_available()

      batch = 1
      seq_len = 8
      hidden = 4
      state = 2

      x = Nx.broadcast(1.0, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.broadcast(-1.0, {hidden, state}) |> Nx.as_type(:f32)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      result = SelectiveScan.scan(x, dt, a, b, c)

      # With constant inputs and positive B/C, output should be non-zero
      mean = result |> Nx.abs() |> Nx.mean() |> Nx.to_number()
      assert mean > 0.0, "Expected non-zero output, got mean=#{mean}"
    end
  end

  describe "forward with states" do
    @tag :requires_nif
    test "scan_with_states returns output and hidden states" do
      skip_unless_nif_available()

      batch = 2
      seq_len = 4
      hidden = 8
      state = 4

      x = Nx.iota({batch, seq_len, hidden}, type: :f32) |> Nx.divide(100)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.iota({hidden, state}, type: :f32) |> Nx.negate() |> Nx.subtract(1)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      {out, h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)

      assert Nx.shape(out) == {batch, seq_len, hidden}
      assert Nx.shape(h_all) == {batch, seq_len, hidden, state}
    end

    @tag :requires_nif
    test "scan and scan_with_states produce same output" do
      skip_unless_nif_available()

      batch = 2
      seq_len = 4
      hidden = 8
      state = 4

      x = Nx.iota({batch, seq_len, hidden}, type: :f32) |> Nx.divide(100)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.iota({hidden, state}, type: :f32) |> Nx.negate() |> Nx.subtract(1)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      out1 = SelectiveScan.scan(x, dt, a, b, c)
      {out2, _h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)

      assert_all_close(out1, out2, atol: 1.0e-5)
    end
  end

  describe "backward pass" do
    @tag :requires_nif
    test "backward returns correct shapes" do
      skip_unless_nif_available()

      batch = 2
      seq_len = 4
      hidden = 8
      state = 4

      x = Nx.iota({batch, seq_len, hidden}, type: :f32) |> Nx.divide(100)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.iota({hidden, state}, type: :f32) |> Nx.negate() |> Nx.subtract(1)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      # Forward pass with state saving
      {_out, h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)

      # Gradient of output (simulating loss gradient)
      dy = Nx.broadcast(1.0, {batch, seq_len, hidden}) |> Nx.as_type(:f32)

      # Backward pass
      {dx, d_dt, d_b, d_c} = SelectiveScan.backward(dy, x, h_all, dt, a, b, c)

      assert Nx.shape(dx) == {batch, seq_len, hidden}
      assert Nx.shape(d_dt) == {batch, seq_len, hidden}
      assert Nx.shape(d_b) == {batch, seq_len, state}
      assert Nx.shape(d_c) == {batch, seq_len, state}
    end

    @tag :requires_nif
    test "backward produces non-zero gradients" do
      skip_unless_nif_available()

      batch = 1
      seq_len = 8
      hidden = 4
      state = 2

      x = Nx.broadcast(1.0, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.broadcast(-1.0, {hidden, state}) |> Nx.as_type(:f32)
      b = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)
      c = Nx.broadcast(1.0, {batch, seq_len, state}) |> Nx.as_type(:f32)

      {_out, h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)
      dy = Nx.broadcast(1.0, {batch, seq_len, hidden}) |> Nx.as_type(:f32)

      {dx, d_dt, d_b, d_c} = SelectiveScan.backward(dy, x, h_all, dt, a, b, c)

      # Gradients should be non-zero
      dx_mean = dx |> Nx.abs() |> Nx.mean() |> Nx.to_number()
      d_dt_mean = d_dt |> Nx.abs() |> Nx.mean() |> Nx.to_number()
      d_b_mean = d_b |> Nx.abs() |> Nx.mean() |> Nx.to_number()
      d_c_mean = d_c |> Nx.abs() |> Nx.mean() |> Nx.to_number()

      assert dx_mean > 0.0, "dx should be non-zero"
      assert d_dt_mean > 0.0, "d_dt should be non-zero"
      assert d_b_mean > 0.0, "d_b should be non-zero"
      assert d_c_mean > 0.0, "d_c should be non-zero"
    end

    @tag :requires_nif
    test "backward gradient numerical check" do
      # Numerical gradient check: compare backward kernel to finite differences
      skip_unless_nif_available()

      batch = 1
      seq_len = 4
      hidden = 4
      state = 2
      eps = 1.0e-4

      # Use deterministic tensors on BinaryBackend — Nx.Random.uniform with EXLA
      # produces shape {2} (PRNG key) instead of the requested shape in some contexts
      x = Nx.iota({batch, seq_len, hidden}, type: :f32) |> Nx.divide(16) |> Nx.add(0.1)
      dt = Nx.broadcast(0.05, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      a = Nx.broadcast(-1.0, {hidden, state}) |> Nx.as_type(:f32)
      b = Nx.iota({batch, seq_len, state}, type: :f32) |> Nx.divide(8) |> Nx.add(0.5)
      c = Nx.iota({batch, seq_len, state}, type: :f32) |> Nx.divide(8) |> Nx.add(0.3)

      # Compute analytical gradient
      {out, h_all} = SelectiveScan.scan_with_states(x, dt, a, b, c)
      dy = Nx.broadcast(1.0, {batch, seq_len, hidden}) |> Nx.as_type(:f32)
      {dx_analytical, _d_dt, _d_b, _d_c} = SelectiveScan.backward(dy, x, h_all, dt, a, b, c)

      # Compute numerical gradient for first element of x using Nx.put_slice
      x_plus = Nx.indexed_put(x, Nx.tensor([[0, 0, 0]]), Nx.tensor([Nx.to_number(x[[0, 0, 0]]) + eps]))

      out_plus = SelectiveScan.scan(x_plus, dt, a, b, c)
      loss_plus = out_plus |> Nx.sum() |> Nx.to_number()
      loss_base = out |> Nx.sum() |> Nx.to_number()

      numerical_grad = (loss_plus - loss_base) / eps
      analytical_grad = dx_analytical[[0, 0, 0]] |> Nx.to_number()

      # Allow for some tolerance due to numerical precision
      assert_in_delta numerical_grad, analytical_grad, 1.0e-2,
        "Gradient mismatch: numerical=#{numerical_grad}, analytical=#{analytical_grad}"
    end
  end

  # Helper functions

  defp skip_unless_nif_available do
    unless SelectiveScan.available?() do
      flunk("NIF not available - run: cd native/selective_scan_nif && cargo build --release --features cuda")
    end
  end

  defp assert_all_close(a, b, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Tensors not close: max difference = #{diff}, tolerance = #{atol}"
  end
end
