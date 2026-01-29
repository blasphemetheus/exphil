# Axon Orthogonal Initialization Fix

**Status:** Planned contribution
**Created:** 2026-01-29
**Target:** [elixir-nx/axon](https://github.com/elixir-nx/axon)

---

## Problem Statement

Axon's `orthogonal` initializer fails when used with RNN layers (LSTM/GRU) due to shape constraints in the QR decomposition approach.

### Error Example

```
** (ArgumentError) length at axis 1 must be less than axis size of 4, got: 32
    (nx 0.10.0) lib/nx/shape.ex:1267: Nx.Shape.do_slice/7
    (axon 0.8.0) lib/axon/initializers.ex:706: Axon.Initializers."__defn:orthogonal_impl__"/2
```

### Root Cause

LSTM recurrent weights have shape `[hidden_size, 4*hidden_size]` (for 4 gates: input, forget, cell, output). When `hidden_size=32`:
- Weight shape: `[32, 128]`
- QR decomposition of `[32, 128]` produces Q of shape `[32, 32]`
- Slicing `[32, 128]` from `[32, 32]` fails

The current implementation assumes `rows >= cols` or similar constraint.

---

## Why This Matters

### Research Background

Orthogonal initialization is the **recommended approach for RNNs** (Saxe et al., 2013):

1. **Gradient preservation:** Orthogonal matrices have eigenvalues with magnitude 1, preventing gradient explosion/vanishing through time
2. **Faster convergence:** Networks reach good solutions faster
3. **Better final performance:** Especially for long sequences

### Current Workarounds

Without orthogonal init, users must rely on:
- Layer normalization (normalizes activations, indirect fix)
- Lower learning rates (slows training)
- Gradient clipping (treats symptom, not cause)

These work but are suboptimal compared to proper initialization.

---

## Proposed Solutions

### Option 1: Pad-and-Slice (Simplest)

Generate a larger square orthogonal matrix, then slice to needed shape.

```elixir
defn orthogonal_impl(shape, type, key, gain) do
  {rows, cols} = shape
  size = max(rows, cols)

  # Generate square orthogonal matrix
  random = Nx.Random.normal(key, shape: {size, size}, type: type)
  {q, _r} = Nx.LinAlg.qr(random)

  # Slice to actual shape
  result = Nx.slice(q, [0, 0], [rows, cols])
  Nx.multiply(result, gain)
end
```

**Pros:**
- Simple implementation
- Maintains orthogonality for the submatrix
- Works for any shape

**Cons:**
- Rows beyond `min(rows, cols)` aren't truly orthogonal to each other
- Wastes computation generating unused rows/cols

**Mathematical note:** Slicing rows from an orthogonal matrix preserves orthonormality of those rows. Slicing columns gives vectors that are still unit-length but may not be mutually orthogonal.

### Option 2: Block-Orthogonal for RNNs (Most Correct)

For LSTM/GRU, initialize each gate's weights as a separate orthogonal matrix.

```elixir
defn orthogonal_block_impl(shape, type, key, gain, num_blocks) do
  {rows, total_cols} = shape
  cols_per_block = div(total_cols, num_blocks)

  # Generate orthogonal matrix for each block
  blocks = for i <- 0..(num_blocks - 1) do
    block_key = Nx.Random.split(key, i)
    random = Nx.Random.normal(block_key, shape: {rows, cols_per_block}, type: type)
    {q, _r} = Nx.LinAlg.qr(random)
    q
  end

  # Concatenate blocks
  Nx.concatenate(blocks, axis: 1) |> Nx.multiply(gain)
end
```

**Pros:**
- Each gate (input, forget, cell, output) has proper orthogonal weights
- Matches how LSTM gates function independently
- Theoretically most sound for RNNs

**Cons:**
- Requires knowing `num_blocks` (4 for LSTM, 3 for GRU)
- More complex implementation
- May need RNN-specific initializer variant

### Option 3: SVD-Based (Most General)

Use SVD to generate orthogonal matrices of any shape.

```elixir
defn orthogonal_svd_impl(shape, type, key, gain) do
  {rows, cols} = shape

  # Generate random matrix
  random = Nx.Random.normal(key, shape: shape, type: type)

  # SVD gives U (m×m orthogonal), S (singular values), Vt (n×n orthogonal)
  {u, _s, vt} = Nx.LinAlg.svd(random)

  # For rows <= cols: use U[:, :cols] @ Vt[:cols, :]
  # For rows > cols: use U[:, :cols]
  result = if rows <= cols do
    Nx.slice(u, [0, 0], [rows, rows])
    |> Nx.dot(Nx.slice(vt, [0, 0], [rows, cols]))
  else
    Nx.slice(u, [0, 0], [rows, cols])
  end

  Nx.multiply(result, gain)
end
```

**Pros:**
- Works for any shape
- Mathematically rigorous
- Produces matrices with orthonormal rows OR columns (depending on shape)

**Cons:**
- SVD is more expensive than QR
- More complex to understand
- May be overkill for most use cases

### Option 4: Approximate via Iterative Orthogonalization

Use iterative methods (Gram-Schmidt or Newton iteration) to orthogonalize a random matrix.

```elixir
defn orthogonal_newton_impl(shape, type, key, gain, iterations \\ 5) do
  {rows, cols} = shape

  # Start with random matrix
  w = Nx.Random.normal(key, shape: shape, type: type)

  # Newton iteration: W = 1.5*W - 0.5*W@W.T@W
  # Converges to orthogonal matrix
  result = Enum.reduce(1..iterations, w, fn _, w ->
    Nx.subtract(
      Nx.multiply(w, 1.5),
      Nx.multiply(Nx.dot(Nx.dot(w, Nx.transpose(w)), w), 0.5)
    )
  end)

  Nx.multiply(result, gain)
end
```

**Pros:**
- Works for any shape
- Doesn't require QR or SVD
- Can be tuned (more iterations = more orthogonal)

**Cons:**
- Approximate, not exact
- Fixed iteration count may not converge for all shapes
- Less standard approach

---

## Recommendation

**Best approach: Option 1 (Pad-and-Slice) with Option 2 as RNN-specific variant**

### Rationale

1. **Option 1 is simple and sufficient for most cases**
   - Easy to implement and review
   - Covers the common failure case (cols > rows)
   - Minimal changes to existing API

2. **Option 2 can be added as `orthogonal_block` for RNN users**
   - More theoretically correct for LSTM/GRU
   - Opt-in for users who want it
   - Backwards compatible

### Proposed API

```elixir
# Fixed general orthogonal (Option 1)
Axon.Initializers.orthogonal(gain: 1.0)

# New RNN-specific variant (Option 2)
Axon.Initializers.orthogonal_block(num_blocks: 4, gain: 1.0)  # For LSTM
Axon.Initializers.orthogonal_block(num_blocks: 3, gain: 1.0)  # For GRU
```

---

## Implementation Plan

### Phase 1: Issue

Open GitHub issue with:
- Problem description and error message
- Minimal reproduction case
- Proposed solution (Option 1)
- Offer to submit PR

**Draft issue title:** `orthogonal initializer fails with RNN weight shapes`

### Phase 2: PR for Option 1

1. Fork elixir-nx/axon
2. Modify `lib/axon/initializers.ex`:
   - Fix `orthogonal_impl` to handle cols > rows
   - Add tests for rectangular shapes
3. Update documentation
4. Submit PR

### Phase 3: Follow-up PR for Option 2 (optional)

If maintainers are interested:
1. Add `orthogonal_block` initializer
2. Document RNN-specific usage
3. Consider auto-detection for RNN layers

---

## Test Cases to Add

```elixir
describe "orthogonal/1" do
  test "works with square matrices" do
    init = Axon.Initializers.orthogonal()
    tensor = init.({64, 64}, {:f, 32}, Nx.Random.key(0))
    assert Nx.shape(tensor) == {64, 64}
    # Verify orthogonality: W @ W.T ≈ I
    product = Nx.dot(tensor, Nx.transpose(tensor))
    assert_all_close(product, Nx.eye(64), atol: 1.0e-5)
  end

  test "works with cols > rows (LSTM case)" do
    init = Axon.Initializers.orthogonal()
    # LSTM: [hidden_size, 4*hidden_size]
    tensor = init.({32, 128}, {:f, 32}, Nx.Random.key(0))
    assert Nx.shape(tensor) == {32, 128}
    # Rows should be orthonormal
    row_products = Nx.dot(tensor, Nx.transpose(tensor))
    assert_all_close(row_products, Nx.eye(32), atol: 1.0e-5)
  end

  test "works with rows > cols" do
    init = Axon.Initializers.orthogonal()
    tensor = init.({128, 32}, {:f, 32}, Nx.Random.key(0))
    assert Nx.shape(tensor) == {128, 32}
    # Columns should be orthonormal
    col_products = Nx.dot(Nx.transpose(tensor), tensor)
    assert_all_close(col_products, Nx.eye(32), atol: 1.0e-5)
  end

  test "respects gain parameter" do
    init = Axon.Initializers.orthogonal(gain: 2.0)
    tensor = init.({32, 32}, {:f, 32}, Nx.Random.key(0))
    # Frobenius norm should be sqrt(32) * 2.0
    norm = Nx.LinAlg.norm(tensor)
    expected = :math.sqrt(32) * 2.0
    assert_in_delta(Nx.to_number(norm), expected, 0.1)
  end
end
```

---

## References

- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). [Exact solutions to the nonlinear dynamics of learning in deep linear networks](https://arxiv.org/abs/1312.6120)
- PyTorch `nn.init.orthogonal_`: [Implementation](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_)
- TensorFlow `Orthogonal`: [Implementation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal)

---

## Timeline

1. **Week 1:** Open issue, gauge maintainer interest
2. **Week 2:** Submit PR for Option 1 if approved
3. **Week 3+:** Iterate based on review feedback
4. **Future:** Option 2 PR if there's demand

---

## Notes

- PyTorch and TensorFlow both handle arbitrary shapes correctly
- This is likely an oversight in Axon, not a design decision
- The Nx ecosystem is actively maintained, PRs are usually reviewed quickly
