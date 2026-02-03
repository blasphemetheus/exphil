# ExPhil Handoff Document

Generated: 2026-01-29

---

<original_task>
User asked about orthogonal initialization for RNNs - specifically whether we should contribute a fix to Axon since research shows orthogonal init is best for RNNs but Axon's implementation fails with LSTM/GRU weight shapes.
</original_task>

<work_completed>

## Documents Created

### 1. `docs/AXON_ORTHOGONAL_INIT_FIX.md` (new file)
Complete design document for contributing orthogonal init fix to Axon upstream:

**Problem documented:**
- Axon's `orthogonal` initializer fails with RNN weight shapes like `[32, 128]` (LSTM: `[hidden_size, 4*hidden_size]`)
- Error: `length at axis 1 must be less than axis size of 4, got: 32`
- Root cause: QR decomposition produces Q of shape `[min(rows, cols), min(rows, cols)]`, can't slice larger shape

**Four solutions analyzed in detail:**

1. **Pad-and-Slice (recommended):**
   - Generate `max(rows, cols)` square orthogonal matrix
   - Slice to needed shape
   - Simple, covers common case
   - Code example provided

2. **Block-Orthogonal:**
   - Initialize each LSTM gate (4 blocks) as separate orthogonal matrix
   - Most theoretically correct for RNNs
   - New API: `orthogonal_block(num_blocks: 4)`

3. **SVD-Based:**
   - Use SVD instead of QR for arbitrary shapes
   - Most general but more expensive
   - Code example provided

4. **Newton Iteration:**
   - Iteratively orthogonalize via `W = 1.5*W - 0.5*W@W.T@W`
   - Approximate, non-standard

**Recommendation:** Option 1 (pad-and-slice) for general fix, Option 2 as optional RNN-specific variant

**Implementation plan:**
- Phase 1: Open GitHub issue with reproduction case
- Phase 2: Submit PR for pad-and-slice fix
- Phase 3: Follow-up PR for block-orthogonal if maintainers interested

**Test cases provided:** 4 test cases covering square matrices, cols > rows, rows > cols, and gain parameter

### 2. `docs/IC_TECH_FEATURE_BLOCK.md` (new file - from earlier in session)
Design document for Ice Climbers tech feature embedding (14-dim minimal or 32-dim full)

### 3. `docs/GOALS.md` (updated)
Added new section "5. Upstream Contributions" with:
- Axon orthogonal init fix task (Planned status)
- IC Tech Feature Block task (Planned status)
- Summary of problem, solutions, and implementation plan

## Git Activity

**Commit pushed:** `500d7c3`
```
docs: Add IC Tech Feature Block and Axon orthogonal init fix plans
```

**Branch:** `main`
**Remote:** `github.com:blasphemetheus/exphil.git`

</work_completed>

<work_remaining>

## Immediate: Axon Contribution

1. **Open GitHub issue on elixir-nx/axon**
   - Title: `orthogonal initializer fails with RNN weight shapes`
   - Include minimal reproduction:
     ```elixir
     init = Axon.Initializers.orthogonal()
     init.({32, 128}, {:f, 32}, Nx.Random.key(0))  # Fails
     ```
   - Reference the design doc analysis
   - Offer to submit PR

2. **Fork and implement fix**
   - Fork `elixir-nx/axon`
   - Modify `lib/axon/initializers.ex` - `orthogonal_impl` function
   - Implement pad-and-slice approach
   - Add test cases from design doc

3. **Submit PR**
   - Reference the issue
   - Include before/after behavior
   - Document any API changes

## Future: Block-Orthogonal Variant

If maintainers are interested:
- Add `orthogonal_block/1` initializer
- Auto-detect RNN layers and suggest block variant
- Document RNN-specific usage patterns

## ExPhil: Use Fixed Orthogonal Init

Once Axon fix is merged:
- Update ExPhil's Axon dependency
- Re-enable orthogonal init for LSTM/GRU backbones
- Remove layer norm workaround if no longer needed
- Benchmark training stability improvement

</work_remaining>

<attempted_approaches>

## Previous Session: Orthogonal Init in ExPhil

**What was tried:**
- Enabled `Axon.Initializers.orthogonal()` for recurrent networks
- Failed with shape constraint error on LSTM weights

**Workaround implemented:**
- Reverted to `glorot_uniform` initialization
- Added layer normalization to stabilize gradients
- Lowered learning rate

**Why workaround works:**
- Layer norm normalizes activations (different mechanism than orthogonal init which preserves gradient norms)
- Combined with gradient clipping, provides stable training
- Not optimal but functional

</attempted_approaches>

<critical_context>

## Why Orthogonal Init Matters

- **Gradient flow:** Orthogonal matrices have eigenvalues = 1, preventing explosion/vanishing
- **Research:** Saxe et al. (2013) showed significant benefits for RNNs and deep networks
- **Industry standard:** PyTorch and TensorFlow both support arbitrary shapes correctly

## Axon Ecosystem Context

- Axon is actively maintained, PRs usually reviewed quickly
- The Nx team (José Valim et al.) is responsive to contributions
- This is likely an oversight, not a design decision
- Fix benefits entire Elixir ML ecosystem, not just ExPhil

## Shape Constraint Details

LSTM weight shapes:
- Input-to-hidden: `[input_size, 4*hidden_size]` (4 gates)
- Hidden-to-hidden: `[hidden_size, 4*hidden_size]`

GRU weight shapes:
- Input-to-hidden: `[input_size, 3*hidden_size]` (3 gates)
- Hidden-to-hidden: `[hidden_size, 3*hidden_size]`

QR decomposition of `[m, n]` produces:
- Q: `[m, min(m, n)]`
- R: `[min(m, n), n]`

When `n > m` (cols > rows), can't slice `[m, n]` from Q.

## References

- Saxe et al. (2013): https://arxiv.org/abs/1312.6120
- PyTorch orthogonal_: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_
- Axon repo: https://github.com/elixir-nx/axon

</critical_context>

<current_state>

## Documentation Status

| Document | Status |
|----------|--------|
| `docs/AXON_ORTHOGONAL_INIT_FIX.md` | ✅ Complete, committed, pushed |
| `docs/IC_TECH_FEATURE_BLOCK.md` | ✅ Complete, committed, pushed |
| `docs/GOALS.md` | ✅ Updated with upstream section |

## Git Status

- **Branch:** `main`
- **Latest commit:** `500d7c3` (pushed to origin)
- **Working tree:** Clean (aside from untracked files)

## Axon Contribution Status

| Step | Status |
|------|--------|
| Design document | ✅ Complete |
| GitHub issue | ❌ Not started |
| Fork Axon | ❌ Not started |
| Implement fix | ❌ Not started |
| Submit PR | ❌ Not started |

## ExPhil Training Status

- Currently using `glorot_uniform` + layer norm as workaround
- Orthogonal init blocked on Axon fix
- Training works but may not be optimal for long sequences

## Next Action

Open GitHub issue on elixir-nx/axon to gauge maintainer interest before investing in PR implementation.

</current_state>
