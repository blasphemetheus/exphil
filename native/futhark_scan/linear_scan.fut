-- Futhark Linear Scan: h = a * h + b
--
-- Uses Futhark's built-in parallel prefix scan (Blelloch-style).
-- The key insight: linear recurrence h = a*h + b is a monoid under composition.
--
-- The associative operator for composing (a1, b1) then (a2, b2):
--   (a_combined, b_combined) = (a1 * a2, a2 * b1 + b2)
--
-- This allows O(log T) parallel scan instead of O(T) sequential.
-- Better for long sequences, higher constant overhead for short ones.
--
-- Inputs:  a_vals [batch][seq_len][hidden] — decay coefficients
--          b_vals [batch][seq_len][hidden] — additive terms
--          h0     [batch][hidden]          — initial hidden state
-- Output:  result [batch][seq_len][hidden] — all hidden states

-- Associative operator for linear recurrence composition
let combine (a1: f32, b1: f32) (a2: f32, b2: f32): (f32, f32) =
  (a1 * a2, a2 * b1 + b2)

-- Single-sequence linear scan using parallel prefix scan
-- a_seq: [seq_len], b_seq: [seq_len], h0_val: f32
-- Returns: [seq_len] of hidden states
let linear_scan_1d [n] (a_seq: [n]f32) (b_seq: [n]f32) (h0_val: f32): [n]f32 =
  -- Prepend identity would shift; instead, incorporate h0 into first element
  -- For scan starting at h0: modify b[0] to be a[0]*h0 + b[0], then scan
  let b_seq' = copy b_seq
  let b_seq'[0] = a_seq[0] * h0_val + b_seq[0]
  -- Run inclusive scan with the associative operator
  let scanned = scan combine (1.0f32, 0.0f32) (zip a_seq b_seq')
  -- Extract the b component (accumulated state) from each pair
  in map (\(_a, b) -> b) scanned

-- Entry point: batched linear scan
-- Processes batch and hidden dims in parallel, scan over seq_len
entry linear_scan [batch][seq_len][hidden]
    (a_vals: [batch][seq_len][hidden]f32)
    (b_vals: [batch][seq_len][hidden]f32)
    (h0:     [batch][hidden]f32)
    : [batch][seq_len][hidden]f32 =
  -- Parallelize over batch and hidden, scan over seq_len
  map3 (\a_b b_b h0_b ->
    -- a_b: [seq_len][hidden], b_b: [seq_len][hidden], h0_b: [hidden]
    -- Transpose to [hidden][seq_len] for per-hidden scans
    let a_t = transpose a_b  -- [hidden][seq_len]
    let b_t = transpose b_b  -- [hidden][seq_len]
    -- Parallel map over hidden dim, each does a scan over seq_len
    let result_t = map3 linear_scan_1d a_t b_t h0_b  -- [hidden][seq_len]
    in transpose result_t  -- [seq_len][hidden]
  ) a_vals b_vals h0

-- Flat entry point: reshape 1D arrays into 3D/2D, run scan, flatten back.
-- NIF passes flat f32 arrays + shape integers; this handles reshaping.
entry linear_scan_flat [n][m]
    (a_flat: [n]f32)
    (b_flat: [n]f32)
    (h0_flat: [m]f32)
    (batch_: i64) (seq_len_: i64) (hidden_: i64)
    : [n]f32 =
  -- Manual reshape: index into flat array
  let batch = i64.i64 batch_
  let seq_len = i64.i64 seq_len_
  let hidden = i64.i64 hidden_
  let a_3d = tabulate_3d batch seq_len hidden (\b t h -> a_flat[b * seq_len * hidden + t * hidden + h])
  let b_3d = tabulate_3d batch seq_len hidden (\b t h -> b_flat[b * seq_len * hidden + t * hidden + h])
  let h0_2d = tabulate_2d batch hidden (\b h -> h0_flat[b * hidden + h])
  let result = linear_scan a_3d b_3d h0_2d
  -- Flatten back to 1D
  in tabulate n (\i ->
    let b = i / (seq_len * hidden)
    let rem = i % (seq_len * hidden)
    let t = rem / hidden
    let h = rem % hidden
    in result[b, t, h]
  )
