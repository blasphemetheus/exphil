# Data scaling analysis — how far the pool grows, and the streaming gap

> **UPDATE 2026-07-23: the streaming EMBED pass is LANDED and flatness-
> PROVEN** via `ExPhil.Data.SituationIndex` (per-file f16 shards, built
> for flywheel P5 but designed as this doc's shard infrastructure).
> Measured peak RSS (self-reported VmHWM, archive/mewtwo corpus):
> 65k frames → 1.48 GB, 415k → 1.55 GB, 1.77M → 1.77 GB — **flat at 27x
> scale, vs 19.9 GB for the all-in-RAM path at the same 2M-frame scale
> (~11x lighter)**. The greg corpus (~50-70M frames) extrapolates to
> ~2 GB RAM / ~30-40 GB of disk shards.
>
> **Still open — the training-side wiring (step 1 below):** dagger_drill
> consuming shards per chunk. Needs: compact per-frame targets in the
> shard (controllers ~100 B/frame — 50M frames fit in ~5 GB), a
> first-pass for relabel/conversion-weights/probe rows, and a shuffle
> buffer over chunks. Do it with a full training-smoke validation, not as
> a drive-by — it touches the production drill path.

Written 2026-07-22 (r16 training). Answers the standing question: how do
we scale up training data without runs that OOM? The memory ledger now
has real anchor points, so the RAM ceiling is a closed-form number, and
the path past it is identifiable.

## The current ceiling: ~5-6M frames on this box

Measured points (`logs/memory_ledger.jsonl`), mamba_2, window 60:

| pool_frames | peak RSS | B/frame (peak) |
|-------------|----------|----------------|
| 2,036,709   | 19.87 GB | ~9,750         |

Post-embed RSS was 10.4 GB (~5,100 B/frame); the peak sits ~2x above
that (the epoch-boundary probe-eval + snapshot + the training graph's
working set). Against a usable budget of ~55 GB (62 GB RAM − headroom;
swap deliberately excluded — a training run in swap is already lost):

    max_frames ≈ 55 GB / 9.75 KB/frame ≈ 5.6M frames

So the **all-in-RAM path caps at ~5.6M frames — about 2.75x r16's pool**.
The `--memory-check` gate now predicts this per-run from the ledger and
refuses (strict) or warns (default) before the embed cost is paid.

This is enough headroom for near-term Mewtwo work (more r-line rollouts,
the full archive/mewtwo 129-game set) but NOT for the five-char program.

## Why the five-char program needs streaming

The greg corpus (#33) is 3,446 .slp across five characters. At a
conservative ~15-20k usable frames/game that is **~50-70M frames** —
roughly 10x the RAM ceiling. Even using a large *fraction* of it blows
past 5.6M. The all-in-RAM precompute cannot get there on this hardware;
more RAM only moves the ceiling linearly, not by the 10x needed.

The lever is **streaming (chunked) embedding**: embed a chunk, train on
it, discard, repeat — so peak RAM is O(chunk), not O(pool).

## The streaming path exists but is NOT wired into the drill

`lib/exphil/training/data.ex` already contains the chunked machinery:
- `create_sequence_batch_lazy/11` + `slice_from_chunks/6` — build
  sequence batches by slicing from fixed-size chunk arrays instead of one
  giant embeddings tensor.
- `ExPhil.Training.Config` carries `--stream-chunk-size` (default nil)
  and a note "only effective with --stream-chunk-size (streaming mode)".

But it is latent: **`dagger_drill.exs` (the drill / newera8 path r16 uses)
calls `precompute_frame_embeddings/2` directly** — the all-in-RAM path —
and never touches the chunked one. `train.exs` doesn't wire it either.
The `⚠ Large dataset — consider using --stream-chunk-size` warning points
at a flag that no training entry point actually consumes.

There is also a SEPARATE lever already built: `EmbeddingCache` /
`precompute_frame_embeddings_cached/2` persists embeddings to disk so a
re-run skips recompute — but it still loads the full set into RAM, so it
saves TIME across runs, not PEAK memory. Orthogonal to the ceiling.

## Recommended path (a build task, testable once GPU frees)

1. **Wire `--stream-chunk-size` into `dagger_drill.exs`.** When set,
   route through the chunked `create_sequence_batch_lazy` path instead of
   `precompute_frame_embeddings`; peak RAM becomes O(chunk + model), flat
   in pool size. This is the single change that lifts the ceiling.
2. **Validate memory flatness with the ledger**: run `--preflight
   --stream-chunk-size N` at 2M, 5M, 10M frames; the ledger points should
   show peak RSS roughly CONSTANT in pool size (the proof streaming
   works). If peak still grows, a full-pool structure (frames list,
   sampling weights, conversion spans) is the residual leak — hunt it.
3. **Watch the second-order costs streaming trades in**:
   - conversion-sampling + probe-reg currently assume the full embedded
     pool in memory; they need a chunk-compatible form or a first pass.
   - shuffle quality drops with chunking (can't globally shuffle a pool
     you never fully hold) — a shuffle-buffer over chunks is the usual
     fix; note it so training dynamics don't silently change.
4. **Pull the greg corpus locally** (`scripts/pull_replays.sh`, B2) only
   once streaming lands — 12.5 GiB of .slp is pointless to hold if the
   embed path still OOMs.

## One-line summary

All-in-RAM caps at ~5.6M frames (2.75x r16); the five-char corpus needs
~10x that; the chunked-embed machinery exists in `data.ex` but isn't
wired into `dagger_drill.exs` — wiring `--stream-chunk-size` into the
drill is the concrete task that unlocks large-scale data, and the
ledger + `--preflight` now make its memory-flatness directly verifiable.
