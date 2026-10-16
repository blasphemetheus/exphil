# Label review — watch the labeler, DAgger and the teacher, one run at a time

Open **http://127.0.0.1:8012/labels/?run=/labelruns/1016_label_review** on this
machine (any export dir under `eval_runs/` works after `/labelruns/`).

Why (Bradley, 2026-10-16): the labeler / DAgger / teacher path is invisible —
its effect shows up only in a trained model, so every bug in it costs an
epoch. This page puts a human eye between the labeler and the GPU: pick a
run, watch the clip, and see the label next to the real press, frame by
frame, with the expert rows the label was sampled from.

## What you see

- **Left**: the runs in the export, filterable by source (a DAgger set = the
  bot's own states + the chained label; an expert replay = the expert's
  states + the expert's press + the teacher-forced label), by readout cell,
  by label edge, and "label edge the real press never made in that cell".
  Chips: cells present, real-press edges (blue), label edges (orange).
- **Pane**: the run's clip (the sim viewer). Nothing plays until a run is
  picked; it loads paused at frame 0. Orange triangles on its seek bar are
  the label's edges.
- **Timeline**: per frame, the cell band (colour), gated frames (red: the
  state was farther than the d² gate from every expert row, so it is
  input-only), context frames (grey), and six lanes — B, jump, shoulder, A,
  stick-up, stick-sideways — each split into the real press (blue, top) and
  the label (orange, bottom). Triangles: edges (label below, real above).
  Click to seek; ← → step; [ ] jump between label edges; n / p change run.
- **Detail**: the frame's state (action, height, distance past the edge,
  jumps, hitstun), the real press, the label, the prev the label conditions
  on, and the k nearest expert rows: what each row presses or holds, its OWN
  cell and its d². A red row is a borrow from another cell — the 10-16
  finding (airdodges offstage-high from wavelands over the stage) reads as
  "off_high_inhand frame, drift_inhand rows" here.

## Export

```sh
scripts/launch_unit.sh exphil-label-review 'env EDIFICE_LOCAL_NX=1 mix run --no-compile scripts/label_review_export.exs \
  --index data/silent_fall/expert_recovery_index_v8.bin \
  --set data/silent_fall/sim_dagger_expert_r9o_2091.frames --set data/silent_fall/slp_dagger_expert_live1014.frames \
  --slp data/silent_fall/heldout_fd_fox_split.json --games 6 --max-runs 80 \
  --out eval_runs/1016_label_review'
```

- `--set` keep-actual DAgger sets (the `:actual` press is the bot's own); the
  stored chained label is shown, the neighbours are recomputed with `--index`
  (so an index newer than the set's shows what it WOULD sample).
- `--slp` a split json (its `validation` list) or a glob of `.slp`: expert
  excursions (60 f before the first readout cell to 30 f after the last),
  teacher-forced with the index.
- `--max-runs` per source (random sample of runs with ≥ `--min-cell-frames`
  frames in a readout cell).

Output: `manifest.json`, `<id>.msltrace.json` (clip), `<id>.json` (rows).

## Server

`scripts/viewer_serve.sh` builds `~/.cache/exphil/viewer_root` (symlinks:
`/tools` → melee-sim-light's built viewer, `/coach` `/ga` `/death-review`
`/labels` → `priv/viewer/*`, `/runs` `/garuns` `/labelruns` `/death-cases` →
`eval_runs`) and runs `python3 -m http.server 8012` on it as the user unit
`exphil-viewer-8012`. `scripts/viewer_serve.sh stop` stops it. The pane is
`priv/viewer/coach/pane.html` (one slippi-viewer per iframe, driven by
postMessage) — the same one the coach and GA pages use.
