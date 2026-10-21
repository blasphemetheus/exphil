# Label review — watch the labeler, DAgger and the teacher, one run at a time

Open **http://127.0.0.1:8012/labels/?run=/labelruns/1016_label_review** on this
machine (any export dir under `eval_runs/` works after `/labelruns/`).

Why (Bradley, 2026-10-16): the labeler / DAgger / teacher path is invisible —
its effect shows up only in a trained model, so every bug in it costs an
epoch. This page puts a human eye between the labeler and the GPU: pick a
run, watch the clip, and see the label next to the real press, frame by
frame, with the expert rows the label was sampled from.

## Who is who

- **Red Fox = port 1 = the one being labelled.** The exporter remaps the
  subject to port 1 whatever port it played on. In a DAgger run that is **the
  bot** (rolled out in the sim; the opponent is the sim's scripted/replayed
  body). In an expert excursion that is **the human expert** from the
  held-out replay. The banner above the clip says which for the picked run.
- **Blue Fox = port 2 = the opponent.** Never labelled, never trained on.
- **Red lanes** = what the red Fox really pressed. **Purple lanes** = the
  label. For a DAgger run the label is the training target (state → label
  pairs; where purple ≠ red is what the mix teaches). For an expert run the
  label is teacher-forced on known-good states — purple ≠ red there is a
  labeler mistake, nothing is trained on it.

## What you see

- **Left**: the runs in the export, filterable by source (a DAgger set = the
  bot's own states + the chained label; an expert replay = the expert's
  states + the expert's press + the teacher-forced label), by readout cell,
  by label edge, and "label edge the real press never made in that cell".
  Chips: red = bot/expert, cells present, real-press edges (red), label edges (purple).
- **Pane**: the run's clip (the sim viewer). Nothing plays until a run is
  picked; it loads paused at frame 0. Purple triangles on its seek bar are
  the label's edges.
- **Timeline** (top to bottom): the readout cell name where a cell starts;
  **doing** = the red Fox's action state, one named block per run of frames
  (dash, turn, jumpsquat, jump, dair, shine, up-b, ledge hang, …), coloured by
  family (ground move / air / attack / special / defence / hit / ledge; the
  full action name is in the hover tooltip); **height** = the y curve against
  a dashed stage-level line, tan = on the ground, blue = airborne, a brown bar
  = offstage past the edge, a red bar = hitstun, a lilac bar = no double jump
  left; then six press lanes — B, jump, L/R, A, stick-up, stick-sideways —
  each split into the real press (red, top) and the label (purple, bottom);
  then the edges (a press that was not there the frame before), real above,
  label below, named when the run has fewer than 40; then frame ticks.
  Gated frames (red: the state was farther than the d² gate from every
  expert row, so input-only) and context frames (grey) tint the whole column.
  Hovering any column writes that frame's action, where it is, and both
  presses on the line above the timeline. Click to seek; ← → step; [ ] jump
  between label edges; n / p change run.
- **Detail**: the frame's state (action, height, distance past the edge,
  jumps, hitstun), the real press, the label, the prev the label conditions
  on, and the k nearest expert rows: what each row presses or holds, its OWN
  cell, its d², and (v10 indexes) the replay and frame it came from. A pink
  row is a borrow from another cell — the 10-16 finding (airdodges
  offstage-high from wavelands over the stage) reads as "off_high_inhand
  frame, drift_inhand rows" here.
- **Source pane** (right, 10-21): the hand-verification. Click a row marked
  ▶ and the right-hand pane plays the expert replay that row came from, paused
  on that very frame, with the row's press marked on its seek bar. The
  question to answer by eye is "is this the same situation as the red Fox's,
  and is what the expert did next sensible?" — a label that differs from the
  real press is fine when the source is a look-alike moment and a sane play;
  it is a labeler fault when the source is a different situation. Rows have
  a clip on every frame with an edge (label or real); on a frame with a label
  edge the first row's source loads without a click. Note that the nearest
  rows are often consecutive frames of ONE expert game (the same moment three
  times), so 3/8 can be one expert, not three.

## Export

Source clips need an index built after 10-21 (`expert_recovery_index_v10.bin`
or later: rows carry `{game, frame}`); `scripts/labeler_v10_source_clips.sh`
builds one and exports with clips. Older indexes export without the ▶ rows.


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
