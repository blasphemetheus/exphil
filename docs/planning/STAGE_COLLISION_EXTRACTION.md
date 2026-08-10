# Stage Collision Extraction — build plan (deep dive 2026-08-11)

Harness task #1. Goal: real collision polylines (walls, slanted
grounds, ceilings/undersides, ledge-grab segments) for the 6 legal
stages, committed as data in `Melee.Stages`, consumed by the rewind
viewer, situation labels, recovery experts, and the coach's scenario
director.

## The format (fully mapped — HSDLib `SBM_Coll_Data.cs`)

Stage collision lives in each stage's `.dat` (an HSD archive) under a
`coll_data` root node:

- **`SBM_CollVertex`** (8 bytes): `float x` @0x00, `float y` @0x04.
- **`SBM_CollLine`** (16 bytes): `s16 vertex1` @0x00, `s16 vertex2`
  @0x02, `s16 next/prev line` @0x04/0x06 (linked chains), alt-group
  links @0x08/0x0A, **`CollPhysics` @0x0C** — `Top` (ceiling), `Bottom`
  (ground), `Right`/`Left` (walls), `Disabled` — **`CollProperty`
  @0x0E** — includes `DropThrough` and **`LedgeGrab`** — and
  `CollMaterial` @0x0F (Rock/Wood/Metal/…).
- **`SBM_Coll_Data`** (44 bytes): vertex array ptr+count @0x00,
  line array ptr+count @0x08, then per-direction offset/count pairs
  (Top/Bottom/Right/Left/Dynamic groupings) @0x10-0x22, line-group
  array @0x24.

Everything we want is first-class: walls are `Right`/`Left` lines,
slants are `Bottom` lines with non-horizontal endpoints, undersides are
`Top` lines, grab-able edges carry `LedgeGrab`. The per-direction
grouping even pre-sorts them.

## Stage files (inside the ISO filesystem)

| Stage | File |
|---|---|
| Battlefield | `GrNBa.dat` |
| Final Destination | `GrNLa.dat` |
| Yoshi's Story | `GrSt.dat` |
| Fountain of Dreams | `GrIz.dat` |
| Dreamland | `GrOp.dat` |
| Pokémon Stadium | `GrPs.dat` |

(Names per the Smashboards stage-hacking documentation; verify at
extraction time — PS may be `GrPst.dat`.)

## The pipeline (4 pieces, ~1 session)

1. **ISO → stage .dat files.** `~/isos/melee.iso` exists. Tooling:
   `wiimms-iso-tools` (`wit extract`, in nixpkgs) or Dolphin's
   filesystem browser (right-click game -> Properties -> Filesystem ->
   extract). One-time, manual is fine.
2. **HSD archive walker + coll_data parser** (new
   `scripts/extract_stage_collision.exs`, pure Elixir binary pattern
   matching — the .dat header is 0x20 bytes: file size, data-block
   size, reloc count, root count; root nodes carry string names, find
   `coll_data`, parse SBM_Coll_Data per the layout above). ~150 lines.
   Formats/facts aren't copyrightable — implementing the documented
   layout fresh has no license interaction. (Alternative zero-code
   path: HSDRawViewer already ships a collision->SVG exporter
   (`ConvSVG.cs`) — usable as a manual cross-check of our parser's
   output.)
3. **JSON → `Melee.Stages`**: per stage, polylines grouped by physics
   class: `grounds` (with slants intact), `walls_left/right`,
   `ceilings`, plus `ledge_grab` segments; committed as data (either
   module attributes or a priv/ JSON the module loads at compile).
   Suite: cross-check the parsed edge x against the existing
   `edge_ground_position` values (they must agree to ~0.01 — a strong
   full-pipeline validation), YS slants present, FD walls' top vertices
   at ledge height.
4. **Consumers**:
   - Viewer: render walls/ceilings/slants exactly (replace STAGE_META
     `walls` stylization); ledge-grab segments as accent ticks.
   - `ExPhil.Situations`: `walltech_available` / `walljump_zone` labels
     become E-difficulty (point-to-segment distance).
   - `FoxRecoveryExpert`: aim targets from real geometry.
   - Coach/scenario director: wall-drill situations.

## Open questions (resolve during build)

- PS transformations change collision (fire/rock/etc. layouts) — the
  base (normal) layout ships first; transformation layouts live in the
  same file's dynamic groups, revisit with task #3's transformation
  events.
- FoD's moving side platforms are dynamic collision (task #3's
  problem, not this one) — extract the static shell only.
- Randall is already handled analytically (not part of stage coll).
