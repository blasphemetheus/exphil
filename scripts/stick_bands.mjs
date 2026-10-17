#!/usr/bin/env node
// Stick-band read of a label review export (2026-10-17, labeler v9 gate):
// how often the LABEL's main stick sits in the partial band
// 0.15 < |x - 0.5| < 0.4 (a walk / tilt for Fox) versus the real press on
// the same frames. A travelling flick (0.50 -> 0.72 -> 0.99) is partial for
// 1–2 frames in the real press; a pre-v9 label sampled a travel row and the
// hold then kept it, so the label's partial share ran well above the real
// one. Pass for v9: label share within ~1.5x of the real share per source.
//
//   node scripts/stick_bands.mjs eval_runs/1016_label_review [eval_runs/1017_label_review_v9 ...]
import fs from "node:fs";
import path from "node:path";

const band = (c) => { const d = Math.abs(c.ms[0] - 0.5); return d < 0.15 ? "neutral" : d < 0.4 ? "partial" : "full"; };

for (const dir of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
  const per = {};
  for (const r of m.runs) {
    const side = JSON.parse(fs.readFileSync(path.join(dir, r.sidecar), "utf8"));
    const s = (per[r.source] ||= { frames: 0, real: { neutral: 0, partial: 0, full: 0 }, label: { neutral: 0, partial: 0, full: 0 }, lagged: 0, changes: 0 });
    let prevLabel = null;
    for (const row of side.rows) {
      if (!row.label || !row.actual) { prevLabel = null; continue; }
      s.frames++; s.real[band(row.actual)]++; s.label[band(row.label)]++;
      // label stick changes that land partial (the slide), among label stick changes
      if (prevLabel && band(row.label) !== band(prevLabel)) { s.changes++; if (band(row.label) === "partial") s.lagged++; }
      prevLabel = row.label;
    }
  }
  console.log(`${dir}  (index ${m.index || "?"})`);
  for (const [src, s] of Object.entries(per)) {
    const pct = (x, n) => (n ? ((100 * x) / n).toFixed(1) : "–");
    console.log(`  ${src.padEnd(32)} ${s.frames} label frames | partial band: real ${pct(s.real.partial, s.frames)}%  label ${pct(s.label.partial, s.frames)}%  (x${(s.label.partial / Math.max(s.real.partial, 1)).toFixed(2)}) | full: real ${pct(s.real.full, s.frames)}%  label ${pct(s.label.full, s.frames)}% | label stick changes landing partial ${pct(s.lagged, s.changes)}% of ${s.changes}`);
  }
}
