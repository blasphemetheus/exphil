// Firefox fire ANGLE and charge-stick STABILITY, bot vs expert (2026-10-09,
// from Bradley's live look at dag34g_x2_e3: 8 of 18 stocks lost were a
// Firefox fired at the wrong angle — away from the stage, straight down,
// straight up from 45 units out, or so shallow it flew under the stage —
// and in 5 of those the stick was fine at charge start and wandered during
// the ~42-frame charge. The pass metric (Firefox-once-spent) scores the
// PRESS only; this scores where it pointed.
//
//   node scripts/recovery_firefox_angle.js ARM [ARM ...]
//
// Per Firefox (a 354 charge run in a recovery_means trace, rows every 3 f):
//   fire stick  = the stick on the last charge row (the game reads the
//                 direction at the end of the charge)
//   ideal       = unit vector from that position to the near ledge (±edge, 0)
//   angle error = degrees between the fire stick and the ideal
//   bucket      = toward-up / vertical / horizontal-toward / away / down
//   stability   = share of charge rows on the same 16-bucket stick as the
//                 fire row, and whether the first charge row already was
// Expert from the traced FD reference; arms from eval_runs/1001_queue/<arm>.
const fs = require("fs");
const EDGE = 85.5657; // FD
const bucket16 = v => Math.round((v - 0.5) * 16);
const key = r => `${bucket16(r[4])},${bucket16(r[5])}`;
function firefoxes(eps) {
  const out = [];
  for (const e of eps) {
    const t = e.trace;
    if (!t || t.length < 2) continue;
    let i = 0;
    while (i < t.length) {
      if (t[i][0] !== 354) { i++; continue; }
      let j = i;
      while (j < t.length && t[j][0] === 354) j++;
      const charge = t.slice(i, j), fired = j < t.length && t[j][0] === 356;
      const last = charge[charge.length - 1], first = charge[0];
      const side = Math.sign(first[1] || 1);
      const dx = last[4] - 0.5, dy = last[5] - 0.5;
      const toward = -dx * side, up = dy;
      const lx = side * EDGE - last[1], ly = 0 - last[2];
      const lm = Math.hypot(lx, ly) || 1, sm = Math.hypot(dx, dy);
      const cos = sm > 0.05 ? (dx * lx + dy * ly) / (sm * lm) : NaN;
      const err = isNaN(cos) ? null : Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
      let b;
      if (sm <= 0.2) b = "neutral";
      else if (up < -0.2) b = "down";
      else if (toward < -0.2) b = "away";
      else if (toward > 0.2 && up > 0.2) b = "toward-up";
      else if (toward > 0.2) b = "horizontal";
      else b = "vertical";
      const fk = key(last), stable = charge.filter(r => key(r) === fk).length / charge.length;
      const under = Math.abs(last[1]) < EDGE && last[2] < -5;
      out.push({ fired, err, b, stable, firstSame: key(first) === fk, nkeys: new Set(charge.map(key)).size,
        y0: first[2], dist0: Math.abs(first[1]) - EDGE, jl: first[8], rows: charge.length, under,
        outcome: e.outcome, toward: lx * side });
      i = j;
    }
  }
  return out;
}
const q = (a, p) => { a = a.filter(v => v != null).sort((x, y) => x - y); return a.length ? a[Math.floor(p * (a.length - 1))] : NaN; };
const pct = (a, b) => b ? (100 * a / b).toFixed(0) + "%" : "-";
function report(label, eps) {
  const ff = firefoxes(eps).filter(f => f.fired);
  const n = ff.length;
  const by = {};
  for (const f of ff) { by[f.b] = by[f.b] || { n: 0, died: 0 }; by[f.b].n++; if (f.outcome === "died") by[f.b].died++; }
  const order = ["toward-up", "vertical", "horizontal", "away", "down", "neutral"];
  const bs = order.filter(k => by[k]).map(k => `${k} ${pct(by[k].n, n)} (died ${pct(by[k].died, by[k].n)})`).join("  ");
  const errs = ff.map(f => f.err);
  const died = ff.filter(f => f.outcome === "died");
  console.log(`${label.padEnd(44)} firefox ${n} / ${eps.length} eps  angle err q50/q75 ${q(errs, 0.5).toFixed(0)}°/${q(errs, 0.75).toFixed(0)}°  >60° ${pct(ff.filter(f => f.err > 60).length, n)}`);
  console.log(`    fire bucket: ${bs}`);
  console.log(`    charge stability: same-as-fire share q50 ${q(ff.map(f => f.stable), 0.5).toFixed(2)}  first==fire ${pct(ff.filter(f => f.firstSame).length, n)}  >=3 distinct sticks ${pct(ff.filter(f => f.nkeys >= 3).length, n)}  | jump in hand at charge ${pct(ff.filter(f => f.jl > 0).length, n)}  start y q50 ${q(ff.map(f => f.y0), 0.5).toFixed(0)}  dist q50 ${q(ff.map(f => f.dist0), 0.5).toFixed(0)}`);
  if (died.length) console.log(`    died after firefox ${died.length}: angle err q50 ${q(died.map(f => f.err), 0.5).toFixed(0)}°  >60° ${pct(died.filter(f => f.err > 60).length, died.length)}  wandered (first!=fire) ${pct(died.filter(f => !f.firstSame).length, died.length)}`);
}
const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
report("expert (FD reference)", Array.isArray(ex) ? ex : ex.episodes);
for (const arm of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${arm}/recovery_means.json`));
  report(arm, m.episodes);
}
