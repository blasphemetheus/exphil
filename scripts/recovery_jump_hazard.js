// Closed-loop double-jump hazard by height: over every offstage episode trace
// (samples every 3 frames: [action,x,y,vy,sx,sy,b,jump,jumps_left]), the share
// of samples that are "offstage, airborne-falling, jump in hand, not in a
// special" where the NEXT sample shows the jump spent (jumps_left drops) or
// the jump button up→down. Expert from the traced FD reference; arms from
// eval_runs/1001_queue/<arm>/recovery_means.json.
//   node scripts/recovery_jump_hazard.js evt2ctx_ck8_off3_dur8e_w240 ...
// 10-07 read: expert 15 → 44 → 60 % (0..-20 / -20..-40 / -40..-60), every arm
// flat 6–26 % — the double-jump defect (INPUT_COHERENCE "10-07 15:30"). The
// compiled ExPhil.Eval.DecisionMap is the per-frame version of this table.
const fs = require("fs");
const band = y => (y > 0 ? "y>0" : y > -20 ? "0..-20" : y > -40 ? "-20..-40" : y > -60 ? "-40..-60" : "<-60");
const BANDS = ["0..-20", "-20..-40", "-40..-60", "<-60"];
function hazard(eps) {
  const n = {}, k = {};
  let deep = 0;
  for (const e of eps) {
    const t = e.trace;
    if (!t || t.length < 2) continue;
    for (let i = 0; i + 1 < t.length; i++) {
      const [a, x, y, vy, , , , j, jl] = t[i], nx = t[i + 1];
      if (y < -60) deep++;
      const offstage = Math.abs(x) > 85.5656967163 || y < -5;
      if (!offstage || a === 35 || a >= 341 || jl < 1 || vy > 0) continue;
      const b = band(y);
      n[b] = (n[b] || 0) + 1;
      if (nx[8] < jl || (nx[7] && !j)) k[b] = (k[b] || 0) + 1;
    }
  }
  return { row: BANDS.map(b => (n[b] ? `${(100 * (k[b] || 0) / n[b]).toFixed(0).padStart(3)} % (n=${n[b]})` : "   -")), deep };
}
const pr = (label, eps) => {
  const h = hazard(eps);
  const slope = (() => { const get = b => { const m = h.row[BANDS.indexOf(b)].match(/(\d+) %/); return m ? +m[1] : null; }; const lo = get("-40..-60"), hi = get("0..-20"); return lo != null && hi ? (lo / hi).toFixed(1) : "-"; })();
  console.log(`${label.padEnd(34)} ${h.row.join("  ")}  slope ${slope}  frames<-60 ${h.deep}`);
};
console.log(`${"".padEnd(34)} ${BANDS.map(b => b.padEnd(16)).join("  ")}`);
pr("expert (FD)", JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json")));
for (const arm of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${arm}/recovery_means.json`));
  pr(arm, m.episodes || m);
}
