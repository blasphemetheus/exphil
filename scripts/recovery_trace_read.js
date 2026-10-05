// Read the per-trip traces in a recovery_means.json (10-05): every 3rd frame of
// each offstage trip as [action, x, y, speed_y, stick_x, stick_y, b, jump, jumps_left].
//
//   node scripts/recovery_trace_read.js eval_runs/1001_queue/NAME/recovery_means.json [high|ledge|low|deep] [n_examples]
//   node scripts/recovery_trace_read.js eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json high
//
// Prints, for decided trips (first_at > 0) in the height band: stick toward/away
// share, stick-down/up share, double jump used, helpless reached, any B / any
// jump — died vs returned — then the deduped action strings of the died trips
// that never used up-B/side-B and a few raw traces. Ledge frames (252..263)
// count as silence here; the hazard-vs-silence analysis in
// INPUT_COHERENCE_2026-10-01.md ("10-05 12:55") excludes them.
const fs = require("fs");
const file = process.argv[2];
const band = process.argv[3] || "high";
const raw = JSON.parse(fs.readFileSync(file));
const eps = Array.isArray(raw) ? raw : raw.episodes;
const pick = eps.filter(e => e.first_at > 0 && e.height === band && e.trace && e.trace.length);

const stats = (e) => {
  const sign = e.x >= 0 ? -1 : 1; // toward stage = move x toward 0
  let tw = 0, aw = 0, dn = 0, up = 0, b = 0, j = 0, n = 0, djUsed = 0, help = 0, lastY = null, minY = 1e9, maxAbsX = 0;
  const j0 = e.trace[0][8];
  for (const t of e.trace) {
    const [a, x, y, vy, sx, sy, bb, jj, jl] = t;
    n++;
    if (sx != null) { if ((sx - 0.5) * sign >= 0.3) tw++; else if ((sx - 0.5) * sign <= -0.3) aw++; }
    if (sy != null) { if (sy <= 0.2) dn++; if (sy >= 0.8) up++; }
    b += bb; j += jj; if (jl < j0) djUsed = 1; if (a === 35) help = 1;
    minY = Math.min(minY, y); maxAbsX = Math.max(maxAbsX, Math.abs(x));
  }
  return { n, tw: tw / n, aw: aw / n, dn: dn / n, up: up / n, b, j, djUsed, help, minY, maxAbsX };
};
const agg = (list) => {
  const k = ["tw", "aw", "dn", "up", "djUsed", "help"], o = {};
  for (const key of k) o[key] = (list.reduce((s, e) => s + e[key], 0) / list.length).toFixed(2);
  o.b_any = (list.filter(e => e.b > 0).length / list.length).toFixed(2);
  o.j_any = (list.filter(e => e.j > 0).length / list.length).toFixed(2);
  o.n = list.length;
  return o;
};
const died = pick.filter(e => e.outcome !== "returned").map(stats);
const ret = pick.filter(e => e.outcome === "returned").map(stats);
console.log(`${band} decided: died`, JSON.stringify(agg(died)));
console.log(`${band} decided: returned`, JSON.stringify(agg(ret)));

// died, no up_b/side_b: deduped action strings and example traces
const nores = pick.filter(e => e.outcome !== "returned" && !/up_b|side_b/.test(e.seq));
const acts = {};
for (const e of nores) { const s = e.trace.map(t => t[0]).filter((a, i, arr) => i === 0 || arr[i - 1] !== a).join(">"); acts[s] = (acts[s] || 0) + 1; }
console.log("died w/o recovery move, action strings (top 12):");
for (const [s, c] of Object.entries(acts).sort((a, b) => b[1] - a[1]).slice(0, 12)) console.log(`  ${c}× ${s}`);
console.log("examples:");
for (const e of nores.slice(0, Number(process.argv[4] || 4))) {
  console.log(` x=${e.x} y=${e.y} jumps=${e.jumps} seq=${e.seq} frames=${e.frames} pre=${e.pre_actions}`);
  console.log("  " + e.trace.map(t => `${t[0]}@(${t[1]},${t[2]})vy${t[3]} s(${t[4]},${t[5]})${t[6] ? "B" : ""}${t[7] ? "J" : ""}j${t[8]}`).join(" | "));
}
