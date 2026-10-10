// Side-B (Fox Illusion, 350..352) offstage trips, bot vs expert (2026-10-10,
// queue 41 close-out). The one death mode that replicated on every seed and
// arm: side-B trips die 100 % (13 / 71 / 15 / 20 per 16 env-min; the expert's
// 5 of 98 trips). "carried-off" in recovery_means is `first_at == 0` — the
// trip BEGAN inside a move — so a carried-off side-B is an Illusion fired on
// stage that carried the bot past the edge, not a hit. This reads where they
// are fired from, which way they fly, what the bot had left, and the 30
// frames before (pre_actions), so the fix can be aimed at the state before
// the press (Bradley 10-09: "there's probably another failure state before
// that we can make better").
//
//   node scripts/recovery_sideb_trips.js ARM [ARM ...]
//
// Per episode with first == side_b: carried (first_at == 0) vs later; sd
// flag (no hit in the 90 f before); jumps left at trip start; facing the
// edge; travel = sign of x motion over the first 4 trace rows (away = flying
// out); start y band; pre_actions tail (the approach). Trace rows are every
// 3 f: [action, x, y, speed_y, stick_x, stick_y, b, jump, jumps_left].
const fs = require("fs");
const EDGE = 85.5657; // FD
const pct = (a, b) => b ? (100 * a / b).toFixed(0) + "%" : "-";
const q = (a, p) => { a = a.filter(v => v != null && !isNaN(v)).sort((x, y) => x - y); return a.length ? a[Math.floor(p * (a.length - 1))] : NaN; };
function describe(e) {
  const t = e.trace || [];
  const side = Math.sign(e.x || 1);
  const r0 = t[0], r3 = t[Math.min(3, t.length - 1)];
  const travel = r0 && r3 && r3[1] != null ? Math.sign((r3[1] - r0[1]) * side) : 0; // +1 = away from the stage
  const jl0 = r0 ? r0[8] : e.jumps;
  // first row after the Illusion ends: what the bot did next (Fall 29 = nothing)
  let k = 0; while (k < t.length && t[k][0] >= 350 && t[k][0] <= 352) k++;
  const after = t[k] ? t[k][0] : null;
  const silentAfter = t.slice(k, k + 8).length > 0 && t.slice(k, k + 8).every(r => r[6] === 0 && r[7] === 0 && Math.hypot(r[4] - 0.5, r[5] - 0.5) < 0.2);
  return { carried: e.first_at === 0, died: e.outcome === "died", sd: !!e.sd, jl0, facing: !!e.pre_facing_edge,
    travel, y0: e.y, dist0: Math.abs(e.x) - EDGE, pre: (e.pre_actions || []).slice(-3).join(">"), stickEdge: e.pre_stick_edge_share, after, silentAfter, h: e.height };
}
function report(label, eps) {
  const sb = eps.filter(e => e.first === "side_b").map(describe);
  const car = sb.filter(s => s.carried), later = sb.filter(s => !s.carried);
  const line = (name, s) => {
    if (!s.length) { console.log(`    ${name}: none`); return; }
    const n = s.length, died = s.filter(x => x.died).length;
    const pre = {}; for (const x of s) pre[x.pre] = (pre[x.pre] || 0) + 1;
    const top = Object.entries(pre).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${k} ${pct(v, n)}`).join(", ");
    const aft = {}; for (const x of s) aft[x.after] = (aft[x.after] || 0) + 1;
    const topAfter = Object.entries(aft).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${pct(v, n)}`).join(", ");
    console.log(`    ${name}: n=${n} died ${pct(died, n)}  sd ${pct(s.filter(x => x.sd).length, n)}  no jump at start ${pct(s.filter(x => x.jl0 === 0).length, n)}  facing edge ${pct(s.filter(x => x.facing).length, n)}  flying away ${pct(s.filter(x => x.travel > 0).length, n)}  y q50 ${q(s.map(x => x.y0), 0.5).toFixed(0)}  start dist past edge q50 ${q(s.map(x => x.dist0), 0.5).toFixed(0)}  pre stick→edge q50 ${q(s.map(x => x.stickEdge), 0.5).toFixed(2)}`);
    console.log(`      approach (last 3 actions): ${top}`);
    console.log(`      after the Illusion: ${topAfter}  silent 8 rows after ${pct(s.filter(x => x.silentAfter).length, n)}  | died with no jump ${pct(s.filter(x => x.died && x.jl0 === 0).length, died || 1)}`);
  };
  console.log(`${label.padEnd(44)} side-B first-means trips ${sb.length} / ${eps.length} eps  (carried ${car.length}, later ${later.length})`);
  line("carried (Illusion fired on stage)", car);
  line("later (fired during the trip)", later);
}
const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
report("expert (FD reference)", Array.isArray(ex) ? ex : ex.episodes);
for (const arm of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${arm}/recovery_means.json`));
  report(arm, m.episodes);
}
