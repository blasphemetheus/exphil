// The silent fall's closed-loop hazard: P(enter silence | active) per 3 f on
// decided recovery trips (first_at > 0), below stage level, by depth × jumps,
// plus high-band decided return. Reads recovery_means.json episode files that
// carry `trace` (scripts/recovery_means.exs / expert_recovery_means.exs, 10-05).
//   node scripts/recovery_enter_silence.js eval_runs/1001_queue/NAME/recovery_means.json \
//        eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json
// trace rows: [action, x, y, speed_y, sx, sy, b, jump, jumps_left]; sticks 0..1, centre 0.5.
// Ledge hangs (CLIFF 252..263) and dead actions (<= 13) are excluded from both sides.
const fs = require("fs");
const DZ = 0.165;
const neutral = r => Math.abs(r[4] - 0.5) <= DZ && Math.abs(r[5] - 0.5) <= DZ && !r[6] && !r[7];
const cliff = a => a >= 252 && a <= 263;
for (const f of process.argv.slice(2)) {
  const raw = JSON.parse(fs.readFileSync(f));
  const eps = Array.isArray(raw) ? raw : raw.episodes;
  const dec = eps.filter(e => e.first_at > 0 && e.trace && e.trace.length > 1);
  const bands = {};
  const add = (k, hit) => { bands[k] = bands[k] || [0, 0]; bands[k][0] += hit; bands[k][1]++; };
  for (const e of dec) {
    const t = e.trace;
    for (let i = 0; i + 1 < t.length; i++) {
      const r = t[i], n = t[i + 1];
      if (r[0] <= 13 || cliff(r[0]) || r[2] >= 0 || neutral(r)) continue;
      const y = r[2], j = r[8] > 0 ? "j1+" : "j0";
      const band = y > -20 ? "ledge" : y > -60 ? "-20..-60" : "<-60";
      add(`${band} ${j}`, neutral(n) && !cliff(n[0]) && n[0] > 13 ? 1 : 0);
    }
  }
  const hi = eps.filter(e => e.first_at > 0 && e.height === "high");
  const hiRet = hi.length ? (hi.filter(e => e.outcome === "returned" || e.outcome === "recovered").length / hi.length) : NaN;
  const outs = [...new Set(eps.map(e => e.outcome))];
  console.log(f.split("/").slice(-2, -1)[0] || f, "decided", dec.length, "high-band decided n", hi.length, "return", hiRet.toFixed(2), "outcomes", outs.join(","));
  for (const k of ["-20..-60 j0", "-20..-60 j1+", "<-60 j0", "<-60 j1+", "ledge j0", "ledge j1+"]) {
    const [h, n] = bands[k] || [0, 0];
    console.log(`  ${k.padEnd(14)} ${(n ? h / n : NaN).toFixed(3)} (n=${n})`);
  }
}
