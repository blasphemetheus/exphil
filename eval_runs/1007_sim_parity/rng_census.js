// Per-frame RNG draw-count delta (retail - sim) vs the GFX ids dispatched in that frame.
// Usage: node rng_census.js census1.log [census2.log ...]
const fs = require("fs");
function step(x) { return (Math.imul(x, 214013) + 2531011) >>> 0; }
function dist(a, b, cap = 5000) { let x = a; for (let i = 0; i < cap; i++) { if (x === b) return i; x = step(x); } return -1; }

const frames = []; // {delta, gfx: {id: count}, simDraws, retailDraws}
for (const file of process.argv.slice(2)) {
  const L = fs.readFileSync(file, "utf8").split("\n");
  let prevRec = null, prevSim = null, gfx = {}, windowStart = null;
  for (const l of L) {
    let m;
    if ((m = l.match(/^GFX_TRACE gfx (\w+) draws (\d+)/))) { gfx[m[1]] = (gfx[m[1]] || 0) + 1; }
    else if ((m = l.match(/^SEED restore#(\d+) sim ([0-9a-f]+) rec ([0-9a-f]+)/))) {
      const sim = parseInt(m[2], 16) >>> 0, rec = parseInt(m[3], 16) >>> 0;
      if (prevRec !== null) {
        const retailDraws = dist(prevRec, rec);          // retail: prev recorded seed -> this one
        const simDraws = dist(prevRec, sim);             // sim: restored prev seed -> state now
        if (retailDraws >= 0 && simDraws >= 0) frames.push({ frame: +m[1] - 123, delta: retailDraws - simDraws, gfx, simDraws, retailDraws, file });
      }
      prevRec = rec; gfx = {};
    }
  }
}
const n = frames.length, drift = frames.filter(f => f.delta !== 0);
console.log(`frames ${n}, drift frames ${drift.length} (${(100 * drift.length / n).toFixed(1)} %)`);
const hist = {}; for (const f of drift) hist[f.delta] = (hist[f.delta] || 0) + 1;
console.log("delta histogram (retail - sim):", JSON.stringify(Object.fromEntries(Object.entries(hist).sort((a, b) => Math.abs(a[0]) - Math.abs(b[0])))));

// Per gfx id: how often it appears in drift vs non-drift frames; mean delta when present alone-ish.
const ids = {};
for (const f of frames) for (const id of Object.keys(f.gfx)) { ids[id] = ids[id] || { frames: 0, drift: 0, deltaSum: 0 }; ids[id].frames++; if (f.delta !== 0) { ids[id].drift++; ids[id].deltaSum += f.delta; } }
console.log("\ngfx id: frames present / drift frames / mean delta in drift frames");
Object.entries(ids).sort((a, b) => b[1].drift - a[1].drift).slice(0, 18).forEach(([id, v]) => console.log(" ", id.padEnd(5), String(v.frames).padStart(5), String(v.drift).padStart(5), (v.drift ? (v.deltaSum / v.drift).toFixed(2) : "-").padStart(7)));

// Least squares: delta ≈ Σ count_id * err_id (ids seen in ≥3 drift frames)
const cand = Object.entries(ids).filter(([, v]) => v.drift >= 3).map(([id]) => id);
if (cand.length) {
  const A = frames.map(f => cand.map(id => f.gfx[id] || 0)), y = frames.map(f => f.delta);
  // normal equations
  const k = cand.length, AtA = Array.from({ length: k }, () => Array(k).fill(0)), Aty = Array(k).fill(0);
  for (let r = 0; r < A.length; r++) for (let i = 0; i < k; i++) { if (!A[r][i]) continue; Aty[i] += A[r][i] * y[r]; for (let j = 0; j < k; j++) AtA[i][j] += A[r][i] * A[r][j]; }
  for (let i = 0; i < k; i++) AtA[i][i] += 1e-6;
  // gaussian elimination
  const M = AtA.map((row, i) => [...row, Aty[i]]);
  for (let c = 0; c < k; c++) { let p = c; for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; if (Math.abs(M[c][c]) < 1e-9) continue; for (let r = 0; r < k; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let j = c; j <= k; j++) M[r][j] -= f * M[c][j]; } }
  const x = M.map((row, i) => row[k] / (row[i] || 1));
  console.log("\nleast-squares per-dispatch error (retail draws - modeled), |err| ≥ 0.3:");
  cand.forEach((id, i) => { if (Math.abs(x[i]) >= 0.3) console.log(" ", id.padEnd(5), x[i].toFixed(2)); });
  // residual
  let res = 0; for (let r = 0; r < A.length; r++) { let pred = 0; for (let i = 0; i < k; i++) pred += A[r][i] * x[i]; res += (y[r] - pred) ** 2; }
  console.log("residual rms:", Math.sqrt(res / A.length).toFixed(3), " raw rms:", Math.sqrt(y.reduce((s, v) => s + v * v, 0) / y.length).toFixed(3));
}
// frames with drift but NO gfx dispatched
const bare = drift.filter(f => Object.keys(f.gfx).length === 0);
console.log("\ndrift frames with no GFX dispatch:", bare.length, "sample:", bare.slice(0, 6).map(f => `f${f.frame}:${f.delta}`).join(" "));
