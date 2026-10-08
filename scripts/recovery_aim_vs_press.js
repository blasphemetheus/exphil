// Among offstage, falling, jump-spent samples (traces every 3 f): how often is the
// stick UP (aim), and how often is B down given the stick is up vs not (press)?
const fs = require("fs");
const EDGE = 85.5656967163;
function load(p) { const d = JSON.parse(fs.readFileSync(p)); return d.episodes || d; }
function stats(eps) {
  let n = 0, up = 0, bUp = 0, bNotUp = 0, notUp = 0, upOnset = 0, upOnsetOpp = 0;
  for (const ep of eps) {
    const tr = ep.trace || ep.samples || ep;
    if (!Array.isArray(tr)) continue;
    let prevUp = null;
    for (const s of tr) {
      const [action, x, y, vy, sx, sy, b, jump, jl] = s;
      const off = Math.abs(x) > EDGE || y < -5;
      const falling = off && action !== 35 && action < 341 && action > 13 && !(action >= 252 && action <= 263);
      const isUp = sy >= 0.75;
      if (falling && jl < 1 && y < -20) {
        n++;
        if (isUp) { up++; if (b) bUp++; } else { notUp++; if (b) bNotUp++; }
        if (prevUp === false) { upOnsetOpp++; if (isUp) upOnset++; }
      }
      prevUp = falling && jl < 1 && y < -20 ? isUp : null;
    }
  }
  const f = (a, b) => b ? (100 * a / b).toFixed(1) + "%" : "-";
  return `n=${n}  stick-up share ${f(up, n)}  up-onset/3f ${f(upOnset, upOnsetOpp)}  P(B|up) ${f(bUp, up)}  P(B|not up) ${f(bNotUp, notUp)}`;
}
console.log("expert   ", stats(load("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json")));
for (const a of process.argv.slice(2)) console.log(a.padEnd(9), stats(load(`eval_runs/1001_queue/${a}/recovery_means.json`)));
