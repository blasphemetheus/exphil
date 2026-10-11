// One curve row per arm (2026-10-11, the full-run readout hook): the two
// numbers the testbed could not move and their pass bars, so a per-epoch
// gate sweep reads as a curve. From eval_runs/1001_queue/<arm>/
// recovery_means.json (+ train.log val, decision_map.json Firefox row).
//
//   node scripts/recovery_curve_row.js ARM [ARM ...]      # one line each
//   node scripts/recovery_curve_row.js --header            # column header
//   node scripts/recovery_curve_row.js --expert            # the expert's row
//
// decided = trips that began outside a move (first_at > 0; the bot chose
// the trip) — return share, pass >= 0.75 (expert 0.916, testbed 0.38–0.60).
// carried = first_at == 0 (the trip began inside a move) — count, died
// share (pass <= 0.60), and the carried side-B count (on-stage Illusion
// past the edge; pass <= 5 per run, testbed 13–22, 100 % fatal).
// ff-40..-60 = DecisionMap special_up j0 share in that band (not worse).
const fs = require("fs");
const args = process.argv.slice(2);
const f3 = v => (v == null || isNaN(v)) ? "-" : v.toFixed(3);
const pad = (s, n) => String(s).padEnd(n);
const cols = [["arm", 40], ["val", 7], ["eps", 5], ["decided", 8], ["ret", 6], ["carried", 8], ["c_died", 7], ["c_sideB", 8], ["died", 6], ["ff-40..-60", 12], ["pass", 20]];
if (args[0] === "--header") { console.log(cols.map(([c, n]) => pad(c, n)).join(" ")); process.exit(0); }

function row(label, E, val, ff) {
  const dec = E.filter(e => e.first_at > 0);
  const ret = dec.filter(e => e.outcome === "returned").length / (dec.length || 1);
  const car = E.filter(e => e.first_at === 0);
  const cDied = car.filter(e => e.outcome === "died").length / (car.length || 1);
  const cSideB = car.filter(e => e.first === "side_b").length;
  const died = E.filter(e => e.outcome === "died").length / (E.length || 1);
  const pass = [ret >= 0.75 ? "ret✓" : "ret✗", cSideB <= 5 ? "sideB✓" : "sideB✗", cDied <= 0.6 ? "cdied✓" : "cdied✗"].join(" ");
  const vals = [label, val || "-", E.length, dec.length, f3(ret), car.length, f3(cDied), cSideB, f3(died), ff || "-", pass];
  console.log(vals.map((v, i) => pad(v, cols[i][1])).join(" "));
}

if (args[0] === "--expert") {
  const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
  row("expert (150 FD games)", ex.episodes || ex, "-", "-");
  process.exit(0);
}
for (const arm of args) {
  const dir = `eval_runs/1001_queue/${arm}`;
  let E;
  try { E = JSON.parse(fs.readFileSync(`${dir}/recovery_means.json`)).episodes; }
  catch (e) { console.log(`${pad(arm, 40)} (no recovery_means.json)`); continue; }
  let val = null;
  try { const m = fs.readFileSync(`${dir}/train.log`, "utf8").match(/val_loss[= ]+([0-9.]+)/g); if (m) val = m[m.length - 1].match(/[0-9.]+$/)[0]; } catch (e) {}
  let ff = null;
  try { const s = JSON.parse(fs.readFileSync(`${dir}/decision_map.json`)).summary["-40..-60:j0"]; if (s) ff = `${s.special_up} [${s.frames}]`; } catch (e) {}
  row(arm, E, val, ff);
}
