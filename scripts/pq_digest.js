#!/usr/bin/env node
// Digest of the 10-24 press-commitment / passivity testbed queue (scripts/overnight_1024.sh stage 1):
// one row per arm from its agreement.json + coherence / closed-loop RESULT lines.
//   node scripts/pq_digest.js [arm ...]   (default: every eval_runs/1001_queue/pq_* dir)
// Columns: val, change-frame option-family agreement, joint, change recall, false change,
// press-edge same-family % (jump / attack / special), calibration AUC (jump / attack),
// false press %, STANDING change recall, closed-loop SD/min and deaths/min.
const fs = require("fs"), path = require("path");
const Q = "eval_runs/1001_queue";
const arms = process.argv.length > 2 ? process.argv.slice(2) : fs.readdirSync(Q).filter((d) => d.startsWith("pq_")).sort((a, b) => fs.statSync(path.join(Q, a)).mtimeMs - fs.statSync(path.join(Q, b)).mtimeMs);
const grab = (file, re) => { try { const m = fs.readFileSync(file, "utf8").match(re); return m ? m[1] : "–"; } catch (e) { return "–"; } };
const f1 = (x) => (x == null ? "–" : typeof x === "number" ? x.toFixed(1) : String(x));
const rows = arms.map((a) => {
  const d = path.join(Q, a); let j = null;
  try { j = JSON.parse(fs.readFileSync(path.join(d, "agreement.json"))); } catch (e) {}
  const st = j && j.by_action_change_frames.find((x) => x.action === "STANDING");
  const pe = (fam) => j ? (j.press_edges.find((x) => x.family === fam) || {}).model_same_press : null;
  const auc = (fam) => j ? (j.press_calibration.find((x) => x.family === fam) || {}).auc : null;
  return {
    arm: a.replace(/^pq_/, ""),
    val: grab(path.join(d, "train.log"), /val_loss[= ]*([0-9.]+)(?![\s\S]*val_loss)/),
    fam: j && f1(j.change.families), joint: j && f1(j.change.joint), recall: j && f1(j.change.change_recall), falsechg: j && f1(j.hold.false_change),
    pj: f1(pe("jump")), pa: f1(pe("attack")), ps: f1(pe("special")), aj: j ? auc("jump") : "–", aa: j ? auc("attack") : "–",
    fp: j && f1(j.false_press_rate_on_no_edge_frames), standing: st ? f1(st.change_recall) : "–",
    sd: grab(path.join(d, "closed_loop.log"), /SD\/min ([0-9.]+)/), deaths: grab(path.join(d, "closed_loop.log"), /deaths\/min ([0-9.]+)/),
    coh: grab(path.join(d, "coherence.log"), /change_recall=([0-9.]+)/),
  };
});
const cols = [["arm", 10], ["val", 6], ["fam%", 5], ["joint", 5], ["recall", 6], ["falseΔ", 6], ["pressJ", 6], ["pressA", 6], ["pressS", 6], ["aucJ", 5], ["aucA", 5], ["falseP", 6], ["STAND", 5], ["SD/m", 5], ["dth/m", 5], ["cohRec", 6]];
const keys = ["arm", "val", "fam", "joint", "recall", "falsechg", "pj", "pa", "ps", "aj", "aa", "fp", "standing", "sd", "deaths", "coh"];
console.log(cols.map(([c, w]) => c.padEnd(w)).join(" "));
for (const r of rows) console.log(keys.map((k, i) => String(r[k] ?? "–").padEnd(cols[i][1])).join(" "));
console.log("\nfam% = option-family agreement on change frames; pressJ/A/S = model's top guess pressed the same family on the expert's press frame; aucJ/A = press-probability ranking; falseP = model press edge with no real press; STAND = change recall in STANDING; SD/m, dth/m = closed-loop sim vs idle; cohRec = offline coherence change recall.");
