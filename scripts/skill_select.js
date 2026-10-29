#!/usr/bin/env node
// Skill-score analysis (2026-10-29): master-master vs untiered Fox games on the play-derived features,
// per-feature AUC, a combined z-score, and the list of untiered Fox games that score like masters.
//   node scripts/skill_select.js eval_runs/1029_skill/scores.jsonl [threshold_quantile=0.5]
// Writes <dir>/selection.json (master files, untiered-selected files, counts) and prints the table.
const fs = require("fs"), path = require("path");
const file = process.argv[2] || "eval_runs/1029_skill/scores.jsonl"; const q = Number(process.argv[3] || 0.5);
const rows = fs.readFileSync(file, "utf8").trim().split("\n").flatMap((l) => { const v = JSON.parse(l); return Array.isArray(v) ? v : [v]; }).filter((r) => !r.error && r.minutes >= 1.5);
const feats = ["l_cancel_rate", "short_hop_share", "apm", "damage_dealt_per_min", "sd_per_min", "offstage_return_rate", "tech_rate", "wavedashes_per_min", "dashes_per_min", "kills_per_min", "deaths_per_min", "neutral_input_share"];
const M = rows.filter((r) => r.tier === "master"), U = rows.filter((r) => r.tier === "untiered");
const auc = (pos, neg) => { const all = [...pos.map((v) => [v, 1]), ...neg.map((v) => [v, 0])].filter(([v]) => v != null).sort((a, b) => a[0] - b[0]); let s = 0, r = 1, np = 0, nn = 0; for (const [, y] of all) { if (y) { s += r; np++; } else nn++; r++; } return np && nn ? (s - np * (np + 1) / 2) / (np * nn) : NaN; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))) || 1; };
console.log(`rows ${rows.length}: master ${M.length}, untiered ${U.length} (Fox games >= 1.5 min)`);
console.log("feature".padEnd(24), "master mean".padStart(12), "untiered".padStart(10), "AUC(m>u)".padStart(9));
const w = {};
for (const f of feats) { const m = M.map((r) => r[f]).filter((v) => v != null), u = U.map((r) => r[f]).filter((v) => v != null); const a = auc(m, u); w[f] = a; console.log(f.padEnd(24), mean(m).toFixed(3).padStart(12), mean(u).toFixed(3).padStart(10), a.toFixed(3).padStart(9)); }
// combined score: z-scores (over everyone) weighted by (AUC - 0.5), sign from direction
const stats = Object.fromEntries(feats.map((f) => { const all = rows.map((r) => r[f]).filter((v) => v != null); return [f, { m: mean(all), s: sd(all) }]; }));
const score = (r) => feats.reduce((acc, f) => r[f] == null ? acc : acc + (w[f] - 0.5) * ((r[f] - stats[f].m) / stats[f].s), 0);
for (const r of rows) r.score = score(r);
const ms = M.map((r) => r.score).sort((a, b) => a - b), thr = ms[Math.floor(ms.length * q)];
const sel = U.filter((r) => r.score >= thr);
console.log(`\ncombined-score AUC master>untiered: ${auc(M.map((r) => r.score), U.map((r) => r.score)).toFixed(3)}; threshold = master q${q} = ${thr.toFixed(3)}`);
console.log(`untiered Fox games scoring like masters: ${sel.length} of ${U.length} (${(100 * sel.length / U.length).toFixed(1)} %) in ${new Set(sel.map((r) => r.file)).size} files; masters above their own q${q}: ${M.filter((r) => r.score >= thr).length}`);
const out = { threshold: thr, quantile: q, weights: w, master_files: [...new Set(M.map((r) => r.file))], selected_untiered_files: [...new Set(sel.map((r) => r.file))], selected_rows: sel.map((r) => ({ file: r.file, port: r.port, score: +r.score.toFixed(3) })) };
fs.writeFileSync(path.join(path.dirname(file), "selection.json"), JSON.stringify(out));
console.log(`-> ${path.join(path.dirname(file), "selection.json")}`);
