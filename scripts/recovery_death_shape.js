// Death shape on self-destruct trips (`sd`, decided + died): input changes per
// trip, trailing identical-input frames before death, held input at death,
// jumps left at death, and the stick direction on every B ONSET (the special's
// identity: up = Firefox, side = Illusion, neutral = laser, down = shine).
// Expert from the traced FD reference; arms from eval_runs/1001_queue/<arm>.
//   node scripts/recovery_death_shape.js evt2ctx_ck8_off3 evt2ctx_ck8_off3_dur8e
// 2026-10-06 read: expert deep B onsets are stick-up 81 %, every arm 17–35 %;
// expert dies with a jump left 7 %, arms 26–30 %. The duration arms did not
// move either number.
// 2026-10-07 decomposition (all died episodes, incl. carried-off): stick
// toward/away the stage offstage, never used a special, jump used, ended in
// an aerial vs passive Fall, Firefox aim (toward/away/up) and start depth.
// pd15 read: the copy-shortcut family (silent holds, direction, Firefox aim
// and depth) moves; "never commits to a special" (55 % vs expert 30 %) and
// the double jump (72 % vs 91 %, identical across every arm) do not.
const fs = require("fs");
const dir = (sx, sy) => {
  const dx = sx - 0.5, dy = sy - 0.5;
  if (Math.abs(dx) < 0.25 && Math.abs(dy) < 0.25) return "neutral";
  if (Math.abs(dy) >= Math.abs(dx)) return dy > 0 ? "up" : "down";
  return "side";
};
const same = (a, b) => a[4] === b[4] && a[5] === b[5] && a[6] === b[6] && a[7] === b[7];
function shape(eps, label) {
  const sd = eps.filter(e => e.sd && e.trace && e.trace.length > 2);
  const died = sd.filter(e => e.outcome === "died");
  const q = a => { a.sort((x, y) => x - y); return [0.25, 0.5, 0.75].map(p => a[Math.floor(p * (a.length - 1))]); };
  const changes = [], tailSilent = [], bdir = {}, jl = {}, lastInputB = 0, lastUp = 0;
  let endsSilent = 0, deadWithJump = 0, bUpDeep = 0, bAnyDeep = 0;
  for (const e of died) {
    const t = e.trace;
    let c = 0;
    for (let i = 1; i < t.length; i++) if (!same(t[i], t[i - 1])) c++;
    changes.push(c);
    // trailing run of identical input (x3 frames each)
    let k = 0;
    for (let i = t.length - 1; i > 0 && same(t[i], t[i - 1]); i--) k++;
    tailSilent.push(k * 3);
    const last = t[t.length - 1];
    const neutral = Math.abs(last[4] - 0.5) < 0.25 && Math.abs(last[5] - 0.5) < 0.25 && !last[6] && !last[7];
    if (neutral) endsSilent++;
    if (last[8] > 0) deadWithJump++;
    for (let i = 1; i < t.length; i++) if (t[i][6] && !t[i - 1][6]) {
      const d = dir(t[i][4], t[i][5]); bdir[d] = (bdir[d] || 0) + 1;
      if (t[i][2] < -40) { bAnyDeep++; if (d === "up") bUpDeep++; }
    }
  }
  console.log(`== ${label}: decided ${sd.length}, died ${died.length} (${(died.length / sd.length).toFixed(2)})`);
  console.log(`  input changes per died trip q25/50/75: ${q(changes)}  trailing identical-input frames before death q: ${q(tailSilent)}`);
  console.log(`  died holding neutral+no buttons: ${endsSilent}/${died.length}  died with a jump left: ${deadWithJump}/${died.length}`);
  console.log(`  B onsets in died trips by stick dir: ${JSON.stringify(bdir)}  below y-40: ${bAnyDeep}, stick up ${bUpDeep}`);
  decompose(eps);
}
function decompose(eps) {
  const died = eps.filter(e => e.outcome === "died" && e.trace && e.trace.length > 1);
  let away = 0, toward = 0, tot = 0, neverSpecial = 0, usedJump = 0, fire = 0, fireAway = 0, fireToward = 0, fireUp = 0, fireDeep = 0, fireHigh = 0, endAttack = 0, endFall = 0;
  for (const e of died) {
    const t = e.trace, side = Math.sign(t[0][1] || 1);
    for (const r of t) if (Math.abs(r[1]) > 85) { tot++; const dx = r[4] - 0.5; if (dx * side > 0.25) away++; else if (dx * side < -0.25) toward++; }
    const la = t[t.length - 1][0];
    if ([65, 66, 67, 68, 69].includes(la)) endAttack++;
    if ([29, 32].includes(la)) endFall++;
    const sp = t.findIndex(r => r[0] >= 341);
    if (sp < 0) neverSpecial++;
    if (la === 35 && sp >= 0 && !(t[sp][0] >= 365 && t[sp][0] <= 368)) {
      fire++; if (t[sp][2] < -40) fireDeep++; if (t[sp][2] > -10) fireHigh++;
      let best = null, bm = 0;
      for (const r of t.slice(sp, sp + 7)) { const dx = r[4] - 0.5, dy = r[5] - 0.5, m = Math.hypot(dx, dy); if (m > bm) { bm = m; best = [dx, dy]; } }
      if (best) { if (best[0] * side > 0.2) fireAway++; else if (best[0] * side < -0.2) fireToward++; else fireUp++; }
    }
    if (t.some(r => r[8] === 0)) usedJump++;
  }
  const pct = (a, b) => b ? (100 * a / b).toFixed(0) + "%" : "-";
  console.log(`  all died ${died.length}: stick toward ${pct(toward, tot)} away ${pct(away, tot)} | never special ${pct(neverSpecial, died.length)} | jump used ${pct(usedJump, died.length)} | ended attacking ${pct(endAttack, died.length)} passive Fall ${pct(endFall, died.length)}`);
  console.log(`  Firefox deaths ${fire}: aimed toward ${pct(fireToward, fire)} away ${pct(fireAway, fire)} up ${pct(fireUp, fire)} | started deep ${pct(fireDeep, fire)} high ${pct(fireHigh, fire)}`);
}
const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
shape(Array.isArray(ex) ? ex : ex.episodes, "expert (FD reference)");
for (const arm of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${arm}/recovery_means.json`));
  shape(m.episodes, arm);
}
