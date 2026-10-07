// Death shape on self-destruct trips (`sd`, decided + died): input changes per
// trip, trailing identical-input frames before death, held input at death,
// jumps left at death, and the stick direction on every B ONSET (the special's
// identity: up = Firefox, side = Illusion, neutral = laser, down = shine).
// Expert from the traced FD reference; arms from eval_runs/1001_queue/<arm>.
//   node scripts/recovery_death_shape.js evt2ctx_ck8_off3 evt2ctx_ck8_off3_dur8e
// 2026-10-06 read: expert deep B onsets are stick-up 81 %, every arm 17–35 %;
// expert dies with a jump left 7 %, arms 26–30 %. The duration arms did not
// move either number.
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
}
const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
shape(Array.isArray(ex) ? ex : ex.episodes, "expert (FD reference)");
for (const arm of process.argv.slice(2)) {
  const m = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${arm}/recovery_means.json`));
  shape(m.episodes, arm);
}
