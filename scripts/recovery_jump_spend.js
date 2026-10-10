// Where the double jump went before a carried-off trip (2026-10-10). Every
// carried side-B trip (Illusion fired on stage, recovery_means first_at == 0)
// starts with no jump, on every arm; the expert's carried trips are 5 / 1788
// episodes. This reads the trip's `pre_trace` (RecoveryMeans, 10-10: every
// 3rd of the 90 frames before the decision frame, oldest first; row =
// [action, x, y, speed_y, stick_x, stick_y, b, jump, jumps_left]) and finds
// the row where jumps_left dropped — how long before the trip, where (x past
// the edge, y), from what (the action on the row before: grounded dash /
// stand / run, or airborne), with what stick — or "earlier" when the jump was
// already gone 90 frames out. Also the approach's last grounded row: how far
// inside the edge the player left the ground. Bot vs expert.
//
//   node scripts/recovery_jump_spend.js ARM [ARM ...]
"use strict";
const fs = require("fs");
const EDGE = 85.57;

const GROUND = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 39, 40, 41, 42, 43]); // wait, walk, dash, run, turn, crouch
const JUMPSQUAT = 24;
const isAir = (a) => !GROUND.has(a) && a !== JUMPSQUAT && a > 13;

function spend(e) {
  const rows = e.pre_trace || [];
  if (rows.length === 0) return null;
  const sign = e.x >= 0 ? 1 : -1;
  let spent = null;
  for (let i = 1; i < rows.length; i++) {
    if (rows[i][8] < rows[i - 1][8]) spent = i; // last drop wins (the jump that mattered)
  }
  const n = rows.length;
  const lastGround = (() => {
    for (let i = n - 1; i >= 0; i--) if (GROUND.has(rows[i][0]) || rows[i][0] === JUMPSQUAT) return rows[i];
    return null;
  })();
  const out = {
    sign, n,
    jumpAtStart: rows[0][8],
    spent: spent == null ? null : {
      framesBefore: (n - spent) * 3,
      dist: Math.abs(rows[spent][1]) - EDGE,
      y: rows[spent][2],
      from: rows[spent - 1][0],
      fromAir: isAir(rows[spent - 1][0]),
      stickOut: (rows[spent][4] - 0.5) * 2 * sign, // >0 = toward the edge side
      stickUp: (rows[spent][5] - 0.5) * 2,
    },
    leftGround: lastGround ? { dist: Math.abs(lastGround[1]) - EDGE, action: lastGround[0], framesBefore: (n - rows.indexOf(lastGround)) * 3 } : null,
  };
  return out;
}

const q = (xs, p) => { if (!xs.length) return NaN; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const r1 = (v) => Number.isFinite(v) ? v.toFixed(1) : "-";
const pct = (a, b) => b ? Math.round(100 * a / b) + "%" : "-";

function report(label, eps) {
  const carried = eps.filter((e) => e.first === "side_b" && e.first_at === 0);
  const noJumpNear = eps.filter((e) => e.jumps === 0 && e.dist === "near" && e.first_at !== null);
  for (const [name, set] of [["carried side-B", carried], ["all near trips starting with no jump", noJumpNear]]) {
    const ss = set.map(spend).filter(Boolean);
    const inWin = ss.filter((s) => s.spent);
    const earlier = ss.filter((s) => !s.spent && s.jumpAtStart === 0);
    const hadJump = ss.filter((s) => !s.spent && s.jumpAtStart > 0); // jump in hand 90 f out and never dropped?? (hit trips: jumps reset)
    const fromAir = inWin.filter((s) => s.spent.fromAir);
    console.log(`${label.padEnd(44)} ${name}: n=${set.length}  jump spent inside the 90 f before: ${inWin.length} (${pct(inWin.length, ss.length)})  already gone 90 f out: ${earlier.length}  still in hand at the window start, no drop: ${hadJump.length}`);
    if (inWin.length) {
      console.log(`    spend: frames before trip q25/50/75 ${q(inWin.map((s) => s.spent.framesBefore), .25)}/${q(inWin.map((s) => s.spent.framesBefore), .5)}/${q(inWin.map((s) => s.spent.framesBefore), .75)}  dist past edge q50 ${r1(q(inWin.map((s) => s.spent.dist), .5))} (q25 ${r1(q(inWin.map((s) => s.spent.dist), .25))}, q75 ${r1(q(inWin.map((s) => s.spent.dist), .75))})  y q50 ${r1(q(inWin.map((s) => s.spent.y), .5))}  from the air ${pct(fromAir.length, inWin.length)}  stick toward edge >=0.6 at the spend ${pct(inWin.filter((s) => s.spent.stickOut >= 0.6).length, inWin.length)}  stick up ${pct(inWin.filter((s) => s.spent.stickUp >= 0.6).length, inWin.length)}`);
      const froms = {};
      for (const s of inWin) froms[s.spent.from] = (froms[s.spent.from] || 0) + 1;
      console.log(`    spent from (action on the row before): ${Object.entries(froms).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${k} ${pct(v, inWin.length)}`).join(", ")}`);
    }
    const lg = ss.filter((s) => s.leftGround);
    if (lg.length) {
      console.log(`    left the ground (last grounded row): ${pct(lg.length, ss.length)} of trips within 90 f  dist past edge q50 ${r1(q(lg.map((s) => s.leftGround.dist), .5))} (q25 ${r1(q(lg.map((s) => s.leftGround.dist), .25))})  frames before q50 ${q(lg.map((s) => s.leftGround.framesBefore), .5)}`);
    }
  }
}

const args = process.argv.slice(2);
const ex = JSON.parse(fs.readFileSync("eval_runs/1002_fidelity/expert_recovery_means_fd_episodes.json"));
report("expert (FD reference)", Array.isArray(ex) ? ex : ex.episodes);
for (const a of args) {
  const j = JSON.parse(fs.readFileSync(`eval_runs/1001_queue/${a}/recovery_means.json`));
  report(a, j.episodes);
}
