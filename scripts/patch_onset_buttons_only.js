// Adds `--onset-buttons-only` (2026-10-09 15:20): keep the onset weight on
// the jump / B press edges, drop its stick-enters-up term. Queue 39 applies
// this in its beam-free window (lib edits are not made while a queue runs);
// idempotent, exact-string, fails loudly if a target line moved.
//   node scripts/patch_onset_buttons_only.js
const fs = require("fs");
function patch(file, pairs) {
  let s = fs.readFileSync(file, "utf8");
  for (const [from, to] of pairs) {
    if (s.includes(to)) continue;
    if (!s.includes(from)) { console.error(`PATCH_FAILED ${file}: missing target`); process.exit(1); }
    s = s.replace(from, to);
  }
  fs.writeFileSync(file, s);
}
patch("lib/exphil/training/config/parser.ex", [[
  `    {"--onset-weight", :onset_weight, :float},\n`,
  `    {"--onset-weight", :onset_weight, :float},\n    {"--onset-buttons-only", :onset_buttons_only, :flag},\n`]]);
patch("lib/exphil/training/config.ex", [[
  `      onset_weight: nil,\n`,
  `      onset_weight: nil,\n      # --onset-buttons-only (2026-10-09): the onset weight on the jump / B\n      # press edges only — its stick-enters-up term (10-08 08:00) taught\n      # "flick up" (y >= 0.75, blind to x) and bent the Firefox fire angle\n      # (recovery_firefox_angle.js: 13° without an onset weight, 30–51° with).\n      onset_buttons_only: false,\n`]]);
patch("lib/exphil/training/pipeline.ex", [[
  `            onset_weight: ropts[:onset_weight]\n`,
  `            onset_weight: ropts[:onset_weight],\n            onset_buttons_only: ropts[:onset_buttons_only] || false\n`]]);
patch("lib/exphil/training/silent_fall_weighting.ex", [
  [`    on_w = Keyword.get(opts, :onset_weight)\n`,
   `    on_w = Keyword.get(opts, :onset_weight)\n    buttons_only = Keyword.get(opts, :onset_buttons_only, false)\n`],
  [`            if off? and on_w != nil and prev_c != nil and falling?(frame) and onset?(prev_c, c, jumps_left(frame)),\n`,
   `            if off? and on_w != nil and prev_c != nil and falling?(frame) and\n                 onset?(prev_c, c, if(buttons_only, do: nil, else: jumps_left(frame))),\n`],
  [`      the aim is its own onset (INPUT_COHERENCE "10-08 08:00").\n`,
   `      the aim is its own onset (INPUT_COHERENCE "10-08 08:00"). Callers pass\n      \`jumps_left\` nil to leave this term out (\`--onset-buttons-only\`, 10-09:\n      the term is blind to x and bent the Firefox fire angle).\n`]]);
patch("docs/guides/TRAINING.md", [[
  "| `--onset-weight X` | nil |",
  "| `--onset-buttons-only` | false | With `--onset-weight`: weight only the jump / B press edges, not the stick entering up once the jump is spent (10-09: that term bent the Firefox fire angle, `recovery_firefox_angle.js`). |\n| `--onset-weight X` | nil |"]]);
console.log("PATCH_OK onset_buttons_only");
