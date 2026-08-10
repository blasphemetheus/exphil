# Situation Labels — expansive brainstorm (2026-08-10)

Status: BRAINSTORM. Deliberately over-inclusive; winnow before
implementing. Born from the edge-SD naming discussion (edgeguard is a
*situation*, not a death outcome) and the fight-state thread's standing
need for richer pressure labels (CLAUDE.md thread 3).

**What a situation label is:** a per-frame (or per-segment) tag derived
mechanically from game state — no human annotation — that names the
strategic context a frame lives in. Uses:
- **Curation/oversampling**: train arms weighted toward underrepresented
  situations (the g10b human-snippet lesson generalized).
- **Conditional evaluation**: per-situation loss/behavior metrics
  ("val loss *in edgeguard situations*" instead of one global number).
- **DAgger routing**: each scripted expert has a jurisdiction; labels ARE
  the routing function (EdgeTurnaround gets dash-off segments,
  FoxRecovery gets offstage-after-hit, LedgeExpert gets cliff states).
- **Interp probes**: "does the trunk represent situation X?" is only
  askable once X is labeled.
- **Event emission**: `Melee.GameEvents`-style live events for
  LiveScorer columns.

Detection difficulty scale: **E** (easy: direct read of action/position
state), **M** (medium: needs a frame window or both players), **H**
(hard: needs intent inference, thresholds, or opponent modeling).

---

## 1. Stage geometry / position

| Label | Definition | Detect |
|---|---|---|
| `onstage_center` | Both feet on main platform, \|x\| < ~40% of edge | E |
| `onstage_corner` | Grounded within N units of own-side edge (the danger margin from the edge-SD arm) | E |
| `offstage` | \|x\| > stage edge or below stage level, not on a platform | E |
| `above_stage` | Airborne over the main platform, y > threshold (juggle zone) | E |
| `on_platform` | Standing on a side/top platform | E |
| `below_ledge` | Offstage AND y below ledge height (recovery is now committal) | E |
| `ledge_hang` | CliffCatch/CliffWait action states (252/253) | E |
| `ledge_occupied_by_opp` | Opponent in cliff states while we're onstage (their invincibility window) | E |
| `platform_underneath` | A platform exists between player and the ground (shark/tech situations) | M (stage geometry table) |
| `near_blastzone` | Within N units of any blastzone (kill-percent context modifier) | E (per-stage blastzone table) |

## 2. Neutral game

| Label | Definition | Detect |
|---|---|---|
| `neutral` | Neither player in hitstun/hitlag/knockdown/ledge/respawn — the default umbrella state | E |
| `dashdance_neutral` | In `neutral` + own action oscillating DASHING with direction flips within a window | M |
| `spacing_contest` | Both grounded, closing distance, inside ~1.5 max-attack-range | M (needs range table) |
| `approach` | Closing distance toward opponent above a speed threshold | M |
| `retreat` | Opening distance away from opponent above a speed threshold | M |
| `zoning` | Repeated projectile use at range (lasers, needles, shadow balls) | M |
| `projectile_incoming` | An opponent projectile exists with a trajectory intersecting us within N frames | H (projectile entity data) |
| `whiff_window` | Opponent in attack lag/endlag within our punish range (whiff-punish opportunity) | M (endlag table) |
| `cross_up` | Players swapped sides within the last N frames at close range | M |
| `stalemate` | `neutral` sustained > N seconds with no hits landed (campy/timeout texture) | M |

## 3. Advantage (we hit them)

| Label | Definition | Detect |
|---|---|---|
| `combo_active` | Opponent in hitstun and we are actionable-or-attacking; ends when they escape | M (hitstun tracking) |
| `juggle` | Opponent airborne above us post-hit, we're under them | E |
| `tech_chase` | Opponent in knockdown/missed-tech/tech-roll family; we're grounded nearby | E |
| `tech_read_window` | The 20-frame window where their tech option is committed but punish is still open | M |
| `edgeguard` | Opponent offstage post-hit; we're onstage/at-ledge acting on their recovery. THE label that started this doc | M |
| `ledge_trap` | Opponent on ledge, we're onstage covering options | E |
| `kill_confirm_range` | Opponent percent × our kill moves ⇒ a kill is available from current state | H (knockback calc or lookup) |
| `conversion_open` | First hit landed from `neutral` within last N frames (the opener→conversion boundary — the 231%-dealt/0-kills failure lives here) | M |
| `shield_pressure_ours` | Opponent shielding, we're attacking/spacing on their shield | E |
| `shield_break_confirm` | Opponent in break family (205..211) — free hit | E |
| `pummel_throw_decision` | We have a grab (CatchWait family) — throw selection situation | E |

## 4. Disadvantage (they hit us)

| Label | Definition | Detect |
|---|---|---|
| `in_hitstun` | Damage-air/damage-ground action families | E |
| `tumble` | Action 38 — tech vs no-tech decision pending | E |
| `sdi_window` | In hitlag from a multihit (SDI decision frames) | M |
| `being_juggled` | Airborne above opponent post-hit (mirror of `juggle`) | E |
| `being_tech_chased` | We're in knockdown family with opponent nearby | E |
| `being_edgeguarded` | We're offstage post-hit, opponent actively positioned to contest (mirror of `edgeguard`); the FoxRecoveryExpert jurisdiction | M |
| `recovery_low` / `recovery_high` | Offstage recovering below/above ledge height — distinct option trees | E |
| `resource_exhausted` | Offstage with 0 jumps and special not yet usable (must-not-get-hit frames) | E |
| `cornered` | Grounded with back to edge, opponent between us and center | E |
| `shield_pressure_theirs` | We're shielding under active attack (OOS decision situation) | E |
| `shield_low` | Shield health below threshold (poke/break risk modifier) | E |
| `crouch_cancel_viable` | Grounded, low percent, opponent approaching with a CC-able move | H (move/percent table) |
| `escape_di_decision` | In hitstun of a known combo-starter (DI mixup frames) | M |

## 5. Ledge / edge micro-situations

| Label | Definition | Detect |
|---|---|---|
| `edge_danger` | Grounded DASHING/RUNNING toward edge inside the danger margin — the EdgeTurnaroundExpert jurisdiction (this session's arm) | E |
| `ledge_option_pending` | CliffWait — getup/attack/roll/jump/drop decision (LedgeExpert jurisdiction) | E |
| `ledge_regrab_cycle` | Repeated cliff-catch within N frames (stalling / invincibility refresh) | M |
| `ledgedash_window` | Frames after ledge-drop where jump+airdodge-in is live | M |
| `both_offstage` | Both players offstage simultaneously (wild west; own recovery priority shifts) | E |

## 6. Timing / game-flow / meta

| Label | Definition | Detect |
|---|---|---|
| `respawn_invincible` | On halo / post-respawn invincibility (either player) | E |
| `post_kill_neutral` | First N seconds after taking a stock (reset discipline; opponent on halo) | E |
| `last_stock_ours` / `last_stock_theirs` / `last_stock_both` | Stock-count context — risk appetite should differ | E |
| `percent_lead` / `percent_deficit` | Same-stock percent differential beyond a threshold | E |
| `kill_percent_us` / `kill_percent_them` | Either player past typical kill threshold for the matchup | M |
| `timeout_relevant` | Game clock under N seconds with a close stock/percent state | E |
| `opponent_habit_window` | Opponent has repeated the same option K times in similar states (adaptation trigger) | H (online opponent model) |

## 7. Character/tech-specific execution windows

(These label OUR execution situations — useful for drill curation and
per-technique loss metrics rather than strategy.)

| Label | Definition | Detect |
|---|---|---|
| `jc_window` | JumpSquat frames (jump-cancel grab/shine window) | E |
| `lcancel_window` | Aerial active near ground contact (L-cancel timing) | M |
| `wavedash_window` | Airborne within airdodge-to-ground range in jumpsquat exit | M |
| `shine_cancellable` | In shine with JC available (the multishine chain state) | E |
| `multishine_chain` | The existing ShineChain tracking, reframed as a situation | E (exists) |
| `ic_desync` | Nana/Popo action states diverged (IC tech situation, the compact-Nana embedding's whole reason) | E |

## 8. Data-hygiene / regime labels

(Not gameplay situations, but the same machinery serves them.)

| Label | Definition | Detect |
|---|---|---|
| `warmup_frames` | First N frames post-game-start (burn-in; probes already skip these) | E |
| `handwarmer` | Detected friendly/non-serious segment in human replays (both idle/taunting) | H |
| `sandbagging` | Sustained one-sided non-engagement in human replays (corpus pollution) | H |
| `lag_spike` | Netplay stale-frame runs (qtrace already measures; as a label it EXCLUDES frames from training) | E (exists in qtrace) |

---

## Implementation notes (for the winnowing pass)

- **Priority heuristic**: value = (how often the bot is currently bad in
  that situation) × (detectability). The known failure list today:
  `edge_danger` (SD loop — in flight), `conversion_open` (231%/0-kills),
  `being_edgeguarded` (E2 recovery post-mortem), `ledge_option_pending`
  (crouch→ledge basin). Those four already have evidence; start there.
- **Emission layer**: extend `Melee.GameEvents` (segment start/end
  events) for live use + a batch labeler over parsed replays for corpus
  curation. Same rule tables, two frontends — keep the rules in ONE
  module so live and post-hoc labels can't drift.
- **Storage**: per-frame bitmask alongside MmapCorpus labels (a u32/u64
  per frame covers 32/64 labels) — mmap-friendly, cheap to filter on.
- **Umbrella hierarchy**: `neutral` / advantage / disadvantage are
  mutually exclusive parents; most other labels are children. Enforcing
  the hierarchy at emission time keeps downstream logic simple.
- **Beware intent inference (H rows)**: everything marked H needs
  thresholds tuned on labeled examples or an opponent model. Ship E
  rows first; M rows need small frame-window state machines; H rows are
  research projects.
- **Naming**: outcome labels stay two-way `:sd`/`:ko`
  (trajectory-based, per the 08-10 classifier discussion);
  `:edgeguard` is reserved for the SITUATION of contesting a recovery,
  never a death kind.
