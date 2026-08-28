# G4 — linear probes on the trunk (40 files, 46,348 train / 9,542 eval rows)

Linear probe balanced accuracy on fox_gen_v1 ep10's GRU trunk hidden state,
vs the raw current-frame embedding (input floor) and shuffled-label floor.
Higher = the feature is linearly decodable from that representation.

| feature | trunk | input | shuffle | majority | verdict |
|---|---|---|---|---|---|
| own_hitstun | 0.924 | 0.999 | 0.477 | 0.500 | **PRESERVED (rich)** |
| opp_hitstun | 0.912 | 1.000 | 0.473 | 0.500 | **PRESERVED (rich)** |
| own_offstage | 0.933 | 0.912 | 0.450 | 0.500 | **PRESERVED (rich)** |
| opp_offstage | 0.827 | 0.926 | 0.488 | 0.500 | **PRESERVED (rich)** |
| opp_percent | 0.524 | 0.923 | 0.243 | 0.250 | DISCARDED (~half lost) |
| stage_identity | 0.489 | 1.000 | 0.125 | 0.333 | DISCARDED (~half lost) |
| own_percent | 0.459 | 0.806 | 0.237 | 0.250 | DISCARDED |
| opp_character | 0.332 | 0.460 | 0.211 | 0.333 | **DEAD (at chance)** |

## Findings (the RL-readiness verdict)

**Fertile — the trunk richly encodes the "punish / edgeguard" signals:**
hitstun (own+opp) and offstage (own+opp) are cleanly decodable (0.83–0.93).
A value model (D2) can sharpen these cheaply.

**Gaps — the trunk discards ~half of:**
- **percent** (own 0.459, opp 0.524 vs input ~0.8–0.9) — the kill-confirm
  signal. A value model reading "is the opponent at kill %?" from the trunk
  will be half-blind.
- **stage identity** (0.489 vs 1.000) — recovery-routing. Note: linear-only.
  G7 showed the *output* does respond to stage (0.40×), so stage must flow
  non-linearly; a *linear* value-model head cannot read it.

**Dead — opponent character (0.332 ≈ majority 0.333).** This disambiguates
G7's dead-channel finding: the raw embedding DOES carry character (input
0.460, weak but above majority), and the corpus DOES have variation (Fox vs
Mewtwo/Ganon/Marth/Peach/Pikachu/ICs/Yoshi — never a ditto). The signal dies
**in the trunk** (0.332, at chance), not at the embedding or in the data.
Making the bot matchup-aware is a representation fix (character must bypass
or survive the trunk), not a data or decode fix.

## Caveats

- Linear probes find only LINEAR decodability; "discarded" ≠ "unused" (stage
  is used non-linearly per G7).
- The 12-file run's stage=0.009 was a small-eval fluke (single eval stage);
  40-file numbers are authoritative.
- The input floor is only reliable with enough rows (12 files → chance due to
  ill-conditioned raw-ID scale; 40 files → clean).
