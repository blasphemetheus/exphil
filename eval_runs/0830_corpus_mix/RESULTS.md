# E1 — corpus mix (by file): fox_gen_v1 imitated port 1, whoever was on it

**2026-08-30.** Raw `.slp` game-start header scan of all 7,911 files in
`replays/erickfm_ranked/FOX/extracted` (0 parse errors; beam-free reader,
external character IDs from the Game Start block).

## The finding

`fox_gen_v1`'s run log has **no `Character:` line** — `--train-character`
was never set — and `--train-character` only FILTERS FILES anyway
(`pipeline.ex:219`: keeps files where ANY player is the character; it
never selects the port). With no `port_map` and no `dual_port`, the
streaming loader parsed **`player_port: 1` for every file**
(`streaming.ex:96`). So v1's imitation target was "whoever sat on port 1":

| fox position | files | share |
|---|---:|---:|
| port 1 only | 3,494 | 44.2% |
| port 2 only | 3,429 | 43.3% |
| ditto (both) | 988 | 12.5% |

**Port-1 character (= what v1 actually imitated):**

| character | files | share |
|---|---:|---:|
| fox | 4,482 | **56.7%** |
| falco | 709 | 9.0% |
| marth | 451 | 5.7% |
| puff | 361 | 4.6% |
| falcon | 319 | 4.0% |
| sheik | 318 | 4.0% |
| peach | 194 | 2.5% |
| ics | 186 | 2.4% |
| yoshi | 177 | 2.2% |
| samus | 168 | 2.1% |
| pikachu | 144 | 1.8% |
| ganon | 133 | 1.7% |
| (13 more) | ~570 | ~7% |

~**43% of v1's demonstrations are non-Fox** (the fox's opponent). The
character embedding conditions the policy, so this is "generalist by
accident" rather than corruption — but it dilutes fox-specific behaviour
mass, and it corrects the EVAL_DIRECTIONS bank note ("port 1 = the
imitated Fox"), which is wrong for this directory.

## Implications

1. **Every eval that used `--expert-port 1` on this corpus** (A1/B2
   situation_hist, entropy, joint-head audit ran on "expert" = port 1)
   mixed ~43% non-fox play into the EXPERT baseline. The fox-only
   re-reads should use per-file character detection (the 8a capture's
   `resolve_port` pattern). Magnitude check needed per instrument —
   the joint-head audit's TC finding is about within-frame controller
   correlation and likely robust; the situation histograms' expert
   denominators are character-mixed.
2. **The 8a head fit** deliberately deviates: per-file fox detection,
   dittos skipped (ambiguous single-port resolution) — a FOX-ONLY,
   correct-demonstrator corpus for both ARhead and INDhead arms
   (symmetric, so the pre-registered comparison stands).
3. **A v1.1/v2 recipe question**: per-file character-aware port selection
   (and optionally both ditto ports, keeping both halves of one game in
   the same train/val split) would give ~5,470 fox demonstrations from
   this dir vs the accidental 4,482-with-3,429-wrong-port mix.

Scanner: inline python in session 2026-08-30 (header offsets: raw element
at 11+4, 0x35 size byte includes itself, Game Start chars at
gs+0x65+0x24*i, types at gs+0x66+0x24*i). Promote to a script if E1 needs
the by-frame version.
