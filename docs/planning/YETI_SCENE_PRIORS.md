# Yeti (St. Louis) scene priors for identity resolution

Bradley, 2026-09-17 (late), verbatim intent: the Yeti corpus is a LOCAL
scene of roughly (probably under) a hundred players, so identity is a
closed-set problem there. Before any fingerprint is consulted,
(character, costume, corpus = Yeti) gives an informed prior over who it
is — "loose confidence". These are Bradley's priors, to be treated as
EVIDENCE with a weight set by calibration (STYLE_IDENTITY.md, "Tags are
EVIDENCE"), never as labels. Scope: within-corpus, handles only.

| Character | Costume | Likely players (Bradley) |
| --- | --- | --- |
| Fox | red | Tim Tempur, Spikenard, Messi |
| Fox | green | OG Swaglord |
| Fox | blue | Trash Machine ("or something") |
| Zelda | — | Greg, Snow Craggy, Mikey (also plays Zelda) |
| Sheik | — | Anubis, Dr. Copter, G-Bats; **playtime as Sheik vs Zelda within the game** (who they start as) separates the Zelda players from the Sheik players |
| Captain Falcon | — | 3Hunna, oh y'all, Spikenard |
| Game & Watch | — | Spikenard, or a Marth player who also plays G&W sometimes |
| Marth | — | Sandsy Boy, Pac-Man |
| any rarely-played character | — | probably Schlag ([INFP], Michael Schlag) playing random |

("And I could go on" — extend this table with Bradley as the oracle; the
reconciliation report should ASK for the next rows, per character with
unresolved clusters.)

## How it enters the model

- Yeti resolution = Bayesian closed-set matching:
  `P(entity | game) ∝ P(entity | character, costume, Yeti) × L(fingerprint | entity centroid)`,
  centroids from tagged games (erickfm + Yeti tags), likelihood scale
  from S2 calibration. Open-set clustering (S3) stays for the erickfm
  hashed games, where no scene prior exists.
- Costume is NOT yet in `Peppi.PlayerMeta` (only port/character/tag/
  netplay name+code). It is in the `.slp` game-start block per port
  (`costume` index); add it to the NIF metadata before S3/S4 on Yeti.
- Sheik/Zelda: the metadata character is the START character; the
  transform playtime split needs the parsed frames (action states /
  character per frame), which the fingerprint pass already reads.
- Session context: consecutive games on one Yeti station/night are
  strong same-player evidence for the winner staying on (bracket
  format); `started_at` from the metadata tail carries it (erickfm
  hashed files have no tail; Yeti files do).
