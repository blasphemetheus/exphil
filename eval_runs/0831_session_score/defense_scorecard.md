# Defense scorecard (F1)

Sets: expert: 166 files · AR_human: 13 files · IND_human: 6 files · B1_human: 7 files. Horizon 240 f.
Episode = contiguous hitstun/tumble run; offstage-class if offstage during the run
or within 30 f after. "Toward stage" = main stick held (|x-0.5| > 0.2) with
sign opposing the subject's x position (stage center 0). Cells with under 20 held
frames / 5 episodes report "–" — do not read them.

| defense | expert (off-eps n=2844) | AR_human (off-eps n=110) | IND_human (off-eps n=40) | B1_human (off-eps n=39) |
|---|---:|---:|---:|---:|
| offstage hitstun episodes / min | 5.76 | 5.29 | 4.81 | 5.47 |
| DI toward stage % (in hitstun) | 83.1 | 67.1 | 57.7 | 63.7 |
| drift toward stage % (post-hitstun) | 91.2 | 87.9 | 83.9 | 87.6 |
| airdodge offstage in episode % | 2.4 | 32.7 | 17.5 | 33.3 |
| died within 90f of that airdodge % | 17.9 (67) | 52.8 (36) | 14.3 (7) | 61.5 (13) |
| episode died % | 23.4 | 50.0 | 47.5 | 59.0 |
| GLOBAL offstage drift toward stage % | 88.8 | 84.1 | 72.3 | 74.1 |
| onstage-hitstun DI toward center % | 63.2 | 56.5 | 45.0 | 45.3 |

Read: the expert column is the DI/drift denominator Bradley described ("hold a
direction and get to ledge"). A bot matching expert drift but dying more is
route-selection (A2's lane); a bot NOT holding toward stage is missing the
survival input itself — a corpus/curation question (G1 target #1), never a
decode mask (standing rule).
