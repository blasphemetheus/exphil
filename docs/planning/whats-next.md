# What's next — pointer doc

**This file used to hold a 2026-01-29 snapshot and went six months stale.**
It is now a pointer, because handoffs are dated files and the newest one is
always the truth.

## Read the newest handoff

    ls -t docs/planning/HANDOFF_*.md | head -1

As of 2026-07-26 that is
[HANDOFF_2026-07-26.md](HANDOFF_2026-07-26.md) — laptop session: task #8
phases 1 and 2, and the root-cause finding that the multishine policy's live
failure is **exposure bias**, not the state-stream shift (GOTCHAS #81 moves
2/288 embedding dims and has no measurable live effect).

Immediately prior: [HANDOFF_2026-07-24e.md](HANDOFF_2026-07-24e.md) (vacation
edition) for the r17a resume point and the multishine teacher narrative.

## Correction to the version this replaced

The old snapshot listed the **Axon orthogonal-init fix** as:

| Step | Old status |
|------|------------|
| GitHub issue | ❌ Not started |
| Fork Axon | ❌ Not started |
| Implement fix | ❌ Not started |

That is wrong and has been for a while. The fix **exists**, on a local-only
branch of the axon fork:

    ~/git/melee/axon   branch: fix/orthogonal-wide-matrix
    commit 504866cc "fix: Handle wide matrices in orthogonal initializer"

It has never been pushed, so there is still no issue and no PR — but the
implementation step is done, and anyone picking this up should start from that
branch rather than from scratch. The design doc it implements is
[../internals/AXON_ORTHOGONAL_INIT_FIX.md](../internals/AXON_ORTHOGONAL_INIT_FIX.md).

Also unpushed and easy to lose: `~/git/melee/nx` carries a commit adding
`whats-next.md` notes for nx PRs #1697/#1691/#1683.

## Where the live task list lives

- [../../TODO.md](../../TODO.md) — infrastructure, datasets, multishine /
  state-stream follow-ups
- [GOALS.md](GOALS.md) — big-picture roadmap
- [../reference/GOTCHAS.md](../reference/GOTCHAS.md) — the accumulated law;
  #81 in particular carries the parsed↔live mapping and the exposure-bias
  root cause
