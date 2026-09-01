# Live look — fox_gen_v1.2_ARrefit (Bradley, 2026-08-31 evening)

Setup: FD only, local d0, T=0.5/0.5, the §6 command; replays in
`2026-08-Mainline/`. Multiple games vs Bradley.

## Verdict: real improvement on survival; positioning blindness is the wall

- **Kills itself a lot less** — clear improvement over prior looks.
  Still occasional SDs, including sometimes the offstage B move, but
  "a lot better than it was."
- **Airdodge panic: visibly reduced** — matches F1's scored halving
  (37.5 → 18.4% panic route). Bradley: "I did see a lot less of" it.
  Couldn't confirm/deny the side-B class specifically.
- **THE NEW HEADLINE COMPLAINT: no positional play.** It does not
  adjust to where Bradley is, does not try to WIN positioning. Plays
  identically at left ledge / right ledge / near-ledge onstage /
  center stage (FD). No stage-control behavior.
- **Attack selection is spam, not targeting**: quickly commits an
  option regardless of opponent position — usually shield grab,
  sometimes multi-jab. Sometimes hits, but it isn't aiming. Never
  dash-dances into jump-cancel upsmash (the obvious Fox punish it
  "probably knows" from the corpus but never selects).

## Reads

- Consistent with F2 (3.2× committal-option volume) and the standing
  no-dash-dance trunk complaint — the whole cluster looks like ONE
  property: the policy doesn't condition its option selection on
  opponent/self stage position.
- Bradley's directive: make "does it value opponent position?" an
  EMPIRICAL interp question (opponent-dependence probe class, W2
  machinery exists: scripts/probe_opponent_dependence.exs), plus
  whatever adjacent questions fall out (position-differentiation, not
  just consultation).

## Gate implications

- Live look done → task 1 fully closed. ARrefit confirmed better than
  v1.1 live on the survival axis.
- `--head` default flip: this look is favorable-but-not-a-crown;
  decision still with Bradley/pre-registered gate.
