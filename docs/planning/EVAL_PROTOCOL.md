
## Regime calibration: pool (sync) vs async runners (2026-08-09, n=8 per arm)

fox_il_v2, temperature 0.3, 120s vs level-1 CPU
(eval_runs/0809_regime_{async2,pool_fast,pool_rt}):

| arm | survive/game | lost (SD/KO) | dmg | taken | actions |
|---|---|---|---|---|---|
| async realtime | 121s (no game ends) | 1.9 (0.8/1.1) | 233 | 0.5 | 67 |
| pool sync, unthrottled | 23.7s | 3.4 (3.3/0.1) | 0 | 0 | 20 |
| pool sync, realtime | 24.2s | 3.5 (3.3/0.2) | 0 | 0 | 20 |

RULES:
1. **Emulation speed is NOT the regime variable** — the two pool arms are
   identical. The sync frame loop's input path is (1-frame-delay
   semantics + per-step sampling cadence). For a delay-untrained policy
   the gap is enormous (SD-collapse vs competent play).
2. **Pool numbers compare only pool-vs-pool** (iteration-speed tool:
   n=8 in ~13 min, live-scored). **Async realtime remains the protocol
   rung for promote-grade behavior claims** until pool is calibrated
   against the deploy rung (human session) — the g10b lesson applied.
3. Pool runs auto-restart games after a 4-stock (~5 games/120s run);
   LiveScorer rows are per-RUN aggregates, replay scoring is per-GAME —
   both agree on stocks/action-diversity (cross-validated 2026-08-09).
4. inputs/min disagrees between live (per-decision) and replay
   (per-recorded-frame) scoring under sampling decode — treat the
   column as decode-mode diagnostics only, never compare across modes.
