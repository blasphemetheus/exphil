# Recurrent inference at deployment latency

All 96 predeclared trials completed: 384 environment-minutes, four paired
seed-level replicas per mode/context. Raw JSON and `summary.json` retain all
outcomes and paired differences. Source, checkpoint and simulator hashes
matched the pre-run manifest at completion. No checkpoint or inference
default changed.

**Keep resync opt-in.** For these window-trained MinGRU weights, resync-30
improves mean deaths and damage over indefinite carry in all four contexts,
with no decrease in mean kills. It does not uniformly improve on windowed
inference. Mamba's tradeoffs depend on delay, stage and opponent; there is
no general quality case for forcing periodic resync.

## Interpretation

- **MinGRU, FD delay 3:** resync lowers deaths against carry in all four seeds,
  averaging 1.0 fewer deaths/minute. Against windowed it lowers mean deaths
  by 0.125/min, but mean kills also fall by 0.3125/min.
- **MinGRU, FD delay 4:** resync has 0.5 more deaths/min than windowed
  (worse in two seeds, tied in two). Mean kills increase, while mean damage
  is almost unchanged. This fails a simple “resync preserves window quality” claim.
- **MinGRU, Battlefield delay 3:** resync lowers deaths relative to windowed
  in all four seeds, but damage falls in three of four, averaging -6.346/min.
  Against the GRU opponent, resync/carry means improve while the
  resync/windowed comparison remains mixed.
- **Mamba, FD delay 3:** the most encouraging Mamba context. Relative to
  windowed, deaths improve in two seeds and tie in two; damage improves in
  three of four. Means: -0.125 deaths/min and +4.9515 damage/min.
- **Mamba, FD delay 4:** resync is worse than windowed in mean deaths,
  kills and damage. Deaths worsen in two seeds and tie in two.
- **Mamba, Battlefield / GRU opponent:** no uniform advantage. On Battlefield,
  resync's mean deaths, kills and damage are all worse than indefinite carry.
  Against GRU, resync matches windowed mean deaths but has fewer kills and
  less damage.

These are fixed-delay NIF simulator screens, not Dolphin/netplay validation
or win-rate measurements. Both actor checkpoints and opponents were trained
at reaction delay 0; the physical delay-3/4 arms are deliberately outside
that training distribution. They identify regressions under latency, not
performance at an aligned trained delay. A deployment decision needs
checkpoints trained for the relevant delay and live harness validation.

## Means across four seeds

FD/BF mean Final Destination/Battlefield. Context suffix identifies the
opponent: windowed MinGRU or carried BPTT GRU. The digit is the action delay
for both players. Each mode runs eight environments for 1,800 frames per seed.
Environment rows are not independent replicas. No significance claim is made.

| Actor | Context | Mode | Deaths/min | Kills/min | Damage dealt/min |
| --- | --- | --- | ---: | ---: | ---: |
| min_gru | fd3_min_gru | windowed | 4.938 | 5.250 | 22.137 |
| min_gru | fd3_min_gru | carried | 5.812 | 4.875 | 14.500 |
| min_gru | fd3_min_gru | resync30 | 4.812 | 4.938 | 23.927 |
| min_gru | fd4_min_gru | windowed | 4.875 | 4.500 | 24.466 |
| min_gru | fd4_min_gru | carried | 5.625 | 5.188 | 15.029 |
| min_gru | fd4_min_gru | resync30 | 5.375 | 5.188 | 24.726 |
| min_gru | bf3_min_gru | windowed | 5.312 | 4.938 | 23.910 |
| min_gru | bf3_min_gru | carried | 5.688 | 4.938 | 8.625 |
| min_gru | bf3_min_gru | resync30 | 4.812 | 5.062 | 17.564 |
| min_gru | fd3_gru | windowed | 3.938 | 1.000 | 22.052 |
| min_gru | fd3_gru | carried | 4.500 | 0.812 | 14.581 |
| min_gru | fd3_gru | resync30 | 4.062 | 1.188 | 21.105 |
| mamba | fd3_min_gru | windowed | 1.188 | 3.812 | 39.680 |
| mamba | fd3_min_gru | carried | 1.250 | 4.000 | 45.017 |
| mamba | fd3_min_gru | resync30 | 1.062 | 3.875 | 44.631 |
| mamba | fd4_min_gru | windowed | 1.125 | 4.312 | 29.540 |
| mamba | fd4_min_gru | carried | 1.188 | 3.875 | 28.700 |
| mamba | fd4_min_gru | resync30 | 1.250 | 4.125 | 28.974 |
| mamba | bf3_min_gru | windowed | 1.125 | 3.562 | 43.586 |
| mamba | bf3_min_gru | carried | 0.938 | 3.812 | 42.858 |
| mamba | bf3_min_gru | resync30 | 1.062 | 3.625 | 41.589 |
| mamba | fd3_gru | windowed | 0.750 | 0.438 | 59.709 |
| mamba | fd3_gru | carried | 0.688 | 0.375 | 53.999 |
| mamba | fd3_gru | resync30 | 0.750 | 0.375 | 55.465 |

## Reproducibility and checks

- `PROTOCOL.md` defines the 96-arm grid, fixed-action FIFO and neutral/reset
  behavior. The resync-30 counter gives a 31-call interval once history is full.
- Action-delay unit tests pin frame-tagged timing at delays 0/1/3/4 and
  partial-reset isolation. A full zero-delay rerun matches every prior
  gameplay metric from `../recurrent_active_20261025/min_gru_windowed_s1003.json`.
- `manifest.json` retains hashes and exact evaluator source. The four-context
  scheduling split was recorded after four completed arms; no experimental
  setting changed. One unfinished arm was restarted during this scheduling
  change. All 96 final artifacts pass protocol validation.
- Four isolated evaluator VMs shared the GPU. Their rollout wall times are
  not inference-speed benchmarks. The uncontended Mamba-2 batching benchmark
  is documented separately in `../independent_batch_20261025/RESULTS.md`.
- The evaluator never calls observe-only Agent inference or enables button
  hysteresis. The separately validated observe/hysteresis correction was
  held until these frozen-source measurements finished.
- Run the summarizer from the recorded source version; it deliberately
  refuses source/checkpoint/simulator changes rather than mixing protocols.

Next quality gate: retain windowed defaults, use resync-30 as an explicit
MinGRU speed/quality candidate, and test correctly delay-trained checkpoints
in the actual deployment harness. Do not infer architecture superiority by
comparing these different checkpoints.

