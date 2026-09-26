"""Keep the Mewtwo handoff status current while the detached campaign runs."""
import datetime
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ROOT / 'eval_runs/0924_mewtwo_il/status.json'
DOC = ROOT / 'docs/planning/MEWTWO_LIVE_STATUS.md'
while True:
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        time.sleep(10)
        continue
    active = subprocess.run(['systemctl', '--user', 'is-active', 'exphil-mewtwo-il-v1'], capture_output=True, text=True).stdout.strip()
    phase = state['phase']
    if active not in ['active', 'activating'] and phase not in ['failed', 'imitation_complete']:
        phase = 'stopped; inspect journal and checkpoints before resuming'
    log = pathlib.Path(state.get('log') or state.get('result', {}).get('log', ''))
    text = log.read_text(errors='replace') if log.is_file() else ''
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    epochs = re.findall(r'--- Epoch (\d+/\d+) ---', text)
    losses = re.findall(r'train_loss\s+([0-9.eE+-]+)\s+│\n│\s+val_loss\s+([0-9.eE+-]+)', text)
    metrics = f'Latest completed epoch: train loss {losses[-1][0]}, validation loss {losses[-1][1]}.' if losses else 'No completed epoch reported in this phase yet.'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    DOC.write_text(f'''# Mewtwo imitation live status

Updated automatically: {now}.

Phase: **{phase}**. Service: `exphil-mewtwo-il-v1` ({active}).
Epoch: {epochs[-1] if epochs else 'preparing data/model'}.
{metrics}

Supervisor PID: {state.get('supervisor_pid')}; latest child PID: {state.get('child_pid', 'finished')}.
Log: `{log}`. These PIDs describe this update and may exit afterward.

Corpus: 418 games, 5.24M frames; 334 train / 42 validation / 42 test.
Model: 2×1024 GRU, 10.46M parameters; up to 30 epochs with early stopping.
Preflight passed: 250 updates, train 4.2093 / validation 4.343, policy exported.
Best full-fit policy: `checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin`
(available after the first completed full-fit epoch).

PPO has not started. Final held-out testing, Dolphin play and Mewtwo sim checks remain.
While training is active, do not run Mix or edit training library files.
Detailed recipe and provenance: [Mewtwo campaign](MEWTWO_IMITATION_2026-09-24.md).
Error, if any: {state.get('error', 'none reported')}.
''')
    if phase in ['failed', 'imitation_complete'] or active not in ['active', 'activating']:
        break
    time.sleep(30)
