"""Run the Mewtwo GRU preflight and full imitation fit with persistent logs/status."""
import datetime
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / 'eval_runs/0924_mewtwo_il'
RUN = ROOT / 'checkpoints/mewtwo_il_v1_20260924'

def status(phase, **extra):
    value = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), supervisor_pid=os.getpid(), **extra)
    (WORK / 'status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)

def run_phase(label, data, destination, val_files, epochs, register):
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / 'model.axon').exists(), 'Refusing to overwrite an existing run'
    args = ['--backbone', 'gru', '--temporal', '--stage-internals', '--hidden-sizes', '1024,1024,512',
        '--num-layers', '2', '--batch-size', '64', '--dropout', '0.1', '--precision', 'f32',
        '--bptt', '--unroll', '80', '--bptt-overlap', '0', '--bptt-val-files', str(val_files),
        '--stream-chunk-size', '64', '--replays', str(data), '--train-character', 'mewtwo',
        '--select-character-port', '--label-delay', '0', '--epochs', str(epochs), '--seed', '924',
        '--head', 'autoregressive', '--save-best', '--save-every-batches', '2000',
        '--label-smoothing', '0.0', '--no-focal-loss', '--button-pos-weight', '1,1,1,1,1,1,1,1',
        '--action-oversample', '1.0', '--entropy-weight', '0.0', '--neutral-weight', '1.0',
        '--stick-edge-weight', '1.0', '--learning-rate', '0.0001', '--lr-schedule', 'constant',
        '--weight-decay', '0.05', '--max-grad-norm', '0.5', '--early-stopping', '--patience', '6',
        '--checkpoint', str(destination / 'model.axon')]
    if not register:
        args.append('--no-register')
    command = [shutil.which('devenv'), 'shell', '--', 'env', 'EXLA_TARGET=cuda',
        'EXPHIL_GPU_MEMORY_FRACTION=0.80', 'EXPHIL_EXLA_PRECISION=highest', 'mix', 'run', 'scripts/train.exs', *args]
    (destination / 'launch.json').write_text(json.dumps(dict(command=command, args=args, corpus=str(WORK/'corpus.json')), indent=2))
    log = destination / 'train.log'
    with log.open('w') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        status(label, child_pid=child.pid, log=str(log), checkpoint=str(destination/'model.axon'))
        code = child.wait()
    text = log.read_text(errors='replace')
    if code != 0 or 'Training complete!' not in text or not (destination/'model_best_policy.bin').exists():
        raise RuntimeError(f'{label} did not complete cleanly: exit={code}; see {log}')
    # Epoch summary must report finite held-out loss before the expensive phase.
    import re
    match = re.search(r'Best val_loss: ([0-9.eE+-]+)', text)
    if not match or not math.isfinite(float(match[1])):
        raise RuntimeError(f'{label}: missing finite validation result')
    return dict(best_val_loss=float(match[1]), log=str(log))

def main():
    corpus = json.loads((WORK/'corpus.json').read_text())
    sample = WORK/'preflight_corpus'
    sample.mkdir(exist_ok=True)
    # Preflight uses only full-run training games; validation/test remain untouched.
    train = [r for r in corpus['rows'] if r['split']=='train'][:20]
    for i, row in enumerate(train):
        path = sample / f'{i:03d}.slp'
        if not path.exists():
            path.symlink_to(row['path'])
    preflight = run_phase('preflight', sample, WORK/'preflight', 4, 1, False)
    (WORK/'preflight_result.json').write_text(json.dumps(preflight, indent=2))
    full = run_phase('training', WORK/'corpus', RUN, corpus['counts']['validation'], 30, True)
    status('imitation_complete', result=full, policy=str(RUN/'model_best_policy.bin'),
        next='Held-out test and Dolphin validation; PPO has not been launched.')

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
