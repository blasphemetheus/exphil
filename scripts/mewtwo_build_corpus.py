"""Build an immutable, hash-deduplicated Mewtwo corpus and game-level splits."""
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / 'eval_runs/0924_mewtwo_il'

def main():
    mode = sys.argv[1]
    if mode == 'candidates':
        rows = []
        for source in ['local_audit', 'public_audit']:
            rows.extend(json.loads(line) for line in (WORK / source / 'metadata.jsonl').read_text().splitlines())
        unique, rejected = {}, []
        for row in rows:
            players = row.get('players', [])
            if 'error' in row or len(players) != 2 or any(p['player_type'] != 'human' or p['team'] is not None for p in players):
                rejected.append(dict(row, rejection='unparseable_or_not_human_singles'))
            elif row['frames'] < 3600 or row['stage'] not in [2, 3, 8, 28, 31, 32]:
                rejected.append(dict(row, rejection='short_or_noncompetitive'))
            elif row['sha256'] in unique:
                unique[row['sha256']]['aliases'].append(row['path'])
            else:
                unique[row['sha256']] = dict(row, aliases=[row['path']])
        dest = WORK / 'candidates'
        dest.mkdir(parents=True, exist_ok=True)
        for sha, row in unique.items():
            target = dest / f'{sha}.slp'
            if not target.exists():
                target.symlink_to(row['path'])
        report = dict(input_rows=len(rows), unique_candidates=len(unique), candidates=list(unique.values()), rejected=rejected)
        (WORK / 'candidates.json').write_text(json.dumps(report, indent=2))
        print(f'{len(rows)} rows -> {len(unique)} unique quality candidates; {len(rejected)} rejections')
    elif mode == 'split':
        candidates = {r['sha256']: r for r in json.loads((WORK / 'candidates.json').read_text())['candidates']}
        quality = [json.loads(line) for line in (WORK / 'quality/manifest.jsonl').read_text().splitlines()]
        print('Quality verdicts:', collections.Counter(r.get('verdict') for r in quality))
        kept = [candidates[pathlib.Path(r['path']).stem] for r in quality if r.get('verdict') == 'keep']
        assert len(kept) >= 100, f'Only {len(kept)} usable replays'
        # Different metadata encodings of one match must not cross splits.
        groups = collections.defaultdict(list)
        for row in kept:
            key = json.dumps([row['started_at'], row['random_seed'], row['stage'], row['frames'],
                sorted((p['port'], p['character']) for p in row['players'])])
            groups[key].append(row)
        ordered = sorted(groups, key=lambda key: hashlib.sha256(('mewtwo-924:' + key).encode()).hexdigest())
        n = max(20, round(len(ordered) * 0.1))
        splits = {'train': ordered[:-2*n], 'validation': ordered[-2*n:-n], 'test': ordered[-n:]}
        rows = []
        for split, keys in splits.items():
            for key in keys:
                # Keep one encoding per exact match signature, with all aliases recorded.
                row = min(groups[key], key=lambda r: r['sha256'])
                directory = WORK / ('test' if split == 'test' else 'corpus')
                directory.mkdir(parents=True, exist_ok=True)
                prefix = 'zz_validation' if split == 'validation' else split
                path = directory / f'{prefix}_{row["sha256"]}.slp'
                if not path.exists():
                    path.symlink_to(row['path'])
                rows.append(dict(row, split=split, training_path=str(path), equivalent_hashes=[r['sha256'] for r in groups[key]]))
        assert len({r['sha256'] for r in rows}) == len(rows)
        counts = collections.Counter(r['split'] for r in rows)
        corpus_paths = sorted(str(p) for p in (WORK / 'corpus').glob('*.slp'))
        assert corpus_paths[-counts['validation']:] == sorted(r['training_path'] for r in rows if r['split'] == 'validation')
        report = dict(seed=924, split_unit='game; exact match signatures grouped',
            limitations=['Players/sessions may overlap splits; not a player-held-out generalization claim.'],
            counts=dict(counts), frames={s:sum(r['frames'] for r in rows if r['split']==s) for s in splits},
            stage_counts=dict(collections.Counter(r['stage'] for r in rows)), rows=rows)
        (WORK / 'corpus.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
    else:
        raise ValueError(mode)

if __name__ == '__main__':
    main()
