"""Fetch the public Mewtwo replay collection; verify LFS SHA256; resume safely."""
import concurrent.futures
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / 'replays/mewtwo_public_20260924'
REPO = 'erickfm/slippi-public-dataset-v3.7'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}'))
    revision = info['sha']
    url = f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/MEWTWO?recursive=true&limit=1000'
    entries = []
    while url:
        with urllib.request.urlopen(url) as response:
            entries.extend(json.load(response))
            link = response.headers.get('Link', '')
            url = next((part.split('<')[1].split('>')[0] for part in link.split(',') if 'rel="next"' in part), None)
    files = [x for x in entries if x['path'].endswith('.slp')]
    (DEST / 'source_manifest.json').write_text(json.dumps({'repo': REPO, 'revision': revision, 'files': files}, indent=2))
    print(f'Pinned {revision}: {len(files)} files, {sum(x["size"] for x in files)/1e6:.1f} MB', flush=True)
    def fetch(entry):
        target = DEST / pathlib.Path(entry['path']).name
        expected = entry['lfs']['oid']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return target.name
        remote = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/' + urllib.parse.quote(entry['path'])
        with urllib.request.urlopen(remote, timeout=120) as response:
            data = response.read()
        assert len(data) == entry['size'] and hashlib.sha256(data).hexdigest() == expected, entry['path']
        temp = target.with_suffix('.part')
        temp.write_bytes(data)
        temp.replace(target)
        return target.name
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, name in enumerate(pool.map(fetch, files), 1):
            print(f'[{i}/{len(files)}] {name}', flush=True)

if __name__ == '__main__':
    main()
