"""Build a small source-only hosting bundle; never include user media or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination, refresh=False):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        receipt_path=destination/'DEPLOYMENT.json'
        if not refresh or not receipt_path.is_file():
            raise ValueError('Choose an empty destination, or refresh an unchanged generated bundle')
        old=json.loads(receipt_path.read_text())['files']
        actual={str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
        if actual != set(old)|{'DEPLOYMENT.json'}:
            raise ValueError('Bundle contains unexpected files; refusing to package them')
        for name,checksum in old.items():
            if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=checksum:
                raise ValueError('Bundle has modified files; choose an empty destination')
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ['app.py', 'requirements.txt', 'README.md']:
        shutil.copy2(ROOT/'deploy/zerogpu'/filename, destination/filename)
    shutil.copytree(ROOT/'src/synthetic_media', destination/'synthetic_media',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name!='DEPLOYMENT.json':
            files[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'hosting': 'Hugging Face ZeroGPU', 'activation': 'not deployed',
               'paid_fallback': False, 'files': files,
               'account_requirement': 'Verified personal account older than 30 days; free quota subject to provider rules'}
    (destination/'DEPLOYMENT.json').write_text(json.dumps(receipt, indent=2))
    return {'bundle': str(destination), 'files': len(files),
            'bytes': sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()),
            'deployed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'.lab/server-package')
    parser.add_argument('--refresh', action='store_true', help='Refresh an unchanged generated bundle only')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output,args.refresh), indent=2))
