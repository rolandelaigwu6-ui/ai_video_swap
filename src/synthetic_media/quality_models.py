"""Small, individually verified optional quality models from official releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zlib

SPECS = {
    'bisenet_resnet_34': {'size': 93632546, 'tag': 'models-3.0.0', 'license': 'MIT', 'source': 'https://github.com/yakhyo/face-parsing'},
    'bisenet_resnet_18': {'size': 53205356, 'tag': 'models-3.1.0', 'license': 'MIT', 'source': 'https://github.com/yakhyo/face-parsing'},
    'xseg_1': {'size': 70324286, 'tag': 'models-3.1.0', 'license': 'GPL-3.0', 'source': 'https://github.com/iperov/DeepFaceLab'},
    'hyperswap_1a_256': {'size': 402742682, 'tag': 'models-3.3.0', 'license': 'ResearchRAIL', 'source': 'https://github.com/facefusion/facefusion-labs/tree/master/hyperswap'},
}


def root():
    return Path(os.environ.get('SYNTHETIC_MEDIA_QUALITY_MODELS', str(Path(__file__).resolve().parents[2] / '.lab/models/quality')))


def model_info(name, verify=False):
    spec = SPECS[name]
    path = root() / (name + '.onnx')
    receipt = root() / (name + '.json')
    if not path.is_file() or not receipt.is_file() or path.stat().st_size != spec['size']:
        raise ValueError(f'Quality model not installed/verified: {name}')
    info = json.loads(receipt.read_text())
    if verify:
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != info['sha256']:
            raise ValueError(f'Quality model checksum mismatch: {name}')
    return info


def available(name):
    try:
        model_info(name)
        return True
    except (OSError, ValueError, KeyError):
        return False


def install(name, verify_only=False):
    spec = SPECS[name]
    folder = root()
    folder.mkdir(parents=True, exist_ok=True)
    base = f'https://github.com/facefusion/facefusion-assets/releases/download/{spec["tag"]}/'
    path = folder / (name + '.onnx')
    partial = folder / (name + '.onnx.part')
    checksum = folder / (name + '.hash')
    for filename, local, expected in [(path.name, partial, spec['size']), (checksum.name, checksum, 8)]:
        if filename == path.name and path.is_file():
            continue
        if not local.is_file() or local.stat().st_size != expected:
            if verify_only:
                raise ValueError(f'Incomplete download: {local}')
            with urllib.request.urlopen(base + filename, timeout=60) as response, local.open('wb') as stream:
                while block := response.read(1024*1024):
                    stream.write(block)
    candidate = path if path.is_file() else partial
    if candidate.stat().st_size != spec['size']:
        raise ValueError(f'Incorrect model size: {name}')
    crc, sha = 0, hashlib.sha256()
    with candidate.open('rb') as stream:
        while block := stream.read(1024*1024):
            crc = zlib.crc32(block, crc)
            sha.update(block)
    if format(crc,'08x') != checksum.read_text().strip():
        raise ValueError(f'Upstream CRC mismatch: {name}')
    info = dict(spec, sha256=sha.hexdigest(), upstream_crc32=format(crc,'08x'), url=base+path.name)
    if candidate != path:
        candidate.replace(path)
    (folder / (name + '.json')).write_text(json.dumps(info, indent=2))
    print(f'Verified {name}', flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', choices=list(SPECS))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for name in args.models:
        install(name, args.verify_only)

if __name__ == '__main__':
    main()
