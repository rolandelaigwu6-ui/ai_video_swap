"""Download only the two required research models and verify upstream CRC + SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zlib

BASE = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/'
MODELS = {'inswapper_128_fp16.onnx': 277680829, 'arcface_w600k_r50.onnx': 174388474}


def install(root: Path, verify_only=False):
    root.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, expected_size in MODELS.items():
        destination = root / name
        partial = root / (name + '.part')
        checksum_file = root / (name.removesuffix('.onnx') + '.hash')
        for remote, local in [(name, partial), (checksum_file.name, checksum_file)]:
            if local == partial and destination.is_file():
                continue
            needs_download = not local.is_file() or (local == partial and local.stat().st_size != expected_size)
            if needs_download:
                if verify_only:
                    raise ValueError(f'Missing {local.name}')
                print(f'Downloading {remote}', flush=True)
                with urllib.request.urlopen(BASE + remote, timeout=60) as response, local.open('wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
        candidate = destination if destination.is_file() else partial
        if candidate.stat().st_size != expected_size:
            raise ValueError(f'Incomplete model {name}: expected {expected_size} bytes')
        crc = 0
        sha = hashlib.sha256()
        with candidate.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                crc = zlib.crc32(block, crc)
                sha.update(block)
        expected_crc = checksum_file.read_text().strip()
        if format(crc, '08x') != expected_crc:
            raise ValueError(f'Upstream checksum mismatch: {name}')
        if candidate != destination:
            candidate.replace(destination)
        verified[name] = {'size': expected_size, 'sha256': sha.hexdigest(), 'upstream_crc32': expected_crc, 'url': BASE + name}
        print(f'Verified {name}', flush=True)
    configuration = {'license': 'non-commercial research only; commercial rights not granted',
                     'license_source': 'https://github.com/deepinsight/insightface#license', 'models': verified}
    temporary = root / 'ready.json.tmp'
    temporary.write_text(json.dumps(configuration, indent=2))
    temporary.replace(root / 'ready.json')
    return configuration


def main():
    from .neural import model_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    install(model_dir(), args.verify_only)

if __name__ == '__main__':
    main()
