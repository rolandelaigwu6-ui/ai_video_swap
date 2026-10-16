"""Install the upstream model plan, streaming integrity checks and reusing links."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zlib

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'vendor/facefusion'
plan=json.loads((folder/'model-plan.json').read_text())
receipts={}
for name,info in plan.items():
    path=Path(info['path']); checksum_path=Path(info['hash_path'])
    if not info['url'] or not info['hash_url']:
        raise ValueError('Model plan has no reachable source URLs; rerun preparation with network access')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not checksum_path.is_file():
        with urllib.request.urlopen(info['hash_url'],timeout=60) as response:
            checksum_path.write_bytes(response.read(128))
    if not path.is_file():
        partial=path.with_suffix('.onnx.part')
        print('Downloading '+name,flush=True)
        with urllib.request.urlopen(info['url'],timeout=60) as response,partial.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        candidate=partial
    else:
        candidate=path
    crc,sha=0,hashlib.sha256()
    with candidate.open('rb') as stream:
        while block:=stream.read(1024*1024):
            crc=zlib.crc32(block,crc);sha.update(block)
    if format(crc,'08x')!=checksum_path.read_text().strip():
        raise ValueError('Upstream checksum mismatch: '+name)
    if candidate!=path:candidate.replace(path)
    receipts[name]={'url':info['url'],'sha256':sha.hexdigest(),'crc32':format(crc,'08x'),'size':path.stat().st_size}
    print('Verified '+name,flush=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip()
(folder/'installed-models.json').write_text(json.dumps({'commit':commit,'models':receipts},indent=2))
print('Upstream model installation verified',flush=True)
