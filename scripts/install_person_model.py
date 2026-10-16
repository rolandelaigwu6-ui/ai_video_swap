"""Install the small pinned Apache-2.0 person segmenter on project disk."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
COMMIT='47534e27c9851bb1128ccc0102f1145e27f23f98'
SHA='552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24'
SIZE=6163938
BASE=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/human_segmentation_pphumanseg/'


def main():
    folder=ROOT/'.lab/models/person';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'pphumanseg.onnx'
    url=BASE+'human_segmentation_pphumanseg_2023mar.onnx'
    if not path.is_file():
        candidate=folder/'pphumanseg.part'
        with urllib.request.urlopen(url,timeout=60) as response,candidate.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
    else:candidate=path
    if candidate.stat().st_size!=SIZE or hashlib.sha256(candidate.read_bytes()).hexdigest()!=SHA:
        raise ValueError('Person model verification failed')
    if candidate!=path:candidate.replace(path)
    if not (folder/'LICENSE').is_file():
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/opencv/opencv_zoo/{COMMIT}/models/human_segmentation_pphumanseg/LICENSE',timeout=30) as response:
            (folder/'LICENSE').write_bytes(response.read())
    (folder/'receipt.json').write_text(json.dumps({'sha256':SHA,'size':SIZE,'commit':COMMIT,
                                                 'url':url,'license':'Apache-2.0'},indent=2))
    print('Person model verified (5.9 MiB on disk); no GPU generation weights installed')


if __name__=='__main__':main()
