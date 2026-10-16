"""Disposable network client so stalled queue threads cannot accumulate locally."""
import json
from pathlib import Path
import sys
from .remote import _request_generation


def main():
    folder=Path(sys.argv[1])
    request=json.loads((folder/'remote-request.json').read_text())
    response=_request_generation(Path(request['reference']),Path(request['target']),folder,request['note'])
    (folder/'remote-response.json').write_text(json.dumps(response))


if __name__=='__main__':main()
