"""Disposable network worker using verified public API schemas."""
import json
from pathlib import Path
import sys
import time
import cv2
from gradio_client import Client,handle_file
from .shared import SPACES,access_token
from .remote import verify_free_runtime
from .instructions import interpret
from .swapping import LandmarkDetector,model_path


def main():
    folder=Path(sys.argv[1]);data=json.loads((folder/'shared-request.json').read_text())
    service=data['service'];space=SPACES[service];token=access_token()
    if service=='shared' and not token:
        raise ValueError('Sign in with hf auth login: this instruction editor requests more GPU time than the anonymous allowance')
    verify_free_runtime(space,token)
    target=cv2.imread(data['target']);reference=cv2.imread(data['reference'])
    height,width=target.shape[:2]
    scale=768/max(height,width);rw,rh=round(width*scale),round(height*scale)
    working=cv2.resize(target,(rw,rh))
    working=cv2.copyMakeBorder(working,0,(-rh)%16,0,(-rw)%16,cv2.BORDER_REPLICATE)
    if data['scope']!='person':
        face=LandmarkDetector(model_path()).single_face(reference,'Reference')
        x,y,w,h=face[:4]
        reference=reference[max(0,int(y-.5*h)):min(reference.shape[0],int(y+1.4*h)),
                            max(0,int(x-.5*w)):min(reference.shape[1],int(x+1.5*w))]
    source_path,target_path=folder/'reference-working.png',folder/'target-working.png'
    cv2.imwrite(str(source_path),reference);cv2.imwrite(str(target_path),working)
    client=Client(space,token=token or False,verbose=False,max_workers=1,analytics_enabled=False,
                  download_files=folder/'download',httpx_kwargs={'timeout':60})
    started=time.monotonic();job=None
    try:
        if service=='shared-head':
            job=client.submit(handle_file(str(source_path)),handle_file(str(target_path)),
                              42,False,working.shape[1],working.shape[0],4,1.0,api_name='/face_swap')
            pair,seed=job.result(timeout=300)
            output_path=pair[1]
        else:
            prompt=interpret(data['instruction'],data['scope'])['generation_prompt']
            images=[{'image':handle_file(str(target_path)),'caption':'Target scene'},
                    {'image':handle_file(str(source_path)),'caption':'Reference identity'}]
            if service=='shared-flux':
                job=client.submit(prompt,images,'Distilled (4 steps)',42,False,
                                  working.shape[1],working.shape[0],4,1.0,False,api_name='/infer')
                output_path,seed=job.result(timeout=300)
            else:
                job=client.submit(images,prompt,42,False,4.0,40,working.shape[0],working.shape[1],False,api_name='/infer')
                gallery,seed=job.result(timeout=300)
                item=gallery[0]
                output_path=item.get('image') if isinstance(item,dict) else item[0]
                if isinstance(output_path,dict):output_path=output_path['path']
        generated=cv2.imread(str(output_path))
        if generated is None or generated.shape[:2]!=working.shape[:2]:raise ValueError('Unexpected shared editor dimensions')
        generated=cv2.resize(generated[:rh,:rw],(width,height))
        cv2.imwrite(str(folder/'generated.png'),generated)
        (folder/'shared-response.json').write_text(json.dumps({'space':space,'shared_service':True,
            'seed':seed,'elapsed_seconds':round(time.monotonic()-started,2),'paid_fallback':False,
            'instruction_forwarded':service in ('shared','shared-flux'),'model_revision_verified':False}))
    except TimeoutError:
        if job is not None:job.cancel()
        raise RuntimeError('Shared editor timed out; cancellation requested; no automatic retry')
    finally:client.close()


if __name__=='__main__':main()
