"""One explicit, free shared-demo trial. No automatic retries or paid APIs."""
import argparse
import json
from pathlib import Path
import time
import uuid
import sys
import synthetic_media
import cv2
import numpy as np
from gradio_client import Client, handle_file
from synthetic_media.swapping import LandmarkDetector, model_path
from synthetic_media.scene_edit import head_mask, composite, pixel_digest
from synthetic_media.neural import run_worker
from synthetic_media.media import mark_footer

SPACE = 'black-forest-labs/FLUX.2-klein-4B'
PROMPT = ('Edit image 1 to show the identity of the person in image 2 naturally photographed '
          'in the scene of image 1. Transfer the reference person facial features, complexion, '
          'short black hairstyle and clean-shaven appearance, with matching natural neck skin. '
          'Keep the head angle, smile, gaze, glasses, body posture, black hoodie, microphone, '
          'camera framing, background and illumination of image 1. Preserve facial proportions '
          'of the reference identity. One coherent realistic photograph.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('target',type=Path)
    parser.add_argument('--authorized',action='store_true')
    parser.add_argument('--license-note',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.authorized or not args.license_note.strip():parser.error('Authorization for both media and external processing is required')
    if not args.worker:
        from synthetic_media.upstream import execute
        root=Path(__file__).resolve().parents[1]
        log=root/'.lab/shared-trial-client.log'
        try:
            execute([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                    log,timeout=360,rss_limit_mib=512,working_directory=root)
        finally:
            if log.is_file():print(log.read_text(),end='')
        return
    folder=Path('.lab/shared-trials')/uuid.uuid4().hex;folder.mkdir(parents=True)
    reference=cv2.imread(str(args.reference));target=cv2.imread(str(args.target))
    if reference is None or target is None:parser.error('Two decodable images are required')
    detector=LandmarkDetector(model_path())
    source_face=detector.single_face(reference,'Reference');face=detector.single_face(target,'Target')
    mask,segmentation=head_mask(target,face,folder,run_worker)
    cv2.imwrite(str(folder/'edit-mask.png'),(mask*255).astype(np.uint8))
    x,y,w,h=source_face[:4]
    crop=reference[max(0,int(y-.5*h)):min(reference.shape[0],int(y+1.4*h)),
                   max(0,int(x-.5*w)):min(reference.shape[1],int(x+1.5*w))]
    cv2.imwrite(str(folder/'reference-crop.png'),crop)
    scale=768/max(target.shape[:2]);width,height=round(target.shape[1]*scale),round(target.shape[0]*scale)
    working=cv2.resize(target,(width,height))
    working=cv2.copyMakeBorder(working,0,(-height)%16,0,(-width)%16,cv2.BORDER_REPLICATE)
    cv2.imwrite(str(folder/'target-working.png'),working)
    from synthetic_media.remote import verify_free_runtime
    verify_free_runtime(SPACE)
    client=Client(SPACE,token=False,verbose=False,analytics_enabled=False,max_workers=1,
                  download_files=folder/'download',httpx_kwargs={'timeout':60})
    started=time.monotonic();job=None
    try:
        images=[{'image':handle_file(str(folder/'target-working.png')),'caption':'Target scene'},
                {'image':handle_file(str(folder/'reference-crop.png')),'caption':'Authorized reference identity'}]
        print('Submitting one shared GPU trial; no paid fallback',flush=True)
        job=client.submit(PROMPT,images,'Distilled (4 steps)',42,False,
                          working.shape[1],working.shape[0],4,1.0,False,api_name='/infer')
        path,seed=job.result(timeout=300)
        generated=cv2.imread(str(path))
        if generated is None or generated.shape[:2]!=working.shape[:2]:
            raise ValueError('Shared service changed generation dimensions')
        generated=cv2.resize(generated[:height,:width],target.shape[:2][::-1])
        result,scene=composite(target,generated,mask)
        cv2.imwrite(str(folder/'synthetic-result.png'),mark_footer(result))
        report={'status':'needs_review','visual_acceptance':'unverified','space':SPACE,
                'model_revision_verified':False,'shared_demo':True,'paid_fallback':False,
                'authorization':args.license_note,'reference_pixel_sha256':pixel_digest(reference),
                'target_pixel_sha256':pixel_digest(target),'seed':seed,
                'elapsed_seconds':round(time.monotonic()-started,2),'scene_integrity':scene,
                'segmentation':segmentation,'output':str(folder/'synthetic-result.png')}
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    except TimeoutError:
        if job is not None:job.cancel()
        (folder/'failure.json').write_text(json.dumps({'status':'failed','error':'Queue response timed out after 300 seconds; cancellation requested','space':SPACE},indent=2))
        raise RuntimeError('Shared GPU trial timed out; cancellation requested')
    except Exception as error:
        (folder/'failure.json').write_text(json.dumps({'status':'failed','error':str(error),'space':SPACE},indent=2))
        raise
    finally:client.close()


if __name__=='__main__':main()
