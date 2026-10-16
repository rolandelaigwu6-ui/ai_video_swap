"""Explicit access to public, quota-limited editors; no hosting-age requirement."""
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import cv2
import numpy as np
from .swapping import ImageFaceSwapper, LandmarkDetector, model_path
from .instructions import edit_mask
from .scene_edit import composite

SPACES={'shared':'Qwen/Qwen-Image-Edit-2511','shared-head':'linoyts/Flux2-Klein-Face-Swap',
        'shared-flux':'black-forest-labs/FLUX.2-klein-4B'}


def access_token():
    from huggingface_hub import get_token
    return os.environ.get('HF_TOKEN') or get_token()


def login_available():
    try:return bool(access_token())
    except ImportError:return False


def request(reference,target,folder,context,service):
    from .upstream import execute
    folder=folder.resolve()
    (folder/'shared-request.json').write_text(json.dumps({'reference':str(reference.resolve()),
        'target':str(target.resolve()),'instruction':context.instruction,'scope':context.edit_scope,
        'service':service}))
    try:
        execute([sys.executable,'-m','synthetic_media.shared_worker',str(folder)],folder/'network.log',
                timeout=330,rss_limit_mib=512,working_directory=Path(__file__).resolve().parents[2])
    except RuntimeError as error:
        from .service_errors import service_error
        log=folder/'network.log'
        if log.is_file():
            shutil.copyfile(log,context.output_dir/'shared-service.log')
        raise service_error(error) from error
    data=json.loads((folder/'shared-response.json').read_text())
    result=cv2.imread(str(folder/'generated.png'))
    if result is None:raise RuntimeError('Shared service returned no decodable image')
    return result,data


class SharedInstructionEditor(ImageFaceSwapper):
    generative=True

    def __init__(self,service='shared'):
        self.service=service;self.backend=service

    def transform(self,reference,target,source_face,target_face,context):
        if not context.instruction.strip():raise ValueError('An edit instruction is required')
        if self.service=='shared-head' and context.edit_scope!='head':raise ValueError('This service only replaces heads')
        with tempfile.TemporaryDirectory(prefix='shared-',dir=context.output_dir) as directory:
            folder=Path(directory)
            mask,segmentation=edit_mask(target,target_face,context.edit_scope,folder)
            source_path,target_path=folder/'reference.png',folder/'target.png'
            for path,image in [(source_path,reference),(target_path,target)]:
                if not cv2.imwrite(str(path),image):raise RuntimeError('Could not prepare shared editor input')
            generated,remote=request(source_path,target_path,folder,context,self.service)
            if generated.shape!=target.shape:raise ValueError('Shared result changed native target dimensions')
            result,scene=composite(target,generated,mask)
            from .quality import evaluate_identity
            report=evaluate_identity(reference,target,result,source_face,target_face,folder)
            report['status']='needs_review';report['scene_integrity']=scene
            if context.edit_scope=='person':report['issues'].append('Whole-body pose has not been independently validated')
            return result,{'instruction':context.instruction,'edit_scope':context.edit_scope,
                'remote':remote,'segmentation':segmentation,'quality_report':report,'scene_integrity':scene,
                '_edit_mask':mask,'generation_device':'shared GPU','compositing_device':'local CPU'}

    def protect_scene(self,target,result,face,metadata):
        mask=metadata.pop('_edit_mask')
        if not np.array_equal(result[mask==0],target[mask==0]):raise ValueError('Background preservation failed')
        return result,metadata['scene_integrity']
