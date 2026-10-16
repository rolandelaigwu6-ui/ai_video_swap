"""Opt-in shared-GPU identity editing; local scene compositing stays bounded."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import sys
import cv2
import numpy as np
from .swapping import ImageFaceSwapper
from .scene_edit import composite, head_mask, pixel_digest

MODEL = 'black-forest-labs/FLUX.2-klein-4B'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'


def status():
    space = os.environ.get('SYNTHETIC_MEDIA_REMOTE_SPACE', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', space):
        return {'available': False, 'reason': 'Configure your deployed Hugging Face Space ID; no server is active'}
    if not os.environ.get('SYNTHETIC_MEDIA_REMOTE_PASSWORD'):
        return {'available': False, 'reason': 'Configure the remote application password locally'}
    if not importlib.util.find_spec('gradio_client'):
        return {'available': False, 'reason': 'Install the remote optional dependencies'}
    from .quality_models import available
    if not available('bisenet_resnet_34'):
        return {'available': False, 'reason': 'Install the head segmentation model first'}
    return {'available': True, 'configured': True, 'connection_verified': False,
            'space': space, 'model': MODEL, 'billing': 'No paid API fallback; subject to free hosting eligibility/quota'}


def request_generation(reference, target, folder, note):
    from .upstream import execute
    folder=folder.resolve()
    (folder/'remote-request.json').write_text(json.dumps({
        'reference':str(reference.resolve()),'target':str(target.resolve()),'note':note}))
    execute([sys.executable,'-m','synthetic_media.remote_worker',str(folder)],
            folder/'remote-client.log',timeout=330,rss_limit_mib=256,
            working_directory=Path(__file__).resolve().parents[2])
    return json.loads((folder/'remote-response.json').read_text())


def _request_generation(reference, target, folder, note):
    from gradio_client import Client, handle_file
    configured = status()
    if not configured['available']:raise ValueError(configured['reason'])
    token=os.environ.get('HF_TOKEN') or None
    verify_free_runtime(configured['space'],token)
    client = Client(configured['space'],
                    auth=('owner', os.environ['SYNTHETIC_MEDIA_REMOTE_PASSWORD']),
                    token=token or False,
                    download_files=str(folder), verbose=False, max_workers=1, analytics_enabled=False,
                    httpx_kwargs={'timeout': 60})
    job = client.submit(handle_file(str(reference)), handle_file(str(target)),
                        True, note, 42, api_name='/generate')
    try:
        return job.result(timeout=300)
    except TimeoutError as error:
        job.cancel()
        raise RuntimeError('Remote generation exceeded five minutes; cancellation requested') from error
    finally:
        client.close()


def verify_free_runtime(space, token=None, api=None):
    """Reject paid/pending-paid hardware and credit-capable account tiers."""
    if api is None:
        from huggingface_hub import HfApi
        api=HfApi(token=token or False)
    runtime=api.get_space_runtime(space)
    hardware=getattr(runtime.hardware,'value',runtime.hardware)
    requested=getattr(runtime.requested_hardware,'value',runtime.requested_hardware)
    if hardware!='zero-a10g' or requested not in (None,'zero-a10g'):
        raise ValueError('Free-only policy: configured Space is not exclusively ZeroGPU hardware')
    if token:
        account=api.whoami()
        if account.get('type')!='user' or account.get('isPro') is not False:
            raise ValueError('Free-only policy: use a free personal account token or anonymous access')


def validate_output(path, metadata, target):
    if metadata.get('protocol') != 1 or metadata.get('synthetic') is not True:
        raise ValueError('Remote output lacks the synthetic-media protocol')
    if metadata.get('model') != MODEL or metadata.get('model_revision') != REVISION:
        raise ValueError('Unexpected remote model or revision')
    if metadata.get('target_pixel_sha256') != pixel_digest(target):
        raise ValueError('Remote output belongs to a different target')
    if metadata.get('content_size') != [target.shape[1], target.shape[0]]:
        raise ValueError('Remote content size changed')
    if Path(path).stat().st_size > 64*1024**2:
        raise ValueError('Remote output exceeds the file limit')
    image = cv2.imread(str(path))
    height, width = target.shape[:2]
    if image is None or image.shape[1] != width or not height+24 <= image.shape[0] <= height+100:
        raise ValueError('Invalid remote image or missing disclosure footer')
    return image[:height].copy()


class RemoteIdentityEditor(ImageFaceSwapper):
    backend = 'remote-reference-conditioned-edit'
    generative = True

    def transform(self, reference, target, source_face, target_face, context):
        from .neural import run_worker
        if not context.license_note.strip():raise ValueError('Reference and target authorization note is required')
        with tempfile.TemporaryDirectory(prefix='remote-', dir=context.output_dir) as directory:
            folder = Path(directory)
            mask, segmentation = head_mask(target, target_face, folder, run_worker)
            source_path, target_path = folder/'reference.png', folder/'target.png'
            for path, image in [(source_path, reference), (target_path, target)]:
                if not cv2.imwrite(str(path), image):raise RuntimeError('Cannot prepare remote media')
            try:
                path, metadata = request_generation(source_path, target_path, folder, context.license_note)
            except (ValueError, RuntimeError):raise
            except Exception as error:
                raise RuntimeError('Remote generation failed; check server availability, credentials and free GPU quota') from error
            generated = validate_output(path, metadata, target)
            result, scene = composite(target, generated, mask)
            quality = {'status': 'needs_review', 'visual_acceptance': 'unverified', 'flawless_certified': False,
                       'issues': ['Reference fidelity and head pose require visual acceptance; remote generation is not certified'],
                       'scene_integrity': scene}
            return result, {'model': MODEL, 'model_revision': REVISION, 'remote': metadata,
                            'segmentation': segmentation, 'scene_integrity': scene,
                            'quality_report': quality, '_edit_mask': mask,
                            'generation_device': 'remote GPU', 'compositing_device': 'local CPU'}

    def protect_scene(self, target, result, face, metadata):
        mask = metadata.pop('_edit_mask')
        if not np.array_equal(target[mask == 0], result[mask == 0]):
            raise ValueError('Remote edit changed pixels outside the head mask')
        return result, metadata['scene_integrity']
