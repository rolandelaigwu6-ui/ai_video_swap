"""Free-tier shared-GPU entrypoint; weights load only on the host."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import synthetic_media  # establishes the decoded-image limit before OpenCV loads
from pathlib import Path
import tempfile
import threading
import time
import uuid
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import Flux2KleinPipeline
from synthetic_media.remote import MODEL, REVISION
from synthetic_media.scene_edit import composite, head_mask, pixel_digest
from synthetic_media.swapping import LandmarkDetector, model_path, read_working_image
from synthetic_media.media import mark_footer
from synthetic_media.neural import run_worker
from synthetic_media.quality_models import available, install

PASSWORD = os.environ.get('APP_PASSWORD', '')
if len(PASSWORD) < 24:
    raise RuntimeError('Set APP_PASSWORD in Space secrets to a random value of at least 24 characters')
OUTPUT_DIR = Path(tempfile.gettempdir())/'synthetic-outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def cleanup_outputs():
    while True:
        for path in OUTPUT_DIR.glob('*.png'):
            try:
                if time.time()-path.stat().st_mtime > 600:path.unlink()
            except FileNotFoundError:pass
        time.sleep(60)


threading.Thread(target=cleanup_outputs, daemon=True).start()
if not available('bisenet_resnet_34'):
    install('bisenet_resnet_34')
pipe = Flux2KleinPipeline.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                                         low_cpu_mem_usage=True, use_safetensors=True)
pipe.to('cuda')  # ZeroGPU emulates startup allocation; real GPU is leased below.

PROMPT = (
    'Edit image 1 using the person in image 2 as the identity reference. '
    'The person from image 2 is naturally photographed in the scene of image 1. '
    'Transfer their facial identity, complexion, hairstyle and facial hair appearance, '
    'including natural matching neck skin. Preserve image 1 head angle, expression, '
    'gaze, body posture, clothing, microphone, framing, background, camera perspective '
    'and illumination. Keep anatomical proportions and photorealistic texture. '
    'Generate one coherent photograph with the reference identity in the target pose.'
)


@spaces.GPU(duration=60)
def render(target, reference, seed):
    with torch.inference_mode():
        return pipe(image=[target, reference], prompt=PROMPT,
                    width=target.width, height=target.height, guidance_scale=1.0,
                    num_inference_steps=4,
                    generator=torch.Generator(device='cuda').manual_seed(int(seed))).images[0]


def working_target(target):
    height, width = target.shape[:2]
    scale = min(1.0, 768/max(height, width))
    resized = cv2.resize(target, (max(1, round(width*scale)), max(1, round(height*scale))))
    h, w = resized.shape[:2]
    padded_w, padded_h = max(256, (w+15)//16*16), max(256, (h+15)//16*16)
    padded = cv2.copyMakeBorder(resized, 0, padded_h-h, 0, padded_w-w, cv2.BORDER_REPLICATE)
    return Image.fromarray(padded[:, :, ::-1]), (w, h)


def generate(reference_path, target_path, authorized, license_note, seed):
    if not authorized or not license_note.strip():
        raise gr.Error('Authorization covering both media files and a consent/license note are required')
    for path in [reference_path, target_path]:
        if not path or Path(path).stat().st_size > 64*1024**2:
            raise gr.Error('Supply two images, each at most 64 MiB')
    reference, _ = read_working_image(Path(reference_path))
    target, _ = read_working_image(Path(target_path), max_edge=None)
    detector = LandmarkDetector(model_path())
    source_face = detector.single_face(reference, 'Reference')
    target_face = detector.single_face(target, 'Target')
    with tempfile.TemporaryDirectory(prefix='synthetic-edit-') as directory:
        mask, segmentation = head_mask(target, target_face, Path(directory), run_worker)
    x, y, w, h = source_face[:4]
    crop = reference[max(0, int(y-.5*h)):min(reference.shape[0], int(y+1.4*h)),
                     max(0, int(x-.5*w)):min(reference.shape[1], int(x+1.5*w))]
    model_target, (model_w, model_h) = working_target(target)
    started = time.monotonic()
    generated = render(model_target, Image.fromarray(crop[:, :, ::-1]), seed)
    generated = cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2BGR)
    generated = cv2.resize(generated[:model_h, :model_w], target.shape[:2][::-1])
    result, scene = composite(target, generated, mask)
    output = OUTPUT_DIR/(uuid.uuid4().hex+'.png')
    if not cv2.imwrite(str(output), mark_footer(result)):
        raise gr.Error('Could not export synthetic image')
    report = {'protocol': 1, 'synthetic': True, 'model': MODEL, 'model_revision': REVISION,
              'content_size': [target.shape[1], target.shape[0]],
              'target_pixel_sha256': pixel_digest(target), 'scene_integrity': scene,
              'elapsed_seconds': round(time.monotonic()-started, 2), 'seed': int(seed),
              'visual_acceptance': 'unverified', 'status': 'needs_review',
              'segmentation': segmentation, 'disclosure': 'added_footer'}
    return str(output), report


with gr.Blocks(delete_cache=(300, 600)) as demo:
    gr.Markdown('# Authorized synthetic-media research\n'
                'Reference identity in the target scene. Outputs require visual review. '
                'Shared GPU availability and daily usage are limited.')
    with gr.Row():
        reference = gr.File(label='Reference identity image', file_types=['image'], type='filepath')
        target = gr.File(label='Target scene image', file_types=['image'], type='filepath')
    authorized = gr.Checkbox(label='I have permission to use both identities and images')
    note = gr.Textbox(label='Consent or license note')
    seed = gr.Number(value=42, precision=0, label='Seed')
    run = gr.Button('Generate synthetic image')
    output = gr.File(label='Synthetic image — requires review')
    report = gr.JSON(label='Generation report')
    run.click(generate, [reference, target, authorized, note, seed], [output, report],
              api_name='generate', concurrency_limit=1)

demo.queue(max_size=4, default_concurrency_limit=1).launch(
    auth=('owner', PASSWORD), show_error=False, max_file_size='64mb')
