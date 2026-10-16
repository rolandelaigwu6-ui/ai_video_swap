"""Semantic face-only compositing with separately executed learned masks."""
import cv2
import numpy as np
from .quality_models import available, model_info, root

# CelebAMask-HQ: skin, brows, eyes, nose, mouth and lips. Preserve glasses,
# hair, ears, neck, hats, and the background from the target image.
FACE_REGIONS = [1, 2, 3, 4, 5, 10, 11, 12, 13]


def masks_available():
    return (available('bisenet_resnet_34') or available('bisenet_resnet_18')) and available('xseg_1')


def semantic_mask(target_crop, generated_crop, folder, worker):
    parser_name = 'bisenet_resnet_34' if available('bisenet_resnet_34') else 'bisenet_resnet_18'
    parser_info = model_info(parser_name, verify=True)
    occluder_info = model_info('xseg_1', verify=True)
    measurements = []
    masks = []
    for crop in (target_crop, generated_crop):
        rgb = cv2.resize(crop, (512,512))[:,:,::-1].astype(np.float32)/255
        tensor = ((rgb - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        np.save(folder / 'mask-input.npy', tensor)
        measurements.append(worker('parse',root()/(parser_name+'.onnx'),folder))
        labels = np.load(folder / 'parse.npy',allow_pickle=False)
        masks.append(cv2.resize(np.isin(labels,FACE_REGIONS).astype(np.float32),target_crop.shape[:2][::-1]))
    image = cv2.resize(target_crop,(256,256)).astype(np.float32)[None]/255
    np.save(folder/'mask-input.npy', image)  # XSeg contract is NHWC, BGR in [0,1].
    measurements.append(worker('occlude',root()/'xseg_1.onnx',folder))
    occlusion = cv2.resize(np.load(folder/'occlude.npy',allow_pickle=False),target_crop.shape[:2][::-1])
    # Profile views can be falsely labeled background by the parser even when
    # the face detector and XSeg recognize the face. Never erase a valid face
    # solely because a semantic parser rejected that pose. Use its regions only
    # with sufficient coverage, and report the fallback explicitly.
    parser_coverages = [float((item>.5).mean()) for item in masks]
    usable = [.15 <= coverage <= .85 for coverage in parser_coverages]
    raw = occlusion.clip(0,1)
    method = 'occlusion-only-parser-rejected'
    limitations = []
    if usable[1]:
        raw = np.minimum(raw,masks[1])
        method = 'occlusion+generated-semantic-face'
    else:
        limitations.append('Generated-face semantic segmentation rejected this pose; using learned XSeg visibility only')
    if usable[0]:
        raw = np.minimum(raw,masks[0])
    else:
        limitations.append('Target semantic segmentation rejected this pose; accessory preservation relies on XSeg')
    raw = cv2.erode(raw,np.ones((3,3),np.uint8))
    mask = cv2.GaussianBlur(raw,(5,5),0).clip(0,1)
    mask = np.minimum(mask,raw)  # no blur bleeding onto preserved regions
    coverage = float((mask>.5).mean())
    if coverage < .05 or coverage > .95:
        raise ValueError(f'Learned masks have implausible face coverage: {coverage:.3f}')
    return mask, {'models':{'parser':parser_info,'occluder':occluder_info},'face_coverage':coverage,
                  'method':method, 'semantic_coverages':parser_coverages,'limitations':limitations,
                  'intended_preservation':['glasses','hair','ears','neck','hat','background'], 'worker_measurements':measurements}
