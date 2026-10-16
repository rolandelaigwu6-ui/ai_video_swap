"""Head/neck edit boundaries for reference-conditioned generation."""
import hashlib
import cv2
import numpy as np

HEAD_LABELS = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 17)


def pixel_digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def head_mask(target, face, folder, worker):
    from .quality_models import model_info, root
    model_info('bisenet_resnet_34', verify=True)
    x, y, w, h = face[:4]
    left, top = max(0, int(x - .6*w)), max(0, int(y - .6*h))
    right = min(target.shape[1], int(np.ceil(x + 1.6*w)))
    bottom = min(target.shape[0], int(np.ceil(y + 1.9*h)))
    crop = target[top:bottom, left:right]
    rgb = cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255
    tensor = ((rgb - np.array([.485, .456, .406], np.float32)) /
              np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None]
    np.save(folder / 'mask-input.npy', tensor)
    measurement = worker('parse', root() / 'bisenet_resnet_34.onnx', folder)
    labels = np.load(folder / 'parse.npy', allow_pickle=False)
    binary = cv2.resize(np.isin(labels, HEAD_LABELS).astype(np.uint8),
                        (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
    coverage = float(binary.mean())
    if not .05 <= coverage <= .90:
        raise ValueError('Head segmentation failed; refusing to edit background')
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    inside = np.clip(distance / max(1, min(w, h)*.025), 0, 1)
    mask = np.zeros(target.shape[:2], np.float32)
    mask[top:bottom, left:right] = inside
    return mask, {'method': 'target-semantic-head-neck', 'parser_coverage': coverage,
                  'labels': list(HEAD_LABELS), 'worker': measurement}


def composite(target, generated, mask):
    if generated.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError('Generation or edit-mask dimensions changed')
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('Invalid edit mask')
    if not np.any(mask > 0) or np.all(mask > 0):
        raise ValueError('Edit mask must preserve scene pixels')
    result = target.copy()
    selected = mask > 0
    alpha = mask[selected, None]
    result[selected] = np.rint(generated[selected]*alpha + target[selected]*(1-alpha)).clip(0, 255).astype(np.uint8)
    return result, {'native_size': [target.shape[1], target.shape[0]],
                    'editable_region': 'segmented head and neck',
                    'editable_pixels': int(selected.sum()),
                    'outside_edit_region_pixels_preserved': bool(np.array_equal(result[~selected], target[~selected])),
                    'mask_sha256': pixel_digest(mask), 'posture_and_background_resampled': False}
