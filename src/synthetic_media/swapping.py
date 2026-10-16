"""CPU image swapping baseline: YuNet alignment and Poisson blending.

This transfers source facial pixels; it is not a generative identity model.
"""
import os
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .components import Artifact, Context
from .media import mark, mark_footer

MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
MAX_EDGE = 1280

MODEL_NAME = 'face_detection_yunet_2023mar.onnx'


def model_path():
    return Path(os.environ.get('SYNTHETIC_MEDIA_YUNET', str(Path(__file__).resolve().parent / 'models' / 'yunet' / MODEL_NAME)))


class LandmarkDetector:
    def __init__(self, path: Path):
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('YuNet model checksum does not match the verified model')
        self.detector = cv2.FaceDetectorYN.create(str(path), '', (320, 320), 0.85, 0.3, 5000)

    def single_face(self, image, role):
        height, width = image.shape[:2]
        base_scale = min(1.0, 1280 / max(height, width))
        # Retry tiny inputs at a higher detection scale, without lowering confidence.
        scales = [base_scale]
        if max(height, width) < 640:
            scales.append(min(3.0, 1280 / max(height, width)))
        for scale in scales:
            small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_CUBIC)
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.detector.detect(small)
            count = 0 if faces is None else len(faces)
            if count > 1:
                raise ValueError(f'{role} must contain exactly one detectable face; found {count}')
            if count == 1:
                face = faces[0].copy()
                face[:14] /= scale
                return face
        raise ValueError(f'{role}: found 0 detectable faces. Use a clear, unobstructed face image')



def blend_face(reference, target, source_face, target_face):
    """Align five landmarks and blend an interior ellipse into the target."""
    source_points = source_face[4:14].reshape(5, 2).astype(np.float32)
    target_points = target_face[4:14].reshape(5, 2).astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('Could not align face landmarks')
    determinant = np.linalg.det(matrix[:, :2])
    if determinant <= 0:
        raise ValueError('Invalid face alignment')
    height, width = target.shape[:2]
    warped = cv2.warpAffine(reference, matrix, (width, height), flags=cv2.INTER_LINEAR)
    # Construct the mask in reference coordinates so it follows the fitted rotation.
    x, y, w, h = source_face[:4]
    mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (round(x + w * .5), round(y + h * .53)),
                (max(1, round(w * .39)), max(1, round(h * .43))), 0, 0, 360, 255, -1)
    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    # Restrict to actual warped source pixels and avoid image-edge clone failures.
    valid = cv2.warpAffine(np.full(reference.shape[:2], 255, np.uint8), matrix, (width, height))
    mask = cv2.bitwise_and(mask, valid)
    mask[:2] = mask[-2:] = 0
    mask[:, :2] = mask[:, -2:] = 0
    bx, by, bw, bh = cv2.boundingRect(mask)
    if bw < 16 or bh < 16:
        raise ValueError('Aligned face is too small or outside the target image')
    center = (bx + bw // 2, by + bh // 2)
    poisson = cv2.seamlessClone(warped, target, mask.copy(), center, cv2.NORMAL_CLONE)
    # Poisson-only blending can erase source appearance, especially in tiny faces.
    # Retain the source pixels, apply bounded illumination correction, and feather
    # only the mask boundary. The mask's interior remains fully opaque.
    inside = mask > 127
    source_pixels = warped[inside].astype(np.float32)
    target_pixels = target[inside].astype(np.float32)
    offset = np.clip(np.median(target_pixels, axis=0) - np.median(source_pixels, axis=0), -24, 24)
    corrected = np.clip(warped.astype(np.float32) + offset * .5, 0, 255)
    content = .85 * corrected + .15 * poisson.astype(np.float32)
    feather_width = max(1.0, min(bw, bh) * .08)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.clip(distance / feather_width, 0, 1)[..., None]
    result = np.clip(content * alpha + target.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
    return result, {'source_landmarks': source_points.tolist(), 'target_landmarks': target_points.tolist(),
                    'affine_matrix': matrix.tolist(), 'blend_box': [bx, by, bw, bh],
                    'blend_method': 'source-preserving-feather-with-poisson', 'source_weight': .85}


def read_working_image(path, max_edge=MAX_EDGE):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('This face-swapping baseline accepts images only, not video or audio')
    original = [image.shape[1], image.shape[0]]
    scale = min(1.0, max_edge / max(image.shape[:2])) if max_edge else 1.0
    if scale < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return image, original


def lock_scene(target,result,face):
    if result.shape!=target.shape:
        raise ValueError('Backend changed target dimensions')
    x,y,w,h=face[:4]
    x0,y0=max(0,int(np.floor(x))),max(0,int(np.floor(y)))
    x1,y1=min(target.shape[1],int(np.ceil(x+w))),min(target.shape[0],int(np.ceil(y+h)))
    locked=target.copy()
    locked[y0:y1,x0:x1]=result[y0:y1,x0:x1]
    return locked,{'native_size':[target.shape[1],target.shape[0]],'editable_face_box':[x0,y0,x1-x0,y1-y0],
                   'outside_face_pixels_preserved':True,'posture_and_background_resampled':False}


class ImageFaceSwapper:
    name = 'face_swapping'
    backend = 'yunet-affine-source-preserving-v2'
    generative = False

    def protect_scene(self, target, result, face, metadata):
        return lock_scene(target, result, face)

    def transform(self, reference, target, source_face, target_face, context):
        return blend_face(reference, target, source_face, target_face)

    def process(self, inputs: dict[str, Artifact], context: Context):
        if 'target' not in inputs:
            raise ValueError('Face swapping requires a target image')
        reference, source_size = read_working_image(inputs['reference'].path)
        target, target_size = read_working_image(inputs['target'].path, max_edge=None)
        detector = LandmarkDetector(model_path())
        source_face = detector.single_face(reference, 'Reference image')
        target_face = detector.single_face(target, 'Target image')
        result, metadata = self.transform(reference, target, source_face, target_face, context)
        warnings = []
        for role, detected in [('reference', source_face), ('target', target_face)]:
            eye_distance = float(np.linalg.norm(detected[4:6] - detected[6:8]))
            if eye_distance < 24:
                warnings.append(f'{role}: only {eye_distance:.1f} pixels between eyes; use a higher-resolution original for better identity detail')
        source_pose = (source_face[8] - (source_face[4] + source_face[6]) / 2) / max(1, np.linalg.norm(source_face[4:6] - source_face[6:8]))
        target_pose = (target_face[8] - (target_face[4] + target_face[6]) / 2) / max(1, np.linalg.norm(target_face[4:6] - target_face[6:8]))
        if not self.generative and abs(source_pose - target_pose) > .15:
            warnings.append('Reference and target have different face angles; this 2D baseline cannot reconstruct hidden facial features')
        detection = {}
        for role, detected in [('reference', source_face), ('target', target_face)]:
            detection[role] = {'box': detected[:4].tolist(), 'confidence': float(detected[14])}
        # Final guard: prevent a backend from changing pose, framing, clothing,
        # or scene pixels outside the detected face rectangle.
        result,scene=self.protect_scene(target,result,target_face,metadata)
        metadata['scene_integrity']=scene
        if 'quality_report' in metadata:
            metadata['quality_report']['scene_integrity']=scene
        labeled=mark_footer(result)
        output = context.output_dir / 'swapped.png' 
        if not cv2.imwrite(str(output), labeled):
            raise RuntimeError('Failed to save swapped image')
        metadata.update(backend=self.backend, detection=detection, warnings=warnings, generative=self.generative, device='cpu',
                        model_sha256=MODEL_SHA256, source_original_size=source_size, target_original_size=target_size,
                        output_size=[labeled.shape[1], labeled.shape[0]], content_size=target_size,
                        disclosure="added_footer", max_reference_working_edge=MAX_EDGE)
        panels = []
        for image, detected, title in [(reference, source_face, 'REFERENCE'), (target, target_face, 'TARGET'), (result, target_face, 'RESULT')]:
            x, y, w, h = detected[:4]
            x0, y0 = max(0, int(x - .15*w)), max(0, int(y - .15*h))
            x1, y1 = min(image.shape[1], int(x + 1.15*w)), min(image.shape[0], int(y + 1.15*h))
            crop = image[y0:y1, x0:x1]
            panel = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(panel, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
            panels.append(panel)
        diagnostic = context.output_dir / 'face-diagnostics.png'
        diagnostic_image = cv2.copyMakeBorder(np.hstack(panels), 0, 36, 0, 0, cv2.BORDER_CONSTANT)
        if not cv2.imwrite(str(diagnostic), mark(diagnostic_image)):
            raise RuntimeError('Failed to save face diagnostics')
        return {'rendered': Artifact(output, 'synthetic_image', metadata),
                'diagnostics': Artifact(diagnostic, 'synthetic_diagnostics') }
