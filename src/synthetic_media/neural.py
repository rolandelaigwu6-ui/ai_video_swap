"""Research-only INSwapper adapter using isolated ONNX CPU inference.

Model I/O and canonical alignment follow the InsightFace reference implementation.
No FaceFusion application code is copied. See docs/NEURAL_BACKEND.md for sources.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import cv2
import numpy as np
from .swapping import ImageFaceSwapper


def model_dir():
    return Path(os.environ.get('SYNTHETIC_MEDIA_NEURAL_MODELS', str(Path(__file__).resolve().parents[2] / '.lab/models/neural')))


def status():
    root = model_dir()
    ready = root / 'ready.json'
    if not ready.is_file():
        return {'available': False, 'reason': 'Neural models have not been installed and verified'}
    if not importlib.util.find_spec('onnxruntime') or not importlib.util.find_spec('onnx'):
        return {'available': False, 'reason': 'Install the neural optional dependencies'}
    try:
        data = json.loads(ready.read_text())
        for name, info in data['models'].items():
            if Path(name).name != name or not (root / name).is_file() or (root / name).stat().st_size != info['size']:
                raise ValueError('Missing or incomplete neural model')
        if set(data['models']) != {'inswapper_128_fp16.onnx', 'arcface_w600k_r50.onnx'}:
            raise ValueError('Incomplete neural model configuration')
        return {'available': True, 'backend': 'inswapper-128-fp16-cpu', 'license': 'non-commercial research', 'models': data['models']}
    except (KeyError, ValueError, OSError) as error:
        return {'available': False, 'reason': str(error)}


def verify_models():
    configuration = status()
    if not configuration['available']:
        raise ValueError(configuration['reason'])
    for name, info in configuration['models'].items():
        with (model_dir() / name).open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != info['sha256']:
            raise ValueError(f'Neural model checksum mismatch: {name}')
    return configuration


def align(image, landmarks, size):
    # InsightFace canonical ArcFace landmarks. INSwapper uses 112-space + x offset 8.
    canonical = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                          [41.5493,92.3655],[70.7299,92.2041]], np.float32)
    if size in (128, 256):
        canonical[:, 0] += 8
        canonical *= size / 128
    elif size != 112:
        raise ValueError("Unsupported alignment size")
    # Least-squares similarity fit over all five points; no landmark is discarded.
    source = np.asarray(landmarks, np.float64)
    source_mean, dest_mean = source.mean(0), canonical.mean(0)
    centered, desired = source - source_mean, canonical - dest_mean
    u, singular, vt = np.linalg.svd(centered.T @ desired)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
        singular[-1] *= -1
    scale = singular.sum() / max(float((centered**2).sum()), 1e-8)
    linear = scale * rotation
    matrix = np.column_stack((linear, dest_mean - linear @ source_mean)).astype(np.float32)
    return cv2.warpAffine(image, matrix, (size, size)), matrix


def run_worker(mode, model, folder, timeout=300, rss_limit_mib=1200):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([sys.executable, '-m', 'synthetic_media.neural_worker', mode,
                                    str(model.resolve()), str(folder.resolve())], env=env, stdout=log, stderr=log)
        started = time.monotonic()
        peak = 0
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise RuntimeError(f'Neural {mode} timed out after {timeout} seconds')
                try:
                    stats = Path(f'/proc/{process.pid}/status').read_text()
                    rss = next((int(line.split()[1]) / 1024 for line in stats.splitlines() if line.startswith('VmRSS:')), 0)
                    peak = max(peak, rss)
                    if rss > rss_limit_mib:
                        raise RuntimeError(f'Neural {mode} exceeded the {rss_limit_mib} MiB worker RAM budget')
                except FileNotFoundError:
                    pass
                time.sleep(.1)
            log.seek(0)
            message = log.read().decode(errors='replace')[-3000:]
            if process.returncode:
                raise RuntimeError(f'Neural {mode} failed: {message}')
            return {'stage': mode, 'observed_peak_rss_mib': round(peak, 1), 'worker_report': message.strip()}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def paste_prediction(target, prediction, matrix, face_mask=None):
    crop = np.clip(prediction[0].transpose(1,2,0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
    # Feather the aligned crop perimeter; source identity comes from the network.
    edge = crop.shape[0]
    mask = np.zeros((edge,edge), np.float32)
    border = max(2, edge//16)
    mask[border:-border, border:-border] = 1
    mask = cv2.GaussianBlur(mask, (2*border+1,2*border+1), 0)
    if face_mask is not None:
        if face_mask.shape != (edge,edge) or not np.isfinite(face_mask).all():
            raise ValueError('Invalid learned face mask')
        mask = np.minimum(mask, face_mask.clip(0,1))
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0,0],[edge,0],[edge,edge],[0,edge]],np.float32)
    mapped = cv2.transform(corners[None],inverse)[0]
    x0,y0 = np.maximum(np.floor(mapped.min(0)).astype(int),0)
    x1,y1 = np.minimum(np.ceil(mapped.max(0)).astype(int),[target.shape[1],target.shape[0]])
    if x1<=x0 or y1<=y0:
        raise ValueError('Generated face is outside target image')
    local = inverse.copy()
    local[:,2] -= [x0,y0]
    roi_size = (int(x1-x0),int(y1-y0))
    alpha = cv2.warpAffine(mask,local,roi_size).clip(0,1)[...,None]
    generated = cv2.warpAffine(crop,local,roi_size)
    original = target[y0:y1,x0:x1]
    blended = np.clip(original.astype(np.float32)*(1-alpha)+generated.astype(np.float32)*alpha,0,255).astype(np.uint8)
    # Copy original bytes wherever alpha is zero; never resample the scene.
    result=target.copy()
    result[y0:y1,x0:x1]=np.where(alpha>0,blended,original)
    return result



class NeuralFaceSwapper(ImageFaceSwapper):
    backend = 'inswapper-128-fp16-cpu'
    generative = True
    generator = 'inswapper'

    def transform(self, reference, target, source_face, target_face, context):
        from .masking import masks_available, semantic_mask
        from .quality import assess
        from .quality_models import model_info, root
        from .swapping import LandmarkDetector, model_path, lock_scene
        configuration = verify_models()
        size = 256 if self.generator == 'hyperswap' else 128
        source, _ = align(reference, source_face[4:14].reshape(5,2), 112)
        crop, matrix = align(target, target_face[4:14].reshape(5,2), size)
        measurements = []
        extra_models = {}
        with tempfile.TemporaryDirectory(prefix='neural-', dir=context.output_dir) as directory:
            folder = Path(directory)
            np.save(folder / 'source.npy', ((source[:,:,::-1].astype(np.float32)/127.5-1).transpose(2,0,1))[None])
            target_tensor = (crop[:,:,::-1].astype(np.float32)/255).transpose(2,0,1)[None]
            np.save(folder / 'target.npy', target_tensor*2-1 if size==256 else target_tensor)
            recognizer = model_dir() / 'arcface_w600k_r50.onnx'
            measurements.append(run_worker('embed', recognizer, folder))
            reference_embedding = np.load(folder/'embedding.npy',allow_pickle=False).copy()
            if self.generator == 'hyperswap':
                extra_models['generator'] = model_info('hyperswap_1a_256',verify=True)
                measurements.append(run_worker('hyperswap',root()/'hyperswap_1a_256.onnx',folder))
            else:
                measurements.append(run_worker('map', model_dir() / 'inswapper_128_fp16.onnx', folder))
                measurements.append(run_worker('swap', model_dir() / 'inswapper_128_fp16.onnx', folder))
            (context.output_dir/'inference-metrics.json').write_text(json.dumps(measurements,indent=2))
            prediction = np.load(folder / 'prediction.npy', allow_pickle=False)
            face_mask, mask_metadata = None, {'method':'box-only', 'limitations':'No learned occlusion or semantic mask installed'}
            if masks_available():
                generated = np.clip(prediction[0].transpose(1,2,0)[:,:,::-1]*255,0,255).astype(np.uint8)
                face_mask, mask_metadata = semantic_mask(crop,generated,folder,run_worker)
            result = paste_prediction(target,prediction,matrix,face_mask)
            result,_ = lock_scene(target,result,target_face)
            evaluated = []
            for image in (target,result):
                aligned,_=align(image,target_face[4:14].reshape(5,2),112)
                np.save(folder/'source.npy',((aligned[:,:,::-1].astype(np.float32)/127.5-1).transpose(2,0,1))[None])
                measurements.append(run_worker('embed',recognizer,folder))
                evaluated.append(np.load(folder/'embedding.npy',allow_pickle=False).copy())
            try:
                detected = LandmarkDetector(model_path()).single_face(result,'Generated image')
            except ValueError:
                detected = None
            report = assess(reference,target,result,source_face,target_face,reference_embedding,evaluated[0],evaluated[1],detected)
            if mask_metadata.get('limitations'):
                report['issues'].extend(mask_metadata['limitations'])
                report['status']='needs_review'
            if face_mask is None:
                report['issues'].append('Learned face-region and occlusion masks are missing')
                report['status']='needs_review'
        return result, {'backend': self.backend, 'generative': True,
                'neural_models': configuration['models'], 'additional_models':extra_models,
                'neural_license': configuration['license'], 'quality_report':report,
                'compositing':mask_metadata, 'aligned_face_size': size, 'worker_measurements': measurements,
                'source_landmarks': source_face[4:14].reshape(5,2).tolist(),
                'target_landmarks': target_face[4:14].reshape(5,2).tolist(), 'affine_matrix': matrix.tolist()}


class HyperSwapFaceSwapper(NeuralFaceSwapper):
    backend = 'hyperswap-1a-256-cpu'
    generator = 'hyperswap'


def candidate_rank(metadata):
    report = metadata['quality_report']
    # Prioritize failed technical checks, then identity with a small drift penalty.
    drift = report.get('normalized_landmark_drift') or 0
    return (-len(report.get('issues',[])), report['identity_cosine'] - .1*drift)


class BestNeuralFaceSwapper(ImageFaceSwapper):
    """Sequential neural comparison; selection is diagnostic, not visual approval."""
    backend = 'neural-auto-comparison'
    generative = True

    def transform(self, reference, target, source_face, target_face, context):
        candidates = [NeuralFaceSwapper(),HyperSwapFaceSwapper()]
        evaluated, failures = [], []
        for candidate in candidates:
            try:
                result,metadata=candidate.transform(reference,target,source_face,target_face,context)
                evaluated.append((result,metadata))
            except (ValueError,RuntimeError,cv2.error) as error:
                failures.append({'backend':candidate.backend,'error':str(error)})
        if not evaluated:
            raise RuntimeError('All neural candidates failed: '+json.dumps(failures))
        chosen=max(evaluated,key=lambda item:candidate_rank(item[1]))
        metadata=dict(chosen[1])
        metadata['selected_backend']=metadata['backend']
        metadata['selection_policy']='Fewest flagged issues, then identity cosine minus 0.1 times normalized landmark drift; not perceptual acceptance'
        metadata['candidate_evaluations']=[{'backend':item[1]['backend'],'quality_report':item[1]['quality_report'],
                'compositing':item[1]['compositing'],'worker_measurements':item[1]['worker_measurements']} for item in evaluated]
        metadata['failed_candidates']=failures
        return chosen[0],metadata
