"""Bounded upstream FaceFusion CLI integration, retaining upstream safeguards."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import cv2
import numpy as np
from .swapping import ImageFaceSwapper, LandmarkDetector, model_path, lock_scene

ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'vendor/facefusion'


def status():
    ready=VENDOR/'installed-models.json'
    python=VENDOR/'.venv/bin/python'
    if not ready.is_file() or not python.is_file() or not (ROOT/'vendor/ffmpeg/ffprobe').is_file():
        return {'available':False,'reason':'Upstream environment/supporting models are not installed and verified'}
    return {'available':True,'backend':'facefusion-upstream','commit':json.loads(ready.read_text())['commit'],
            'safeguards':'upstream checks unmodified','cache_policy':'one model module at a time'}


def command(source,target,output,working):
    return [str(VENDOR/'.venv/bin/python'),str(ROOT/'scripts/facefusion_runtime.py'),'headless-run',
            '--source-paths',str(source),'--target-path',str(target),'--output-path',str(output),
            '--temp-path',str(working/'temp'),'--jobs-path',str(working/'jobs'),
            '--processors','face_swapper','--face-swapper-model','hyperswap_1a_256',
            '--face-swapper-pixel-boost','256x256','--face-swapper-weight','0.9',
            '--face-detector-model','yolo_face','--face-landmarker-model','peppa_wutz',
            '--face-selector-mode','one','--face-mask-types','box','occlusion',
            '--execution-providers','cpu','--execution-thread-count','1',
            '--output-image-quality','100','--output-image-scale','1.0',
            '--workflow-strategy','memory','--video-memory-strategy','strict','--log-level','info']


def prepare_target(target):
    # Upstream rounds image dimensions to even numbers. Pad without resampling
    # and remove only that padding afterward, retaining original coordinates.
    height,width=target.shape[:2]
    return cv2.copyMakeBorder(target,0,height%2,0,width%2,cv2.BORDER_REPLICATE)


def restore_target(result,target):
    padded_shape=prepare_target(target).shape
    if result.shape!=padded_shape:
        raise ValueError('Upstream changed padded target dimensions')
    return result[:target.shape[0],:target.shape[1]].copy()


def tree_rss(pid):
    total,queue,seen=0,[pid],set()
    while queue:
        current=queue.pop()
        if current in seen:continue
        seen.add(current)
        try:
            lines=Path(f'/proc/{current}/status').read_text().splitlines()
            total+=next((int(line.split()[1])/1024 for line in lines if line.startswith('VmRSS:')),0)
            queue.extend(int(item) for item in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):pass
    return total


def execute(args,log_path,timeout=900,rss_limit_mib=1200,working_directory=None):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic();peak=0
    with log_path.open('wb') as log:
        process=subprocess.Popen(args,cwd=working_directory or VENDOR,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                rss=tree_rss(process.pid);peak=max(peak,rss)
                if rss>rss_limit_mib:raise RuntimeError(f'Upstream worker exceeded {rss_limit_mib} MiB RAM budget')
                if time.monotonic()-started>timeout:raise RuntimeError('Upstream processing timed out')
                time.sleep(.1)
            if process.returncode:
                with log_path.open('rb') as stream:
                    stream.seek(max(0,log_path.stat().st_size-3000))
                    detail=stream.read().decode(errors='replace')
                raise RuntimeError(f'Upstream processor failed ({process.returncode}): {detail}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
    return {'observed_peak_process_tree_rss_mib':round(peak,1),'elapsed_seconds':round(time.monotonic()-started,2)}


class FaceFusionSwapper(ImageFaceSwapper):
    backend='facefusion-upstream'
    generative=True

    def transform(self,reference,target,source_face,target_face,context):
        from .neural import align,run_worker,model_dir
        from .quality import assess
        installed=status()
        if not installed['available']:raise ValueError(installed['reason'])
        with tempfile.TemporaryDirectory(prefix='upstream-',dir=context.output_dir) as directory:
            folder=Path(directory).resolve()
            source_path,target_path,output=folder/'reference.png',folder/'target.png',folder/'output.png'
            for path,image in [(source_path,reference),(target_path,prepare_target(target))]:
                if not cv2.imwrite(str(path),image):raise RuntimeError('Failed to prepare upstream input')
            measurement=execute(command(source_path,target_path,output,folder),context.output_dir/'upstream.log')
            result=cv2.imread(str(output))
            if result is None:raise RuntimeError('Upstream returned no decodable output')
            result=restore_target(result,target)
            result,scene=lock_scene(target,result,target_face)
            embeddings=[];evaluation_workers=[]
            for image,face in [(reference,source_face),(target,target_face),(result,target_face)]:
                crop,_=align(image,face[4:14].reshape(5,2),112)
                np.save(folder/'source.npy',((crop[:,:,::-1].astype(np.float32)/127.5-1).transpose(2,0,1))[None])
                evaluation_workers.append(run_worker('embed',model_dir()/'arcface_w600k_r50.onnx',folder))
                embeddings.append(np.load(folder/'embedding.npy',allow_pickle=False).copy())
            try:detected=LandmarkDetector(model_path()).single_face(result,'Generated image')
            except ValueError:detected=None
            quality=assess(reference,target,result,source_face,target_face,*embeddings,detected)
        return result,{'backend':self.backend,'generative':True,'upstream':installed,'upstream_measurements':measurement,
                       'quality_report':quality,'scene_integrity':scene,'evaluation_workers':evaluation_workers,
                       'generator':'hyperswap_1a_256','identity_weight':.9,'content_checks':'upstream retained'}
