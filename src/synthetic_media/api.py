"""Local research API. Bind to loopback; authentication is required before remote deployment."""
import json
from pathlib import Path
from uuid import uuid4
import os
import threading
import shutil
from typing import Literal
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from .pipeline import registry, run
from .service_errors import SharedServiceError, public_manifest

ROOT = Path(os.environ.get("SYNTHETIC_MEDIA_DATA", ".lab")).resolve()
MAX_BYTES = 64 * 1024 * 1024
JOB_LOCK = threading.Lock()
app = FastAPI(title="Synthetic Media Lab", version="0.1.0")

@app.get('/')
def studio():
    return FileResponse(Path(__file__).with_name('studio.html'))


@app.post('/edits/analyze')
async def analyze_edit(reference:UploadFile=File(...),target:UploadFile=File(...),
                       authorized:bool=Form(...),license_note:str=Form(...),instruction:str=Form(...),
                       scope:Literal['auto','face','head','person']=Form('auto'),
                       backend:Literal['auto','shared','shared-head','shared-flux']=Form('auto'),allow_external:bool=Form(False)):
    return await process_uploads(reference,target,authorized,license_note,'instruction_analysis',
                                 backend,False,instruction,scope,allow_external)


@app.post('/edits/generate')
async def generate_edit(reference:UploadFile=File(...),target:UploadFile=File(...),
                        authorized:bool=Form(...),license_note:str=Form(...),instruction:str=Form(...),
                        scope:Literal['auto','face','head','person']=Form('auto'),
                        backend:Literal['auto','shared','shared-head','shared-flux']=Form('auto'),allow_external:bool=Form(False)):
    return await process_uploads(reference,target,authorized,license_note,'instruction_edit',
                                 backend,False,instruction,scope,allow_external)

@app.get("/capabilities")
def capabilities():
    return registry().capabilities()

@app.post("/jobs")
async def create_job(reference: UploadFile = File(...), authorized: bool = Form(...), license_note: str = Form(...)):
    return await process_uploads(reference, None, authorized, license_note, "face_processing")


@app.post("/jobs/swap")
async def swap_job(reference: UploadFile = File(...), target: UploadFile = File(...),
                   authorized: bool = Form(...), license_note: str = Form(...),
                   backend: Literal["auto", "neural", "baseline", "hyperswap", "best", "facefusion", "remote"] = Form("auto"), allow_low_detail: bool = Form(False)):
    """Swap one face between images. Authorization must cover both media files."""
    return await process_uploads(reference, target, authorized, license_note, "face_swapping", backend, allow_low_detail)


async def process_uploads(reference, target, authorized, license_note, stage, backend="auto", allow_low_detail=False,instruction='',scope='auto',allow_external=False):
    files = [reference] + ([target] if target is not None else [])
    paths = []
    locked = False
    try:
        if not authorized or not license_note.strip():
            raise HTTPException(422, "Authorization and a license/consent note covering all inputs are required")
        locked = JOB_LOCK.acquire(blocking=False)
        if not locked:
            raise HTTPException(429, "Another job is processing; retry when it finishes")
        uploads = ROOT / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        for upload in files:
            path = uploads / uuid4().hex
            paths.append(path)
            size = 0
            with path.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Each upload must be at most 64 MiB")
                    stream.write(chunk)
        if stage in ('instruction_analysis','instruction_edit'):
            from .instructions import analyze_job,execute_job
            function=analyze_job if stage=='instruction_analysis' else execute_job
            return await run_in_threadpool(function,paths[0],paths[1],ROOT/'jobs',authorized,license_note,
                                          instruction,scope,backend,allow_external)
        return await run_in_threadpool(run, paths[0], ROOT / "jobs", authorized, license_note,
                                      stage, paths[1] if target is not None else None, backend, allow_low_detail)
    except SharedServiceError as error:
        raise HTTPException(error.status, {'code':error.code,'message':str(error)}) from error
    except (ValueError, RuntimeError, cv2.error) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        if locked:
            JOB_LOCK.release()
        for upload in files:
            await upload.close()

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return read_manifest(job_id)


def read_manifest(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(404, "Job not found")
    path = ROOT / "jobs" / job_id / "manifest.json"
    if not path.is_file():
        raise HTTPException(404, "Job not found")
    return public_manifest(json.loads(path.read_text()))

@app.get("/jobs/{job_id}/preview")
def preview(job_id: str):
    manifest = read_manifest(job_id)
    name = manifest["outputs"].get("labeled_preview")
    if not name:
        raise HTTPException(404, "Preview unavailable")
    return FileResponse(ROOT / "jobs" / job_id / name)


@app.get("/jobs/{job_id}/diagnostics")
def diagnostics(job_id: str):
    manifest = read_manifest(job_id)
    name = manifest['outputs'].get('face_diagnostics')
    if not name:
        raise HTTPException(404, 'Diagnostics unavailable for this job')
    return FileResponse(ROOT / 'jobs' / job_id / name)


@app.get('/backends')
def backends():
    from .neural import status
    from .quality_models import available
    from .masking import masks_available
    from .upstream import status as upstream_status
    from .remote import status as remote_status
    from .shared import SPACES,login_available
    neural = status()
    return {'selected': registry().require('face_swapping').backend, 'neural': neural, 'semantic_masks': masks_available(), 'hyperswap_installed': available('hyperswap_1a_256'),'upstream':upstream_status(), 'remote':remote_status(),
            'shared':{'services':SPACES,'login_available':login_available(),'hosting_required':False,
                      'free_hardware_verified_per_request':True,'visual_quality_approved':False},
            'resource_policy': {'cpu_only': True, 'active_jobs': 1, 'worker_rss_limit_mib': 1200}}


@app.get('/studio/status')
def studio_status():
    from .shared import login_available
    info={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:','MemAvailable:')):
            key,value,*_=line.split();info[key.rstrip(':')]=round(int(value)/1024)
    return {'memory_mib':info,'free_disk_gib':round(shutil.disk_usage(ROOT.parent).free/1024**3,1),
            'shared_login_available':login_available(),'local_generation':'CPU face swapping',
            'shared_generation':'Public services; free quota and availability checked per request',
            'foundation_model_training':False}


@app.get('/jobs')
def recent_jobs():
    manifests=sorted((ROOT/'jobs').glob('*/manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]
    items=[]
    for path in manifests:
        try:
            data=public_manifest(json.loads(path.read_text()))
            items.append({k:data.get(k) for k in ['job_id','stage','status','links','quality','error']})
        except (ValueError,OSError):continue
    return items
