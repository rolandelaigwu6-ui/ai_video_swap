import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .components import Artifact, Context, Registry
from .media import FaceProcessor, render_labeled
from .swapping import ImageFaceSwapper, model_path


def registry(backend="auto"):
    result = Registry()
    result.register(FaceProcessor())
    if model_path().is_file():
        from .neural import NeuralFaceSwapper, HyperSwapFaceSwapper, BestNeuralFaceSwapper, status
        from .quality_models import available as quality_available
        if backend not in ('auto', 'neural', 'baseline', 'hyperswap', 'best', 'facefusion', 'remote', 'shared', 'shared-head','shared-flux'):
            raise ValueError('Backend must be auto, neural, hyperswap, best, facefusion, remote, or baseline')
        available = status()['available']
        if backend == 'neural' and not available:
            raise ValueError('Neural backend is not installed; see /backends for details')
        if backend in ('shared','shared-head','shared-flux'):
            from .shared import SharedInstructionEditor
            result.register(SharedInstructionEditor(backend))
        elif backend == 'remote':
            from .remote import RemoteIdentityEditor, status as remote_status
            if not remote_status()['available']:raise ValueError(remote_status()['reason'])
            result.register(RemoteIdentityEditor())
        elif backend == 'facefusion':
            from .upstream import FaceFusionSwapper,status as upstream_status
            if not upstream_status()['available']:raise ValueError(upstream_status()['reason'])
            result.register(FaceFusionSwapper())
        elif backend == 'best' or (backend == 'auto' and available and quality_available('hyperswap_1a_256')):
            if not available or not quality_available('hyperswap_1a_256'):
                raise ValueError('Both neural model sets are required for best mode')
            result.register(BestNeuralFaceSwapper())
        elif backend == 'hyperswap':
            if not available or not quality_available('hyperswap_1a_256'):
                raise ValueError('HyperSwap models are not installed and verified')
            result.register(HyperSwapFaceSwapper())
        else:
            result.register(NeuralFaceSwapper() if available and backend != 'baseline' else ImageFaceSwapper())
    return result


def run(reference: Path, root: Path, authorized: bool, license_note: str, stage="face_processing", target: Path | None = None, backend="auto", allow_low_detail=False, instruction='', edit_scope='face'):
    if not authorized or not license_note.strip():
        raise ValueError("Authorization confirmation and a license/consent note are required")
    component = registry(backend).require(stage)
    reference = reference.resolve(strict=True)
    if not reference.is_file():
        raise ValueError("Reference must be a file")
    if stage == "face_swapping" and target is None:
        raise ValueError("Face swapping requires a target image")
    if target is not None:
        target = target.resolve(strict=True)
        if not target.is_file():
            raise ValueError("Target must be a file")
    job_id = uuid4().hex
    output = root / job_id
    output.mkdir(parents=True)
    manifest = {"job_id": job_id, "status": "running", "stage": stage, "synthetic": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "authorization": {"attested": True, "note": license_note},
                "source_sha256": digest(reference), "outputs": {},
                "links": {"manifest": f"/jobs/{job_id}", "preview": f"/jobs/{job_id}/preview"}}
    path = output / "manifest.json"
    try:
        inputs = {"reference": Artifact(reference, "reference")}
        if target is not None:
            inputs["target"] = Artifact(target, "target")
            manifest["target_sha256"] = digest(target)
        results = component.process(inputs, Context(job_id, output, allow_low_detail=allow_low_detail, license_note=license_note, instruction=instruction, edit_scope=edit_scope))
        manifest["analysis"] = {key: value.metadata for key, value in results.items()}
        labeled = results["rendered"].path if "rendered" in results else render_labeled(reference, output / "labeled")
        manifest["outputs"] = {"labeled_preview": labeled.name}
        if 'diagnostics' in results:
            manifest['outputs']['face_diagnostics'] = results['diagnostics'].path.name
            manifest['links']['diagnostics'] = f'/jobs/{job_id}/diagnostics'
        if 'rendered' in results and 'quality_report' in results['rendered'].metadata:
            manifest['quality'] = results['rendered'].metadata['quality_report']
        quality=manifest.get('quality')
        pending_review=quality is not None and (
            quality.get('status')=='needs_review' or quality.get('visual_acceptance')!='accepted')
        manifest["status"] = "needs_review" if pending_review else "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
