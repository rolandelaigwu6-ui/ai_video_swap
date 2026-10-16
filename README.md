# Synthetic Media Lab

Open **http://127.0.0.1:8000/** for the instruction studio: upload both images,
type a face/head/person replacement request, analyze the editable region, select
local or immediate shared-GPU generation, and inspect quality reports. See
[validation and immediate free alternatives](docs/STUDIO.md). Current portrait
experiments remain unapproved; successful inference is not visual acceptance.

An initial working slice of a modular, authorized synthetic-media research application. This version implements **face processing, image face swapping, and labeled media exports**. A research neural backend uses INSwapper/ArcFace when its optional models are installed; the classical baseline remains selectable. Voice cloning, reenactment, video face swapping, and lip synchronization remain future components.

## Run

Python 3.11+ is required. Installation downloads OpenCV and a packaged FFmpeg executable; no system FFmpeg installation is needed.

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/synthetic-media capabilities
.venv/bin/synthetic-media prepare reference.mp4 --authorized --license-note "My own recorded likeness and voice"
.venv/bin/uvicorn synthetic_media.api:app --host 127.0.0.1 --port 8000
```

The local API documentation is at http://127.0.0.1:8000/docs. POST `/jobs` accepts multipart fields `reference`, `authorized`, and `license_note`. Processing runs in a worker thread and the request returns when completed. GET `/jobs/{id}` returns its manifest; GET `/jobs/{id}/preview` downloads the labeled preview. Upload limit is 64 MiB per file. API input uploads are deleted after processing; face crops and results remain under `.lab/jobs`. Keep this local until authentication, resource isolation, and a durable queue are added.

## Implemented behavior

- Detect frontal faces in images, or sample video once per second; retain bounding boxes and face crops for downstream preparation.
- Export visibly labeled image/video previews and a manifest with authorization attestation, source SHA-256, processing status, and analysis results.
- Preserve video frame rate and frame count, encode H.264, retain audio when present, and pad odd dimensions for encoder compatibility.
- Report installed components honestly through the capabilities endpoint and CLI.

The detector is a CPU Haar cascade baseline: it can miss profiles and difficult lighting, does not produce identity embeddings, and does not establish ownership or verify consent. Face crops are intermediate reference assets; share the labeled previews for the research workflow. A preview is the original media with a synthetic-media research label, not a generated identity transformation. Variable-frame-rate inputs are normalized to the decoder's reported frame rate; exact original timestamps are not preserved.

## Component architecture and next steps

`components.py` defines `Artifact`, `Context`, and a `Component` protocol. Model adapters register by stage and pass typed artifact envelopes through `process(inputs, context)`. Unknown or unavailable components fail explicitly. `Context.device` provides a device-selection hook; the shipped detector runs on CPU and no GPU model is bundled.

Development order:

1. Face processing: replace/extend the baseline with landmarks, identity embeddings, alignment, and evaluation fixtures.
2. Face swapping: CPU landmark-and-blending image baseline implemented; next evaluate a licensed generative model.
3. Video consistency: tracking, face association, smoothing, scene-cut handling, and timestamp-aware decoding.
4. Voice cloning: licensed voice adapter with audio preparation and speech synthesis.
5. Lip synchronization: mouth-region synthesis with audio alignment.
6. Facial reenactment: expression and pose transfer.
7. Multimodal integration: dependency-aware job orchestration, cancellation, retries, isolated GPU workers.
8. Quality optimization: temporal/identity evaluation, audio drift checks, restoration, and resolution controls.

Model weights, model licenses, GPU runtime, and quality benchmarks must be selected and validated as each component is integrated. No unimplemented stage silently returns a simulated result.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover authorization enforcement, unavailable stages, image labeling and manifests, video frame rate/count plus retained audio, real YuNet inference on a blank image, controlled-landmark swapping, output provenance, upload limits, job serialization, native scene preservation, and upstream worker failures/timeouts. Generated fixtures validate processing contracts. User-supplied portraits are evaluated separately; previous trial outputs failed visual acceptance. Passing tests does not establish portrait quality.

## Image face swapping (second milestone)

A CPU baseline now swaps facial pixels between a reference image and a target image using a bundled, MIT-licensed YuNet detector (five landmarks), affine alignment, and Poisson blending. It requires exactly one detectable face in each image. This is a classical image-compositing baseline, not a generative identity model; strong pose/expression differences and occlusions can produce poor results. Video swapping is not supported yet.

```sh
.venv/bin/synthetic-media swap reference.jpg target.jpg --authorized --license-note "Permission for both people and both images"
```

Or use **POST /jobs/swap** at http://127.0.0.1:8000/docs. Upload both images, confirm authorization covering both inputs, and supply a consent/license note. A completed job returns `links.preview`, which can be appended to `http://127.0.0.1:8000`; paste the actual returned link rather than a placeholder. GET job results also include the links.

Restart Uvicorn after updating to expose the new endpoint. The bundled detector is ~228 KiB, with no Torch/CUDA dependency. Model provenance, checksum, and license are packaged under `src/synthetic_media/models/yunet`.

Swaps retain the target at its native resolution. The reference working edge is bounded to 1280 pixels; detection works on bounded images and neural models work on face crops. A final scene guard copies only the detected face rectangle into the original target: pixels outside that rectangle remain exactly unchanged. The synthetic disclosure is added in a separate footer below the content. The manifest records the editable rectangle, original content size, and exported size. The API allows one active job per process and 64 MiB per uploaded file; run **one Uvicorn worker**. OpenCV's decoded-image limit is 16 million pixels. CLI processes do not share the API lock. Resource preferences and measurements are recorded in PROJECT_NOTES.md.

### Source-preserving blending update

The v2 image swap keeps source facial pixels with bounded illumination correction, feathered edges, and a small Poisson contribution. Small inputs get an additional upscaled detection attempt at the same confidence threshold; this improves detection opportunities but cannot recover missing detail. Job analysis reports detected face boxes/confidence, small eye-distance warnings, and a coarse face-angle mismatch warning. `links.diagnostics` opens a labeled reference/target/result face comparison at GET `/jobs/{job_id}/diagnostics`. Previously generated outputs are not rewritten. Face-angle differences still require a generative/3D approach for substantially better reconstruction.


## Neural identity transfer

The project now includes an optional CPU ONNX adapter using the INSwapper-128 FP16 generator and compatible ArcFace W600K R50 identity embeddings, based on published InsightFace model contracts and the model pairing used by FaceFusion. This replaces pixel warping for neural swaps. See [the integration research and resource policy](docs/NEURAL_BACKEND.md) for sources, model licenses, download sizes, setup, and limitations.

`GET /backends` reports whether neural inference is installed and selected. In POST `/jobs/swap`, use `backend=neural` to require it explicitly; unavailable models or inference errors return an error instead of pretending the baseline is neural output. The neural backend is for **non-commercial research only** under the supplied pretrained model terms. Generated images remain visibly labeled. Neither neural integration nor passing contract tests establishes flawless portrait quality; inspect the comparison on your actual inputs.

### Quality evaluation and automatic candidate comparison

When the optional quality models are installed, `backend=best` compares INSwapper and HyperSwap sequentially and records which candidate was selected and why. `auto` selects this mode when both are installed. Learned BiSeNet/XSeg masks aim to preserve target accessories and scene pixels; a semantic segmentation failure is reported and uses learned XSeg visibility rather than silently erasing the detected face. These learned components can fail on unfamiliar poses.

```sh
.venv/bin/python -m synthetic_media.quality_models bisenet_resnet_34 xseg_1 hyperswap_1a_256
```

A job's processing status can now be `needs_review`. Its `quality` report includes identity transfer, face-detail measurements, output redetection and landmark drift. Even a technical pass remains visually unverified; no metric or output is certified flawless. Models have individual MIT/GPL/ResearchRAIL/research-only terms recorded in their receipts and in the research notes. The full application roadmap is still unfinished: image quality evaluation does not implement video swapping, voice cloning, reenactment, or lip sync.

Small detectable reference faces are processed and receive detail warnings. Detail measurements do not block a swap or certify fidelity. The older `allow_low_detail` parameter remains accepted for compatibility.

### Optional remote generation

`backend=remote` connects to your explicitly configured, password-protected Hugging Face Space. The source-only ZeroGPU package uses a pinned reference-conditioned image editor, with semantic head/neck compositing to retain original scene pixels. No remote service is active until deployed. No paid fallback is implemented. See [free hosting setup and eligibility](docs/FREE_HOSTING.md); the user's current $0 budget and lack of an eligible account prevent immediate activation. GPU inference and visual acceptance remain untested.

### Official FaceFusion adapter

`backend=facefusion` runs the vendored official CLI in its separate environment, using HyperSwap, upstream face analysis, alignment, and occlusion masking. Its upstream checks remain enabled. Model modules are loaded sequentially and the complete process tree is limited to 1200 MiB resident memory with a 900-second timeout. Supporting models are verified against upstream hashes, and compatible weights are shared with the other adapters. See [upstream integration](docs/UPSTREAM_BACKEND.md). Installation or a successful CLI exit alone does not establish visual acceptance.
