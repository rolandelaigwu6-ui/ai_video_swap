# Official FaceFusion integration

Source: https://github.com/facefusion/facefusion (3.9.1).
The pinned Git commit and downloaded model SHA-256/CRC32 values are stored in
`vendor/facefusion/installed-models.json` after successful installation.
Upstream source remains unmodified. Its OpenRAIL-AS license and individual
model licenses remain applicable; this adapter does not grant additional rights.

The isolated upstream environment avoids dependency conflicts with the API.
FFmpeg/FFprobe are stored under `vendor/ffmpeg`, with their release digest and
origin in `ORIGIN.json`. Existing compatible model weights are symlinked rather
than duplicated. Supporting downloads are selected by `scripts/prepare_facefusion.py`
and verified by `scripts/download_facefusion.py`.

`scripts/facefusion_runtime.py` constrains ONNX CPU threads and session-cache
residency, disables graph optimization/CPU arenas, and returns freed heap pages
between modules. Upstream content analysis, its thresholds, model checks, face
analysis and generation remain enabled. There are no guard bypasses. A parent
watchdog measures the whole worker tree and terminates it above 1200 MiB RSS
or after 900 seconds. The API serializes jobs. Do not run simultaneous CLI jobs
on the development machine's 3.7 GiB RAM.

Select `backend=facefusion` in POST `/jobs/swap`, or:

```sh
.venv/bin/synthetic-media swap reference.png target.png --backend facefusion --authorized --license-note "Authorized reference and target media"
```

Odd dimensions are temporarily padded to even sizes, then cropped back without
resampling, because the upstream image exporter normalizes dimensions.
The target supplies pose, expression and scene. A final native-resolution guard
retains all original target pixels outside its detected face rectangle. A
synthetic disclosure is added below the image. Hair and body outside that
rectangle are retained; this is not full-person generation. This restriction
protects the scene but does not prove that the generated face is natural.
ArcFace similarity, landmark drift and detail reports are diagnostics. Only
inspection of real output can establish whether a particular result is acceptable.
The app does not certify flawless results.

## Current portrait evaluation

The supplied 113x282 reference / 423x321 target trial completed in 67.66 seconds
at 879.9 MiB observed worker-tree peak RSS. Saved content retained every pixel
outside the editable face rectangle and native dimensions. The result was
visually rejected: target complexion and facial hair still dominate. Its
identity cosine was .5571 against the reference, versus -.0932 for the original
target. The adapter is installed and callable but remains an explicit backend;
it has not met the user's identity-fidelity acceptance requirement.
