# Validate the application now

Open http://127.0.0.1:8000/ (the studio), rather than `/docs` (API documentation).
If needed, start the local server from the project root:

```sh
cd /home/don1/ai_video_swap
.venv/bin/uvicorn synthetic_media.api:app --host 127.0.0.1 --port 8000 --workers 1
```

1. Choose the **reference identity** and **target scene** images.
2. Enter an instruction, such as:
   - `Swap just the face. Keep the target hair, expression and background.`
   - `Replace the head and hair with the reference identity. Keep the target pose and clothes.`
   - `Replace the whole person with the reference. Keep the target sitting posture and background.`
3. Confirm authorization and enter a consent/license note.
4. Click **Analyze images and instruction**. This runs locally. Inspect the pink
   editable region, inferred scope, detection measurements, input-detail warnings,
   execution plan and blockers. Unknown/conflicting requests require clarification.
5. Choose a generation engine. Generating on an external service requires the
   external-processing checkbox. No data is uploaded for local analysis.
6. Click **Generate synthetic image**. Inspect the labeled result and its identity,
   landmark-drift and scene-preservation reports. Rejected automatic checks are
   shown as rejected. A generated file never establishes flawless quality.

The history panel links to actual prior masks, output images, comparisons and
reports. Previous rejected experiments remain marked rejected. The status bar
shows available RAM, disk and local account-login availability.

## Instruction handling and scene analysis

This project integrates pretrained models; it has not trained a new foundation
model. The lightweight scope interpreter is explicitly rule-based and supports
face/head/person replacement phrases. It is not general language understanding.
YuNet measures face boxes/landmarks/confidence, BiSeNet segments head/neck, and
OpenCV PPHumanSeg segments the whole person. The latter is a verified 6,163,938-byte
Apache-2.0 model from OpenCV Zoo; its receipt is `.lab/models/person/receipt.json`.
It is installed on this machine. For another installation, use
`.venv/bin/python scripts/install_person_model.py` from the project root.

The **instruction editors receive the actual instruction text and both images**.
The local face engine and trained head-swap preset have narrower capabilities;
their descriptions and execution plans say so. Unsupported scope/backend
combinations are blocked rather than silently reduced to a face swap.

The final composite retains original target pixels outside the chosen mask and
keeps native dimensions. Semantic masks can be wrong: inspect them around hair,
hands, held objects and clothing. This constraint cannot establish exact identity
or preserve a new person's silhouette outside the original edit region.

## Immediate free alternatives

Using an existing demo is distinct from hosting a new Space. The account-age
restriction on free hosting does not require waiting to use existing demos:

- **Shared FLUX instruction editor**: `black-forest-labs/FLUX.2-klein-4B`.
  Supports the full instruction and both reference images. Anonymous free quota
  is limited; no automatic retry or paid fallback occurs.
- **Shared Qwen instruction editor**: `Qwen/Qwen-Image-Edit-2511`.
  Supports multi-image editing and the full instruction. Its GPU task duration
  requires free-account quota, so authenticate locally with a read-only token:

  ```sh
  cd /home/don1/ai_video_swap
  .venv/bin/hf auth login
  ```

  This stores a small credential file. It does not download AI weights or keep
  a model resident in local RAM. Do not put the token in chat or source files.
- **Trained head preset**: `linoyts/Flux2-Klein-Face-Swap`. Uses a 9B model and
  head-swap LoRA with a fixed prompt; it does not accept arbitrary appearance
  instructions. Intended only for head replacement.

The studio links to the public demos for browser use as well. Before API uploads,
the client verifies free ZeroGPU hardware and rejects credit-capable account tiers.
Queues, service changes, quota errors and failures remain possible. Downloading
large generation models locally is not part of this workflow.

Sources:
- https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511
- https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/spaces/linoyts/Flux2-Klein-Face-Swap
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg

## What has actually passed

Instruction routing/blocking, mask previews, local resource reporting, cloud
worker isolation and API contracts are tested separately from image quality.
Face-only and person masks were exercised on the supplied target at native
423x321 resolution. Public head and FLUX generation returned outputs, but both
failed reference-identity checks. No result from this image pair is approved.
Qwen generation requires account authentication and has not been validated here.
