# Neural face-swap integration research

The project previously used 2D pixel warping, which cannot reconstruct an identity at a new angle. The neural backend instead computes a 512-dimensional ArcFace identity embedding and conditions an INSwapper generator on that identity and an aligned target crop. It then blends the generated crop back into the target scene. This is the approach documented by InsightFace and used in FaceFusion's INSwapper integration. It does not guarantee flawless output or recover true details missing from compressed reference images.

## Primary sources reviewed

- InsightFace example and model requirements: https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
- InsightFace model I/O and latent mapping: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py
- ArcFace alignment template: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
- Code and model licensing: https://github.com/deepinsight/insightface#license
- FaceFusion swap models and licensing metadata: https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py
- FaceFusion model distribution: https://github.com/facefusion/facefusion-assets/releases/tag/models-3.0.0
- FaceFusion CPU execution and memory strategies: https://docs.facefusion.io/usage/cli-arguments/execution and https://docs.facefusion.io/usage/cli-arguments/memory
- MobileFaceSwap research alternative: https://arxiv.org/abs/2201.03808 (a different runtime/training stack; not integrated)

The implementation is an independent ONNX adapter to documented model contracts, not a copy of FaceFusion application code. It uses the InsightFace canonical numerical template and latent normalization. InsightFace source code is MIT licensed; its published pretrained models have separate non-commercial research restrictions. These weights are used for this project's authorized research workflow; commercial deployment needs separately cleared licenses. HyperSwap was also reviewed but is not bundled; FaceFusion identifies its license as ResearchRAIL. Do not infer model rights from application code licenses.

## Installation and resource policy

```sh
.venv/bin/pip install -e '.[neural]'
.venv/bin/python -m synthetic_media.neural_install
```

Only two weights are downloaded, not a full model bundle: INSwapper-128 FP16 (277,680,829 bytes) and the compatible ArcFace W600K R50 recognizer (174,388,474 bytes), totaling about 431 MiB. No Torch, CUDA, Gradio, or restoration model is needed. Downloads are verified using upstream CRC32 files and then recorded with SHA-256 in `.lab/models/neural/ready.json`; SHA-256 is rechecked before inference. Failed or partial downloads do not enable the backend.

Inference runs recognition, mapping extraction, and generation in three sequential short-lived processes. Each process has one inference/BLAS thread, CPU-only providers, disabled optional graph optimization and memory arenas, a 2 GiB address-space limit, a 1200 MiB resident-memory watchdog, and a 300-second timeout per stage. These limits can cause explicit failures on this small machine; they are not a guarantee of fitting every workload. Models and temporary identity embeddings are released after processing. Temporary tensors and embeddings are deleted; final manifests include model hashes and measured worker memory, never embeddings.

The API still admits one job per process. GET `/backends` reports the selected engine. POST `/jobs/swap` accepts `backend=auto|neural|baseline`; auto selects neural only when models/dependencies are present. An inference error never silently substitutes the classical result. CLI example:

```sh
.venv/bin/synthetic-media swap reference.jpg target.jpg --backend neural --authorized --license-note "Authorized research inputs"
```

Output remains visibly synthetic. YuNet supplies face landmarks. This integration does not include segmentation, occlusion masks, face restoration, or temporal tracking yet. In particular, hats, glasses, hands, extreme poses, and very small faces still require evaluation and further components.


## First portrait trial (2026-10-15)

Job `08d420ae1a27430d843895f3f25468ed` uses the user-provided reference and target image areas extracted from screenshots. End-to-end neural inference completed on CPU. Worker-reported peaks: recognition 269.0 MiB, mapping extraction 570.7 MiB, generation 831.2 MiB. The parent's sampled generator peak was 831.5 MiB. These are worker measurements, not total system RAM or a universal upper bound.

The reference-to-target ArcFace cosine similarity was 0.2877 before swapping and reference-to-generated-result similarity was 0.8629 afterward. This is a limited diagnostic using the same recognizer used to condition the generator; it is not independent evaluation, proof of identity, a visual-quality score, or a promise of similar performance on other people. Visual inspection remains required. The trial preserves the target pose/expression substantially better than the baseline, while the low-resolution inputs and crop mask still limit quality.

All 19 integration/contract tests passed with the optional runtime installed. No temporal consistency, lip sync, or voice model is included in this milestone.

## Quality work in progress (2026-10-16)

The user requested no intermediate images presented as finished work and no claims of flawlessness. Keep drafts internal and explicitly separate processing completion from visual acceptance. The current input screenshots contain only about 36–37 pixels between the detected eyes. Their missing native detail cannot be certified by upscaling or generated restoration.

Additional primary sources reviewed:

- FaceFusion learned face masking and class selection: https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py and https://docs.facefusion.io/usage/cli-arguments/face-masker
- Parser inference contract and MIT source: https://github.com/yakhyo/face-parsing
- XSeg architecture/training source and GPL license: https://github.com/iperov/DeepFaceLab
- HyperSwap training/export source and ResearchRAIL license: https://github.com/facefusion/facefusion-labs/tree/master/hyperswap
- Face restoration reference: https://github.com/TencentARC/GFPGAN (not installed; restoration can invent details and change identity)
- Identity-specific training alternative: https://github.com/deepfakes/faceswap (not installed; different training/runtime workflow)

Research model candidates: BiSeNet ResNet18 (53,205,356 bytes), BiSeNet ResNet34 (93,632,546 bytes), XSeg1 (70,324,286 bytes), HyperSwap1a256 (402,742,682 bytes). Keep receipts with origin, model license, exact size, upstream CRC, and SHA256. No application source from FaceFusion was copied; model I/O contracts and independently implemented compositing are used.

Initial ResNet18/XSeg masking trial **failed** identity diagnostics (0.4167 versus prior unmasked 0.8629). It must not be promoted as an improvement. Investigate and compare the default ResNet34 parser; do not assume a downloaded component improves a result. All learned models remain isolated under the existing memory watchdog.

Quality reports use explicit heuristic criteria: source/result embedding cosine >=0.65; gain >=0.15 when original identity differs; eye separation >=40 pixels; image-face Laplacian variance >=15; and generated-face redetection. These research heuristics are not calibrated acceptance thresholds, independent benchmarks, or a definition of flawlessness. A neural job with issues gets processing status `needs_review`, not a final-quality approval. Even a technical pass leaves `visual_acceptance=unverified` and `flawless_certified=false`.

### Subsequent candidate evaluations

All quality models were downloaded and verified. HyperSwap inference fits the original worker memory budget (observed ~795–811 MiB on these trials). The default parser works on the new frontal target but fails the earlier profile view. The compositor therefore applies semantic masks only with sufficient coverage; otherwise it explicitly reports learned XSeg-only masking, rather than erasing a detected face or silently pretending segmentation succeeded. This fallback is flagged for review.

New supplied files: target 423×321; reference 113×282. Reference face ~35×46 with 17.65 pixels between eyes; target eye separation ~63.16 pixels. For these inputs, masked HyperSwap source/result cosine was 0.5089 and masked INSwapper was 0.5354, versus source/original-target -0.0932. Both were correctly marked `needs_review`, below the experimental identity threshold and with inadequate reference detail. No new trial is certified flawless or presented as an accepted final image.

`backend=best` runs INSwapper and HyperSwap sequentially, records their reports/failures, and chooses the candidate with fewer flagged issues, then higher identity cosine minus a small normalized landmark-drift penalty. `auto` uses this comparison when both model sets are installed; explicit `neural`, `hyperswap`, or `baseline` remain selectable. This selection policy is heuristic and does not establish visual approval. Single-candidate memory limits remain unchanged; execution time increases because both models run. Model errors are recorded; if all candidates fail the job fails.

### Visual review of the new pair

The selected INSwapper result was visually rejected: the target complexion and facial-hair appearance remain dominant, so improved cosine similarity alone did not satisfy the user's requirement that the reference person appear naturally in the target shot. This review is recorded in the affected trial manifests. Do not use passing contract tests or a generator's output file as evidence that visual fidelity has been achieved.

Additional identity-conditioned generation sources reviewed: https://github.com/instantX-research/InstantID, https://github.com/ToTheBeginning/PuLID, and https://github.com/TencentARC/PhotoMaker. These are diffusion-generation/customization approaches with different base models, runtimes and resource requirements; they are not drop-in guarantees of a faithful swap. They have not been installed. Evaluate them separately on suitable hardware and with model-specific licensing rather than loading an unmeasured large stack onto this 3.7GiB-RAM machine. A high-detail native reference is still required for credible fidelity evaluation.

Strict preflight now rejects reference eye separation below 40 pixels before model inference. This policy implements the user's request not to pass off inadequate outputs as completed fidelity work and avoids wasting CPU/RAM on known failing detail checks. Experimental opt-in remains available through `allow_low_detail` for researchers; it does not waive quality reporting. The threshold is a heuristic, not an assurance of flawless output at or above 40 pixels.
