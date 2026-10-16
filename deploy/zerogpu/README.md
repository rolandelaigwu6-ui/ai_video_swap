---
title: Authorized Synthetic Media Lab
emoji: 🎬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.3.0
python_version: 3.12.12
app_file: app.py
---

Research deployment package, not a validated identity-fidelity result.

Requires an eligible Hugging Face personal account, **ZeroGPU hardware**, and
an `APP_PASSWORD` Space secret with at least 24 random characters. No paid GPU
hardware, paid API, or paid fallback is requested by this application.

The host downloads FLUX.2-klein-4B (Apache-2.0) at pinned revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294` and the MIT-licensed BiSeNet parser.
GPU weights never load on the user's laptop. All images must be authorized.
Generated files have a synthetic disclosure footer. Files are temporary;
Gradio periodically deletes cached files, and this is not archival storage.

Generation may change head angle, expression or identity. Every output is
marked `needs_review`; no flawless or exact identity guarantee is made.
The semantic head/neck composite preserves pixels outside its edit region.
Segmentation failures stop processing. The model's built-in mitigations remain
unchanged; no arbitrary user prompt or model override is exposed.

Current free personal eligibility and quota:
https://huggingface.co/docs/hub/spaces-zerogpu

Model: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
