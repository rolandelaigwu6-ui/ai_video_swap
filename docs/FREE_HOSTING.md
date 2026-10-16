# Free server deployment

Budget: **$0 only**, as requested. The user now has the `relaigwu` account, but
no private server has been deployed. GPU generation on this private-server
package has not been tested. Immediate use of existing shared editors is now
integrated separately; see [the instruction studio](STUDIO.md).

Hugging Face currently allows eligible personal accounts to host up to two
ZeroGPU Gradio Spaces for free: verified email and account older than 30 days.
Free accounts have a daily shared-GPU quota; queues and availability apply.
Creating ordinary Docker/Gradio compute Spaces generally requires a paid plan,
so the former free CPU Docker route is not selected.

Sources checked 2026-10-16:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/docs/diffusers/api/pipelines/flux2

The user is now signed in as `relaigwu`. Verify the personal account email in
https://huggingface.co/settings/account. Creating an account does not grant immediate
free-hosting eligibility. The account age requirement is an external activation
blocker; this project does not create alternate accounts or bypass eligibility.

## Prepared package

```sh
cd /home/don1/ai_video_swap
.venv/bin/python scripts/prepare_server.py --refresh
```

The source-only bundle is `.lab/server-package`; it contains no user photos,
job artifacts, local model cache, passwords or account tokens. `DEPLOYMENT.json`
records the source hashes. All large model downloads happen on the host, not
on the user's 3.7 GiB laptop. The selected FLUX.2-klein-4B model supports
multiple input images and is Apache-2.0 licensed. Reference conditioning can
edit appearance but does not guarantee exact identity or pose.

Once the account is eligible, create a Gradio Space with **ZeroGPU** hardware,
upload only this bundle, and set an `APP_PASSWORD` secret of at least 24 random
characters. Do not choose paid hardware or buy credits. This code never requests
a paid API or paid hardware fallback. Hosting policy/quota must still be checked
at activation time; source-code configuration is not a provider spending cap.

From the signed-in homepage:

1. Open https://huggingface.co/new-space. Use owner `relaigwu`, name
   `synthetic-media-lab`, and SDK **Gradio** (blank template).
2. Select **ZeroGPU** only if the account is eligible and the interface offers
   free hosting. If it is unavailable or a paid subscription is required, stop:
   do not select a paid GPU or CPU hosting plan.
3. In the created Space, open **Settings**, find **Variables and secrets**, and
   add a **Secret** named `APP_PASSWORD`. Generate a local random value with
   `openssl rand -hex 24`; keep the value private for the local connector too.
4. In **Files**, choose **Add file → Upload files**. Upload the *contents* of
   `/home/don1/ai_video_swap/.lab/server-package/`: `app.py`, `requirements.txt`,
   `README.md`, and the entire `synthetic_media/` directory. These must appear
   at the Space root, not inside a `server-package` subfolder. Do not upload
   photos, local job artifacts, `.venv`, or the entire project.
5. Commit the files, then inspect build/runtime logs. Successful activation
   means the Space reaches **Running** on ZeroGPU and shows the app login.
   The application username is `owner`; use the generated password. A Space
   existing on the website does not by itself mean generation is operational.

Install the small optional client locally:

```sh
.venv/bin/pip install -e '.[remote]'
```

Configure `SYNTHETIC_MEDIA_REMOTE_SPACE=your-account/your-space` and
`SYNTHETIC_MEDIA_REMOTE_PASSWORD` in the local server process environment.
An optional `HF_TOKEN` lets the platform identify a free personal account's quota.
Before upload, the client verifies both active/requested hardware are ZeroGPU
and rejects tokens from PRO/organization or unknown account tiers that could
use paid credits. This is not a substitute for provider-side billing settings.
Keep passwords/tokens out of chat, Git and screenshots. No configuration means
the remote backend is explicitly unavailable; local inference does not silently
send media elsewhere. Select `backend=remote` through the API or CLI only after
deployment. Authorization notes cover sending the supplied media to that host.

## Behavior and limits

The shared GPU runs the pinned, four-step image-editing model with target and
reference images. The app restricts requests to authorized identity replacement.
CPU parsing segments the head and neck. The final composite retains the target's
native dimensions and every pixel outside that mask, including clothing and
background. Hair outside the original segmented silhouette cannot be guaranteed.
The synthetic label is in a separate footer. Pose, expression and likeness need
visual review: output status remains `needs_review`.

Requests are serialized. The local client times out after five minutes and
requests cancellation. Its disposable process tree has a 330-second outer
deadline and 256 MiB RAM ceiling, preventing stalled client threads from
accumulating locally. There is no automatic retry, alternate endpoint, or
paid fallback. Remote files are temporary, not private archival storage. The
password restricts application access; the hosting provider processes the media.
Package preparation does not upload media. A separate, explicitly authorized
trial sent the supplied target and reference crop to the model author's public
shared demo. No result returned within 300 seconds; cancellation was requested,
and the remaining local client process was stopped. A later trial using the
current API client returned a full-instruction result in11.6seconds, but failed
identity checks. It is now available as an explicit, quota-limited shared editor,
not a private or guaranteed service. The earlier failure record is under
`.lab/shared-trials/a262c01c37dc46d0b83f17ad04288532/failure.json`.

Validation before calling this deployed: authenticate to the actual Space,
verify its hardware remains the free ZeroGPU tier, run the user's authorized
portrait pair, check mask boundaries/pose/identity, and measure GPU quota use.
