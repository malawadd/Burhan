# Burhan Smart Inspection backend

This service is separate from the desktop computer-control API. The phone keeps CameraX,
ONNX detection, evidence, and reports offline. This service receives an occasional permitted
snapshot, interprets the surroundings, and chooses a bounded inspection action. It never
runs model-proposed shell commands, confirms defects, or declares structural safety.

## Start on this Windows computer

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File burhan/backend/start.ps1 -Install
```

For an extracted field kit, open PowerShell in the extracted app folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File backend/start.ps1 -Install
```

The outer folder may have any name. On first use, the script reuses an existing project
environment if available, or creates `backend/.venv` using `python` on PATH. Install Python
3.12 or 3.13 first on a new computer. Later launches can omit `-Install`. The script starts
`python -m backend` from the app folder, so it does not depend on its directory being named
`burhan`. It prefers an existing `backend/.venv` over the parent project environment.

Dependencies can also be installed with `.venv/Scripts/python -m pip install -r
burhan/backend/requirements.txt`, then start with `.venv/Scripts/python -m burhan.backend`.
For manual startup in an extracted kit, install dependencies into an environment of your
choice, enter the app folder, and run that environment's Python with `-m backend`.

Copy `.env.example` to `.env` in this directory to change settings. Existing process environment
variables take precedence. The defaults listen only on `127.0.0.1:3210`; the existing tunnel
must forward `https://merhaba-api.kiyan.finance` to that address. No tunnel credentials are
stored in the app. The server disables proxy-header trust and request access logging.

Startup prints an eight-character pairing code, valid for one hour. It is also in
`backend/data/pairing.json`, accessible only to the operator. Enter it in the app's Smart
Inspection connection panel. A paired device receives a 30-day bearer token; only token
hashes are stored in SQLite. Restart to rotate the pairing code. Do not publish the code or
the data directory. `GET /api/v1/health` is public and reveals no pairing credentials.

## Switch providers

```dotenv
SMART_INSPECTION_PROVIDER=codex
```

Choose `codex`, `aws`, or `off`, then restart the backend. The installed app and its URL do
not change. Provider changes do not erase sessions, tokens, or previous traces. In-flight
turns are cancelled on restart; the app can submit a fresh view to the same session.

**codex:** Install Codex CLI and run `codex login` as the same operating-system account that
runs the server. It uses that CLI login, not a key embedded in the application. Each turn
uses `codex --no-daemon exec --ephemeral --ignore-user-config --ignore-rules --sandbox
read-only --image ... --output-schema ...`, a dedicated temporary workspace, bounded history,
and stdin. Shell, browser, apps, plugins, agents, hooks and unrelated tools are disabled.
Only the final validated plan and usage metrics are retained. Local development still sends
the selected image and inspection context to OpenAI for online inference. `CODEX_MODEL` is
optional; omit it to use the CLI's default. `ready: true` establishes binary availability,
not guaranteed account quota or model access. CLI documentation: [official reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli).

**aws:** Set `AWS_REGION=us-east-1` and
`AWS_BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0`. Use the normal AWS credential chain
(`AWS_PROFILE`, SSO, or an instance/task role). Never copy credentials into the APK.
The actual boto3 Bedrock Converse call sends a JPEG and context and forces a
`submit_inspection_plan` tool response, validated against the same schema and capability
registry. Model access and quotas must be enabled for that AWS account. There is no fallback
from AWS to Codex; failures stay visible so submission traces cannot misrepresent execution.
The shared prompt and tool loop follow the [Nova 2 Converse API](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html).

**off:** Health reports provider `off` and `ready: false`; planning returns 503 with a clear
offline-detection message. Existing local inspection remains usable.

## API and orchestration

Interactive OpenAPI documentation: `http://127.0.0.1:3210/docs`. Every session route requires
`Authorization: Bearer <device token>` and enforces device ownership.

| Method/path | Purpose |
|---|---|
| `GET /api/v1/health` | Process status, provider, configured availability |
| `POST /api/v1/pair` | `{code, deviceName}` → `{token}` |
| `POST /api/v1/sessions` | Inspection metadata → `{id, provider}` |
| `POST /api/v1/sessions/{id}/turns` | Snapshot + scene version + context → 202 `{id,status}` |
| `GET /api/v1/sessions/{id}/turns/{turnId}` | Poll queued/running/completed/failed/superseded/cancelled |
| `POST /api/v1/sessions/{id}/tool-results` | Record acknowledged/completed/skipped/failed actions |
| `GET /api/v1/sessions/{id}/trace` | Export perception, decisions, tool results and measured usage |
| `DELETE /api/v1/sessions/{id}` | Close session and cancel its active work |

Snapshots are JPEG/PNG data URLs up to 4 MB decoded, 32–4096 pixels per side and at most
12 megapixels. Headers are checked before pixel decoding. OpenCV 5 resizes them to at most
1280 pixels, measures focus/exposure, and encodes JPEG for both providers. Metrics are
heuristics; they never prove defect absence. Requests are limited to about 6 MB including
base64. Inspection observations and tool results have separate size limits.

Submit `sceneVersion` as a monotonically increasing integer when the camera context changes.
Older submissions return 409. A newer scene cancels running older work. Only the newest
pending turn per session remains queued; the single global worker runs one Codex process
at a time. Reusing optional `requestId` returns the original job instead of spending another
turn. Keep one planning request active on the client and discard responses for obsolete
scenes. Do not change sceneVersion for every frame: a stable close-up belongs to the same
scene. A no-image context/tool turn reuses the previous snapshot if it still exists.

Defaults are one submitted turn every 10 seconds, 60 turns per inspection and 300 across
the whole backend in a rolling 24 hours. Creating additional paired devices cannot bypass
the daily cap. `AGENT_TIMEOUT_SECONDS=120` bounds provider work. Cancellation kills the
entire Codex process tree on Windows. Bedrock requests have bounded SDK socket timeouts;
a cancelled network request may finish remotely, but its result is discarded.

Temporary snapshots are deleted after 24 hours during startup and a periodic one-minute sweep.
CLI workspaces are removed after each turn. SQLite retains inspection metadata, bounded
observation context, plans, tool results, timing, token counts, and provider identifiers for
evaluation. It does not retain raw CLI streams or credentials. Store `data` on a private disk;
exclude it from source archives and shared field kits. Closing a session stops work but is
not a deletion request. Delete the backend data directory only when its records are no
longer needed; this revokes paired tokens as well.

Both providers constrain generated task identifiers to exact catalogue IDs. The orchestrator
still rejects unknown tasks, false automatic capabilities, invalid detector/task combinations,
and changes to locked tasks. Non-detector instructions use `mode: null`; an echoed detector
mode on a view/manual instruction is normalized to null without changing its known task ID
or running a detector. A `plan_normalized` trace records that adjustment. Invalid plans fail
closed and produce a bounded `rejected_plan` diagnostic containing a precise reason code,
the rejected typed plan and the offending action/capability. These inspection-only diagnostics
are visible solely to the owning paired device through the authenticated trace endpoint.

## AWS deployment

For local testing with real AWS inference, select `aws` on this backend and supply AWS
credentials. This already executes the planner on AWS Bedrock; the phone remains offline
for detection. No account or deployment is created automatically by these scripts.

For an AWS-hosted backend, `deploy/aws-ec2.yaml` creates an outbound-only, SSM-managed
Ubuntu host with encrypted storage, an instance role for the selected Bedrock model and
the exact deployment artifact, and a systemd service. This keeps the SQLite/job-worker
single-process design valid. It is an alternative to container hosting, not a serverless
Lambda template. Do not run multiple uvicorn workers against this queue implementation.

1. Run `python burhan/backend/deploy/package.py` to create a source-only `backend.zip`.
2. Upload it to a private S3 bucket as `burhan/backend.zip`.
3. Deploy the template in `us-east-1`, supplying bucket, subnet and VPC parameters and
   acknowledging IAM creation. Choose a subnet with outbound access to S3, PyPI, SSM and
   Bedrock. Confirm model access and current inference-profile region permissions.
4. Use SSM Session Manager to inspect `systemctl status burhan` and read the local pairing
   file `/var/lib/burhan/pairing.json`.
5. For a workstation health test, use the stack's SSM port-forward command and open
   `http://127.0.0.1:3211/api/v1/health`. For phones, configure an authenticated HTTPS tunnel
   or reverse proxy to the host's `127.0.0.1:3210`, preserving the app's supplied public domain.
   No inbound security-group ports are opened by default.
6. Delete the CloudFormation stack when finished, and separately delete the deployment
   artifact if desired. The root EBS volume is deleted with the instance; export traces first.

`Dockerfile` is another deployable option: from `burhan`, build with
`docker build -f backend/Dockerfile -t burhan-smart .`. Mount `/var/lib/burhan` persistently and
use an AWS task/instance role. This image defaults to `aws`; it does not contain a Codex
login. Keep it behind HTTPS and a gateway rather than exposing port 3210 publicly.

AWS charges include inference, compute and storage. This expanded image-aware planner is
not covered by the earlier infrastructure-only estimate. Caps limit usage but are not a
billing guarantee. Keep AWS budget alerts configured for the account.

## Verification and troubleshooting

```powershell
.\.venv\Scripts\python.exe -m pytest burhan/backend/tests -q
```

Tests cover pairing ownership, stale/out-of-order scenes, queue cancellation, request
idempotency, quotas, bad images, capability/locked-task rejection, persisted session memory,
provider failures, restarts and the Bedrock request contract. The AWS client test is a stub
and does not establish real account access. Actual provider/tunnel tests belong in the
delivery validation record, with real timing and known limitations.

| Symptom | Check |
|---|---|
| Public domain returns 502 | Backend is listening on 3210; tunnel target is `http://127.0.0.1:3210` |
| Browser cannot connect, native can | Add only the actual browser origin to `CORS_ORIGINS`; restart |
| Pairing rejected | Use current local pairing file; restart if older than one hour |
| 429 while moving camera | Keep one active turn; respect ten-second interval and usage caps |
| Codex turn failed | Run `codex login status` as backend user; check model/account quota |
| Codex missing with desktop app closed | Put CLI on PATH or set absolute `CODEX_COMMAND` |
| AWS health not ready | Check `AWS_PROFILE`/SSO or instance-role credentials |
| AWS turn failed | Check model access, InvokeModel permissions and cross-region profile |
| New guidance replaces a close-up | Send confirmation-aware context and guidanceTaskId; never apply a different task’s instruction |
| Backend restarted | Submit a fresh view; paired token and previous plans persist |
