# Edge vision product plan

Status: target architecture with an implemented first local prototype. See [README](../README.md), [implemented architecture](runtime-architecture.md) and [verification](verification.md) for actual features and measured limits. AWS management, field qualification and persistent calibration remain outstanding.

Date: 20 October 2026.

Working name: InspectFlow. Placeholder name; availability has not been researched.

## Product definition

One locally installed vision runtime with three inspection profiles: factory production, freight document capture, and road inspection.

The product observes camera feeds, analyzes them on the device, makes a local decision, and records useful events. Customers do not submit footage to a remote inspection service. Capture, OpenCV processing, model inference, temporal logic, alerts and the operator interface run on-site or in the vehicle.

AWS provides device management, versioned profiles and models, rollout status and fleet summaries. It is outside the detection path. Internet loss must not pause inspection, local guidance, alerts or local review.

This plan supersedes the upload-and-cloud-worker architecture. The [market study](../research/market-study-2026-10-20/market-study.md) remains background research, but its upload-based MVPs and server cost scenarios are not the deployment plan.

## What is shared

The common product is an installable runtime, camera connectivity, performance supervision, local events and evidence, and fleet management. Domain profiles supply their own models, thresholds, timing requirements and actions.

| Shared foundation | Profile-specific behavior |
|---|---|
| Camera and local replay sources | Production trigger, document capture session or vehicle stream |
| OpenCV 5 processing and inference interfaces | Board inspection, page checks or road distress model |
| Timestamped buffers and runtime supervision | Item deadlines, latest-frame guidance or road coverage accounting |
| Local operator interface and event history | Product result, document readiness or road observation |
| Local artifact store and asynchronous sync | What evidence may leave the site and when |
| Versioned packages and update validation | Models, calibration, input support and thresholds |

Factory quality teams remain the first buyer hypothesis. Freight operators and road inspection teams are separate customer segments. Shared software is an engineering opportunity; demand for one buyer purchasing all three has not been established.

## Deployment in each environment

### Factory production

A camera and suitable lighting observe a defined inspection station. A local industrial PC or other qualified edge computer receives the stream or triggered frames, runs inspection and presents the result immediately.

The initial supported task is visible product-fault inspection for one product family. The system associates the result with an item or trigger and records a suspect region. The local output can drive an operator alert or a simulated reject signal in the demo. A later line integration maps validated results to the site's controller.

Product inspection does not establish internal machine condition. If machinery faults are added later, they need their own observable signals, data and evaluation.

### Freight document capture

A dock camera or local capture station checks pages while an operator presents them. Boundaries, framing and image quality are measured continuously; the display immediately guides positioning. A stable image is selected locally, then checked against the required packet pages and regions.

The MVP uses a camera attached to the same edge computer. A future native phone application could run equivalent processing on-device, but a desktop Python runtime does not automatically constitute a mobile implementation.

The first freight profile still concerns the researched document workflow. Physical cargo damage or package sorting would be a separate profile requiring its own data.

### Road inspection

A vehicle camera connects to an onboard computer. The runtime analyzes the moving feed, tracks candidate observations through overlapping frames, and stores local defect events. GNSS position may be attached when an actual synchronized sensor is available.

The local result is a recorded inspection observation. A passenger/operator display or later review is appropriate; the system does not require a driver to interact while moving.

No manual photograph submission is part of ordinary operation. Missing location remains unknown. Detection from a monocular image does not establish pothole depth, road safety or repair priority.

## Architecture

~~~mermaid
flowchart TB
    subgraph SITE[On site or in vehicle]
        CAM[Camera or local replay source] --> CAP[Capture and timestamps]
        CAP --> BUF[Bounded profile-specific buffer]
        BUF --> CV[OpenCV 5 and local model inference]
        CV --> TEMP[Tracking and temporal confirmation]
        TEMP --> POLICY[Local decision policy]
        POLICY --> OUT[Local guidance alarm or recorded observation]
        POLICY --> EV[Local event writer]
        EV --> DB[(Local database and bounded evidence store)]
        DB --> UI[Local operator dashboard]
        CV --> UI
        DB --> SYNC[Independent sync worker]
        CACHE[Verified cached model and profile] --> CV
    end
    subgraph AWS[AWS management services]
        REG[Versioned packages in S3]
        CTRL[Device desired state and rollout status]
        METRICS[Fleet health and event summaries]
    end
    SYNC -. event metadata when connected .-> METRICS
    CTRL -. approved update .-> SYNC
    REG -. verified download outside inspection path .-> SYNC
    SYNC -. stage validate and activate .-> CACHE
~~~

The local dashboard is served by the edge device and is available through localhost or an approved site LAN. Preview access is local; it does not require a cloud video stream.

Use separate capture, inference, evidence-writing and synchronization execution paths. Bounded queues and explicit overload behavior prevent slow storage, rendering or cloud retries from creating an invisible backlog of stale inspection frames.

The initial software direction is Python with OpenCV 5, local compatible model inference, a local API, SQLite for events/configuration, and a small web interface. Pin versions after a runtime test. Select acceleration only after measuring supported hardware; do not promise a throughput figure for an unspecified device.

## Real time requirements

Real time means a decision is useful before the task's deadline. Display FPS is not sufficient evidence.

| Profile | Buffer and timing policy | Behavior under overload |
|---|---|---|
| Factory | Associate every triggered item with required frames, a deadline and a final inspection status | Mark affected items as inspection missed or late; invoke the configured local handling policy |
| Freight | Use recent frames for guidance; preserve a stable selected capture for detailed checks | Discard stale preview frames and report current guidance delay |
| Roads | Use a bounded stream pipeline with timestamps and coverage accounting | Record dropped frames and unobserved intervals; do not imply complete route coverage |

For a line with camera-to-action distance d and conveyor speed v, d/v is an illustrative physical time budget. Exposure, acquisition, inference, item tracking, communications, actuator response and margin all consume that budget. Actual integration must account for changing speed, triggers and conveyor geometry. A generic latest-frame strategy cannot guarantee every product was inspected.

Return an explicit status such as inspected clear for supported checks, suspect, inconclusive, inspection missed or late. A missed inspection must not become a clear result. Software on a general-purpose computer has no hard real-time guarantee from this plan.

Measure capture-to-decision p50/p95/p99 latency, maximum observed latency, deadline misses, sustained throughput, dropped frames, CPU/memory use and run duration. Distinguish hardware exposure time from host receipt time when the camera does not provide trustworthy timestamps.

Use a monotonic clock within a runtime session for latency and ordering, plus UTC timestamps for event records. Keep source sequence numbers and session identifiers. Replay uses source timestamps or an explicitly simulated clock, so it does not pretend fast file processing is live-camera latency.

## Local operation and evidence

Normal processing is automatic after a source and profile are configured:

1. Capture frames and attach timestamps and source identity.
2. Validate source health, focus, exposure and profile support.
3. Run supported local analysis.
4. Associate results with an item, document session or road observation.
5. Apply bounded temporal and action rules.
6. Display or emit a local result.
7. Save the event and selected evidence asynchronously.
8. Synchronize permitted metadata later if configured.

Raw continuous video stays local by default. Keep only a bounded rolling buffer and selected event frames/clips according to the site's policy. Media synchronization is opt-in; counts, device health and deployment status do not require customer footage.

The event writer has a bounded queue and disk budget. Disk-full, evidence loss, camera disconnect and stalled inference produce visible health states. Storage failure policy must be explicit for each profile, especially when traceability is required. Cloud backpressure never blocks the inspection path.

Local review records a person's confirmation or rejection separately from the automatic observation. Review is useful for uncertain events and later correction; it is not a prerequisite for every live detection.

## Module boundaries

| Profile | First automatic task | Public-data prototype | Required field validation |
|---|---|---|---|
| Factory | One supported product family and visible fault scope | VisA images plus clearly labelled timed sequence replay | Real optics, illumination, triggers, line speed, item association and fault examples |
| Freight | Live framing guidance and two-page packet readiness | SmartDoc15 geometry samples and original fictional form sequences | Actual station or mobile optics, operator behavior and customer packet rules |
| Roads | One supported distress class from a local stream | RDD2022 images and separately identified sequence/replay tests | Vehicle vibration, speed, weather, exposure, camera mounting and sensor timing |

VisA and RDD2022 are principally still-image evaluation routes. Replaying them through a timed source can test inference and event plumbing, but does not establish production-line tracking or continuous-road coverage. Synthetic sequences stay separate from real-video validation.

No hardware or site access has been established. Start with the existing computer as an edge development host. If it has an accessible camera, test it; otherwise use local timed replay through the same source interface. Do not select an appliance or promise industrial performance until a representative workload is measured.

## Domain behavior

### Factory profile

- Configure supported product/reference, camera region and trigger or item-entry rule.
- Check input quality and alignment where the selected method requires it.
- Run the validated fault method on the local device.
- Associate evidence with an item; avoid emitting one alert for every frame of the same item.
- Confirm results according to the profile's timing and evidence rules.
- Emit a local result before its deadline or an explicit missed/late status.
- Record evidence without waiting for network synchronization.

A light, sound or local integration event is sufficient for the prototype. Closed-loop reject hardware needs validated timing, item association and a site-defined handling policy. An LLM or cloud request must not determine a time-critical line decision.

### Freight profile

- Analyze the current camera preview for boundaries, local quality and stability.
- Give immediate local guidance such as reposition page or hold steady.
- Select a stable original capture for rectification and required-region checks.
- Track which accepted page fulfills each packet requirement.
- Detect duplicate captures and show the next missing page locally.
- Complete the configured packet checks while offline.

Blur or contrast values are readability proxies unless evaluated against actual text outcomes. Packet readiness does not authenticate signatures or authorize payment.

### Roads profile

- Analyze a local camera stream using a tested supported detector.
- Link overlapping observations conservatively to reduce repeated events.
- Preserve frame/timestamp ranges and the source region.
- Record coverage gaps and source health.
- Attach measured sensor position and timing quality when available.
- Keep depth, physical severity and unmeasured location unspecified.

A tracker ID is not proof of a unique physical defect across disconnected views. Positioning, calibration and observation overlap determine what deduplication claims are defensible.

## Adaptive perception

The immediate path uses bounded local tools and policies. A slower local supervisor may adjust the next inspection based on evidence without delaying the required output.

Examples: obtain another frame before the item deadline; inspect a supported crop at original resolution; ask a freight operator to reposition a page; flag a road interval as unusable; propose a camera exposure adjustment within tested limits.

If no time or sufficient evidence remains, return inconclusive or missed instead of waiting indefinitely. Do not repeatedly change thresholds until a desired result appears.

For an agentic demonstration, record the observation, selected next tool/action, outcome and deadline behavior. Compare against fixed processing. Calling a state machine an agent does not establish special-award eligibility.

## AWS role

AWS is a meaningful management component, with no synchronous dependency in inspection.

Proposed minimum cloud functions:

- Store immutable, versioned model/profile packages and manifests.
- Track device desired/active versions and deployment outcome.
- Receive health, aggregate counts and permitted event metadata.
- Demonstrate staged update, acknowledgement and rollback to a known working version.

[AWS IoT Greengrass V2](https://docs.aws.amazon.com/greengrass/v2/developerguide/what-is-iot-greengrass.html) supports local processing and device application deployment. [AWS IoT Jobs](https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs.html) supports operations dispatched to connected devices. Evaluate one integration route rather than implement two device-management systems.

For the build, implement a minimal AWS package registry and device status/update path, using S3 plus an AWS-hosted management API and persistent device state. Greengrass V2 is an alternative if setup fits the development host. Verify supported OS and packaging before selecting it; do not assume the Windows development environment supports every edge deployment route.

Updates are downloaded and verified in the background. Activate only at a permitted inspection boundary after compatibility checks, preserve the previous working package, and report the result. A partially downloaded package must never replace the active model.

The runtime boots from its last working local configuration when disconnected. Event synchronization is durable and idempotent by device/session/event ID. Cloud commands do not directly issue time-critical reject decisions.

Demonstrate actual AWS registry/state behavior in the submission and describe it precisely. Competition eligibility depends on the implemented meaningful AWS component and the published rules, not on this design alone.

## Interface and data contracts

Main local screens:

1. Station setup: source, profile, calibration and supported operating conditions.
2. Live monitor: preview, current status, local alerts and latency/coverage health.
3. Event history: evidence, model/profile version and optional human review.
4. Device health: source status, storage, active package and pending synchronization.

The optional cloud screen shows fleet health and deployment versions. It does not need raw video to operate.

Core records:

| Record | Required fields |
|---|---|
| Device/session | Device ID, boot/session ID, hardware/runtime version and health |
| Source/profile | Source ID, adapter, calibration, model/configuration version and timing policy |
| Frame reference | Source sequence, capture/receipt timestamps, original dimensions and transforms |
| Observation | Type, region, measurement units, method-specific score and source |
| Domain unit | Item/trigger ID, packet/session ID or road observation interval |
| Decision | Domain result, deadline, actual latency, next local action and limitations |
| Event/evidence | Stable event ID, selected local artifacts, retention and sync permissions |
| Review | Event reference, reviewer decision, reason and time |
| Package | Version, artifact hashes, compatibility, active/previous state and rollout result |

Keep live inference state separate from review and synchronization state. There is no per-frame remote upload API in the product path. Separate domain scores and labels; do not create a universal confidence threshold.

## Implementation order

1. Establish one local source interface for available cameras and clocked replay.
2. Build the bounded capture/inference loop, freshness metrics and local health supervision.
3. Validate each thin domain method on the development host.
4. Add item/session/observation association and each profile's overload policy.
5. Add local outputs, event storage and local live UI.
6. Add the independent AWS package/status/synchronization path.
7. Test internet loss, source loss, inference slowdown, disk pressure and rollback.
8. Freeze scope, evaluate, and document measured limits.

Continuous local capture is core scope. Defer cloud video streaming, a mobile port, custom appliances, a training platform, city maps, full PLC actuation and broad analytics.

## Demonstration and acceptance

The five-minute story is one edge runtime running three local profiles. Show live camera input where available and label replay honestly. Show factory local detection, freight immediate guidance and road local observations. Disconnect external networking and show inspection continuing. Reconnect and show queued summaries synchronizing. Show an AWS-managed package update and a known-working rollback.

Keep per-profile detection evaluation and runtime evaluation separate. Use validation data for model selection and a final untouched test. Report missed inspections, abstentions and coverage gaps as well as correct detections.

Minimum acceptance:

- Each declared automatic task runs locally on current input.
- No internet connection is required for inspection or local UI.
- A slow sync/evidence worker cannot accumulate unbounded stale inference.
- Factory items have a result or explicit missed/late state.
- Freight guidance responds to current frames and accepted captures remain local.
- Road dropped frames and coverage gaps are recorded.
- AWS performs demonstrated management work, without receiving raw media by default.
- Hardware, sources, rates, latencies and unvalidated field assumptions are documented.

## Commercial impact and remaining decisions

The product is locally deployed software, potentially paired with qualified hardware and installation. It needs a different commercial study from a server-upload service: appliance and camera costs, calibration, support, updates, installation and field reliability become central. The original study's hypothetical cloud subscription economics do not establish this offering's cost or price.

Resolve the actual camera/source access, target hardware, line timing, supported models and AWS account/budget before deployment commitments. Customer discovery should test whether an existing industrial or fleet solution already supplies the required local workflow. No new market demand or device-performance claim is established by this architectural correction.
