# Edge runtime implementation backlog

Status: shared local runtime and three real-input CPU adapters implemented, with public-data/model studies. See [real-input verification](real-input-verification.md) for current evidence; broader field/deployment criteria remain pending. The checklist below retains target scope; see [verification](verification.md) and [implemented architecture](runtime-architecture.md) for tested behavior. Unchecked items are not claims of completion.

All detection and immediate decisions run on the customer device. AWS manages versions, health and optional event summaries asynchronously.

## P0 Local feasibility and timing contract

- [ ] Record development hardware, available camera access, team capacity and build hours.
- [x] Pin and verify OpenCV 5 and selected local inference dependencies.
- [x] Define camera and clocked-local-replay source contracts.
- [x] Verify actual capture backend behavior; measure frame age rather than trusting a requested buffer setting.
- [x] Retrieve KolektorSDD, CORD/public invoice fixtures and RDD2022 subsets with attribution, fixed selections and explicit split/overlap caveats. VisA and SmartDoc were not used.
- [x] Test one narrow factory fault method and one road class detector locally.
- [x] Generate original fictional freight form sequences for capture/packet tests.
- [x] Record model weight licenses separately from data licenses.
- [x] Declare each profile's supported input, result types, timing policy and overload behavior.

Exit: useful local predictions for every promised automatic task, and an explicit statement of what camera/field validation remains unavailable.

## P1 Streaming runtime foundation

- [x] Independent capture, inference and event-writing execution paths. Sync is deferred and disabled.
- [x] Bounded timestamped frame buffers with profile-specific policies.
- [x] Frame sequences, monotonic timing, UTC event times and boot/session IDs.
- [ ] Camera disconnect/stall detection and bounded reconnect handling.
- [x] Frame-age, throughput, queue and inference health metrics.
- [x] Event-driven local observation and UI output; physical actuation is deferred.
- [x] Keep disk writes outside inference and cloud calls absent. Preview encoding still consumes local pipeline CPU and is documented.
- [x] Distinguish source timestamps from host receipt timestamps and simulated replay time.

Exit: the development host processes a stream continuously and exposes real frame freshness, latency and overload state.

## P2 Three local profiles

### Factory

- [x] Supported product/reference and inspection-region setup.
- [x] Local fault method, input quality and alignment checks as required.
- [x] Item/trigger association and per-item result/deadline accounting.
- [x] Temporal rules that avoid repeated alerts for one item.
- [x] Explicit inspected, suspect, inconclusive, missed and late outcomes.
- [ ] Local operator alert and simulated downstream output.
- [x] Test missed triggers, overload and stale evidence with clearly labelled simulation.

Exit: every simulated/observed item has a traceable local outcome; no silent frame dropping masquerades as full product inspection.

### Freight

- [x] Local live boundaries, framing, quality and stability checks.
- [x] Immediate positioning guidance in the local interface.
- [ ] Select and preserve a stable original capture.
- [x] Page rectification, asynchronous local OCR, heading/label rules and duplicate capture detection. Trained field-region validation remains future work.
- [x] Local packet checklist and next-page prompt.
- [ ] Re-evaluate a corrected capture while disconnected.

Exit: the operator can complete the supported packet through local capture without submitting it to a remote service.

### Roads

- [x] Local stream inference for one supported distress class.
- [x] Conservative association across overlapping observations.
- [ ] Evidence frame ranges, timestamps and original-coordinate regions.
- [x] Dropped-frame counters; geographic/elapsed unobserved coverage remains unmeasured.
- [ ] Optional measured GNSS input with timing quality; unknown location otherwise.
- [x] Distinguish predictions, human annotations and benchmark labels.

Exit: observations are generated locally, with explicit coverage limits and no invented position, depth or severity.

## P3 Local operations and evidence

- [x] Local station setup and live monitor.
- [x] Per-profile status, on-screen alerts and runtime health.
- [x] Local event database and bounded selected-image retention; continuous video ring is deferred.
- [x] Event detail and source/model/profile attribution.
- [ ] Persist human review decisions.
- [x] Disk budget, retention, write-error state and evidence-loss counters.
- [ ] Define local behavior when required traceability cannot be retained.
- [ ] Preserve decisions and accepted captures across process restart.

Exit: operators can monitor and review the station locally without internet.

## P4 Independent AWS management

- [ ] Resolve authorized AWS account, region and budget.
- [ ] Select one device-management route compatible with the host.
- [ ] Store immutable package/profile versions and manifests in S3.
- [ ] Deploy AWS device desired/active version and health/status handling.
- [ ] Download and verify packages outside the inspection path.
- [ ] Activate updates only at a permitted boundary; preserve the working version.
- [ ] Demonstrate failed-update rejection and rollback.
- [ ] Persist and idempotently synchronize permitted event metadata.
- [ ] Keep raw media local by default, with explicit optional sharing.
- [ ] Ensure boot, detection, alerts and local review use cached configuration offline.

Exit: AWS performs visible management work, while network loss has no functional dependency in local inspection.

## P5 Evaluation and failure tests

- [ ] Freeze models and thresholds using validation data before the final test.
- [x] Report available public-data diagnostics, factory holdout recall/false alarms/abstentions, OCR field checks and supported scopes. Roads training overlap is unknown; no general accuracy claim.
- [x] Separate still-image accuracy from stream and field validation.
- [x] Measure host-receipt-to-decision percentiles and maximum; camera exposure/backend delay remains unmeasured.
- [ ] Report throughput, deadline misses, dropped frames, CPU/memory and test duration.
- [ ] Test network loss/recovery without pausing detection.
- [ ] Test slow sync, slow evidence writing, bounded memory and disk pressure.
- [ ] Test source disconnection and inference overload with explicit domain outcomes.
- [ ] Test updating/rolling back a package without confusing active-version attribution.
- [ ] Verify no raw media leaves the device under the default synchronization policy.
- [x] Clearly label replay, synthetic sequences and untested industrial/vehicle conditions.

Exit: the system's local operation and limits are demonstrated with measurements, not inferred from average FPS.

## P6 Submission

- [x] Document edge installation, source setup, model/data rights and reproduction steps.
- [ ] Include edge and AWS architecture diagrams and actual management traces.
- [x] Show all three local profiles and one evidence-driven next action per profile.
- [ ] Demonstrate offline continuity, reconnect synchronization and package rollback.
- [x] Include per-profile results and field-validation limitations.
- [ ] Finish the five-minute video and judge-accessible demonstration.
- [ ] Submit before the earlier deadline displayed by the competition platform.

## Deferred features

Native phone port, custom appliance procurement, industrial reject actuation, cloud video streaming, large-scale training, route planning, city maps, multi-tenant billing and broad analytics.

Continuous local capture, offline inspection, local alerts and profile-specific timing are core requirements and must not be deferred.
