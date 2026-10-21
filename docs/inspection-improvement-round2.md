# Factory localization and meaningful construction video evidence

Measured on the user's Windows laptop, 20 October 2026. All model execution is local. This report connects the factory experiment, independent localization audit, sustained runtime checks and construction viewer changes. Raw JSON reports remain alongside the Markdown explanations.

## What changed

The factory experiment replaces a fixed random-forest feature representation with an end-to-end spatial MobileNet U-Net. A rejected SDD1-only run is preserved. A second run adds real industrial SDD2 training data and training-only hard-negative patches, with calibration selecting the operating point before the reused SDD1 comparison is opened. [Training and calibration](factory-accuracy-round2.md) and [data/model research](factory-model-data-research-round2.md) give the precise architecture, hashes, licenses and split boundaries.

Factory localization is evaluated against original-resolution defect masks. Any-pixel overlap is retained for historical comparison, but pixel precision/recall/IoU, 10%/50% true-pixel coverage and one-to-one component IoU matching reveal weak or misplaced detections. The audit can save **every defective view and every normal false alarm**, with separate raw input, publisher mask, model mask and TP/FN/FP panels. Exact original-resolution masks are saved as PNGs. No favorable-example filtering is used.

Construction now uses genuine field-camera sequences containing visible cracks and spalling. Original pixels, frame order and encoded presentation timestamps are preserved. These are actual recorded exposures, not still-image slideshows or generated video. The [construction study](construction-actionable-inspection.md) describes the original dataset, timestamp/annotation association, model comparisons and remaining unsupported inspections.

## Construction interface

The **Damage cases and inspection evidence** panel defaults to recordings with external damage annotations. Earlier interview, worker and steel-joint clips are retained behind **Include failure diagnostics and context footage**. They no longer imply successful inspection. The same filter applies to construction public-input presets.

Each damage case shows the condition, why it needs inspection, the next check, publisher-labelled frame count, localized matches, unmatched annotations/model boxes and unsupported annotations. The annotation count can exceed the labelled-frame count because a frame can contain multiple boxes. Correlated frames are not independent buildings or physical defects.

Solid green boxes represent matched model predictions; solid amber boxes are unmatched model candidates. Dashed cyan/red boxes are external publisher annotations for matched/missed damage, with gray for unsupported damage. The labels never become model inputs. Below the player, **View localized damage** and **View missed damage** seek to actual source timestamps and explain that exact frame. A partial crack box can fail whole-instance IoU 0.50 even where damage exists; the interface explains this rather than asserting that every unmatched box depicts a nonexistent crack.

**Load this source for local inspection** loads the original video and its explicit model settings into the shared runtime. Start remains a separate action. The spalling case has no such supported spalling detector; its annotations remain explicitly unsupported. Live construction output hides generic context objects by default and provides plain-language explanations for issue candidates. `show_context_objects: true` enables a diagnostic overlay when desired.

The [evidence API](construction-evidence-api.md) allows only registry-listed local files. It validates frame structure, finite geometry, class/ID association, the actual IoU of claimed matches and one-to-one assignment. This verifies internal consistency; it does not establish annotation truth or training independence.

## Timing and validation boundaries

The laptop has an Intel i7-8550U, Intel UHD 620 and 24 GB RAM. Exclusive timing windows prevent competing training/inference jobs from contaminating measurements. Training, model-only latency, full-profile latency and shared-runtime throughput are separate measurements.

Factory runtime replay cycles predecoded original public images at a software clock. It includes the shared queue, profile processing, preview/evidence workers and deadline accounting, but excludes real camera exposure, moving objects and physical line actuation. Repetitions add speed observations, not independent accuracy examples. The repeatedly exposed SDD1 comparison remains development data. The original SDD2 official test is sealed; SDD2 pretraining is not a claimed independent test result or a direct supported input domain for the SDD1 geometry guard.

The [factory video-data search](factory-video-data-gap-round2.md) distinguishes genuine industrial sequences from images, simulations and unrelated surveillance. No immediately usable commutator production video with timestamped spatial defect labels and a verified public license was established. That missing acceptance evidence is not replaced by an image slideshow.

HTML playback and offline construction frame scoring use original VFR presentation timestamps. The shared runtime's current file scheduler uses nominal FPS, so its replay is not exact VFR source-time validation. Sequential mode avoids intentional source-frame skipping, while the separate inference queue can still drop decoded frames; its counters must be checked.

## Reproduce the final factory checks

```powershell
.venv/Scripts/python.exe -X utf8 scripts/factory_localization_audit.py --model models/factory/kolektor_surface_rf.joblib --output output/optimization/factory-round2/baseline-localization-audit.json --visuals-dir output/optimization/factory-round2/baseline-masks
# Use the frozen candidate path recorded in factory-accuracy-round2.md for the matching after audit.
.venv/Scripts/python.exe -X utf8 scripts/benchmark_runtime.py --varied --fps 16 --seconds 60 --deadline 100 --preview-fps 6 --model models/factory/kolektor_surface_rf.joblib --output output/optimization/factory-round2/baseline-runtime-16fps-60s.json
```

Annotated construction video reproduction and its pinned models are documented in the construction guide. Run neural jobs sequentially on this laptop when reproducing speed measurements.

## Verification record

### Frozen factory comparison

The 80 original views contain 11 defective and 69 normal views from ten physical items. Neither the added SDD2 data nor repeated timing passes increase that comparison sample. No parameter was retuned after these results.

| Actual original-mask measure | Baseline RF | Spatial U-Net + seeded mask |
|---|---:|---:|
| Defective views with any real-mask overlap | 8/11 (72.73%) | **10/11 (90.91%)** |
| Views with foreground IoU ≥0.10 | 3/11 | **10/11** |
| Views covering ≥50% of annotated defect pixels | 0/11 | **7/11** |
| Foreground pixel recall | 11.37% | **54.44%** |
| Foreground pixel precision | **96.88%** | 68.73% |
| Micro foreground IoU | 0.1133 | **0.4363** |
| One-to-one defect region recall at IoU ≥0.50 | 0/11 | **5/11** |
| Normal false alarm views | **1/69** | 5/69 |
| Flagged-view precision | **88.89%** | 66.67% |
| Image F1 | **0.8000** | 0.7692 |
| Image accuracy, dominated by normal views | **95.00%** | 92.50% |
| Quality abstentions | 0 | 0 |

This improves recall and actual localization, while increasing false alarms and reducing classification precision. It is a **manual-review candidate**, not an across-the-board accuracy gain or a production acceptance model. `kos02/Part6.jpg` and `kos38/Part1.jpg` are recovered; `kos12/Part5.jpg` remains missed. Normal alarms are `kos07/Part2.jpg`, `kos26/Part1.jpg`, `kos32/Part6.jpg`, `kos45/Part0.jpg` and `kos45/Part6.jpg`.

The [complete localization gallery](factory-localization-gallery-round2.md) links every defective view and normal false alarm, including exact predicted and publisher masks. Raw audit reports are `output/optimization/factory-round2/baseline-localization-audit.json` and `spatial-localization-audit.json`.

Before the final empty-mask optimization, two paired timing repeats provide 160 full-profile observations per model, with identical decoded inputs and alternating execution order. Mean/P95/max were **30.86/45.95/55.76 ms** for the RF and **38.58/48.69/59.71 ms** for the spatial candidate. Serial capacity was 32.40/s and 25.92/s respectively. These are profile capacity results, not shared-runtime throughput. Raw report: `paired-profile-comparison.json`.

### Actual construction damage comparison

| Whole-instance box localization, IoU ≥0.50 | Previous crack model | Joint-aware model |
|---|---:|---:|
| Branching crack, true matches / missed / unmatched model boxes | 14 / 148 / 157 | **136 / 26 / 87** |
| Branching crack recall / precision | 8.64% / 8.19% | **83.95% / 60.99%** |
| Thin crack challenge, true matches / missed / unmatched boxes | 0 / 8 / 8 | **0 / 8 / 9** |
| Both crack clips, recall / precision | 8.24% / 7.82% | **80.00% / 58.62%** |
| Spalling | Unsupported | Unsupported |

The candidate passed the rule written before its inference: aggregate recall across both clips improves and precision does not fall. No case-specific model choice, box enlargement or IoU relaxation was used. The same exposed clips form a development comparison, not independent accuracy evidence. Joint predictions are a separate context class and are never credited as cracks. The hard thin-crack failure and all 72 unsupported spalling annotations remain explicit.

Source replay uses FFV1 grayscale/NUT files. All **258** OpenCV-decoded BGR frames and PyAV-decoded gray frames equal the original arrays exactly, with unchanged counts and **0.0 µs** timestamp discrepancy. The annotated browser MP4s are derived display artifacts. `output/evaluation/construction-lossless-source-verification.json` records pixel/PTS checks; `construction-video-model-comparison.json` records the paired model results. The edge file scheduler's nominal-FPS limitation still applies independently of container timestamp fidelity.

### Runtime tail investigation

The first spatial 16 FPS / 60-second run averaged 15.9 processed FPS but had **4 drops and 14 late frames** at a 100 ms deadline, maximum 208.3 ms. Its median/P95 were 36.3/45.9 ms. This failed clean-coverage acceptance and is retained as `spatial-runtime-16fps-60s.json`.

A separate cold 15 FPS / 30-second diagnostic delivered 450/450 frames with no drops, but **2 decisions exceeded 66.667 ms**, maximum 71.1 ms. The first profile call took 58.46 ms; slower calls occurred at 2.808 and 5.814 seconds. These traces do not support a startup-only cause.

Three constructor-thread prediction warmups then moved visible initialization work before capture, without changing weights or masks. That 16 FPS run delivered 960/960 with no drops, but **2 decisions still exceeded 100 ms**, maximum 106.3 ms. The first inference-worker call took 102.4 ms despite constructor warmup; another stall occurred at 33.564 seconds. Median/P95/P99 were 36.1/47.0/53.0 ms. Compilation/request setup took 9,530.63 ms and warmup 121.60 ms, reported separately from the capture clock. These results are retained in `spatial-warmed-runtime-16fps-60s.json`; warmup is not presented as a guarantee against later OS/GPU stalls.

### Final common-runtime result

Worker preparation now runs native OpenCV quality operations and three neural predictions on the actual inference thread before capture is constructed. Its bounded startup barrier prevents captured frames from competing with initialization. Failed/timed-out preparation never opens the source. Later CPU/GPU scheduling delays remain possible.

The exact empty-mask optimization still runs inference on **every** observation. When no high-score component meets the frozen seed rule, it avoids unnecessary lower-threshold labeling and final component statistics. It changes no scores, thresholds or foreground mask. The final accuracy pass confirmed **all 80 original-resolution binary masks are hash-identical**, with identical source/annotation/model/metadata hashes and all aggregate metrics. See `fastpath-accuracy-identity.json` and `spatial-final-localization-audit.json`.

Both final runs use the common worker-barrier runtime, 16 scheduled FPS, 60 seconds, 100 ms deadline and 6 FPS previews. Event metadata persistence is enabled; event-image saving is off. Controlled training, inference, encoding and test jobs were sequenced for timing; operating-system and browser activity were not isolated.

| Shared runtime measure | Baseline RF | Final spatial model |
|---|---:|---:|
| Captured / processed | 960 / 960 | 960 / 960 |
| Input window / reported session duration | 60 / 60.2 s | 60 / 60.2 s |
| Processed / session duration | ≈15.95/s | **≈15.95/s** |
| Dropped / source-skipped / inspection errors | 0 / 0 / 0 | 0 / 0 / 0 |
| Host-to-decision median | 36.1 ms | **33.0 ms** |
| Host-to-decision P95 | 48.5 ms | **47.3 ms** |
| Host-to-decision P99 | 54.3 ms | 70.4 ms |
| Host-to-decision maximum | 94.4 ms | 130.4 ms |
| Decisions beyond 100 ms | 0 | **3/960** |

Raw reports: `baseline-final-runtime-16fps-60s.json` and `spatial-fastpath-runtime-16fps-60s.json`. Earlier runs remain in the [complete factory study](factory-accuracy-round2.md). Three late decisions mean the laptop still does **not** demonstrate a hard every-item production deadline, despite satisfying the sustained at-least-15 processed FPS target without frame drops.

### Review the actual inputs

In the dashboard, select a **Factory · spatial model** public-data example and start inspection. Recovered defects, a normal view and the remaining miss have separate presets; the outputs request human review. The RF baseline presets remain named explicitly for comparison.

For construction, select **Construction**, then the **Visible branching crack** case under **Damage cases and inspection evidence**. Use **View localized damage** or **View missed damage** to inspect exact source frames and separate publisher/model boxes. **Load this source for local inspection** loads the lossless original video and explicit joint-aware model settings; **Start inspection** runs fresh local inference without feeding annotations into the model. The thin-crack failure and unsupported spalling case remain available.

### Final application verification

- **191 tests passed in 50.52 seconds**, with one upstream Starlette/AnyIO deprecation warning. The first full run caught one startup regression: a profile without a preparation hook was described twice. The fix refreshes the description only after an actual hook; its camera race test and all fourteen targeted checks passed before the complete suite rerun. Logs remain `output/evaluation/round2-pytest.txt` (first run) and `round2-pytest-final.txt` (189-test passing rerun) and `round2-pytest-final-191.txt` (final 191-test pass).
- JavaScript syntax validation passed; the runtime environment reports no broken package requirements.
- The dashboard loaded the actual lossless `crack_10` video and explicitly selected `joint_aware_candidate` using **Load this source for local inspection**, then **Start inspection**. Fresh local inference processed **162/162 original frames**, with **zero drops, late decisions, source skips or inference errors**, ending at original frame 161. The source was scheduled at **8 FPS in sequential mode**, completing in 20.4 seconds; this is a complete-frame application integration check, not a claim of real-time 40-FPS source coverage. Actual host P95 was 70.1 ms and maximum 125.4 ms against its configured 200 ms deadline.
- The live output reported the pinned candidate SHA `4d48232d7b1427ac468022d2cc11a344885629cac71ad48bd2ff4c6659302246`, human-review status and plain-language what/location/why/next-check explanations. Annotations were not an input. The full end-of-source snapshot is `output/evaluation/construction-dashboard-live-input.json`.
- The station was stopped after verification. No camera was opened. The local server remains at `http://127.0.0.1:8765/`; the verification session uses `runtime-data-round2`.

Browser checks confirm the default list contains only three real damage cases, old context videos require the diagnostic toggle, spalling shows **Not evaluated** plus 72 unsupported annotations, and the crack source loader selects the verified lossless file and explicit model variant. Complete source timelines use the current displayed frame's PTS interval, rather than a future nearest annotation. Recorded annotated playback is labelled offline; fresh inference is a separate action.

Final source/config/package hashes are in `output/optimization/factory-round2/final-code-manifest.json`. The model and policy hashes remain those frozen before comparison. See the [construction guide](construction-actionable-inspection.md), [API contract](construction-evidence-api.md) and [complete factory experiment record](factory-accuracy-round2.md) for reproducibility and limits.

The final visual review corrected 39 diagnostic IoU values for unmatched construction boxes: overlap now includes annotations already claimed by another box. One-to-one assignments, prediction geometry, all TP/FN/FP counts and benchmark accuracy are unchanged. Duplicate boxes and class separation have two additional passing geometry tests. [Final browser evidence](../output/construction-damage-evidence-final.jpg) shows actual damage, model/publisher identities, complete counts and the corrected overlap; recorded playback advanced normally and the browser reported no console errors.
