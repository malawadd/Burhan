# Freight CPU performance study

Implemented and measured 2026-10-20 on this Windows laptop: Intel i7-8550U, CPU only, Python 3.13.5, OpenCV 5.0.0, RapidOCR 3.9.0, ONNX Runtime 1.23.2. No OCR model, detector score threshold, parser, recognition height, minimum recognition width, orientation step or input page was removed to obtain the selected result. No new runtime dependency is required.

## Result and operating limit

Full offline OCR for the identical six public pages improved from **43.319 s to 19.289 s, 2.25× faster**. Warm mixed-page p95 improved from **11,771 ms to 4,856 ms, 2.42× faster**. The first cold page improved from **7,034 ms to 3,291 ms**. All 12 exact invoice-field checks and 55/58 annotated CORD word-substring hits were preserved, as were 3/3 invoice headings and3/3 out-of-scope receipt rejections.

This is still **seconds per complete page**: selected warm pages took 1.56–5.08 seconds. It is not full OCR at video frame rate and does not meet a production-line per-item OCR deadline. The camera quality/preview loop continues separately, and one page job is allowed at a time; its actual wall time remains visible as `ocr_job_ms`. Bounded async work existed before this optimization and is not counted as an inference speedup.

## Changes actually selected

- Enable the ONNX CPU memory arena so recognition calls reuse allocated buffers.
- Recognize one sorted text crop per batch, retaining `[3,48,320]` normalization. This reduces padding to the widest line of a multi-line batch; it does not skip a text region.
- Use 4 intra-op threads and 1 inter-op thread for each explicit CPU session. Disable intra-op and inter-op helper spinning on all 3 model sessions, so idle stage helpers do not compete with active work. Graph optimization remains `ORT_ENABLE_ALL`.
- Keep detection, 0/180 orientation classification and English recognition. Keep the prior detector sizing and recognition filter0.50, without capping page detector resolution.
- Export separate detector/orientation/recognizer stage times, session-load time and effective runtime settings with OCR evidence. `ocr_ms` is engine work, excluding initialization; outer `ocr_job_ms` includes initialization and parsing.

RapidOCR 3.9 documents injected ONNX sessions in its installed `OrtInferSession` implementation. Its OmegaConf parameter loader cannot directly encode an opaque Python session, so the local adapter uses the pinned initialization hook with `allow_objects` to inject 3 explicit sessions. There are tests asserting the actual ONNX options and that every model uses CPUExecutionProvider. Unsupported configuration keys are not silently passed to the RapidOCR engine. This narrow adapter must be reviewed when upgrading RapidOCR.

Profile version: `ppocr_v4_english_mobile_document_rules_v2_cpu`. Session/reset generation checks, matching page hashes, single in-flight job, bounded close and original-frame EOF finalization remain unchanged.

## Identical-image before/after

| Public input | Before wall ms | After wall ms | Before/after accuracy |
|---|---:|---:|---|
| cord\test-000.jpg | 7,034.0 | 3,290.8 | 23/23 word hits; unknown freight |
| cord\test-001.jpg | 3,100.1 | 1,564.2 | 12/12 word hits; unknown freight |
| cord\test-002.jpg | 3,319.0 | 1,893.1 | 20/23 word hits; unknown freight |
| invoice-pack\01-simple-english\invoice.png | 8,823.2 | 3,948.4 | 4/4 exact fields; invoice heading |
| invoice-pack\11-freight-duty\invoice.png | 12,508.1 | 5,082.4 | 4/4 exact fields; invoice heading |
| invoice-pack\14-poor-scan\invoice.png | 8,534.6 | 3,509.7 | 4/4 exact fields; invoice heading |

The first row is cold. Every other row uses the already initialized engine. Timings measure actual synchronous `read` plus parsing, not preview frame time; image decoding, filesystem hashing and report writing are outside the timed call. The before and selected measurements each ran in a fresh Python process during an exclusive CPU window: the other model workers were paused and no other model benchmark ran concurrently. OpenCV used2 threads in this standalone OCR study. There was no physical network disconnect, power-plan change, GPU inference or fabricated repeat count.

One complete pass per selected configuration was recorded; there are 1 cold and 5 warm observations per before/after set. The mixed-page percentile is a compact sample summary, not a statistical latency guarantee, sustained-load benchmark or repeated latency estimate for each page. Sequential runs may differ with laptop temperature, clock rate and OS scheduling. Source SHA256 values match for all 6 comparisons.

## Accuracy scope

The set is 3 genuine photographed CORD receipts plus 3 externally authored synthetic invoice fixtures. The 12 field checks are reference/date/currency/textual total for the 3 invoices; they are not a real shipment-document accuracy estimate. The55/58 measure normalizes alphanumerics, concatenates recognized text and checks annotated-word substrings. It can over-credit repeats/substrings and is not WER or CER. No genuine bill-of-lading, packing-list or delivery-note holdout was added in this optimization. See [dataset/model study](freight-public-evaluation.md) for source, attribution, licensing and previous scope limitations.

No model was retrained or quantized. Model hashes and licenses remain the same, and the parser was not altered to fit these six outputs. Unverified fields still require human review.

## Rejected measured variants

| Variant | Six-page wall s | Fields | CORD word hits | Decision |
|---|---:|---:|---:|---|
| Original baseline:2 threads, arena off, batch6 | 43.319 | 12/12 | 55/58 | Before |
| Arena on only | 29.042 | 12/12 | 55/58 | Improvement, superseded |
| Arena on,1 thread, batch6 | 39.421 | 12/12 | 55/58 | Slower than2-thread arena |
|1024 detector cap,1 thread, batch12, minimum recognition width160 | 30.553 | 12/12 | 53/58 | Rejected: lost2 annotated words |
| Arena on,2 threads, batch1, minimum recognition width48 | 19.771 | 12/12 | 54/58 | Rejected: lost1 annotated word |
| Arena on,2 threads, batch1, width320 | 22.654 | 12/12 | 55/58 | Preserved checks, superseded |
| Arena on,2 threads, no spinning, batch1, width320 | 24.038 | 12/12 | 55/58 | Did not improve2-thread page speed |
| Arena on,4 threads, no spinning, batch1, width320 | **19.289** | **12/12** | **55/58** | **Selected** |

The cap experiment used a narrow detector adapter because RapidOCR 3.9 `get_preprocess` ignores the configured side length in its `max` branch and chooses 960/1500/2000 internally. That adapter was used only for the rejected experiment and is off by default. It preserves full-resolution recognition crops, but the changed detector boxes still reduced word recall. No such cap ships in the selected default.

## Reproduce and evidence

```powershell
.venv/Scripts/python.exe scripts/benchmark_freight_speed.py --candidates baseline optimized --repeats 2 --output output/optimization/freight-rerun.json
.venv/Scripts/python.exe -m pytest tests/test_freight.py tests/test_freight_real.py tests/test_freight_speed.py -q
```

The command intentionally requests 2 complete passes for a larger rerun; the report delivered here records the actual 1-pass fresh-process runs described above. Do not run timing experiments concurrently with other model benchmarks. `baseline` reconstructs the old model settings; `optimized` uses the shipped 4-thread no-spin settings. Individual rejected candidates remain named in the script so they can be reproduced.

- [Consolidated measured before/after](../output/optimization/freight-speed.json)
- [Baseline + arena/thread/cap exploration](../output/optimization/freight-exploration.json)
- [Selected 4-thread no-spin run](../output/optimization/freight-nospin4-exploration.json)
- [Single-line crop batch](../output/optimization/freight-single-exploration.json)
- [Rejected compact-width run](../output/optimization/freight-compact-exploration.json)
- [2-thread no-spin run](../output/optimization/freight-nospin-exploration.json)

Lightweight regression verification after the change:15 passed,1 actual-model test deferred to the parent final suite to avoid CPU contention. Tests cover engine settings, true no-spinning session options, detector experiment coordinate bounds, parser evidence, async lifecycle/reset/stale-job behavior, and original-frame EOF finalization. The existing actual public-image test blocks Python socket connections and remains available in the full suite.

Metadata is constructed completely before a background job starts and remains unchanged while sessions initialize or pages run. A regression test checks the same published metadata object before/after initialization and that caller-owned override lists cannot change reported settings. This prevents a describe/API serializer from racing a worker dictionary update.
