# Freight public model and dataset study

For the current optimized CPU runtime, use the [before/after performance study](freight-performance.md). Timings below document the earlier implementation on a busy machine and are retained as historical evidence, not the new benchmark baseline.

Recorded 2026-10-20 for the actual Windows laptop (Intel i7-8550U, CPU inference, Python 3.13.5, OpenCV 5.0.0). This is an implementation and small reproducible evaluation, not a market survey or a production qualification report.

## Selection

| Candidate | Suitability on this laptop | Decision |
|---|---|---|
| RapidOCR 3.9.0 + ONNX Runtime 1.23.2 CPU | Python 3.13 wheels, compact models, regular OpenCV dependency preserves working camera backend | Selected |
| RapidOCR 3.10.0 | Works with Python 3.13, but requires headless OpenCV; this laptop lost the verified Media Foundation backend when that wheel replaced regular OpenCV | Rejected for this installation |
| rapidocr-onnxruntime 1.4.4 | Small bundled older models, but published Python constraint is `<3.13` | Rejected; did not bypass the Python constraint |
| Bundled PP-OCRv6 small | Actual OCR worked, but measured roughly 4–9 seconds/page on this machine | Evaluated; not selected runtime model |
| English PP-OCRv4 mobile detector/recognizer | Small explicit local models and narrower language scope; actual OCR/fields verified. Still seconds/page here | Selected with bounded asynchronous processing, not video-rate OCR |
| Donut/LayoutLM/VLM document extraction | Larger inference/training stacks and more memory/latency; narrow freight data needed regardless | Not installed or benchmarked; no comparative speed claim |
| Tesseract | No binary was installed in standard locations. Another credible offline alternative | Not installed/benchmarked; avoided a separate Windows executable setup |

Selection references: [RapidOCR official repository](https://github.com/RapidAI/RapidOCR), [RapidOCR 3.9.0 package metadata](https://pypi.org/project/rapidocr/3.9.0/), [legacy package constraint](https://pypi.org/project/rapidocr-onnxruntime/), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [PaddleOCR license](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE). The selected model URLs and SHA256 values come from the installed official RapidOCR 3.9.0 `default_models.yaml`, copied into the reproducible setup script.

The detector is a learned text-region model and the recognizer is a learned text model. Heading classification and field extraction are our explicit English rules; this project did not train a freight classifier or a key-information model.

## Downloaded public evaluation inputs

| Dataset | What was downloaded | Purpose | License and caveat |
|---|---|---|---|
| [CORD v2, official NAVER CLOVA host](https://huggingface.co/datasets/naver-clova-ix/cord-v2) / [research repository](https://github.com/clovaai/cord) | First three `test` rows, 0–2, plus their annotations | OCR on genuine photographed paper receipts; reject receipt content as unknown freight type | CC BY 4.0; Park et al. These are Indonesian receipts, not freight documents |
| [Synthetic Shopify Invoice Test Pack](https://github.com/SalorWorks/shopify-invoice-test-pack), commit `bb2e531cfa504e6f7200243620121c523f9f7622` | `01-simple-english`, `11-freight-duty`, `14-poor-scan`: PNG + expected JSON | Externally authored invoice layouts, noisy scan variant, heading/field checks | CC BY 4.0; SalorWorks, version 1.0.0. Explicitly synthetic/fictional; no real shipment claim |

Only six images were downloaded. The CORD dataset has 1,000 rows overall; the full 2.31 GB archive was not downloaded. The public sample script fetches selected rows through the dataset viewer API, not remote inference. `datasets/freight/manifest.json` records paths, sizes, SHA256, attribution, row IDs and source references; license text copies are stored with it. The invoice fixture is additional public data, distinct from the application's authored marker demo.

Other data considered: [FATURA](https://zenodo.org/records/8261508), 10,000 generated invoice images across 50 templates, archive about 363.5 MB; [DocLayNet](https://github.com/DS4SD/DocLayNet), diverse layout data with a 28 GiB core archive under CDLA-Permissive-1.0. Neither was downloaded. Neither alone establishes freight packet correctness. No genuine public bill of lading, delivery note or packing-list holdout has been tested in this change.

## Measured results

Reproduce:

```powershell
.venv/Scripts/python.exe scripts/evaluate_freight_public.py
```

The machine-readable output is [freight-public.json](../output/evaluation/freight-public.json). The set was deliberately selected as a tiny functional smoke subset, not randomly sampled to estimate global accuracy; no training or fine-tuning used it.

Actual latest PP-OCRv4 CPU measurements on this busy laptop:

| Input | Wall time | Result |
|---|---:|---|
| CORD test 0 (cold first call) | 6,282 ms | 23/23 annotated words found as normalized substrings; unknown freight type |
| CORD test 1 | 4,100 ms | 12/12 annotated words found as normalized substrings; unknown freight type |
| CORD test 2 | 5,669 ms | 20/23 annotated words found as normalized substrings; unknown freight type |
| External simple invoice | 9,484 ms | Invoice heading recognized |
| External freight/duty invoice | 11,530 ms | Invoice heading recognized |
| External poor-scan invoice | 11,796 ms | Invoice heading recognized |

The substring measure removes non-alphanumeric characters, uppercases, concatenates OCR text, and checks whether each annotated word appears. It can over-credit repeats and substrings. It is **not** character error rate, word error rate, a freight extraction accuracy score, or a calibrated confidence measure. The CORD annotations label selected receipt content rather than every printed word. All three receipts were correctly left outside the supported freight document classes in this sample.

The simple public invoice flowed through `FreightProfile` and produced `INV-2026-001`, `2026-08-01`, `AED`, and `AED 63.00` as unverified suggestions with textual evidence. A configured one-invoice checklist completed and a repeated view was suppressed. All 12 exact checks (reference, date, currency, textual total for three invoice fixtures) passed. The poor-scan text initially exposed a parser error: a table-column `Total` was followed by quantity `24` in OCR reading order. The parser now prefers explicit currency-bearing total evidence, with a regression test. The final extraction checks were rerun against saved actual OCR text; the report marks this reparse and retains the original model timings. Reproduce this extraction-only pass with `--reparse`, or rerun the full evaluator to read the pixels again. An actual public-image test runs with Python socket `connect` blocked and successfully reads the page, establishing that preinstalled inference needs no connection. It does not claim a physical disconnected-network field trial.

These timings were recorded during parallel development, with other model work active, and are small samples rather than sustained benchmarks. Each cold OCR engine includes ONNX session initialization. The selected English v4 models did **not** outperform the original bundled v6 small models in these runs. The change therefore uses asynchronous page processing rather than claiming an unsupported latency improvement: the preview/quality loop keeps running, one OCR job is in flight, and results are consumed only for a matching page. `ocr_ms` reports the separate OCR work. Expect several seconds, potentially over ten, for a full page on this laptop. Hold a camera page in view or use the single-image source while it runs.

## Asynchronous real profile verification

`scripts/smoke_freight_async.py` exercised the public simple invoice through the actual default asynchronous profile while Python socket connections were blocked. It processed 76 guidance frames while the cold OCR job ran, then accepted the invoice, completed a configured one-invoice packet, and suppressed a repeat. It extracted the same four supported fields. [Recorded output](../output/evaluation/freight-async-smoke.json): guidance p50 114.35 ms, p95 200.64 ms, maximum 1,076.56 ms; separate cold OCR job 17,507.95 ms, total run 18.028 seconds. The one maximum is a real observed delay, not hidden or called an on-time result. CPU contention and cold startup make a strict low-latency guarantee inappropriate.

The module's source completion contract preserves the retained actual final frame, source sequence, and original capture time when OCR finishes after file EOF. `finish_source(timeout)` waits boundedly and returns a final result only when ready; repeated finalization never starts more OCR jobs. The shared runtime records completion as asynchronous document work rather than a newly captured frame or a guidance timing sample. `close(timeout)` reports whether the native job has actually exited, preventing worker accumulation across station switches.

## Model files and provenance

`scripts/fetch_freight_models.py` downloads just the two selected models to ignored `models/freight`, checks published SHA256 and writes a manifest. The orientation model is already shipped in the pinned RapidOCR wheel. `model_inventory()` checks all three hashes and reports the installed files in the profile descriptor and OCR provenance.

| File | Bytes | Published SHA256 |
|---|---:|---|
| `ch_PP-OCRv4_det_mobile.onnx` | 4,745,517 | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` |
| `en_PP-OCRv4_rec_mobile.onnx` | 7,653,044 | `e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318` |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | See generated inventory | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

RapidOCR and PaddleOCR use Apache-2.0; license copies are under `models/freight`. ONNX Runtime is MIT. Public dataset data remains subject to its own CC BY attribution requirements, independently of our application source. Runtime explicitly selects local model paths; it never offers automatic downloads. Missing/corrupted models produce a visible guidance error and no accepted page.

## What is still required

Collect customer-approved documents, label real freight page types and fields, and create a held-out set spanning suppliers, languages, camera conditions and layouts. Measure page-level precision/recall, field exact match, rejection quality, duplicate collisions, full-packet consistency and latency under the intended station load. A human must verify the current OCR suggestions and shipment contents. Legal, authenticity, signature, customs, dangerous-goods and financial correctness are outside this implementation.
