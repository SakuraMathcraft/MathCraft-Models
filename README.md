# MathCraft Models

MathCraft Models is the model artifact repository for **MathCraft OCR**, the ONNX-only OCR runtime used by LaTeXSnipper.

The goal of this repository is to make OCR model delivery explicit, reproducible, and independent from upstream implicit cache behavior. Every active model is distributed as a versioned release asset with a stable MathCraft-owned model ID, a manifest entry, and a SHA-256 checksum.

## Abstract

MathCraft OCR is designed for mathematical documents where text, inline formulae, display equations, headers, page numbers, and multi-column layouts have to be recovered together. The runtime uses a compact ONNX model set and a structured block contract instead of relying on heavyweight PyTorch runtime imports or hidden first-run model downloads.

The current v1 model set targets three practical workloads:

| Profile | Components | Output |
| --- | --- | --- |
| `formula` | formula detector + formula recognizer | LaTeX formula text |
| `text` | text detector + text recognizer | plain OCR text |
| `mixed` | formula detector + formula recognizer + text detector + text recognizer + layout merge | structured Markdown-ready blocks |

## Model Set

Active release: `v1.0.0`

| Model ID | Runtime | Files | Purpose |
| --- | --- | --- | --- |
| `mathcraft-formula-det` | ONNX | `mathcraft-mfd.onnx` | Mathematical formula region detection |
| `mathcraft-formula-rec` | ONNX | encoder/decoder ONNX models, tokenizer and generation configs | Formula-to-LaTeX recognition |
| `mathcraft-text-det` | ONNX | `ppocrv5_mobile_det.onnx` | Fast multilingual text detection |
| `mathcraft-text-rec` | ONNX | `ppocrv5_mobile_rec.onnx`, `ppocrv5_keys.txt` | Fast multilingual text recognition |

Removed from the active v1 runtime:

- PyTorch-only layout models
- PyTorch-only table recognition models
- legacy cache-compatible model names
- hidden dependency on `torch`

## Design Principles

1. **ONNX-only inference**

   Runtime inference is performed through ONNX Runtime providers. GPU acceleration is selected through `CUDAExecutionProvider` when available, with CPU as the portable baseline.

2. **Manifest-first model management**

   Model availability is checked by a local manifest. Missing or incomplete files are detected before runtime initialization and repaired by downloading the corresponding release asset.

3. **Stable model directories**

   Model directories use MathCraft-owned names. This prevents upstream cache layout changes from breaking LaTeXSnipper.

4. **Structured OCR blocks**

   The runtime returns block-level geometry and roles, not only plain text. This allows the PDF document engine to reason about reading order, formulas, headings, headers, page numbers, and columns.

5. **No hidden first-run side effects**

   Model downloads, cache repair, provider selection, and warmup are explicit. A broken cache is treated as a recoverable state, not as a vague recognition failure.

## Runtime Architecture

```text
image / PDF page
  |
  |-- formula detector -----------+
  |                               |
  |-- text detector -- mask/split +--> OCR regions
                                  |
                                  +--> formula recognizer
                                  +--> text recognizer
                                  |
                                  v
                         MathCraftBlock stream
                                  |
                                  v
              page-aware layout ordering and role assignment
                                  |
                                  v
                  Markdown / TeX document export pipeline
```

Key layout features:

- page-aware block ordering
- column-aware reading order
- formula/text overlap arbitration
- display formula promotion for large formulas, matrices, aligned equations, and array-like structures
- header, footer, and page-number filtering
- debug visualization through `debug_blocks.png` and `debug_blocks.html`

## Baseline: Block/Layout Regression v4

The following metrics come from the local `block_layout_regression_v4` suite. They are **layout regression telemetry**, not manually labeled OCR accuracy scores. They are intended to track throughput, block structure, role assignment, and document-engine stability across representative English, Chinese, formula-heavy, and mixed-layout pages.

Environment:

- Provider: `CUDAExecutionProvider`
- Runtime: MathCraft OCR v1, ONNX-only
- Scope: single-page mixed OCR + structured block layout + Markdown export
- Cases: 10 pages across 5 PDF sources

Aggregate telemetry:

| Metric | Value |
| --- | ---: |
| Pages | 10 |
| Total blocks | 495 |
| Total text characters | 21,417 |
| Non-empty Markdown lines | 304 |
| Mean page time | 8.34 s |
| Fastest page | 1.33 s |
| Slowest page | 18.53 s |

Role distribution:

| Role | Count |
| --- | ---: |
| paragraph | 348 |
| formula | 127 |
| heading | 7 |
| header | 5 |
| formula_anchor | 4 |
| formula_label | 2 |
| page_number | 2 |

Signal flags:

| Flag | Count |
| --- | ---: |
| top_margin | 22 |
| display_formula | 20 |
| low_score | 16 |
| bottom_margin | 10 |
| header | 5 |
| page_number | 2 |

Per-page telemetry:

| Case | Page | Time | Blocks | Characters | Lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| `brouwer_p001` | 1 | 18.53 s | 33 | 1,571 | 20 |
| `brouwer_p004` | 4 | 8.67 s | 84 | 2,087 | 32 |
| `abstract_algebra_p001` | 1 | 1.33 s | 2 | 40 | 2 |
| `abstract_algebra_p018` | 18 | 11.82 s | 145 | 3,285 | 64 |
| `limits_p001` | 1 | 3.29 s | 11 | 162 | 7 |
| `limits_p012` | 12 | 8.11 s | 37 | 2,283 | 29 |
| `qing_shu_p001` | 1 | 4.95 s | 23 | 235 | 15 |
| `largest_chinese_p009` | 9 | 6.97 s | 39 | 1,252 | 33 |
| `dynamics_p001` | 1 | 13.97 s | 100 | 5,720 | 64 |
| `dynamics_p005` | 5 | 5.72 s | 21 | 4,782 | 38 |

## Representative Debug Visualizations

The following images are generated from MathCraft's structured block contract. Each bounding box is annotated with role, column/order metadata, score, and layout flags.

### Abstract Algebra, page 18

Formula-heavy English mathematical prose with dense inline and display formulae.

<img width="1700" height="2200" alt="debug_blocks" src="https://github.com/user-attachments/assets/855d0694-4068-4e32-ac5e-80f0c4178332" />

### Dynamics journal, page 5

Formula-dominant journal page with display equations, anchors, labels, headers, and page numbers.

<img width="1700" height="2200" alt="debug_blocks" src="https://github.com/user-attachments/assets/b1498ee9-32fd-4d56-bbfb-876881758587" />

### Chinese lecture note, page 1

Chinese mathematical document page with mixed text and formula blocks.

<img width="1654" height="2339" alt="debug_blocks" src="https://github.com/user-attachments/assets/5d153b2d-d089-4ee2-9ee7-c7f52eb099c9" />

### Limits and series, page 1

Sparse title/cover-style page used to check that the document engine does not overfit dense pages.

<img width="1221" height="1898" alt="debug_blocks" src="https://github.com/user-attachments/assets/6c6404e0-bea4-4811-b135-feff3a063a18" />

## Why MathCraft Performs Well

MathCraft is optimized for LaTeXSnipper's document workflow rather than generic OCR demos.

| Area | Implementation |
| --- | --- |
| Model lifecycle | Manifest-driven release assets with SHA-256 validation |
| Runtime dependency | ONNX Runtime only; no PyTorch import path in active OCR |
| First-run behavior | Explicit warmup and automatic repair for incomplete model caches |
| Text speed | Mobile PP-OCRv5 text detector/recognizer for everyday OCR latency |
| Formula handling | Formula detection is used before text OCR so formula regions can be masked/split |
| Batch behavior | Formula crops are recognized in batches where possible |
| Layout | Page-aware block ordering, role assignment, and display formula promotion |
| Debuggability | Every regression page can emit `structured.json`, `debug_blocks.png`, and `debug_blocks.html` |

Compared with implicit upstream cache-based OCR stacks, MathCraft's main advantage is not a single model trick. The improvement comes from a controlled runtime contract: deterministic model roots, explicit provider selection, recoverable downloads, structured blocks, and document-aware post-processing.

## Reproducing the Regression Suite

From the LaTeXSnipper repository:

```powershell
cd E:\LaTexSnipper
src\deps\python311\python.exe scripts\run_mathcraft_pdf_regression.py `
  --provider gpu `
  --output test_pdf\outputs\block_layout_regression_v4
```

Each case produces:

```text
document_engine.md
structured.json
debug_blocks.png
debug_blocks.html
```

## Release Asset Contract

Each model is published as a ZIP asset:

```text
mathcraft-formula-det.zip
mathcraft-formula-rec.zip
mathcraft-text-det.zip
mathcraft-text-rec.zip
SHA256SUMS.txt
```

The LaTeXSnipper manifest points to these release URLs and validates the expected files. A missing or incomplete user cache is repaired by downloading the affected model into the writable user model root, while bundled offline models remain read-only.

Default user model root:

```text
%APPDATA%\MathCraft\models
```

Bundled offline model root:

```text
<LaTeXSnipper>\_internal\MathCraft\models
```

## License and Provenance

This repository distributes model artifacts for the MathCraft OCR runtime. Upstream model provenance should be tracked per release asset and preserved in release notes. MathCraft-owned directory names are used for runtime stability; internal filenames may retain upstream-compatible names where that improves traceability.
