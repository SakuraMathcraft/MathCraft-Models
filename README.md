# MathCraft Models

Model assets for **MathCraft OCR**, the ONNX-only OCR runtime used by LaTeXSnipper.

MathCraft OCR recognizes formulae, text, and mixed mathematical documents with a compact ONNX model set. This repository provides the model release assets and the source package for the PyPI package `mathcraft-ocr` used by LaTeXSnipper.

## Quick Start

Current PyPI release: `mathcraft-ocr 0.1.7`.

Install the library and CLI without choosing an ONNX Runtime backend:

```powershell
pip install mathcraft-ocr
mathcraft --help
```

Install exactly one ONNX Runtime backend before running OCR inference.

CPU:

```powershell
pip install "mathcraft-ocr[cpu]"
```

GPU:

```powershell
pip install "mathcraft-ocr[gpu]"
```

Use only one ONNX Runtime backend in the same environment. Do not install `onnxruntime` and `onnxruntime-gpu` together.

Upgrade the current release with a chosen backend:

```powershell
pip install -U "mathcraft-ocr[gpu]"
mathcraft --help
```

Check the runtime:

```powershell
mathcraft doctor --provider auto
mathcraft models check
mathcraft warmup --profile mixed --provider auto
```

Recognize an image:

```powershell
mathcraft ocr "C:\path\to\formula.png" --profile formula --provider auto --json
```

Mixed OCR to Markdown:

```powershell
mathcraft ocr "C:\path\to\page.png" --profile mixed --provider auto --output result.md
mathcraft ocr "C:\path\to\page.png" --profile mixed --provider auto --output-dir "D:\MathCraft\outputs"
```

When a file is written, the CLI prints the resolved output path:

```text
[MATHCRAFT_OUTPUT] written to D:\MathCraft\outputs\page.md
```

PowerShell custom model cache:

```powershell
$env:MATHCRAFT_HOME="D:\MathCraft\models"
mathcraft doctor --provider auto
```

Persistent user-level cache path:

```powershell
setx MATHCRAFT_HOME "D:\MathCraft\models"
```

Open a new terminal after `setx`.

Restore the default cache path:

```powershell
[Environment]::SetEnvironmentVariable("MATHCRAFT_HOME", $null, "User")
Remove-Item Env:\MATHCRAFT_HOME -ErrorAction SilentlyContinue
mathcraft doctor --provider auto
```

Open a new terminal after removing the persistent variable. The default root is:

```text
%APPDATA%\MathCraft\models
```

## Python API

```python
from mathcraft_ocr import MathCraftRuntime

runtime = MathCraftRuntime(provider_preference="auto")
result = runtime.recognize_mixed(r"C:\path\to\page.png")

print(result.text)
for block in result.blocks:
    print(block.role, block.kind, block.text[:80])
```

## Profiles

| Profile | Use Case | Output |
| --- | --- | --- |
| `formula` | Formula screenshots | LaTeX formula text |
| `text` | Plain text OCR | Text |
| `mixed` | Text + formula documents | Markdown-ready structured text |

## Model Set

Active release: `v1.0.0`

| Model ID | Runtime | Purpose |
| --- | --- | --- |
| `mathcraft-formula-det` | ONNX | Mathematical formula region detection |
| `mathcraft-formula-rec` | ONNX | Formula-to-LaTeX recognition |
| `mathcraft-text-det` | ONNX | Fast multilingual text detection |
| `mathcraft-text-rec` | ONNX | Fast multilingual text recognition |

Release assets:

```text
mathcraft-formula-det.zip
mathcraft-formula-rec.zip
mathcraft-text-det.zip
mathcraft-text-rec.zip
SHA256SUMS.txt
```

Default writable model root:

```text
%APPDATA%\MathCraft\models
```

The runtime checks the manifest before initialization. Missing or incomplete model folders are repaired automatically by downloading only the affected model asset.

Interrupted downloads are resumable. Partial archives are stored under the active writable model root:

```text
<MATHCRAFT_HOME>\.downloads\<model_id>.zip.part
```

After a model archive is fully downloaded, verified, and extracted, the `.part` file is removed automatically.

## Results

The examples below are generated from MathCraft's structured block output. Boxes show detected roles, order, column metadata, score, and layout flags.

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

Sparse title/cover-style page used to check layout stability.

<img width="1221" height="1898" alt="debug_blocks" src="https://github.com/user-attachments/assets/6c6404e0-bea4-4811-b135-feff3a063a18" />

## Benchmark Snapshot

Local `block_layout_regression_v4` telemetry:

| Metric | Value |
| --- | ---: |
| Pages | 10 |
| Total blocks | 495 |
| Text characters | 21,417 |
| Markdown lines | 304 |
| Mean page time | 8.34 s |
| Fastest page | 1.33 s |
| Slowest page | 18.53 s |

Environment:

```text
Provider: CUDAExecutionProvider
Runtime: MathCraft OCR v1
Backend: ONNX Runtime
```

## Why It Is Stable

- ONNX Runtime only, no active PyTorch inference dependency.
- Stable MathCraft-owned model IDs and folders.
- Manifest-based file checks and cache repair.
- Resumable model downloads for slow or interrupted networks.
- Formula detection before text OCR.
- Structured blocks for headings, paragraphs, display formulae, headers, page numbers, and columns.

## LaTeXSnipper

LaTeXSnipper already integrates MathCraft OCR. Normal users do not need to install this package manually. Use this repository when you need standalone OCR, mirrored model assets, or an offline package.

Bundled offline model root:

```text
<LaTeXSnipper>\_internal\MathCraft\models
```

Missing or repaired files are written to the user model root, not into the bundled read-only directory.
