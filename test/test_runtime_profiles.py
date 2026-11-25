"""Check incomplete manifests and automatic repair without network or ONNX sessions."""

import hashlib
from dataclasses import replace

import pytest

import mathcraft_ocr.runtime as runtime_module
from mathcraft_ocr.errors import ModelCacheError
from mathcraft_ocr.manifest import load_manifest
from mathcraft_ocr.profiles import CPU_PROVIDER, CUDA_PROVIDER, model_id_for_role
from mathcraft_ocr.providers import ProviderInfo
from mathcraft_ocr.runtime import MathCraftRuntime


@pytest.fixture(autouse=True)
def cpu_provider(monkeypatch):
    provider = ProviderInfo(
        available_providers=("CPUExecutionProvider",),
        active_provider="CPUExecutionProvider",
        device="cpu",
        gpu_requested=False,
        gpu_runtime_ok=False,
    )
    monkeypatch.setattr(runtime_module, "detect_providers", lambda **kwargs: provider)
    monkeypatch.setattr(runtime_module, "confirm_provider_device", lambda value: value)
    return provider


def write_model(root, manifest, model_id):
    spec = manifest.models[model_id]
    manifest.models[model_id] = replace(
        spec,
        files=tuple(
            replace(item, sha256=hashlib.sha256(b"weights").hexdigest())
            for item in spec.files
        ),
    )
    for item in spec.files:
        path = root / model_id / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")


@pytest.mark.parametrize("profile", ["formula", "text", "mixed"])
@pytest.mark.parametrize("auto_download", [False, True])
def test_custom_manifest_reports_undeclared_model_and_accepts_completion(
    tmp_path, monkeypatch, profile, auto_download
):
    manifest = load_manifest()
    roles = runtime_module.PROFILE_MODEL_IDS[profile]
    required = tuple(model_id_for_role(role, CPU_PROVIDER) for role in roles)
    missing_id = required[0]
    missing_spec = manifest.models.pop(missing_id)
    for model_id in required[1:]:
        write_model(tmp_path, manifest, model_id)
    warmed = []
    monkeypatch.setattr(
        runtime_module,
        "ONNX_WARMUP_HANDLERS",
        {role: lambda path, provider: warmed.append(path.name) for role in roles},
    )
    monkeypatch.setattr(
        runtime_module,
        "download_model_archive",
        lambda *args, **kwargs: pytest.fail("Cannot download an undeclared model"),
    )
    runtime = MathCraftRuntime(
        cache_dir=tmp_path,
        manifest=manifest,
        provider_preference="cpu",
        auto_download=auto_download,
    )
    plan = runtime.warmup(profile)
    assert not plan.ready
    assert plan.missing_models == (missing_id,)
    assert set(warmed) == set(required[1:])
    with pytest.raises(ModelCacheError, match=missing_id):
        getattr(runtime, f"recognize_{profile}")(None)

    manifest.models[missing_id] = missing_spec
    write_model(tmp_path, manifest, missing_id)
    assert runtime.warmup(profile).ready


@pytest.mark.parametrize("profile", ["formula", "text", "mixed"])
def test_declared_missing_files_are_downloaded_before_warmup(
    tmp_path, monkeypatch, profile
):
    manifest = load_manifest()
    roles = runtime_module.PROFILE_MODEL_IDS[profile]
    required = tuple(model_id_for_role(role, CPU_PROVIDER) for role in roles)
    downloaded, warmed = [], []

    def download(spec, *, target_root, **kwargs):
        downloaded.append(spec.model_id)
        write_model(target_root, manifest, spec.model_id)

    monkeypatch.setattr(runtime_module, "download_model_archive", download)
    monkeypatch.setattr(
        runtime_module,
        "ONNX_WARMUP_HANDLERS",
        {role: lambda path, provider: warmed.append(path.name) for role in roles},
    )
    runtime = MathCraftRuntime(
        cache_dir=tmp_path, manifest=manifest, auto_download=True
    )
    plan = runtime.warmup(profile)
    assert plan.ready
    assert tuple(downloaded) == required
    assert tuple(warmed) == required


def test_cuda_mixed_selects_one_matching_formula_pair(
    tmp_path, monkeypatch, cpu_provider
):
    manifest = load_manifest()
    expected = (
        "mathcraft-formula-det-cuda-fp16",
        "mathcraft-formula-rec-cuda-fp16",
        "mathcraft-text-det",
        "mathcraft-text-rec",
    )
    for model_id in expected:
        write_model(tmp_path, manifest, model_id)
    provider = replace(cpu_provider, active_provider=CUDA_PROVIDER, device="gpu")
    monkeypatch.setattr(runtime_module, "detect_providers", lambda **_kwargs: provider)
    warmed = []
    monkeypatch.setattr(
        runtime_module,
        "ONNX_WARMUP_HANDLERS",
        {
            role: lambda path, info: warmed.append((path.name, info.active_provider))
            for role in runtime_module.PROFILE_MODEL_IDS["mixed"]
        },
    )
    runtime = MathCraftRuntime(
        cache_dir=tmp_path, manifest=manifest, auto_download=False
    )
    plan = runtime.warmup("mixed")
    assert plan.ready
    assert plan.required_models == expected
    assert warmed == [(name, CUDA_PROVIDER) for name in expected]


@pytest.mark.parametrize(
    ("backend", "requested"),
    [
        ("TensorrtExecutionProvider", "auto"),
        ("DmlExecutionProvider", "auto"),
        ("CoreMLExecutionProvider", "auto"),
        ("OpenVINOExecutionProvider", "auto"),
        (CPU_PROVIDER, "fp32"),
        (CUDA_PROVIDER, "fp32"),
        (CPU_PROVIDER, "fp32-v2"),
        (CUDA_PROVIDER, "fp32-v2"),
        ("DmlExecutionProvider", "fp32-v2"),
        ("CoreMLExecutionProvider", "fp32-v2"),
        ("OpenVINOExecutionProvider", "fp32-v2"),
        ("TensorrtExecutionProvider", "fp32-v2"),
    ],
)
def test_standalone_formula_selects_fp32(
    tmp_path, monkeypatch, cpu_provider, backend, requested
):
    manifest = load_manifest()
    role = "mathcraft-formula-rec"
    expected = f"{role}-fp32-v2" if requested == "fp32-v2" else role
    write_model(tmp_path, manifest, expected)
    provider = replace(cpu_provider, active_provider=backend)
    monkeypatch.setattr(runtime_module, "detect_providers", lambda **_: provider)
    seen = []
    monkeypatch.setitem(
        runtime_module.ONNX_WARMUP_HANDLERS,
        role,
        lambda path, info, **options: seen.append(
            (path.name, info.active_provider, options.get("model_variant", "auto"))
        ),
    )
    runtime = MathCraftRuntime(
        cache_dir=tmp_path,
        manifest=manifest,
        formula_variant=requested,
        auto_download=False,
    )
    plan = runtime.warmup("formula")
    assert plan.ready
    assert plan.required_models == (expected,)
    assert seen == [(expected, backend, requested)]
