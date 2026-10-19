"""Check incomplete manifests and automatic repair without network or ONNX sessions."""

import hashlib
from dataclasses import replace

import pytest

import mathcraft_ocr.runtime as runtime_module
from mathcraft_ocr.errors import ModelCacheError
from mathcraft_ocr.manifest import load_manifest
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
    required = runtime_module.PROFILE_MODEL_IDS[profile]
    missing_id = required[0]
    missing_spec = manifest.models.pop(missing_id)
    for model_id in required[1:]:
        write_model(tmp_path, manifest, model_id)
    warmed = []
    monkeypatch.setattr(
        runtime_module,
        "ONNX_WARMUP_HANDLERS",
        {
            model_id: lambda path, provider: warmed.append(path.name)
            for model_id in required
        },
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
    required = runtime_module.PROFILE_MODEL_IDS[profile]
    downloaded, warmed = [], []

    def download(spec, *, target_root, **kwargs):
        downloaded.append(spec.model_id)
        write_model(target_root, manifest, spec.model_id)

    monkeypatch.setattr(runtime_module, "download_model_archive", download)
    monkeypatch.setattr(
        runtime_module,
        "ONNX_WARMUP_HANDLERS",
        {
            model_id: lambda path, provider: warmed.append(path.name)
            for model_id in required
        },
    )
    runtime = MathCraftRuntime(
        cache_dir=tmp_path, manifest=manifest, auto_download=True
    )
    plan = runtime.warmup(profile)
    assert plan.ready
    assert tuple(downloaded) == required
    assert tuple(warmed) == required
