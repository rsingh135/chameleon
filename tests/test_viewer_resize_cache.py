"""Content identity, immutable replay and bounded cache behavior without rendering."""

import base64
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from viewer.resize_cache import (
    ResourceBindings,
    ResourceChangedError,
    SerializedCache,
    result_key,
    source_key,
)


def _png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _source(asset="a.png"):
    return {
        "canvas": {"width": 100, "height": 80},
        "elements": [
            {
                "id": "image",
                "type": "image",
                "asset_id": asset,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 80,
            }
        ],
    }


def _result():
    return {
        "design": _source(),
        "source_png": _png(),
        "output_png": _png(),
        "warnings": ["source warning"],
        "report": {
            "violations": [],
            "total": 1.0,
            **{
                key: {}
                for key in (
                    "metrics",
                    "style",
                    "relationships",
                    "proportions",
                    "focal",
                    "rules",
                    "reading_order",
                    "occlusion",
                    "layout_state",
                )
            },
        },
        "elapsed_ms": 123456,
    }


@pytest.fixture
def resources(tmp_path):
    assets, fonts = tmp_path / "assets", tmp_path / "fonts"
    assets.mkdir()
    fonts.mkdir()
    # Binding hashes exact bytes; renderer/font validation belongs to the caller.
    (assets / "a.png").write_bytes(b"old1")
    (assets / "b.png").write_bytes(b"BBBB")
    (fonts / "face.ttf").write_bytes(b"font-A")
    (fonts / "font-manifest.json").write_text("[]", encoding="utf-8")
    return assets, fonts


def test_four_target_caller_reuses_one_source_render_but_keeps_target_results_separate(resources):
    assets, fonts = resources
    bindings = ResourceBindings()
    source_cache, result_cache = SerializedCache(kind="source"), SerializedCache(kind="result")
    execution = {"loaded_code": "frozen-A", "runtime": "test-runtime"}
    source_renders = result_computations = 0
    seen_source_keys, seen_result_keys = set(), set()
    for target in (
        {"width": 50, "height": 40},
        {"width": 100, "height": 80},
        {"width": 200, "height": 160},
        {"width": 300, "height": 240},
    ):
        identity = bindings.bind(_source(), assets, fonts)
        skey = source_key(_source(), resources=identity, execution_identity=execution)
        rkey = result_key(_source(), target, resources=identity, execution_identity=execution)
        seen_source_keys.add(skey)
        seen_result_keys.add(rkey)
        source_render = source_cache.get(skey)
        if source_render is None:
            source_renders += 1
            source_render = {"png": _png(), "warnings": []}
            assert source_cache.put_success(skey, source_render)
        result = result_cache.get(rkey)
        if result is None:
            result_computations += 1
            result = _result()
            result["design"]["canvas"] = target
            assert result_cache.put_success(rkey, result)
    assert source_renders == 1 and result_computations == 4
    assert len(seen_source_keys) == 1 and len(seen_result_keys) == 4


def test_keys_bind_source_target_attribution_resource_roots_and_execution(resources, tmp_path):
    assets, fonts = resources
    identity = ResourceBindings().bind(_source(), assets, fonts)
    options = {"resources": identity, "execution_identity": {"revision": "A"}}
    key = result_key(_source(), {"width": 100, "height": 80}, **options)
    variants = [
        result_key(_source(), {"width": 200, "height": 160}, **options),
        result_key(
            {**_source(), "note": "different source"}, {"width": 100, "height": 80}, **options
        ),
        result_key(
            _source(), {"width": 100, "height": 80}, attribution={"author": "New"}, **options
        ),
        result_key(
            _source(),
            {"width": 100, "height": 80},
            resources=identity,
            execution_identity={"revision": "B"},
        ),
    ]
    other_fonts = tmp_path / "other-font-root"
    other_fonts.mkdir()
    for path in fonts.iterdir():
        (other_fonts / path.name).write_bytes(path.read_bytes())
    variants.append(
        result_key(
            _source(),
            {"width": 100, "height": 80},
            resources=ResourceBindings().bind(_source(), assets, other_fonts),
            execution_identity={"revision": "A"},
        )
    )
    assert all(value != key for value in variants)
    assert len(set(variants)) == len(variants)
    assert source_key(_source(), **options) != key
    assert source_key(_source(), **options, attribution=None) != source_key(
        _source(), **options, attribution={}
    )


def test_different_asset_subsets_are_normal_and_receipts_cannot_be_modified(resources):
    assets, fonts = resources
    bindings = ResourceBindings()
    first = bindings.bind(_source(), assets, fonts)
    second = bindings.bind(_source("b.png"), assets, fonts)
    assert first.digest != second.digest
    receipt = first.hashes()
    receipt["assets"]["a.png"] = "poisoned"
    assert first.hashes()["assets"]["a.png"] != "poisoned"
    assert bindings.bind(_source(), assets, fonts) == first


def test_a_fresh_process_binding_changes_keys_when_resource_bytes_change_at_the_same_paths(
    resources,
):
    assets, fonts = resources
    first = ResourceBindings().bind(_source(), assets, fonts)
    options = {"execution_identity": "same-loaded-code"}
    original_key = source_key(_source(), resources=first, **options)
    (assets / "a.png").write_bytes(b"new2")
    # A new registry models a restarted process/new native-resource lifetime.
    restarted = ResourceBindings().bind(_source(), assets, fonts)
    assert source_key(_source(), resources=restarted, **options) != original_key


@pytest.mark.parametrize("relative", ["assets/a.png", "fonts/face.ttf", "fonts/font-manifest.json"])
def test_same_size_same_mtime_resource_changes_require_a_restart(resources, relative):
    assets, fonts = resources
    bindings = ResourceBindings()
    bindings.bind(_source(), assets, fonts)
    root, name = relative.split("/")
    path = (assets if root == "assets" else fonts) / name
    before = path.stat()
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ResourceChangedError, match="Restart"):
        bindings.bind(_source(), assets, fonts)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_font_inventory_changes_are_not_hidden_by_unchanged_referenced_faces(resources, change):
    assets, fonts = resources
    bindings = ResourceBindings()
    bindings.bind(_source(), assets, fonts)
    if change == "add":
        (fonts / "new-face.otf").write_bytes(b"new-face")
    else:
        (fonts / "face.ttf").unlink()
    with pytest.raises(ResourceChangedError, match="Font inventory"):
        bindings.bind(_source(), assets, fonts)


def test_binding_budget_fails_closed_instead_of_eviction_forgetting_native_resource_identity(
    resources,
):
    assets, fonts = resources
    bindings = ResourceBindings(max_paths=3)
    first = bindings.bind(_source(), assets, fonts)
    with pytest.raises(ResourceChangedError, match="budget is full"):
        bindings.bind(_source("b.png"), assets, fonts)
    assert bindings.bind(_source(), assets, fonts) == first


def test_model_requests_explicitly_bypass_result_cache_without_a_model_execution_identity(
    resources,
):
    assets, fonts = resources
    identity = ResourceBindings().bind(_source(), assets, fonts)
    key = result_key(
        _source(),
        {"width": 100, "height": 80},
        resources=identity,
        execution_identity={"revision": "A"},
        model_enabled=True,
    )
    cache = SerializedCache(kind="result")
    assert key is None and cache.get(key) is None
    assert not cache.put_success(key, _result())
    assert cache.info()["entries"] == 0


def test_cached_successes_own_no_mutable_caller_objects_and_never_replay_old_elapsed_time():
    cache = SerializedCache(kind="result")
    original = _result()
    assert cache.put_success("job", original)
    original["design"]["elements"][0]["asset_id"] = "poisoned-after-put"
    first = cache.get("job")
    assert "elapsed_ms" not in first
    first["elapsed_ms"] = 0.25  # The caller supplies this request's measured elapsed time.
    first["design"]["elements"][0]["asset_id"] = "poisoned-after-get"
    first["report"]["metrics"]["new"] = "poisoned"
    first["warnings"].append("poisoned")
    second = cache.get("job")
    assert second["design"]["elements"][0]["asset_id"] == "a.png"
    assert second["report"]["metrics"] == {}
    assert second["warnings"] == ["source warning"]
    assert "elapsed_ms" not in second


def test_completed_source_png_and_warnings_are_also_returned_as_fresh_values():
    cache = SerializedCache(kind="source")
    value = {"png": _png(), "warnings": ["kept"]}
    assert cache.put_success("source", value)
    value["warnings"].clear()
    cache.get("source")["warnings"].append("not retained")
    assert cache.get("source") == {"png": _png(), "warnings": ["kept"]}


@pytest.mark.parametrize("kind", ["source", "result"])
def test_errors_partials_and_nonserializable_values_remain_misses(kind):
    cache = SerializedCache(kind=kind)
    values = [{"error": "failed"}, {}, {"png": _png()}, RuntimeError("failed"), {"x": float("nan")}]
    if kind == "result":
        failed = _result()
        failed["report"]["violations"] = ["off_canvas"]
        partial = _result()
        partial["output_png"] = "data:image/png;base64,iVBORw=="
        values.extend([failed, partial, {**_result(), "status": "running"}])
    for index, value in enumerate(values):
        assert not cache.put_success(str(index), value)
        assert cache.get(str(index)) is None
    assert cache.info()["entries"] == 0


def test_lru_entry_budget_and_oversized_raw_values_are_bounded():
    cache = SerializedCache(kind="source", max_bytes=2048, max_entries=2, max_entry_bytes=512)
    value = {"png": _png(), "warnings": []}
    assert cache.put_success("A", value) and cache.put_success("B", value)
    assert cache.get("A") == value
    assert cache.put_success("C", value)
    assert cache.get("B") is None and cache.get("A") == value and cache.get("C") == value
    # Highly compressible data still obeys the uncompressed allocation bound.
    assert not cache.put_success("too-large", {"png": _png(), "warnings": ["x" * 10000]})
    assert cache.get("too-large") is None
    cache.clear()
    assert cache.info()["entries"] == cache.info()["stored_bytes"] == 0


def test_byte_budget_evicts_even_when_entry_count_would_allow_more():
    cache = SerializedCache(kind="source", max_bytes=700, max_entries=100, compress=False)
    value = {"png": _png(), "warnings": ["large warning " * 10]}
    for index in range(10):
        assert cache.put_success(str(index), value)
        assert cache.info()["stored_bytes"] <= 700
    assert cache.get("0") is None and cache.get("9") == value


def test_double_lookup_under_existing_engine_lock_coalesces_parallel_misses():
    cache = SerializedCache(kind="result")
    lock, barrier = threading.Lock(), threading.Barrier(8)
    computations = []

    def caller(_):
        value = cache.get("same-job")
        barrier.wait(timeout=5)  # All callers see the initial miss before the owner computes.
        if value is None:
            with lock:
                value = cache.get("same-job")
                if value is None:
                    computations.append(True)
                    value = _result()
                    assert cache.put_success("same-job", value)
                    value = cache.get("same-job")
        return value

    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(caller, range(8)))
    assert computations == [True]
    replies[0]["warnings"].append("caller mutation")
    assert all(reply["warnings"] == ["source warning"] for reply in replies[1:])


def test_a_failed_computation_is_not_cached_and_a_later_retry_can_succeed():
    cache = SerializedCache(kind="result")

    def caller(fail):
        value = cache.get("job")
        if value is None:
            if fail:
                raise OSError("injected renderer failure")
            value = _result()
            assert cache.put_success("job", value)
        return value

    with pytest.raises(OSError, match="renderer failure"):
        caller(True)
    assert cache.get("job") is None
    assert caller(False)["design"] == _source()
    assert cache.get("job")["design"] == _source()


def test_key_validation_requires_frozen_execution_and_rejects_nonfinite_provenance(resources):
    assets, fonts = resources
    identity = ResourceBindings().bind(_source(), assets, fonts)
    with pytest.raises(ValueError, match="execution identity"):
        source_key(_source(), resources=identity, execution_identity={})
    with pytest.raises(ValueError):
        source_key(_source(), resources=identity, execution_identity={"value": float("inf")})
    before = deepcopy(_source())
    source_key(before, resources=identity, execution_identity="frozen")
    assert before == _source()


def test_legacy_safe_missing_image_is_explicit_incomplete_and_uncacheable(resources):
    assets, fonts = resources
    source = _source("missing.png")
    before = deepcopy(source)
    bindings = ResourceBindings()
    with pytest.raises(ValueError, match="missing"):
        bindings.bind(source, assets, fonts)
    identity = bindings.bind(source, assets, fonts, allow_missing=True)
    assert identity.missing_assets == ("missing.png",) and identity.complete is False
    assert "missing.png" in identity.hashes()["assets"]
    assert source == before
    assert len(source_key(source, resources=identity, execution_identity="frozen")) == 64
    assert (
        result_key(
            source,
            {"width": 100, "height": 80},
            resources=identity,
            execution_identity="frozen",
        )
        is None
    )
    # The source key still binds mutation checks; incomplete previews do not
    # acquire a storage key in the caller's legacy warning-only path.
    source_cache = SerializedCache(kind="source")
    storage_key = None if not identity.complete else "source-render"
    assert not source_cache.put_success(storage_key, {"png": _png(), "warnings": ["missing"]})
    assert source_cache.info()["entries"] == 0


def test_a_previously_missing_image_appearing_requires_restart_even_for_a_strict_bind(resources):
    assets, fonts = resources
    bindings = ResourceBindings()
    bindings.bind(_source("missing.png"), assets, fonts, allow_missing=True)
    (assets / "missing.png").write_bytes(b"newly available bytes")
    with pytest.raises(ResourceChangedError, match="Restart"):
        bindings.bind(_source("missing.png"), assets, fonts)


@pytest.mark.parametrize(
    "name",
    [
        "../missing.png",
        "folder/../missing.png",
        "/missing.png",
        "C:\\missing.png",
        "C:missing.png",
        "\\\\host\\share\\missing.png",
        "absolute-inside-root",
    ],
)
def test_allow_missing_never_relaxes_relative_path_security(resources, name):
    assets, fonts = resources
    if name == "absolute-inside-root":
        name = str(assets / "missing.png")
    with pytest.raises(ValueError):
        ResourceBindings().bind(_source(name), assets, fonts, allow_missing=True)


def test_allow_missing_rejects_symlink_escape_before_missing_file_fallback(resources, tmp_path):
    assets, fonts = resources
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (assets / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable on this host")
    with pytest.raises(ValueError, match="leaves"):
        ResourceBindings().bind(_source("escape/missing.png"), assets, fonts, allow_missing=True)


def test_allow_missing_images_keeps_font_directory_validation_strict(resources):
    assets, fonts = resources
    with pytest.raises(ValueError, match="directories must exist"):
        ResourceBindings().bind(
            _source("missing.png"), assets, fonts / "not-present", allow_missing=True
        )
