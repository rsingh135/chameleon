"""External poster evidence is reproducible, red-capable, and keeps reserves untouched."""

import hashlib
import json
from copy import deepcopy

import pytest
from PIL import Image

from tools import poster_audit as audit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "license.txt").write_text("Test fixture license", encoding="utf-8")
    entries = []
    for case_id, split in [("tested", "reviewed"), ("reserve", "reserve")]:
        prepared = root / "prepared" / case_id
        (prepared / "resources/assets").mkdir(parents=True)
        _write(prepared / "resources/fonts/font-manifest.json", [])
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120"/>'
        original = root / f"{case_id}.svg"
        original.write_bytes(svg)
        (prepared / "original.svg").write_bytes(svg)
        Image.new("RGBA", (100, 120), "white").save(prepared / "native.png")
        source = {
            "canvas": {"width": 100, "height": 120},
            "elements": [
                {
                    "id": "panel",
                    "type": "shape",
                    "x": 40,
                    "y": 40,
                    "width": 20,
                    "height": 30,
                    "shape_type": "rectangle",
                    "fill_color": "#cc3300",
                }
            ],
        }
        _write(prepared / "source.json", source)
        _write(
            prepared / "import-report.json",
            {
                "status": "ready",
                "failures": [],
                "node_mapping": [],
                "editable_text_count": 0,
                "vector_art_count": 1,
            },
        )
        entries.append(
            {
                "id": case_id,
                "title": "<Native poster> " + case_id,
                "split": split,
                "author": "Fixture author",
                "source_url": "https://example.org/poster",
                "license_id": "CC0-1.0",
                "license_url": "https://example.org/license",
                "license_path": "license.txt",
                "original_path": f"{case_id}.svg",
                "sha256": hashlib.sha256(svg).hexdigest(),
                "prepared_path": f"prepared/{case_id}",
                "invariants": [
                    {
                        "name": "Authored central panel",
                        "kind": "center_x",
                        "element_ids": ["panel"],
                        "max_drift": 0.02,
                    }
                ],
            }
        )
    path = root / "catalogue.json"
    _write(path, {"schema_version": 1, "examples": entries})
    return path, entries


@pytest.fixture
def runtime(monkeypatch):
    calls = []

    def resize(source, target):
        calls.append(deepcopy(source))
        design = deepcopy(source)
        design["canvas"] = target
        scale = min(
            target["width"] / source["canvas"]["width"],
            target["height"] / source["canvas"]["height"],
        )
        dx = (target["width"] - source["canvas"]["width"] * scale) / 2
        dy = (target["height"] - source["canvas"]["height"] * scale) / 2
        for element in design["elements"]:
            element.update(
                x=element["x"] * scale + dx,
                y=element["y"] * scale + dy,
                width=element["width"] * scale,
                height=element["height"] * scale,
            )
        return {
            "design": design,
            "state": {
                "name": "fixture-selection",
                "cluster_axes": {},
                "source_digest": audit._digest(source),
            },
        }

    monkeypatch.setattr(audit, "_resize", resize)
    monkeypatch.setattr(
        audit,
        "_render",
        lambda design: (Image.new("RGBA", tuple(design["canvas"].values()), "white"), []),
    )
    return calls, resize


def test_reviewed_formats_and_reserve_imports_are_distinct(corpus, runtime, tmp_path):
    manifest, _entries = corpus
    before = manifest.read_bytes()
    out = tmp_path / "gallery"
    report = audit.run_audit(manifest, out)
    assert len(runtime[0]) == 3
    assert report["summary"]["reviewed_posters"] == 1
    assert report["summary"]["reserve_posters_untested"] == 1
    assert report["summary"]["target_results"] == 3
    assert report["technical_pass"] and report["visual_approval"] is False
    tested, reserve = report["examples"]
    assert tested["source_immutable"]
    assert [cell["name"] for cell in tested["targets"]] == list(audit.TARGETS)
    assert all(cell["technical_pass"] for cell in tested["targets"])
    assert all(cell["visual_review"]["status"] == "pending" for cell in tested["targets"])
    assert reserve["status"] == "reserve_untested" and reserve["targets"] == []
    assert not list((out / "reserve").glob("*.plan.json"))
    assert not (out / "reserve/converted-source.png").exists()
    assert (out / "index.html").is_file() and (out / "contact_sheet.png").is_file()
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "&lt;Native poster&gt;" in page and "<Native poster>" not in page
    assert "reserve_untested" in page and "http://127.0.0.1" not in page
    assert manifest.read_bytes() == before


def test_reserve_only_and_gallery_only_never_call_engine(corpus, runtime, tmp_path, monkeypatch):
    manifest, _ = corpus
    out = tmp_path / "gallery"
    audit.run_audit(manifest, out)

    def forbidden(*_args):
        raise AssertionError("The engine/render backend must not run")

    monkeypatch.setattr(audit, "_resize", forbidden)
    monkeypatch.setattr(audit, "_render", forbidden)
    saved = audit.run_audit(manifest, out, gallery_only=True)
    assert saved["technical_pass"]
    assert saved["summary"]["target_results"] == 3
    reserve = audit.run_audit(manifest, tmp_path / "reserve-only", case_ids=["reserve"])
    assert reserve["technical_pass"] and reserve["summary"]["target_results"] == 0


def test_source_authored_invariant_can_fail_otherwise_valid_geometry(
    corpus, runtime, tmp_path, monkeypatch
):
    manifest, _ = corpus

    def shifted(source, target):
        plan = runtime[1](source, target)
        plan["design"]["elements"][0]["x"] = target["width"] * 0.8
        return plan

    monkeypatch.setattr(audit, "_resize", shifted)
    report = audit.run_audit(manifest, tmp_path / "gallery")
    assert not report["technical_pass"]
    assert report["summary"]["technical_check_failures"] == 3
    for target in report["examples"][0]["targets"]:
        assert target["checks"]["contract"]["passed"]
        assert target["invariants"]["passed"] is False
        assert target["invariants"]["rules"][0]["value"] > 0.02


def test_partial_engine_and_render_failures_remain_explicit(corpus, runtime, tmp_path, monkeypatch):
    manifest, _ = corpus
    original_render = audit._render

    def resize(source, target):
        if target["width"] == target["height"]:
            raise RuntimeError("square engine failed")
        return runtime[1](source, target)

    def render(design):
        if design["canvas"] == audit.TARGETS["story"]:
            raise RuntimeError("story renderer failed")
        return original_render(design)

    monkeypatch.setattr(audit, "_resize", resize)
    monkeypatch.setattr(audit, "_render", render)
    report = audit.run_audit(manifest, tmp_path / "gallery")
    landscape, square, story = report["examples"][0]["targets"]
    assert landscape["png_file"] == "landscape.png"
    assert square["failures"][0]["phase"] == "engine"
    assert story["failures"][0]["phase"] == "render"
    assert not report["technical_pass"]


def test_mutating_engine_is_red_without_changing_source_artifacts(
    corpus, runtime, tmp_path, monkeypatch
):
    manifest, _ = corpus
    source = manifest.parent / "prepared/tested/source.json"
    before = source.read_bytes()

    def mutate(original, target):
        result = runtime[1](original, target)
        original["elements"][0]["x"] = 999
        return result

    monkeypatch.setattr(audit, "_resize", mutate)
    report = audit.run_audit(manifest, tmp_path / "gallery")
    assert not report["technical_pass"]
    assert report["examples"][0]["source_immutable"]
    assert all(
        "mutated" in cell["failures"][0]["error"] for cell in report["examples"][0]["targets"]
    )
    assert source.read_bytes() == before


def test_unsupported_import_counts_as_failure_and_never_calls_engine(corpus, runtime, tmp_path):
    manifest, _ = corpus
    _write(
        manifest.parent / "prepared/tested/import-report.json",
        {
            "status": "unsupported",
            "failures": [{"code": "missing_font", "message": "Original face unavailable"}],
        },
    )
    report = audit.run_audit(manifest, tmp_path / "gallery")
    assert runtime[0] == []
    assert report["summary"]["import_failures"] == 1
    assert not report["technical_pass"]
    assert "missing_font" in report["examples"][0]["failures"][0]["error"]
    assert report["examples"][1]["status"] == "reserve_untested"


def test_gallery_only_refuses_stale_prepared_or_png_evidence(corpus, runtime, tmp_path):
    manifest, _ = corpus
    out = tmp_path / "gallery"
    audit.run_audit(manifest, out)
    (out / "tested/landscape.png").write_bytes(b"changed")
    report = audit.run_audit(manifest, out, gallery_only=True)
    assert not report["technical_pass"]
    landscape = report["examples"][0]["targets"][0]
    assert "png_file" not in landscape
    assert landscape["failures"][0]["phase"] == "saved_evidence"
    source_path = manifest.parent / "prepared/tested/source.json"
    source = json.loads(source_path.read_text())
    source["elements"][0]["x"] += 1
    _write(source_path, source)
    stale = audit.run_audit(manifest, out, gallery_only=True)
    assert stale["examples"][0]["status"] == "untested"
    assert stale["examples"][0]["targets"] == []


def test_original_hash_failure_is_visible_before_engine(corpus, runtime, tmp_path):
    manifest, _ = corpus
    (manifest.parent / "tested.svg").write_bytes(b"replaced")
    report = audit.run_audit(manifest, tmp_path / "gallery")
    assert not runtime[0]
    assert report["examples"][0]["status"] == "import_failed"
    assert "SHA-256" in report["examples"][0]["failures"][0]["error"]


def test_json_original_and_author_thumbnail_are_labeled_truthfully(corpus, runtime, tmp_path):
    manifest, entries = corpus
    entry = entries[0]
    original = b'{"editable":"author page data"}'
    entry.update(
        original_path="tested.json",
        sha256=hashlib.sha256(original).hexdigest(),
        native_reference={"role": "author thumbnail"},
        attribution_text="Required visible credit",
    )
    (manifest.parent / "tested.json").write_bytes(original)
    (manifest.parent / "prepared/tested/original.json").write_bytes(original)
    _write(manifest, {"schema_version": 1, "examples": entries})
    out = tmp_path / "gallery"
    report = audit.run_audit(manifest, out)
    assert (out / "tested/original.json").read_bytes() == original
    assert report["examples"][0]["original_file"] == "original.json"
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "Required visible credit" in page
    assert "original app thumbnail" in page


def test_unknown_invariant_is_a_red_configuration_error():
    source = {"canvas": {"width": 100, "height": 100}, "elements": []}
    result = audit.evaluate_invariants(source, source, [{"kind": "invented"}], "landscape")
    assert result["passed"] is False
    assert "Unknown invariant" in result["rules"][0]["error"]


def test_source_pixel_threshold_is_explicit_and_red_capable(corpus, runtime, tmp_path, monkeypatch):
    manifest, entries = corpus
    entries[0]["invariants"].append(
        {
            "kind": "source_pixel_mae",
            "targets": ["source"],
            "maximum": 0.01,
        }
    )
    _write(manifest, {"schema_version": 1, "examples": entries})
    monkeypatch.setattr(
        audit,
        "_render",
        lambda design: (Image.new("RGBA", tuple(design["canvas"].values()), "red"), []),
    )
    report = audit.run_audit(manifest, tmp_path / "gallery")
    assert report["summary"]["source_check_failures"] == 1
    assert report["examples"][0]["source_pixels"]["rgb_mae"] > 0.01
    assert not report["technical_pass"]
    assert all(cell["technical_pass"] for cell in report["examples"][0]["targets"])


def test_native_raster_failure_is_distinct_from_import_and_engine(corpus, runtime, tmp_path):
    manifest, _ = corpus
    (manifest.parent / "prepared/tested/native.png").write_bytes(b"invalid PNG")
    report = audit.run_audit(manifest, tmp_path / "gallery")
    case = report["examples"][0]
    assert case["failures"][0]["phase"] == "native_render"
    assert case["import"]["status"] == "ready"
    assert len(case["targets"]) == 3 and len(runtime[0]) == 3
    assert not report["technical_pass"]
    assert "native_file" not in case


def test_visual_approval_requires_matching_evidence_and_named_reviewer():
    entry = {
        "visual_reviews": {
            "story": {"status": "approved", "reviewer": "Designer", "evidence_digest": "old"}
        }
    }
    assert audit._review(entry, "story", "new")["status"] == "pending"
    entry["visual_reviews"]["story"]["evidence_digest"] = "new"
    assert audit._review(entry, "story", "new")["status"] == "approved"


def test_filters_and_output_cannot_overwrite_corpus(corpus, tmp_path):
    manifest, _ = corpus
    with pytest.raises(ValueError, match="unknown"):
        audit.run_audit(manifest, tmp_path / "gallery", case_ids=["missing"])
    with pytest.raises(ValueError, match="corpus root"):
        audit.run_audit(manifest, manifest.parent / "gallery")
