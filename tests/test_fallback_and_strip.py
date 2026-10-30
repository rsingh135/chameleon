"""The engine must always produce a valid output, and extreme canvases must get
band layouts rather than a shrunken poster."""

import json
from pathlib import Path

import pytest

from engine.eval import build_context, check_contract
from engine.eval.catalogue import CATALOGUE
from engine.layout.fallback import safe_candidate
from engine.layout.model import build_context as layout_context
from engine.layout.strip import TALL_RATIO, WIDE_RATIO, generate_strip_candidates
from engine.resize import resize

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {
    p.stem.removeprefix("chameleon_sample_"): json.loads(p.read_text())["source"]
    for p in sorted((REPO_ROOT / "input").glob("*.json"))
}
EXTREME = {
    k: v
    for k, v in CATALOGUE.items()
    if k in ("leaderboard", "mobile_banner", "wide_skyscraper", "linkedin_banner")
}


@pytest.mark.parametrize("name", sorted(SAMPLES))
@pytest.mark.parametrize("target", list(CATALOGUE.values()), ids=list(CATALOGUE))
def test_safe_candidate_always_passes_contract(name, target):
    source = SAMPLES[name]
    design = safe_candidate(source, target)
    assert check_contract(source, design, target, build_context(source)) == []


def test_safe_candidate_handles_unbreakable_word():
    source = {
        "canvas": {"width": 1000, "height": 1000},
        "elements": [
            {
                "id": "t",
                "type": "text",
                "x": 100,
                "y": 100,
                "width": 800,
                "height": 200,
                "content": "Supercalifragilisticexpialidocious-Solstice-Limited",
                "font_family": "Solstice Sans",
                "font_size": 90,
                "color": "rgba(0,0,0,1)",
            }
        ],
    }
    target = {"width": 300, "height": 250}
    design = safe_candidate(source, target)
    assert check_contract(source, design, target, build_context(source)) == []


@pytest.mark.parametrize("name", sorted(SAMPLES))
@pytest.mark.parametrize("target", list(EXTREME.values()), ids=list(EXTREME))
def test_strip_candidates_are_valid(name, target):
    source = SAMPLES[name]
    ratio = target["width"] / target["height"]
    candidates = generate_strip_candidates(source, target, layout_context(source))
    assert candidates, "extreme targets must get band proposals"
    ctx = build_context(source)
    floor = min(10.0, 0.08 * min(target.values()))
    for design in candidates:
        assert check_contract(source, design, target, ctx) == []
        sizes = [el["font_size"] for el in design["elements"] if el["type"] == "text"]
        assert min(sizes) >= 1.0
        # The headline-sized copy must be readable; fine print may not be.
        assert max(sizes) >= floor
    assert ratio >= WIDE_RATIO or ratio <= TALL_RATIO


def test_no_strip_candidates_for_square():
    source = SAMPLES["level2_core"]
    assert (
        generate_strip_candidates(source, {"width": 1080, "height": 1350}, layout_context(source))
        == []
    )


def test_resize_survives_pipeline_failure(monkeypatch):
    """Any exception inside candidate generation/selection degrades to the
    safe fallback instead of propagating."""
    import engine.resize as mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(mod, "_candidate_pool", boom)
    mod._resize_cached.cache_clear()
    source = SAMPLES["level3_advanced"]
    target = {"width": 728, "height": 90}
    design = resize(source, target)
    assert check_contract(source, design, target, build_context(source)) == []
    mod._resize_cached.cache_clear()


def test_main_writes_output_even_when_resize_raises(monkeypatch, tmp_path):
    import engine.main as main_mod

    task = {"source": SAMPLES["level1_warmup"], "target_canvas": {"width": 300, "height": 250}}
    inp, out = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    (inp / "t.json").write_text(json.dumps(task))
    monkeypatch.setattr(main_mod, "INPUT_DIR", inp)
    monkeypatch.setattr(main_mod, "OUTPUT_DIR", out)
    monkeypatch.setattr(main_mod, "ASSETS_DIR", REPO_ROOT / "brand_kit/assets")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(main_mod, "resize", boom)
    assert main_mod.main() == 0
    assert (out / "t_output.json").stat().st_size > 0
    assert (out / "t_output.png").stat().st_size > 0
    design = json.loads((out / "t_output.json").read_text())
    assert [e["id"] for e in design["elements"]] == [e["id"] for e in task["source"]["elements"]]
