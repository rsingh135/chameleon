"""Static Doory conversion preserves native data and rejects unsupported features."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from tools import doory_import as adapter

ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _base(kind="rect", **kwargs):
    return {
        "id": "native-node",
        "type": kind,
        "originX": "center",
        "originY": "center",
        "left": 50,
        "top": 50,
        "width": 40,
        "height": 30,
        "fill": "#ff0000",
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "opacity": 1,
        **kwargs,
    }


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "LICENSE").write_text("Fixture AAL license", encoding="utf-8")
    (root / "NOTICE").write_text("Fixture author notice", encoding="utf-8")
    (root / "OFL.txt").write_text("Fixture font license", encoding="utf-8")
    font = ROOT / "brand_kit/fonts/SolsticeSans-Regular.ttf"
    data = font.read_bytes()
    (root / "regular.ttf").write_bytes(data)
    entry = {
        "id": "fixture-doory",
        "original_path": "original.json",
        "prepared_path": "prepared/fixture-doory",
        "title": "Fixture poster",
        "author": "Fixture author",
        "attribution_text": "Required fixture credit",
        "source_url": "https://example.org/original.json",
        "license_id": "LicenseRef-Doory-AAL",
        "license_url": "https://example.org/LICENSE",
        "license_path": "LICENSE",
        "notice_path": "NOTICE",
        "font_requirements": [
            {
                "family": "Solstice Sans",
                "weight": "400",
                "style": "normal",
                "path": "regular.ttf",
                "postscript_name": "SolsticeSans-Regular",
                "sha256": hashlib.sha256(data).hexdigest(),
                "license_path": "OFL.txt",
            }
        ],
        "asset_requirements": [],
    }
    document = {
        "schemaVersion": 1,
        "width": 100,
        "height": 120,
        "unit": "in",
        "pages": [
            {
                "id": "page-1",
                "name": "Page one",
                "background": {"type": "solid", "color": "#ffffff"},
                "objects": [_base()],
            }
        ],
    }

    def prepare(nodes=None, changes=None, page_index=None):
        source = deepcopy(document)
        if nodes is not None:
            source["pages"][0]["objects"] = nodes
        if changes:
            source.update(changes)
        _write(root / "original.json", source)
        entry["sha256"] = hashlib.sha256((root / "original.json").read_bytes()).hexdigest()
        out = root / "prepared/fixture-doory"
        return adapter.prepare_doory(entry, root, out, page_index), out, source

    return root, entry, document, prepare


def _text(**changes):
    return _base(
        "textbox",
        width=80,
        height=22.6,
        left=10,
        top=15,
        originX="left",
        originY="top",
        text="Literal message",
        fontFamily='"Solstice Sans", sans-serif',
        fontWeight=400,
        fontStyle="normal",
        fontSize=20,
        lineHeight=1.2,
        charSpacing=100,
        textAlign="left",
        styles=[],
        **changes,
    )


def test_editable_text_and_license_travel_in_the_fresh_bundle(corpus):
    root, entry, _, prepare = corpus
    report, out, original = prepare([_text()])
    assert report["status"] == "ready", report
    assert report["editable_text_count"] == 1
    assert report["font_provenance"]["historical_source_bytes_verified"] is False
    bundle = json.loads((out / "bundle.json").read_text())
    assert bundle["poster_id"] == bundle["task_id"] == entry["id"]
    assert bundle["attribution"]["text"] == entry["attribution_text"]
    assert len(bundle["attribution"]["licenses"]) == 3
    assert any("notice" in item["text"].lower() for item in bundle["attribution"]["licenses"])
    text = bundle["source"]["elements"][1]
    assert text["type"] == "text" and text["content"] == original["pages"][0]["objects"][0]["text"]
    assert text["font_family"] == "SolsticeSans-Regular"
    assert text["font_size"] == 20 and text["letter_spacing"] == 0
    assert text["native_text_effects"]["tracking_em"] == 0.1
    assert text["line_height"] == pytest.approx(1.356)
    assert report["node_mapping"][1]["native_frame_height"] == 22.6
    assert (out / "original.json").read_bytes() == (root / "original.json").read_bytes()
    assert bundle["document"]["original_document"] == original


def test_origin_rotation_and_nested_groups_preserve_centers(corpus):
    _, _, _, prepare = corpus
    node = _base(
        "rect",
        id="child",
        width=20,
        height=10,
        originX="left",
        originY="top",
        left=0,
        top=0,
        angle=90,
    )
    group = _base("group", id="parent", left=60, top=60, objects=[node])
    report, out, _ = prepare([group])
    assert report["status"] == "ready"
    element = json.loads((out / "source.json").read_text())["elements"][1]
    assert element["x"] + element["width"] / 2 == pytest.approx(55)
    assert element["y"] + element["height"] / 2 == pytest.approx(70)
    assert element["rotation"] == pytest.approx(90)
    assert element["group_id"] == "doory:parent"


def test_plain_doryline_is_an_editable_round_capsule(corpus):
    _, _, _, prepare = corpus
    node = _base(
        "doryline",
        width=40,
        height=0,
        stroke="#123456",
        strokeWidth=4,
        strokeUniform=True,
        strokeLineCap="round",
        scaleX=2,
        scaleY=2,
        angle=90,
    )
    report, out, _ = prepare([node])
    assert report["status"] == "ready"
    element = json.loads((out / "source.json").read_text())["elements"][1]
    assert element["type"] == "shape" and element["width"] == 84
    assert element["height"] == 4 and element["corner_radius"] == 2
    assert element["rotation"] == pytest.approx(90)


def test_hard_shadow_keeps_editable_literal_copy_as_one_group(corpus):
    _, _, _, prepare = corpus
    node = _text(effects={"shadow": {"color": "#000000", "blur": 0, "offsetX": 4, "offsetY": 6}})
    report, out, _ = prepare([node])
    assert report["status"] == "ready"
    elements = json.loads((out / "source.json").read_text())["elements"][1:]
    assert len(elements) == 2 and all(e["type"] == "text" for e in elements)
    assert elements[0]["content"] == elements[1]["content"] == "Literal message"
    assert elements[0]["x"] - elements[1]["x"] == 4
    assert elements[0]["y"] - elements[1]["y"] == 6
    assert elements[0]["group_id"] == elements[1]["group_id"]


@pytest.mark.parametrize(
    "change,code",
    [
        (
            {"styles": [{"start": 0, "end": 2, "style": {"fontStyle": "italic"}}]},
            "missing_font",
        ),
        ({"fill": {"type": "linear"}}, "unsupported_paint"),
        ({"effects": {"shadow": {"color": "#000000", "blur": 2}}}, "unsupported_text_effect"),
        ({"scaleX": 2, "scaleY": 1}, "unsupported_transform"),
    ],
)
def test_active_unsupported_features_are_reported_without_a_raster_text_fallback(
    corpus, change, code
):
    _, _, _, prepare = corpus
    node = _text()
    node.update(change)
    report, out, _ = prepare([node])
    assert report["status"] == "unsupported" and report["failures"][0]["code"] == code
    assert not (out / "source.json").exists() and not (out / "bundle.json").exists()
    assert (out / "original-data.json").is_file()
    assert (out / "licenses/0-LICENSE").is_file()


def test_isolated_cubic_artwork_uses_true_curve_bounds(corpus):
    _, _, _, prepare = corpus
    path = [["M", 10, 10], ["C", 10, 80, 90, 80, 90, 10], ["Z"]]
    assert adapter._path(path)[1] == pytest.approx((10, 10, 80, 52.5))
    node = _base("path", path=path, left=50, top=36.25, width=80, height=52.5)
    report, out, _ = prepare([node])
    assert report["status"] == "ready", report
    assert report["vector_art_count"] == 1
    element = json.loads((out / "source.json").read_text())["elements"][1]
    assert element["type"] == "image" and element["x"] == 10
    assert element["y"] == 10 and element["height"] == 53
    assert report["node_mapping"][1]["curve_editable_in_engine"] is False
    assert element["source_art"]["kind"] == "svg-nontext"
    assert element["source_art"]["node_id"] == "native-node"
    assert not element["source_art"]["curve_editable"]


def test_photo_slot_clips_the_panned_image_without_moving_its_frame(corpus):
    root, entry, _, prepare = corpus
    image = Image.new("RGBA", (100, 100), "#0088ff")
    image.save(root / "photo.png")
    entry["asset_requirements"] = [
        {
            "source_reference": "/photo.png",
            "path": "photo.png",
            "sha256": hashlib.sha256((root / "photo.png").read_bytes()).hexdigest(),
        }
    ]
    node = _base(
        "image",
        src="/photo.png",
        width=100,
        height=100,
        left=55,
        top=50,
        slot={"cx": 50, "cy": 50, "width": 40, "height": 40, "angle": 0, "mask": "circle"},
    )
    report, out, _ = prepare([node])
    assert report["status"] == "ready", report
    element = json.loads((out / "source.json").read_text())["elements"][1]
    assert (element["x"], element["y"], element["width"], element["height"]) == (30, 30, 40, 40)
    with Image.open(out / "resources/assets" / element["asset_id"]) as cropped:
        assert cropped.getpixel((0, 0))[3] == 0
        assert cropped.getpixel((20, 20)) == (0, 136, 255, 255)
    assert len(list((out / "original-resources").iterdir())) == 1
    assert element["source_art"]["kind"] == "masked-photo"


def test_independent_author_thumbnail_is_retained_without_upscaling(corpus):
    root, entry, _, prepare = corpus
    Image.new("RGBA", (50, 60), "#123456").save(root / "reference.png")
    entry["native_reference"] = {
        "path": "reference.png",
        "sha256": hashlib.sha256((root / "reference.png").read_bytes()).hexdigest(),
        "page_index": 0,
    }
    report, out, _ = prepare()
    assert report["status"] == "ready" and report["source_parity_status"] == "pending"
    assert report["native_reference"]["independent_of_importer"]
    assert not report["native_reference"]["full_resolution"]
    assert Image.open(out / "native.png").size == (50, 60)
    assert (out / "native-reference.png").read_bytes() == (root / "reference.png").read_bytes()


def test_selected_page_is_explicit_and_old_native_reference_is_removed(corpus):
    root, entry, document, prepare = corpus
    Image.new("RGB", (50, 60), "white").save(root / "reference.png")
    entry["native_reference"] = {
        "path": "reference.png",
        "sha256": hashlib.sha256((root / "reference.png").read_bytes()).hexdigest(),
        "page_index": 0,
    }
    prepare()
    page = deepcopy(document["pages"][0])
    page["id"] = "page-2"
    report, out, _ = prepare(changes={"pages": [document["pages"][0], page]}, page_index=1)
    assert report["status"] == "ready" and report["selected_page"]["index"] == 1
    assert report["selected_page"]["page_count"] == 2
    assert not (out / "native.png").exists()
    assert report["native_reference"]["kind"] == "unavailable"


def test_rotated_text_padding_does_not_move_the_native_top_left(corpus):
    _, _, _, prepare = corpus
    node = _text(angle=45)
    report, out, _ = prepare([node])
    assert report["status"] == "ready"
    element = json.loads((out / "source.json").read_text())["elements"][1]
    cx, cy = element["x"] + element["width"] / 2, element["y"] + element["height"] / 2
    angle = math.radians(element["rotation"])
    top_left = (
        cx - element["width"] / 2 * math.cos(angle) + element["height"] / 2 * math.sin(angle),
        cy - element["width"] / 2 * math.sin(angle) - element["height"] / 2 * math.cos(angle),
    )
    assert top_left == pytest.approx((node["left"], node["top"]))


def test_live_outline_and_gradient_keep_literal_editable_text(corpus):
    _, _, _, prepare = corpus
    outline = _text(
        fill="#00000000",
        stroke="#9fb4ff",
        strokeWidth=3,
        effects={"outline": {"color": "#9fb4ff", "width": 3}},
    )
    gradient = _text(
        id="gradient",
        fill={
            "type": "linear",
            "gradientUnits": "percentage",
            "coords": {"x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1},
            "colorStops": [{"offset": 0, "color": "#ffe27a"}, {"offset": 1, "color": "#f4a640"}],
        },
    )
    report, out, _ = prepare([outline, gradient])
    assert report["status"] == "ready", report
    elements = json.loads((out / "source.json").read_text())["elements"][1:]
    assert all(
        element["type"] == "text" and element["content"] == "Literal message"
        for element in elements
    )
    assert elements[0]["native_text_effects"]["outline"]["width_em"] == pytest.approx(3 / 20)
    assert not elements[0]["native_text_effects"]["outline"]["fill"]
    assert elements[1]["native_text_effects"]["gradient"]["height_factor"] < 1


def test_exact_italic_run_remains_live_and_carries_its_original_face_bytes(corpus):
    root, entry, _, prepare = corpus
    data = (
        ROOT / "fixtures/open_posters/fonts/crimson-text/crimson-text-latin-400-italic.ttf"
    ).read_bytes()
    (root / "italic.ttf").write_bytes(data)
    entry["font_requirements"].append(
        {
            "family": "Crimson Text",
            "weight": "400",
            "style": "italic",
            "path": "italic.ttf",
            "postscript_name": "CrimsonText-Italic",
            "sha256": hashlib.sha256(data).hexdigest(),
            "license_path": "OFL.txt",
        }
    )
    node = _text()
    node["styles"] = [
        {"start": 8, "end": 15, "style": {"fontFamily": "Crimson Text", "fontStyle": "italic"}}
    ]
    report, out, _ = prepare([node])
    assert report["status"] == "ready", report
    bundle = json.loads((out / "bundle.json").read_text())
    element = bundle["source"]["elements"][1]
    assert element["content"] == node["text"] and element["font_family"] == "SolsticeSans-Regular"
    assert element["native_text_effects"]["runs"] == [
        {"start": 8, "end": 15, "font_family": "CrimsonText-Italic"}
    ]
    assert {font["postscript_name"] for font in bundle["fonts"]} == {
        "SolsticeSans-Regular",
        "CrimsonText-Italic",
    }


def test_gradient_background_uses_authored_horizontal_axis(corpus):
    _, _, document, prepare = corpus
    page = deepcopy(document["pages"][0])
    page["background"] = {
        "type": "gradient",
        "kind": "linear",
        "angle": 0,
        "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}],
    }
    report, out, _ = prepare(changes={"pages": [page]})
    assert report["status"] == "ready", report
    element = json.loads((out / "source.json").read_text())["elements"][0]
    with Image.open(out / "resources/assets" / element["asset_id"]) as image:
        assert image.getpixel((0, 50))[0] > image.getpixel((99, 50))[0]
        assert image.getpixel((0, 50))[2] < image.getpixel((99, 50))[2]
    assert report["node_mapping"][0]["mode"] == "isolated_gradient_page_background"


def test_dashed_ellipse_art_keeps_visible_gaps(corpus):
    _, _, _, prepare = corpus
    node = _base(
        "ellipse",
        fill="#00000000",
        stroke="#123456",
        strokeWidth=3,
        strokeDashArray=[4, 5],
        rx=20,
        ry=15,
    )
    report, out, _ = prepare([node])
    assert report["status"] == "ready", report
    element = json.loads((out / "source.json").read_text())["elements"][1]
    with Image.open(out / "resources/assets" / element["asset_id"]) as image:
        alpha = image.getchannel("A")
        assert alpha.getextrema() == (0, 255)


@pytest.mark.parametrize(
    "case_id,content",
    [
        ("doory_bold-youtube-thumbnail", "30 DAYS"),
        ("doory_bold-sale-announcement", "WEEKEND"),
        ("doory_night-bloom-concert-poster", "PAPER TIGERS"),
    ],
)
def test_real_anton_and_poppins_ink_matches_independent_author_baseline(tmp_path, case_id, content):
    """Author thumbnail pixels establish expected ink, independent of our layout."""
    import io

    import skia
    from PIL import ImageColor

    from engine.resources import get_renderer, native_render_example, render_input, using_resources
    from renderer.converter.convert import convert_chameleon_to_crello

    corpus_root = ROOT / "fixtures/open_posters"
    entries = json.loads((corpus_root / "catalogue.json").read_text(encoding="utf-8"))["examples"]
    entry = next(entry for entry in entries if entry["id"] == case_id)
    report = adapter.prepare_doory(entry, corpus_root, tmp_path / case_id)
    assert report["status"] == "ready", report
    out = tmp_path / case_id
    source = json.loads((out / "source.json").read_text(encoding="utf-8"))
    element = next(
        e for e in source["elements"] if e.get("content") == content and "--shadow" not in e["id"]
    )
    isolated = {"canvas": source["canvas"], "elements": [element]}
    with using_resources(assets_dir=out / "resources/assets", fonts_dir=out / "resources/fonts"):
        renderer = get_renderer()
        example, warnings = convert_chameleon_to_crello(
            render_input(isolated), renderer.font_manager, str(out / "resources/assets")
        )
        assert not warnings
        rendered = renderer.render(
            native_render_example(example, isolated),
            max_size=max(source["canvas"].values()),
            format="png",
            canvas_color=skia.ColorTRANSPARENT,
        )
    with Image.open(out / "native.png") as reference:
        reference = reference.convert("RGB")
    current = (
        Image.open(io.BytesIO(rendered))
        .convert("RGBA")
        .resize(reference.size, Image.Resampling.LANCZOS)
    )
    sx, sy = (
        reference.width / source["canvas"]["width"],
        reference.height / source["canvas"]["height"],
    )
    # The leading letters avoid unrelated photographic whites in the wider
    # authored frame. Region comes from the source object, never the candidate.
    left, top = int(element["x"] * sx), int(element["y"] * sy)
    right = int((element["x"] + element["width"] * 0.3) * sx)
    bottom = int((element["y"] + element["height"]) * sy)
    color = ImageColor.getrgb(element["color"])

    def ink_rows(image, alpha=False):
        rows = []
        for y in range(top, bottom):
            count = 0
            for x in range(left, right):
                pixel = image.getpixel((x, y))
                if (
                    pixel[3] > 180
                    if alpha
                    else max(abs(pixel[i] - color[i]) for i in range(3)) < 45
                ):
                    count += 1
            if count:
                rows.append((y, count))
        assert rows, "The independent native reference must contain visible text ink"
        # The author's script above WEEKEND has a small descender entering its
        # frame. Select the dominant contiguous ink band in each image rather
        # than treating another authored message as this headline's first row.
        bands = []
        for y, count in rows:
            if not bands or y > bands[-1][-1][0] + 1:
                bands.append([])
            bands[-1].append((y, count))
        dominant = max(bands, key=lambda band: sum(count for _, count in band))
        return dominant[0][0], dominant[-1][0]

    assert ink_rows(current, alpha=True) == pytest.approx(ink_rows(reference), abs=3)

    # Render a real similarity transform, including tracked Oswald on its plate.
    # This rejects preserving an absolute pixel gap while halving the glyphs.
    scaled = deepcopy(isolated)
    scaled["canvas"] = {key: value // 2 for key, value in source["canvas"].items()}
    for key in ("x", "y", "width", "height", "font_size"):
        scaled["elements"][0][key] /= 2
    with using_resources(assets_dir=out / "resources/assets", fonts_dir=out / "resources/fonts"):
        example, warnings = convert_chameleon_to_crello(
            render_input(scaled), renderer.font_manager, str(out / "resources/assets")
        )
        assert not warnings
        half = renderer.render(
            native_render_example(example, scaled),
            max_size=max(scaled["canvas"].values()),
            format="png",
            canvas_color=skia.ColorTRANSPARENT,
        )
    full_ink = Image.open(io.BytesIO(rendered)).getchannel("A").getbbox()
    half_ink = Image.open(io.BytesIO(half)).getchannel("A").getbbox()
    assert half_ink == pytest.approx([value / 2 for value in full_ink], abs=2)
    if content == "PAPER TIGERS":
        index = source["elements"].index(element)
        plate = source["elements"][index - 1]
        assert plate["type"] == "image"
        assert plate["x"] / 2 <= half_ink[0]
        assert half_ink[2] <= (plate["x"] + plate["width"]) / 2
        assert plate["y"] / 2 <= half_ink[1]
        assert half_ink[3] <= (plate["y"] + plate["height"]) / 2
