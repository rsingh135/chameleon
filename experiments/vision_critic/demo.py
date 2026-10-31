"""Create original synthetic collection inputs for local framework tests.

These generated shapes and short text are not real design preferences, an
independent held-out corpus, or approved training material. No labels are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import collect


def create_inputs(destination: Path) -> Path:
    if destination.exists():
        raise ValueError("Demo input directory must not already exist")
    destination.mkdir(parents=True)
    sources = []
    for index, split in enumerate(("train", "validation", "test")):

        def shape(identity, x, y, width, height, fill, group=None):
            # Native shape rasterization accepts rgb/rgba, rather than hex paint.
            channels = [int(fill[start : start + 2], 16) for start in (1, 3, 5)]
            value = {
                "id": identity,
                "type": "shape",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "shape_type": "rectangle",
                "fill_color": f"rgba({channels[0]}, {channels[1]}, {channels[2]}, 1)",
            }
            if group:
                value["group_id"] = group
            return value

        def text(identity, content, x, y, width, height, size, group=None):
            value = {
                "id": identity,
                "type": "text",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "font_family": "Solstice Sans",
                "font_size": size,
                "color": "#21372e",
                "content": content,
            }
            if group:
                value["group_id"] = group
            return value

        design = {
            "canvas": {"width": 520, "height": 640},
            "elements": [
                shape("paper", 0, 0, 520, 640, ("#f1efde", "#e9eef5", "#f8e9df")[index]),
                text("eyebrow", "LOCAL FRAMEWORK FIXTURE", 36, 35, 445, 24, 15),
                text(
                    "headline",
                    ("Shape study", "Color study", "Space study")[index],
                    36,
                    86,
                    445,
                    76,
                    42,
                ),
                text("description", "Original shapes. Synthetic test data.", 36, 180, 445, 64, 21),
                shape("art-a", 36 + index * 10, 280, 250, 150, "#819b84"),
                shape("art-b", 310, 280 + index * 12, 170, 150, "#c9aa6e"),
                shape("action-box", 36, 480, 200, 60, "#cbd9c3", "action"),
                text("action-label", "Compare layouts", 48, 496, 176, 30, 20, "action"),
                text("footer", "Infrastructure validation only", 36, 587, 445, 24, 14),
            ],
        }
        name = f"synthetic-{split}.json"
        (destination / name).write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
        sources.append(
            {
                "source_id": f"synthetic-{split}",
                "family_id": f"fixture-{split}",
                "split": split,
                "source": name,
                "source_parity_verified": False,
                "training_rights_reviewed": True,
                "rights_reference": (
                    "Original shape/text generator: experiments/vision_critic/demo.py; "
                    "bundled Poppins/OFL font provenance: brand_kit/README.md. "
                    "Synthetic infrastructure only; not human design labels."
                ),
                "targets": [{"width": 640, "height": 360}],
            }
        )
    path = destination / "inputs.json"
    path.write_text(
        json.dumps({"schema_version": 1, "purpose": "synthetic", "sources": sources}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/vision-framework-demo"))
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    manifest = create_inputs(args.out / "inputs")
    if args.inputs_only:
        print(manifest)
    else:
        print(json.dumps(collect(manifest, args.out / "collection", pool="shadow"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
