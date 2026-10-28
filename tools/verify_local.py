"""Exercise a running local viewer through its public HTTP API.

    python -m tools.verify_local --url http://127.0.0.1:8765

Checks use the supplied sample fixtures, decoded PNGs, and the design schema
when jsonschema is installed. No renderer imports or network services are needed.
The machine-readable result is saved to output/local-verification.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MUTABLE_FIELDS = {"x", "y", "width", "height", "font_size", "corner_radius", "stroke_width"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class Client:
    def __init__(self, url: str, timeout: float):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict | bytes | None = None):
        data = (
            json.dumps(payload, allow_nan=False).encode() if isinstance(payload, dict) else payload
        )
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = Request(self.url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, response.headers, response.read()
        except HTTPError as exc:
            try:
                return exc.code, exc.headers, exc.read()
            finally:
                exc.close()

    def json(self, path: str, payload: dict | bytes | None = None, status: int = 200):
        actual, headers, body = self.request(path, payload)
        require(actual == status, f"{path}: HTTP {actual}, expected {status}; {body[:300]!r}")
        require(
            "application/json" in headers.get("Content-Type", "").lower(),
            f"{path}: response is not JSON",
        )
        result = json.loads(body.decode("utf-8"))
        require(isinstance(result, dict), f"{path}: expected a JSON object")
        return result


def png_dimensions(data_url: str, expected: dict) -> list[int]:
    prefix = "data:image/png;base64,"
    require(isinstance(data_url, str) and data_url.startswith(prefix), "Expected a PNG data URL")
    raw = base64.b64decode(data_url[len(prefix) :], validate=True)
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG signature")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG", "Decoded image is not PNG")
        expected_size = (expected["width"], expected["height"])
        require(image.size == expected_size, f"PNG is {image.size}, expected {expected_size}")
        size = list(image.size)
        image.verify()
    return size


def validate_design(source: dict, design: dict, target: dict, validator) -> None:
    require(isinstance(design, dict), "Missing output design")
    if validator is not None:
        validator.validate(design)
    require(design.get("canvas") == target, "Output canvas differs from requested target")
    elements = design.get("elements")
    require(isinstance(elements, list), "Output elements must be a list")
    expected_ids = Counter(element["id"] for element in source["elements"])
    actual_ids = Counter(element.get("id") for element in elements)
    require(actual_ids == expected_ids, "Element IDs or element counts changed")
    require(all(count == 1 for count in actual_ids.values()), "Output contains duplicate IDs")
    by_id = {element["id"]: element for element in elements}
    for original in source["elements"]:
        element = by_id[original["id"]]
        for field, value in original.items():
            if field not in MUTABLE_FIELDS:
                require(element.get(field) == value, f"{original['id']}.{field} changed")
        for field in ("x", "y", "width", "height"):
            value = element.get(field)
            require(
                isinstance(value, (int, float)) and math.isfinite(value),
                f"{original['id']}.{field} is not finite geometry",
            )
        require(element["width"] > 0 and element["height"] > 0, "Nonpositive element dimensions")
        width, height = element["width"], element["height"]
        angle = math.radians(element.get("rotation", 0))
        rotated_width = abs(width * math.cos(angle)) + abs(height * math.sin(angle))
        rotated_height = abs(width * math.sin(angle)) + abs(height * math.cos(angle))
        left = element["x"] + (width - rotated_width) / 2
        top = element["y"] + (height - rotated_height) / 2
        require(
            left >= -0.5
            and top >= -0.5
            and left + rotated_width <= target["width"] + 0.5
            and top + rotated_height <= target["height"] + 0.5,
            f"{original['id']} has rotated bounds outside the target",
        )
        if original["type"] == "image" and original.get("fit", "contain") != "cover":
            source_ratio = original["width"] / original["height"]
            drift = abs(width / height / source_ratio - 1)
            require(drift <= 0.05, f"{original['id']} image aspect ratio drifted by {drift:.1%}")
        if original["type"] == "text":
            size = element.get("font_size")
            require(
                isinstance(size, (int, float)) and math.isfinite(size) and size >= 1,
                f"{original['id']} has an invalid font size",
            )


def verify(url: str, timeout: float) -> dict:
    client = Client(url, timeout)
    fixtures = {
        path: path.read_bytes() for path in sorted((ROOT / "input").glob("chameleon_sample_*.json"))
    }
    tasks = {path.stem: json.loads(content) for path, content in fixtures.items()}
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        validator = None
    else:
        schema = json.loads(
            (ROOT / "chameleon_design_schema_v0.9.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema)

    report = {
        "url": client.url,
        "started_at": datetime.now(UTC).isoformat(),
        "schema_validation": "jsonschema"
        if validator is not None
        else "built-in structural checks",
        "cases": [],
    }
    catalogue = {}

    def check(name, operation):
        started = time.perf_counter()
        row = {"name": name}
        try:
            row["details"] = operation() or {}
            row["passed"] = True
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        row["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["cases"].append(row)
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}")
        if not row["passed"]:
            print(f"       {row['error']}")

    def static_root():
        status, headers, body = client.request("/")
        require(status == 200, f"Root returned HTTP {status}")
        require("text/html" in headers.get("Content-Type", "").lower(), "Root is not HTML")
        require(bool(body.strip()), "Root HTML is empty")
        return {"status": status, "bytes": len(body)}

    def static_asset(route: str):
        status, headers, body = client.request(route)
        require(status == 200, f"{route} returned HTTP {status}")
        require(bool(body), f"{route} response is empty")
        local_path = (
            ROOT / "brand_kit/fonts" / route.removeprefix("/fonts/")
            if route.startswith("/fonts/")
            else ROOT / "viewer" / route.lstrip("/")
        )
        require(body == local_path.read_bytes(), f"{route} does not match the current local asset")
        content_type = headers.get("Content-Type", "").lower()
        if route.endswith(".css"):
            require("text/css" in content_type, f"{route} has the wrong MIME type")
        elif route.endswith(".js"):
            require("javascript" in content_type, f"{route} has the wrong MIME type")
        elif route.endswith(".html"):
            require("text/html" in content_type, f"{route} has the wrong MIME type")
        return {"status": status, "bytes": len(body), "content_type": content_type}

    def health():
        result = client.json("/api/health")
        require(result.get("status") == "ok", f"Health status is {result.get('status')!r}")
        return result

    def samples():
        result = client.json("/api/samples")
        supplied = result.get("samples")
        targets = result.get("targets")
        require(isinstance(supplied, list) and len(supplied) >= 3, "Missing the three samples")
        require(len(tasks) == 3, f"Expected three local sample fixtures, found {len(tasks)}")
        require(isinstance(targets, list) and bool(targets), "Target catalogue is empty")
        for sample in supplied:
            require(all(key in sample for key in ("id", "label", "task")), "Malformed sample entry")
        for target in targets:
            require(
                all(key in target for key in ("id", "label", "width", "height")),
                "Malformed target entry",
            )
            require(target["width"] > 0 and target["height"] > 0, "Invalid catalogue target size")
        for name, task in tasks.items():
            require(
                any(
                    sample["task"].get("source") == task["source"]
                    and sample["task"].get("target_canvas") == task["target_canvas"]
                    for sample in supplied
                ),
                f"API catalogue omitted or altered {name}",
            )
        catalogue.update(deepcopy(result))
        return {"samples": len(supplied), "targets": len(targets)}

    def resize_task(task: dict, target: dict):
        source = deepcopy(task["source"])
        before = deepcopy(source)
        result = client.json("/api/resize", {"source": source, "target_canvas": target})
        validate_design(before, result.get("design"), target, validator)
        require(source == before, "Request source was mutated")
        evaluation = result.get("report")
        require(isinstance(evaluation, dict), "Missing evaluation report")
        require(evaluation.get("violations") == [], f"Contract violations: {evaluation}")
        require(isinstance(evaluation.get("metrics"), dict), "Missing quality metrics")
        total = evaluation.get("total")
        require(isinstance(total, (int, float)) and 0 <= total <= 1, "Invalid quality score")
        style = evaluation.get("style")
        require(isinstance(style, dict), "Missing source-style report")
        require(
            isinstance(style.get("score"), (int, float)) and 0 <= style["score"] <= 1,
            "Invalid source-style score",
        )
        require(isinstance(style.get("metrics"), dict), "Missing source-style metrics")
        require(isinstance(style.get("warnings"), list), "Missing source-style warnings")
        source_ids = {element["id"] for element in before["elements"]}
        for warning in style["warnings"]:
            require(
                isinstance(warning, dict)
                and bool(warning.get("code"))
                and bool(warning.get("message"))
                and isinstance(warning.get("element_ids"), list)
                and bool(warning["element_ids"])
                and set(warning["element_ids"]) <= source_ids,
                "Malformed source-style warning",
            )
        warnings = result.get("warnings")
        require(isinstance(warnings, list), "Missing warnings list")
        missing = {
            element["asset_id"]
            for element in before["elements"]
            if element["type"] == "image"
            and not (ROOT / "brand_kit/assets" / element["asset_id"]).is_file()
        }
        warning_text = json.dumps(warnings).lower()
        for asset in missing:
            require(asset.lower() in warning_text, f"Missing-asset warning absent for {asset}")
        elapsed = result.get("elapsed_ms")
        require(
            isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0,
            "Invalid elapsed_ms",
        )
        return {
            "elements": len(result["design"]["elements"]),
            "source_png": png_dimensions(result.get("source_png"), before["canvas"]),
            "output_png": png_dimensions(result.get("output_png"), target),
            "total": total,
            "style": style,
            "elapsed_ms": elapsed,
            "warnings": warnings,
        }

    def unchanged_inputs():
        require(bool(catalogue), "Cannot compare inputs because the samples request failed")
        require(client.json("/api/samples") == catalogue, "Server sample catalogue was mutated")
        for path, content in fixtures.items():
            require(path.read_bytes() == content, f"Sample fixture {path.name} was modified")
        return {"sample_files": len(fixtures), "catalogue_unchanged": True}

    def invalid_request(payload):
        result = client.json("/api/resize", payload, status=400)
        require(bool(result.get("error")), "HTTP 400 did not include a JSON error")
        return {"status": 400, "error": result["error"]}

    check("static viewer", static_root)
    for route in (
        "/styles.css",
        "/app.js",
        "/fonts/SolsticeSans-Regular.ttf",
        "/fonts/SolsticeSans-Bold.ttf",
        "/inspector.html",
    ):
        check(f"static {route}", lambda route=route: static_asset(route))
    check("health", health)
    check("sample and target catalogues", samples)
    for name, task in tasks.items():
        check(name, lambda task=task: resize_task(task, task["target_canvas"]))
    if tasks:
        task = next(iter(tasks.values()))
        check("custom 913x517 target", lambda: resize_task(task, {"width": 913, "height": 517}))
        check(
            "full-size 2560x1440 target",
            lambda: resize_task(task, {"width": 2560, "height": 1440}),
        )
        check("malformed JSON rejected", lambda: invalid_request(b'{"source":'))
        check(
            "missing source rejected",
            lambda: invalid_request({"target_canvas": task["source"]["canvas"]}),
        )
        check(
            "zero target rejected",
            lambda: invalid_request(
                {"source": task["source"], "target_canvas": {"width": 0, "height": 100}}
            ),
        )
    else:
        check("sample fixtures available", lambda: require(False, "No sample fixtures found"))
    check("inputs preserved", unchanged_inputs)
    report["passed"] = sum(case["passed"] for case in report["cases"])
    report["failed"] = len(report["cases"]) - report["passed"]
    report["success"] = report["failed"] == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Running local viewer URL")
    parser.add_argument(
        "--timeout", type=float, default=60, help="Timeout per HTTP request in seconds"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "output/local-verification.json")
    args = parser.parse_args()
    report = verify(args.url, args.timeout)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{report['passed']} passed; {report['failed']} failed. Report: {args.report}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
