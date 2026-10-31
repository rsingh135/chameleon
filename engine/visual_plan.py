"""Closed, source-bound menus for target-conditioned local visual planning.

The model selects an opaque menu ID. Trusted code owns all geometry, identities
and crop permissions. Source digests match candidate snapshots (spaced JSON);
other artifact digests use compact canonical JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path, PureWindowsPath

from engine.layout.relationships import build_relationship_graph
from engine.resources import get_resources, using_resources

POLICY_VERSION = "visual-layout-menu-v1"
MAX_PLANS = 6


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Visual plans must be finite JSON") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path, maximum=64_000_000):
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Visual-plan resource is missing or exceeds its byte budget")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Visual-plan resource exceeds its byte budget")
    return hashlib.sha256(data).hexdigest()


def _resources(source):
    current = get_resources()
    assets, fonts = {}, {}
    root = current.assets_dir.resolve()
    for element in source["elements"]:
        if element["type"] != "image":
            continue
        name = element.get("asset_id")
        if not isinstance(name, str) or not name or PureWindowsPath(name).drive:
            raise ValueError("Visual-plan assets must be local relative paths")
        path = (root / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Visual-plan asset leaves its resource directory")
        # Bind an absent supplied asset just as explicitly as existing bytes.
        # The renderer already supports missing artwork with a placeholder;
        # adding or removing that artwork must still invalidate this menu.
        assets[name] = _file_hash(path, 32_000_000) if path.exists() else {"status": "missing"}
    root = current.fonts_dir.resolve()
    if not root.is_dir():
        raise ValueError("Visual-plan font resources are missing")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".json"}:
            if not path.resolve().is_relative_to(root) or len(fonts) >= 200:
                raise ValueError("Visual-plan font directory is unsafe or exceeds its budget")
            fonts[path.relative_to(root).as_posix()] = _file_hash(path)
    return {"assets": assets, "fonts": fonts}


def _inputs(source, target):
    canonical(source)
    if not isinstance(source, dict) or not isinstance(source.get("elements"), list):
        raise ValueError("Visual planning requires a source design")
    if not 1 <= len(source["elements"]) <= 256:
        raise ValueError("Visual-plan source element budget exceeded")
    if not all(isinstance(e, dict) for e in source["elements"]):
        raise ValueError("Visual-plan source elements must be objects")
    ids = [e.get("id") for e in source["elements"]]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Visual planning requires unique source IDs")
    for canvas in (source.get("canvas"), target):
        if (
            not isinstance(canvas, dict)
            or set(canvas) != {"width", "height"}
            or any(
                isinstance(canvas[k], bool)
                or not isinstance(canvas[k], (int, float))
                or not math.isfinite(canvas[k])
                or canvas[k] <= 0
                for k in ("width", "height")
            )
        ):
            raise ValueError("Visual planning requires finite positive canvas dimensions")


def _bindings(source, target):
    _inputs(source, target)
    graph = build_relationship_graph(source)
    semantic = {key: asdict(graph)[key] for key in ("units", "edges", "clusters", "ambiguities")}
    units = []
    for unit in graph.units:
        text = " ".join(
            e.get("content", "") for e in source["elements"] if e["id"] in unit.element_ids
        )
        units.append(
            {
                "unit_id": f"u{unit.index}",
                "element_ids": list(unit.element_ids),
                "role": unit.role,
                "box": [
                    unit.box.x / graph.width,
                    unit.box.y / graph.height,
                    unit.box.width / graph.width,
                    unit.box.height / graph.height,
                ],
                "text_preview": text[:160],
            }
        )
    root = Path(__file__).resolve().parents[1]
    implementation = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (root / "engine", root / "renderer/converter")
        for path in sorted(folder.rglob("*.py"))
    }
    return graph, {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "source_digest": source_digest(source),
        "resource_digest": digest(_resources(source)),
        "graph_digest": digest(semantic),
        "implementation_digest": digest(implementation),
        "target_canvas": dict(target),
        "units": units,
    }


def descriptor(family, parameters, label):
    spec = {"family": family, "parameters": parameters, "label": label}
    return {"plan_id": "plan-" + digest(spec)[:16], **spec}


@lru_cache(maxsize=32)
def _menu_options(source_json, target_json, assets_dir, fonts_dir, binding_digest):
    """Cache only immutable specs, keyed by actual resources and implementation."""
    from engine.layout.visual_plan import candidate_specs, compile_spec

    with using_resources(assets_dir, fonts_dir):
        source, target = json.loads(source_json), json.loads(target_json)
        graph = build_relationship_graph(source)
        plans = []
        for spec in candidate_specs(source, target, graph):
            if compile_spec(source, target, graph, spec) is not None:
                plans.append(spec)
            if len(plans) == MAX_PLANS:
                break
        return canonical(plans)


def build_plan_menu(source, target, *, assets_dir=None, fonts_dir=None):
    """Return at most six realizable options plus an explicit keep/abstain ID."""
    original = deepcopy(source)
    with using_resources(assets_dir, fonts_dir):
        graph, body = _bindings(source, target)
        from engine.resize import _resolution_state

        resolution = _resolution_state(source, target) is not None
        plans = [
            {
                "plan_id": "keep",
                "label": "Keep deterministic layout",
                "family": "baseline",
                "parameters": {},
            }
        ]
        if not resolution:
            resources = get_resources()
            plans.extend(
                json.loads(
                    _menu_options(
                        canonical(source),
                        canonical(target),
                        str(resources.assets_dir),
                        str(resources.fonts_dir),
                        digest(body),
                    )
                )
            )
        body.update(
            status="resolution" if resolution else "ready" if len(plans) > 1 else "unsupported",
            plans=plans,
        )
        body["menu_digest"] = digest(body)
    if source != original:
        raise RuntimeError("Visual menu construction mutated its source")
    return deepcopy(body)


def validate_decision(decision, menu):
    required = {"schema_version", "menu_digest", "plan_id"}
    if (
        not isinstance(decision, dict)
        or set(decision) - required - {"rationale"}
        or not required <= decision.keys()
    ):
        raise ValueError("Visual decision has missing or unknown fields")
    canonical(decision)
    if type(decision["schema_version"]) is not int or decision["schema_version"] != 1:
        raise ValueError("Unsupported visual decision version")
    if decision["menu_digest"] != menu.get("menu_digest"):
        raise ValueError("Visual decision belongs to another menu")
    if not isinstance(decision["plan_id"], str) or decision["plan_id"] not in {
        p["plan_id"] for p in menu["plans"]
    }:
        raise ValueError("Visual decision names an unknown plan")
    if "rationale" in decision and (
        not isinstance(decision["rationale"], str) or len(decision["rationale"]) > 1000
    ):
        raise ValueError("Visual rationale must be short text")
    return deepcopy(decision)


def validate_bound_menu(source, target, menu):
    """Rebind actual resources/code and reject injected geometry or unit tokens."""
    from engine.layout.visual_plan import candidate_specs

    if not isinstance(menu, dict) or menu.get("menu_digest") != digest(
        {k: v for k, v in menu.items() if k != "menu_digest"}
    ):
        raise ValueError("Visual menu content differs from its digest")
    graph, bindings = _bindings(source, target)
    if set(menu) != set(bindings) | {"status", "plans", "menu_digest"}:
        raise ValueError("Visual menu has missing or unknown fields")
    if any(canonical(menu[key]) != canonical(value) for key, value in bindings.items()):
        raise ValueError("Visual menu source, graph, target, code or resource identity is stale")
    plans = menu["plans"]
    if not isinstance(plans, list) or not 1 <= len(plans) <= MAX_PLANS + 1:
        raise ValueError("Visual menu exceeds its plan budget")
    allowed = [
        {
            "plan_id": "keep",
            "label": "Keep deterministic layout",
            "family": "baseline",
            "parameters": {},
        }
    ]
    allowed += candidate_specs(source, target, graph)
    tokens = {canonical(spec) for spec in allowed}
    if len({p.get("plan_id") for p in plans}) != len(plans) or any(
        canonical(p) not in tokens for p in plans
    ):
        raise ValueError("Visual menu contains an unsupported plan or unit reference")
    from engine.resize import _resolution_state

    resolution = _resolution_state(source, target) is not None
    expected = "resolution" if resolution else "ready" if len(plans) > 1 else "unsupported"
    if menu["status"] != expected or resolution and len(plans) != 1:
        raise ValueError("Visual menu has inconsistent resize authority")
    return graph


def validate_plan_decision(menu, decision):
    """Menu-first convenience alias for the public resize integration."""
    return validate_decision(decision, menu)
