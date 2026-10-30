"""Band layouts for extreme aspect ratios.

A 728x90 leaderboard or a 320x50 mobile banner cannot hold a poster: the brief's
own reference output turns the design into one horizontal band -- brand mark,
product, headline with its support copy, then the call to action -- and the
rubric scores readable text above everything but collisions. Scaling the
poster down into a small island in the middle of the strip keeps the shape of
the composition and loses the message.

These proposals join the regular candidate pool, so the source-aware selector
still decides. They only exist for targets at least ~2.2x wider than tall or
at least ~2.2x taller than wide.
"""

from __future__ import annotations

from .candidates import _ORDER, _base, _finish, _place
from .geometry import Box
from .model import DesignContext, Unit
from .text import get_measurer

WIDE_RATIO = 2.2
TALL_RATIO = 1 / 2.2


def _by_role(context: DesignContext) -> dict[str, list[Unit]]:
    roles: dict[str, list[Unit]] = {}
    for unit in sorted(context.units, key=lambda u: (_ORDER[u.role], u.box.y, u.box.x)):
        roles.setdefault(unit.role, []).append(unit)
    return roles


def _aspect(unit: Unit) -> float:
    return unit.box.width / max(unit.box.height, 1e-6)


def _fit_unit_height(unit: Unit, box: Box, min_font: float) -> float:
    """Height a text unit needs at the largest font that keeps its longest
    line inside `box.width`, bounded by `box.height`."""
    measurer = get_measurer()
    total = 0.0
    for element in unit.text:
        size = measurer.fit(element, box.width, box.height, element["font_size"] * 10)
        size = max(min_font, min(size, box.height / 1.25))
        total += measurer.measure(element, size, box.width).height
    return min(box.height, max(total, 1.0))


def _place_fit(unit: Unit, box: Box, min_font: float) -> list[dict]:
    """Place a unit in `box`, then raise text sizes to use the room."""
    placed = _place(unit, box, 1.0)
    measurer = get_measurer()
    for element in placed:
        if element["type"] == "text":
            element["font_size"] = max(
                min_font,
                measurer.fit(element, element["width"], element["height"], box.height),
            )
    return placed


def _wide_band(
    source: dict,
    context: DesignContext,
    target: dict,
    hero_share: float,
    badges_inline: bool,
) -> dict | None:
    width, height = float(target["width"]), float(target["height"])
    roles = _by_role(context)
    pad = max(2.0, height * 0.10)
    gap = max(2.0, height * 0.08)
    inner = Box(pad, pad, width - 2 * pad, height - 2 * pad)
    min_font = max(6.0, inner.height * 0.12)
    elements = _base(context, target)

    x = inner.x
    # Brand mark: fits the band height, keeps its proportion.
    for logo in roles.get("logo", []):
        logo_h = inner.height * 0.6
        logo_w = logo_h * _aspect(logo)
        if logo_w > inner.width * 0.22:
            logo_w = inner.width * 0.22
            logo_h = logo_w / _aspect(logo)
        elements.extend(
            _place(logo, Box(x, inner.y + (inner.height - logo_h) / 2, logo_w, logo_h), 1.0)
        )
        x += logo_w + gap

    # Product: full band height, never more than `hero_share` of the width.
    for hero in roles.get("hero", []):
        hero_h = inner.height
        hero_w = min(hero_h * _aspect(hero), inner.width * hero_share)
        hero_h = hero_w / _aspect(hero)
        elements.extend(
            _place(hero, Box(x, inner.y + (inner.height - hero_h) / 2, hero_w, hero_h), 1.0)
        )
        x += hero_w + gap
    # Secondary art (stamps, watermarks) stays small, beside the product.
    for extra in roles.get("support", []):
        sup_h = inner.height * 0.45
        sup_w = min(sup_h * _aspect(extra), inner.width * 0.08)
        sup_h = sup_w / _aspect(extra)
        elements.extend(
            _place(extra, Box(x, inner.y + (inner.height - sup_h) / 2, sup_w, sup_h), 1.0)
        )
        x += sup_w + gap

    # Call to action at the right edge, badges stacked beside it if not inline.
    right = inner.x + inner.width
    actions = roles.get("action", [])
    if actions:
        action = actions[0]
        act_h = inner.height * 0.55
        act_w = min(inner.width * 0.18, act_h * _aspect(action))
        act_h = act_w / _aspect(action)
        right -= act_w
        elements.extend(
            _place_fit(
                action,
                Box(right, inner.y + (inner.height - act_h) / 2, act_w, act_h),
                min_font * 0.9,
            )
        )
        right -= gap
        for extra in actions[1:]:
            act_w2 = act_w * 0.9
            act_h2 = act_w2 / _aspect(extra)
            right -= act_w2
            elements.extend(
                _place_fit(
                    extra,
                    Box(right, inner.y + (inner.height - act_h2) / 2, act_w2, act_h2),
                    min_font * 0.8,
                )
            )
            right -= gap

    badges = roles.get("badge", [])
    if badges and not badges_inline:
        # A small stack of pills between the copy and the action, never taller
        # than a third of the band each.
        n = len(badges)
        row_h = min(inner.height * 0.3, (inner.height - gap * 0.5 * (n - 1)) / n)
        col_w = min(inner.width * 0.12, max(_aspect(b) for b in badges) * row_h)
        right -= col_w
        stack_h = row_h * n + gap * 0.5 * (n - 1)
        y = inner.y + (inner.height - stack_h) / 2
        for badge in badges:
            w = min(col_w, row_h * _aspect(badge))
            elements.extend(
                _place_fit(badge, Box(right + (col_w - w) / 2, y, w, row_h), min_font * 0.75)
            )
            y += row_h + gap * 0.5
        right -= gap

    # Copy column takes what is left: headline, then body, then an inline
    # badge row, then legal, each as little height as it needs.
    copy_w = max(10.0, right - x)
    column: list[tuple[Unit, float]] = []
    for role in ("headline", "body"):
        for unit in roles.get(role, []):
            column.append(
                (
                    unit,
                    _fit_unit_height(
                        unit,
                        Box(
                            x, inner.y, copy_w, inner.height * (0.6 if role == "headline" else 0.4)
                        ),
                        min_font,
                    ),
                )
            )
    inline_row_h = 0.0
    if badges and badges_inline:
        inline_row_h = inner.height * 0.28
    legal = roles.get("legal", [])
    legal_h = inner.height * 0.16 if legal else 0.0

    needed = (
        sum(h for _, h in column)
        + inline_row_h
        + legal_h
        + gap * 0.5 * (len(column) + bool(inline_row_h) + bool(legal_h) - 1)
    )
    factor = min(1.0, inner.height / max(needed, 1e-6))
    y = inner.y + (inner.height - needed * factor) / 2
    for unit, h in column:
        h *= factor
        elements.extend(_place_fit(unit, Box(x, y, copy_w, h), min_font))
        y += h + gap * 0.5 * factor
    if inline_row_h:
        h = inline_row_h * factor
        n = len(badges)
        bw = (copy_w - gap * 0.5 * (n - 1)) / n
        bx = x
        for badge in badges:
            w = min(bw, h * _aspect(badge))
            elements.extend(_place_fit(badge, Box(bx, y, w, h), min_font * 0.75))
            bx += w + gap * 0.5
        y += h + gap * 0.5 * factor
    for unit in legal:
        h = legal_h * factor
        elements.extend(_place_fit(unit, Box(x, y, copy_w, h), max(4.0, min_font * 0.6)))
        y += h

    return _finish(source, target, elements)


def _tall_stack(
    source: dict, context: DesignContext, target: dict, hero_share: float
) -> dict | None:
    width, height = float(target["width"]), float(target["height"])
    roles = _by_role(context)
    pad = max(2.0, width * 0.08)
    gap = max(2.0, width * 0.06)
    inner = Box(pad, pad, width - 2 * pad, height - 2 * pad)
    min_font = max(6.0, inner.width * 0.07)
    elements = _base(context, target)

    order: list[tuple[Unit, float]] = []
    for logo in roles.get("logo", []):
        order.append((logo, inner.width / _aspect(logo)))
    for hero in roles.get("hero", []) + roles.get("support", []):
        h = min(inner.height * hero_share, inner.width / _aspect(hero))
        order.append((hero, h))
    for role in ("headline", "body"):
        for unit in roles.get(role, []):
            order.append(
                (
                    unit,
                    _fit_unit_height(
                        unit, Box(inner.x, inner.y, inner.width, inner.height * 0.3), min_font
                    ),
                )
            )
    for badge in roles.get("badge", []):
        order.append((badge, min(inner.width / _aspect(badge), inner.height * 0.06)))
    for action in roles.get("action", []):
        order.append((action, min(inner.width / _aspect(action), inner.height * 0.09)))
    for unit in roles.get("legal", []):
        order.append(
            (
                unit,
                _fit_unit_height(
                    unit,
                    Box(inner.x, inner.y, inner.width, inner.height * 0.1),
                    max(4.0, min_font * 0.6),
                ),
            )
        )

    needed = sum(h for _, h in order) + gap * (len(order) - 1)
    factor = min(1.0, inner.height / max(needed, 1e-6))
    y = inner.y + (inner.height - needed * factor) / 2
    for unit, h in order:
        h *= factor
        w = inner.width if unit.text else min(inner.width, h * _aspect(unit))
        if unit.role in {"badge", "action"}:
            w = min(inner.width, h * _aspect(unit))
        box = Box(inner.x + (inner.width - w) / 2, y, w, h)
        elements.extend(
            _place_fit(unit, box, min_font * (0.75 if unit.role == "badge" else 1.0))
            if unit.text
            else _place(unit, box, 1.0)
        )
        y += h + gap * factor
    return _finish(source, target, elements)


def generate_strip_candidates(source: dict, target: dict, context: DesignContext) -> list[dict]:
    ratio = target["width"] / target["height"]
    out: list[dict] = []
    try:
        if ratio >= WIDE_RATIO:
            # A product at full band height needs more of the width the
            # taller the band is: ~15% of a 728x90, ~40% of a 1500x500.
            base_share = max(0.1, min(0.45, 1.2 / ratio))
            for hero_share in (base_share * 0.8, base_share * 1.2):
                for inline in (True, False):
                    design = _wide_band(source, context, target, hero_share, inline)
                    if design:
                        out.append(design)
        elif ratio <= TALL_RATIO:
            for hero_share in (0.25, 0.35):
                design = _tall_stack(source, context, target, hero_share)
                if design:
                    out.append(design)
    except (ValueError, ZeroDivisionError, KeyError):
        return out
    return out
