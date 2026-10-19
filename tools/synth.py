"""Synthetic resize tasks for regression testing.

The hidden scoring set is large and unseen, so breadth matters more than the
three samples. This generates designs from a handful of layout *families*
with randomised content, counts, ids, groups, rotations and missing assets,
and pairs each with a catalogue target. Some families and some target
formats are reserved as a holdout so sample-specific tuning shows up as a
gap between seen and held-out scores.

Every generated source is self-consistent: text is sized so it fits its box
by the renderer's own measurement, and nothing non-decorative starts
off-canvas, so the source itself passes the contract check.

    python -m tools.synth --n 40 --seed 0 --out eval/cases
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from engine.eval.catalogue import CATALOGUE
from engine.eval.text import get_measurer

SOURCE_CANVASES = [
    (1080, 1080),
    (1080, 1350),
    (1080, 1920),
    (1200, 628),
    (1920, 1080),
    (800, 1200),
    (1500, 500),
]
HOLDOUT_TARGETS = {"wide_skyscraper", "mobile_banner", "a4_landscape", "half_page"}
HOLDOUT_FAMILIES = {"split_hero", "multi_image"}

SERIF, SANS = "Solstice Serif", "Solstice Sans"
WHITE, GOLD, ORANGE = "rgba(255,255,255,1)", "rgba(240,178,66,1)", "rgba(232,106,58,1)"
DARK, CREAM = "rgba(28,16,24,1)", "rgba(246,239,228,1)"
ASSETS = {
    "bg": "solstice_bg_gradient.png",
    "bottle": "solstice_bottle.png",
    "logo": "solstice_logo_lockup.png",
}
MISSING_ASSETS = ["solstice_holiday_badge_2025.png", "promo_sticker.png", "qr_code.png"]

WORDS = [
    "chase",
    "the",
    "light",
    "botanical",
    "sparkling",
    "infusions",
    "crafted",
    "for",
    "golden",
    "hour",
    "introducing",
    "limited",
    "batch",
    "summer",
    "line",
    "small",
    "wild",
    "citrus",
    "bloom",
    "sunset",
    "sale",
    "award",
    "winning",
    "new",
    "flavor",
    "shop",
    "now",
    "discover",
    "taste",
    "refresh",
    "every",
    "moment",
    "bright",
    "crisp",
    "natural",
    "ingredients",
    "zero",
    "sugar",
]
LONG_WORDS = ["Limited-Batch-Solstice", "#ChaseTheLightEverywhere", "SUMMERLINE2026"]


def _phrase(rng: random.Random, n_words: int, title: bool = False) -> str:
    words = [rng.choice(WORDS) for _ in range(n_words)]
    if rng.random() < 0.15:
        words[rng.randrange(n_words)] = rng.choice(LONG_WORDS)
    text = " ".join(words)
    return text.title() if title else text[0].upper() + text[1:]


class _Builder:
    """Accumulates elements for one design, sizing text to fit its box."""

    def __init__(self, rng: random.Random, width: int, height: int, opaque_ids: bool):
        self.rng = rng
        self.w, self.h = width, height
        self.opaque = opaque_ids
        self.elements: list[dict] = []
        self.measurer = get_measurer()
        self._z = 0
        self.next_y = 0  # first free row below everything placed by a family

    def _id(self, name: str) -> str:
        return f"el_{len(self.elements) + 1}" if self.opaque else name

    def _next_z(self, z: int | None) -> int:
        if z is None:
            self._z += 1
            return self._z
        self._z = max(self._z, z)
        return z

    def image(self, name, asset, x, y, w, h, fit="contain", z=None, **extra) -> dict:
        el = {
            "id": self._id(name),
            "type": "image",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "z_index": self._next_z(z),
            "asset_id": asset,
            "fit": fit,
            **extra,
        }
        self.elements.append(el)
        return el

    def shape(self, name, kind, x, y, w, h, fill, z=None, **extra) -> dict:
        el = {
            "id": self._id(name),
            "type": "shape",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "z_index": self._next_z(z),
            "shape_type": kind,
            "fill_color": fill,
            **extra,
        }
        self.elements.append(el)
        return el

    def text(self, name, content, x, y, w, h, family, size, color, z=None, **extra) -> dict:
        el = {
            "id": self._id(name),
            "type": "text",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "z_index": self._next_z(z),
            "content": content,
            "font_family": family,
            "font_size": float(size),
            "color": color,
            **extra,
        }
        fitted = self.measurer.fit_font_size(el, min_size=6.0, max_size=float(size))
        if fitted is None:
            # Box too small for the content even at 6px: widen it to the
            # canvas and try once more; failing that, accept the overflow.
            el["x"], el["width"] = 40, self.w - 80
            fitted = self.measurer.fit_font_size(el, min_size=6.0, max_size=float(size))
        el["font_size"] = float(fitted if fitted is not None else 6.0)
        self.elements.append(el)
        return el

    def button(self, name, label, x, y, w, h, fill=ORANGE, z=None) -> None:
        gid = self._id(name)
        self.shape(
            f"{name}_bg", "rectangle", x, y, w, h, fill, z=z, corner_radius=h / 2, group_id=gid
        )
        pad = h * 0.25
        self.text(
            f"{name}_label",
            label,
            x,
            y + pad,
            w,
            h - 2 * pad,
            SANS,
            size=h * 0.42,
            color=WHITE,
            font_weight="bold",
            text_align="center",
            group_id=gid,
        )

    def design(self) -> dict:
        return {"canvas": {"width": self.w, "height": self.h}, "elements": self.elements}


# ----------------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------------


def poster_stack(b: _Builder) -> None:
    rng, w, h = b.rng, b.w, b.h
    b.image("bg", ASSETS["bg"], 0, 0, w, h, fit="cover", z=0)
    if rng.random() < 0.5:
        s = int(min(w, h) * rng.uniform(0.3, 0.6))
        b.shape("deco", "ellipse", -s // 3, -s // 3, s, s, WHITE, z=1, opacity=0.12)
    lw = int(w * 0.3)
    b.image("logo", ASSETS["logo"], (w - lw) // 2, int(h * 0.04), lw, int(lw * 0.34))
    hero = int(min(w, h) * rng.uniform(0.3, 0.45))
    b.image("hero", ASSETS["bottle"], (w - hero) // 2, int(h * 0.18), hero, hero)
    y = int(h * 0.18) + hero + int(h * 0.03)
    tw = int(w * 0.84)
    headline = _phrase(rng, rng.randint(2, 8), title=True)
    hh = int(h * 0.14)
    b.text(
        "headline",
        headline,
        (w - tw) // 2,
        y,
        tw,
        hh,
        SERIF,
        size=min(w, h) * 0.08,
        color=WHITE,
        font_weight="bold",
        text_align="center",
    )
    y += hh + int(h * 0.01)
    if rng.random() < 0.8:
        sh = int(h * 0.07)
        b.text(
            "subhead",
            _phrase(rng, rng.randint(5, 14)) + ".",
            (w - int(w * 0.74)) // 2,
            y,
            int(w * 0.74),
            sh,
            SANS,
            size=min(w, h) * 0.03,
            color="rgba(255,255,255,0.85)",
            text_align="center",
        )
        y += sh + int(h * 0.02)
    bw, bh = int(w * 0.4), int(min(w, h) * 0.085)
    if y + bh < h - int(h * 0.03):
        b.button("cta", _phrase(rng, rng.randint(2, 4), title=True), (w - bw) // 2, y, bw, bh)
        y += bh + int(h * 0.02)
    b.next_y = y


def badge_row(b: _Builder) -> None:
    poster_stack(b)
    rng, w, h = b.rng, b.w, b.h
    n = rng.randint(1, 5)
    bw = int((w * 0.86) / n) - 10
    bh = int(min(w, h) * 0.045)
    y = b.next_y
    if y + bh <= h - int(h * 0.06):  # room above the legal line
        for i in range(n):
            b.button(
                f"badge{i + 1}",
                _phrase(rng, rng.randint(1, 3)).upper(),
                int(w * 0.07) + i * (bw + 10),
                y,
                bw,
                bh,
                fill=ORANGE,
            )
        b.next_y = y + bh + int(h * 0.02)
    if rng.random() < 0.7:
        rw, rh = int(w * 0.28), int(min(w, h) * 0.05)
        gid = b._id("ribbon")
        b.shape("ribbon_bg", "rectangle", w - rw - 40, 30, rw, rh, GOLD, rotation=-12, group_id=gid)
        b.text(
            "ribbon_label",
            _phrase(rng, 2).upper(),
            w - rw - 40,
            30 + rh * 0.2,
            rw,
            rh * 0.6,
            SANS,
            size=rh * 0.45,
            color=DARK,
            font_weight="bold",
            text_align="center",
            rotation=-12,
            group_id=gid,
        )
    if rng.random() < 0.6 and b.next_y <= h - 110:
        b.image("watermark", rng.choice(MISSING_ASSETS), w - 220, h - 110, 180, 80, z=2)
    b.text(
        "legal",
        _phrase(rng, rng.randint(8, 16)) + ".",
        60,
        h - 40,
        w - 120,
        30,
        SANS,
        size=12,
        color="rgba(255,255,255,0.6)",
        text_align="center",
        z=6,
    )


def split_hero(b: _Builder) -> None:
    rng, w, h = b.rng, b.w, b.h
    b.shape("bg", "rectangle", 0, 0, w, h, CREAM, z=0)
    hero_w = int(w * 0.45)
    b.image("hero", ASSETS["bottle"], int(w * 0.04), int(h * 0.1), hero_w, int(h * 0.8))
    x = int(w * 0.52)
    tw = int(w * 0.44)
    lw = int(tw * 0.5)
    b.image("logo", ASSETS["logo"], x, int(h * 0.1), lw, int(lw * 0.34))
    b.text(
        "headline",
        _phrase(rng, rng.randint(2, 6), title=True),
        x,
        int(h * 0.28),
        tw,
        int(h * 0.25),
        SERIF,
        size=min(w, h) * 0.09,
        color=DARK,
        font_weight="bold",
    )
    b.text(
        "subhead",
        _phrase(rng, rng.randint(6, 14)) + ".",
        x,
        int(h * 0.56),
        tw,
        int(h * 0.14),
        SANS,
        size=min(w, h) * 0.03,
        color="rgba(28,16,24,0.8)",
    )
    b.button("cta", _phrase(rng, 2, title=True), x, int(h * 0.76), int(tw * 0.7), int(h * 0.09))


def text_heavy(b: _Builder) -> None:
    rng, w, h = b.rng, b.w, b.h
    b.image("bg", ASSETS["bg"], 0, 0, w, h, fit="cover", z=0)
    n = rng.randint(3, 6)
    y = int(h * 0.08)
    slot = int(h * 0.84 / n)
    for i in range(n):
        family = SERIF if i == 0 else SANS
        size = min(w, h) * (0.07 if i == 0 else rng.uniform(0.022, 0.04))
        words = rng.randint(2, 6) if i == 0 else rng.randint(6, 22)
        b.text(
            f"text{i + 1}",
            _phrase(rng, words, title=(i == 0)),
            int(w * 0.08),
            y,
            int(w * 0.84),
            slot - int(h * 0.02),
            family,
            size=size,
            color=WHITE if i else GOLD,
            font_weight="bold" if i == 0 else "regular",
            text_align=rng.choice(["left", "center"]),
        )
        y += slot


def multi_image(b: _Builder) -> None:
    rng, w, h = b.rng, b.w, b.h
    b.shape("bg", "rectangle", 0, 0, w, h, DARK, z=0)
    n = rng.randint(2, 4)
    gap = int(w * 0.04)
    cell = int((w - gap * (n + 1)) / n)
    y = int(h * 0.12)
    for i in range(n):
        asset = ASSETS["bottle"] if rng.random() < 0.8 else rng.choice(MISSING_ASSETS)
        x = gap + i * (cell + gap)
        b.image(f"img{i + 1}", asset, x, y, cell, cell)
        b.text(
            f"caption{i + 1}",
            _phrase(rng, rng.randint(1, 4), title=True),
            x,
            y + cell + 10,
            cell,
            int(h * 0.06),
            SANS,
            size=min(w, h) * 0.025,
            color=WHITE,
            text_align="center",
        )
    lw = int(w * 0.25)
    b.image("logo", ASSETS["logo"], (w - lw) // 2, int(h * 0.02), lw, int(lw * 0.34))
    b.text(
        "headline",
        _phrase(rng, rng.randint(2, 5), title=True),
        int(w * 0.1),
        int(h * 0.78),
        int(w * 0.8),
        int(h * 0.12),
        SERIF,
        size=min(w, h) * 0.07,
        color=GOLD,
        font_weight="bold",
        text_align="center",
    )


def minimal(b: _Builder) -> None:
    rng, w, h = b.rng, b.w, b.h
    b.image("bg", ASSETS["bg"], 0, 0, w, h, fit="cover", z=0)
    b.text(
        "headline",
        _phrase(rng, rng.randint(2, 5), title=True),
        int(w * 0.1),
        int(h * 0.35),
        int(w * 0.8),
        int(h * 0.3),
        SERIF,
        size=min(w, h) * 0.12,
        color=WHITE,
        font_weight="bold",
        text_align="center",
        style_runs=[{"start": 0, "end": 5, "color": GOLD}],
    )
    lw = int(w * 0.2)
    b.image("logo", ASSETS["logo"], (w - lw) // 2, int(h * 0.06), lw, int(lw * 0.34))


FAMILIES = {
    "poster_stack": poster_stack,
    "badge_row": badge_row,
    "text_heavy": text_heavy,
    "minimal": minimal,
    "split_hero": split_hero,
    "multi_image": multi_image,
}


def generate(n: int, seed: int = 0, holdout_fraction: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    seen_families = [f for f in FAMILIES if f not in HOLDOUT_FAMILIES]
    seen_targets = [t for t in CATALOGUE if t not in HOLDOUT_TARGETS]
    for i in range(n):
        holdout = rng.random() < holdout_fraction
        if holdout:
            # A held-out case uses a held-out family, a held-out target, or both.
            which = rng.choice(("family", "target", "both"))
            family = rng.choice(sorted(HOLDOUT_FAMILIES) if which != "target" else seen_families)
            target_name = rng.choice(sorted(HOLDOUT_TARGETS) if which != "family" else seen_targets)
        else:
            family = rng.choice(seen_families)
            target_name = rng.choice(seen_targets)
        w, h = rng.choice(SOURCE_CANVASES)
        b = _Builder(rng, w, h, opaque_ids=rng.random() < 0.4)
        FAMILIES[family](b)
        tasks.append(
            {
                "task_id": f"synth_{seed}_{i:03d}",
                "difficulty": "synthetic",
                "family": family,
                "holdout": holdout,
                "target_name": target_name,
                "source": b.design(),
                "target_canvas": dict(CATALOGUE[target_name]),
            }
        )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval/cases"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task in generate(args.n, args.seed):
        (args.out / f"{task['task_id']}.json").write_text(json.dumps(task, indent=2))
    print(f"wrote {args.n} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
