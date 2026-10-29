"""Document-local assets and exact native fonts without changing vendored code.

Resource directories are immutable once used: importers materialize each document
under a content-addressed folder. Caches therefore key on resolved directories,
not mutable request state. Context variables keep concurrent documents isolated.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cr_renderer import CrelloV5Renderer
    from cr_renderer.fonts import FontManager

_BRAND_KIT = Path(__file__).resolve().parents[1] / "brand_kit"
DEFAULT_ASSETS_DIR = (_BRAND_KIT / "assets").resolve()
DEFAULT_FONTS_DIR = (_BRAND_KIT / "fonts").resolve()
_CACHE_SIZE = 16
_LOCK = RLock()


@dataclass(frozen=True)
class ResourceContext:
    assets_dir: Path = DEFAULT_ASSETS_DIR
    fonts_dir: Path = DEFAULT_FONTS_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets_dir", Path(self.assets_dir).resolve())
        object.__setattr__(self, "fonts_dir", Path(self.fonts_dir).resolve())


_DEFAULT_RESOURCES = ResourceContext()
_RESOURCES = ContextVar("chameleon_resources", default=_DEFAULT_RESOURCES)


def get_resources() -> ResourceContext:
    """Return this task/thread's absolute resource directories."""
    return _RESOURCES.get()


@contextmanager
def using_resources(
    assets_dir: str | Path | None = None, fonts_dir: str | Path | None = None
) -> Iterator[ResourceContext]:
    """Temporarily override resources; omitted directories inherit the context."""
    current = get_resources()
    resources = ResourceContext(
        current.assets_dir if assets_dir is None else Path(assets_dir),
        current.fonts_dir if fonts_dir is None else Path(fonts_dir),
    )
    token = _RESOURCES.set(resources)
    try:
        yield resources
    finally:
        _RESOURCES.reset(token)


def _manifest_path(fonts_dir: Path) -> Path:
    manifest = fonts_dir / "font-manifest.json"
    if not manifest.resolve().is_relative_to(fonts_dir):
        raise ValueError("Font manifest must remain inside its document fonts directory.")
    return manifest


def _font_bytes(fonts_dir: Path, filename: object) -> bytes:
    import skia

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Each font manifest entry requires a nonempty file path.")
    # Treat both separators consistently on Windows and the Linux test/runtime image.
    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(filename).drive:
        raise ValueError(f"Font file path must be relative to its fonts directory: {filename!r}")
    path = (fonts_dir / relative).resolve()
    if not path.is_relative_to(fonts_dir):
        raise ValueError(f"Font file path escapes its fonts directory: {filename!r}")
    if path.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError(f"Document font must be an original .ttf or .otf file: {filename!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Document font file is missing: {path}")
    data = path.read_bytes()
    if len(data) < 12 or data[:4] not in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"}:
        raise ValueError(f"Invalid TrueType/OpenType font header: {filename!r}")
    if skia.Typeface.MakeFromData(data) is None:
        raise ValueError(f"Invalid TrueType/OpenType font data: {filename!r}")
    return data


def _manifest_manager(fonts_dir: Path, manifest: Path) -> FontManager:
    from cr_renderer.fonts import FontManager, normalize_family

    class DocumentFontManager(FontManager):
        """Native aliases must never silently become a different family or face."""

        def lookup(
            self,
            font_family: str,
            font_weight: str = "regular",
            font_style: str = "regular",
        ) -> bytes:
            key = normalize_family(font_family)
            family = self._fonts.get(key)
            if not family:
                raise LookupError(f"Document font family is not available: {font_family!r}")
            for font in family:
                if font["fontWeight"] == font_weight and font["fontStyle"] == font_style:
                    return font["bytes"]
            raise LookupError(
                f"Document font face is not available: {font_family!r}, "
                f"weight={font_weight!r}, style={font_style!r}"
            )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("fonts") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Font manifest must be a font-entry list or an object containing fonts.")
    fonts: dict[str, list[dict]] = {}
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each font manifest entry must be an object.")
        family = entry.get("font_family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Each font manifest entry requires a nonempty font_family alias.")
        weight, style = entry.get("weight"), entry.get("style")
        if weight not in {"regular", "bold"} or style not in {"normal", "italic"}:
            raise ValueError("Font manifest weight/style must be regular|bold and normal|italic.")
        renderer_style = (
            "bolditalic"
            if weight == "bold" and style == "italic"
            else "bold"
            if weight == "bold"
            else "italic"
            if style == "italic"
            else "regular"
        )
        key = normalize_family(family)
        identity = (key, weight, renderer_style)
        if identity in seen:
            raise ValueError(f"Duplicate document font alias/face: {family!r}, {weight}, {style}")
        seen.add(identity)
        fonts.setdefault(key, []).append(
            {
                "fontFamily": family,
                "fontWeight": weight,
                "fontStyle": renderer_style,
                "bytes": _font_bytes(fonts_dir, entry.get("file")),
            }
        )
    manager = DocumentFontManager.__new__(DocumentFontManager)
    manager._fonts = fonts
    return manager


@lru_cache(maxsize=_CACHE_SIZE)
def _load_font_manager(fonts_dir: Path) -> FontManager:
    from renderer.converter.brand_fonts import load_brand_font_manager

    manifest = _manifest_path(fonts_dir)
    # Bundled defaults retain the original filename loader and its established faces.
    if fonts_dir != DEFAULT_FONTS_DIR and manifest.is_file():
        return _manifest_manager(fonts_dir, manifest)
    return load_brand_font_manager(str(fonts_dir))


def load_font_manager(fonts_dir: str | Path | None = None) -> FontManager:
    """Load exact manifest aliases, or preserve the existing brand-folder loader."""
    directory = get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
    with _LOCK:
        renderer = _renderers.get(str(directory))
        if renderer is not None:
            return renderer.font_manager
        return _load_font_manager(directory)


_renderers: OrderedDict[str, CrelloV5Renderer] = OrderedDict()


def get_renderer(fonts_dir: str | Path | None = None) -> CrelloV5Renderer:
    """Use the vendored renderer cache with a document's exact immutable fonts."""
    from renderer.converter import convert

    directory = get_resources().fonts_dir if fonts_dir is None else Path(fonts_dir).resolve()
    key = str(directory)
    with _LOCK:
        if directory == DEFAULT_FONTS_DIR:
            return convert._get_renderer(key)
        renderer = _renderers.get(key)
        if renderer is not None:
            _renderers.move_to_end(key)
            return renderer
        manager = load_font_manager(directory)
        if _manifest_path(directory).is_file():
            # The vendored constructor scans flat filename-derived aliases. Copy its
            # template so nested/empty manifests work without mutating the default.
            renderer = copy(convert._get_renderer(str(DEFAULT_FONTS_DIR)))
            renderer.font_manager = manager
            convert._renderer_cache[key] = renderer
        else:
            renderer = convert._get_renderer(key)
            renderer.font_manager = manager
        renderer = convert._get_renderer(key)
        _renderers[key] = renderer
        if len(_renderers) > _CACHE_SIZE:
            evicted_key, evicted = _renderers.popitem(last=False)
            if convert._renderer_cache.get(evicted_key) is evicted:
                del convert._renderer_cache[evicted_key]
        return renderer


def clear_resource_caches() -> None:
    """Clear owned custom caches without changing the bundled/default renderer."""
    from renderer.converter import convert

    with _LOCK:
        for key, renderer in _renderers.items():
            if convert._renderer_cache.get(key) is renderer:
                del convert._renderer_cache[key]
        _renderers.clear()
        _load_font_manager.cache_clear()
