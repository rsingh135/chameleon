"""Catalogue of common real-world canvas formats.

RULES.md: hidden target sizes are drawn from a fixed catalogue of platform
formats "described the way a designer would". The official list has not
been shared yet; this is the standard set, used for tests and the harness.
"""

from __future__ import annotations

CATALOGUE: dict[str, dict[str, int]] = {
    # Social: square / portrait
    "instagram_post": {"width": 1080, "height": 1080},
    "instagram_portrait": {"width": 1080, "height": 1350},
    "instagram_story": {"width": 1080, "height": 1920},
    "facebook_post": {"width": 1200, "height": 630},
    "facebook_cover": {"width": 820, "height": 312},
    "linkedin_post": {"width": 1200, "height": 627},
    "linkedin_banner": {"width": 1584, "height": 396},
    "x_post": {"width": 1600, "height": 900},
    "x_header": {"width": 1500, "height": 500},
    "pinterest_pin": {"width": 1000, "height": 1500},
    "tiktok_video": {"width": 1080, "height": 1920},
    # Video platforms
    "youtube_thumbnail": {"width": 1280, "height": 720},
    "youtube_banner": {"width": 2560, "height": 1440},
    # Display ads (IAB)
    "leaderboard": {"width": 728, "height": 90},
    "medium_rectangle": {"width": 300, "height": 250},
    "wide_skyscraper": {"width": 160, "height": 600},
    "half_page": {"width": 300, "height": 600},
    "billboard": {"width": 970, "height": 250},
    "mobile_banner": {"width": 320, "height": 50},
    # Print / presentation
    "a4_portrait": {"width": 2480, "height": 3508},
    "a4_landscape": {"width": 3508, "height": 2480},
    "us_letter": {"width": 2550, "height": 3300},
    "presentation_16_9": {"width": 1920, "height": 1080},
    "presentation_4_3": {"width": 1024, "height": 768},
    "poster_24x36": {"width": 2400, "height": 3600},
    "business_card": {"width": 1050, "height": 600},
}


def aspect(canvas: dict[str, int]) -> float:
    return canvas["width"] / canvas["height"]


def by_aspect_distance(source: dict[str, int], n: int | None = None) -> list[str]:
    """Catalogue names sorted from most similar to most different aspect
    ratio relative to `source` (log-ratio distance)."""
    import math

    src = math.log(aspect(source))
    ranked = sorted(CATALOGUE, key=lambda k: abs(math.log(aspect(CATALOGUE[k])) - src))
    return ranked[:n] if n else ranked
