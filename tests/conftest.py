"""Shared fixtures: the sample tasks in input/ and a catalogue of realistic
target canvases (RULES.md: hidden targets look like real platform formats)."""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TASKS = sorted((REPO_ROOT / "input").glob("*.json"))

# Real-world formats the engine should cope with, including the extreme
# ratios (tall -> very wide, square -> very tall) that break naive approaches.
TARGET_CANVASES = {
    "instagram_post": {"width": 1080, "height": 1080},
    "instagram_story": {"width": 1080, "height": 1920},
    "youtube_thumbnail": {"width": 1280, "height": 720},
    "twitter_header": {"width": 1500, "height": 500},
    "pinterest_pin": {"width": 1000, "height": 1500},
    "leaderboard_ad": {"width": 728, "height": 90},
}


def _task_id(path: Path) -> str:
    return path.stem.removeprefix("chameleon_sample_")


@pytest.fixture(params=SAMPLE_TASKS, ids=_task_id)
def source(request) -> dict:
    """The `source` design of one sample task."""
    return json.loads(request.param.read_text())["source"]


@pytest.fixture(params=TARGET_CANVASES.values(), ids=list(TARGET_CANVASES))
def target_canvas(request) -> dict:
    return dict(request.param)
