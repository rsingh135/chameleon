"""Small authored development tasks, never held-out or human taste labels."""

from copy import deepcopy


def _text(identity, content, x, y, width, height, size, group=None):
    return {
        "id": identity,
        "type": "text",
        "content": content,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "font_family": "Solstice Sans",
        "font_size": size,
        "color": "#111111",
        "z_index": 2,
        **({"group_id": group} if group else {}),
    }


def fundamental_tasks():
    """Return fresh task copies; these synthetic cases cannot qualify a trained VLM."""
    simple = {
        "canvas": {"width": 320, "height": 240},
        "elements": [
            _text("message", "Keep the message", 24, 32, 272, 48, 24),
            _text("detail", "Source copy and fonts stay intact.", 24, 104, 272, 40, 16),
        ],
    }
    grouped = {
        "canvas": {"width": 320, "height": 480},
        "elements": [
            _text("headline", "A clear headline", 24, 32, 272, 56, 28),
            _text("body", "Supporting details stay readable.", 24, 120, 272, 64, 18),
            {
                "id": "action-plate",
                "type": "shape",
                "shape_type": "rectangle",
                "fill_color": "#eeeeee",
                "x": 24,
                "y": 240,
                "width": 220,
                "height": 56,
                "group_id": "action",
                "z_index": 1,
            },
            _text("action-label", "Read more", 40, 252, 180, 28, 20, "action"),
        ],
    }
    rows = [
        {"id": "identity", "source": simple, "target_canvas": {"width": 320, "height": 240}},
        {"id": "proportional", "source": simple, "target_canvas": {"width": 640, "height": 480}},
        {
            "id": "portrait-to-landscape",
            "source": grouped,
            "target_canvas": {"width": 640, "height": 360},
        },
        {"id": "action-group", "source": grouped, "target_canvas": {"width": 480, "height": 640}},
    ]
    return deepcopy(rows)
