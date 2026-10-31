"""Optional image-role binding text for a freshly frozen recipe, not old runs.

This pure helper creates chat content only. The caller supplies actual images
separately in the same order and freezes the selected layout in its recipe.
It accepts no element IDs, labels, paths, predictions, or training annotations.
"""

from __future__ import annotations

LEGACY = "legacy"
INTERLEAVED_V1 = "interleaved-image-roles-v1"


def build_image_role_content(task, prompt, *, image_count, layout=LEGACY, purpose_views=2):
    """Return user content with unchanged prompt text and ordered placeholders.

    Legacy exactly matches image placeholders followed by final prompt. V1
    inserts fixed known input-role text immediately before each placeholder;
    it does not supply a preferred candidate or inferred element purpose.
    """
    if task not in ("purpose", "critic") or layout not in (LEGACY, INTERLEAVED_V1):
        raise ValueError("Image-role content needs a known task and explicit layout version")
    if not isinstance(prompt, str) or not 0 < len(prompt) <= 64_000:
        raise ValueError("The final prompt must be bounded raw text")
    if type(purpose_views) is not int or purpose_views not in (2, 3):
        raise ValueError("Purpose views must be integer two or three")
    expected = 3 if task == "critic" else purpose_views
    if type(image_count) is not int or image_count != expected:
        raise ValueError("Actual image count differs from the frozen task/view contract")
    if layout == LEGACY:
        return [{"type": "image"} for _ in range(expected)] + [{"type": "text", "text": prompt}]
    labels = (
        ("ORIGINAL SOURCE:", "CANDIDATE A:", "CANDIDATE B:")
        if task == "critic"
        else ("ORIGINAL SOURCE:", "TARGET CONTEXT:", "ISOLATED TARGET ON NEUTRAL CHECKER:")
    )
    content = []
    for label in labels[:expected]:
        content.extend(({"type": "text", "text": label}, {"type": "image"}))
    content.append({"type": "text", "text": prompt})
    return content
