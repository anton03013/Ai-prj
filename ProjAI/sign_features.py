"""Shared MediaPipe landmark features for the ASL digit classifier."""
from __future__ import annotations

import numpy as np

FEATURE_SIZE = 2 * 21 * 3  # left and right hands, 21 landmarks, x/y/z


def landmarks_to_features(result) -> np.ndarray | None:
    """Return scale-normalized landmarks in fixed Left/Right hand slots.

    Fixed slots make a two-handed sign such as ASL digit 0 usable by the
    classifier while keeping one-handed signs at the same vector length.
    """
    slots = {"Left": np.zeros((21, 3), dtype=np.float32),
             "Right": np.zeros((21, 3), dtype=np.float32)}
    found = False

    for index, hand in enumerate(result.hand_landmarks):
        if index >= len(result.handedness) or not result.handedness[index]:
            continue
        side = result.handedness[index][0].category_name
        if side not in slots:
            continue

        points = np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32)
        points -= points[0]  # wrist becomes the origin
        scale = np.linalg.norm(points[:, :2], axis=1).max()
        if scale > 1e-6:
            points /= scale
        slots[side] = points
        found = True

    if not found:
        return None
    return np.concatenate((slots["Left"].ravel(), slots["Right"].ravel()))
