import os
from typing import List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def landmarks_to_feature(landmarks: List[float]) -> List[float]:
    """Return a flatten list of landmarks (already normalized x,y,z values).

    landmarks: sequence of (x,y,z) floats flattened or list of objects with x,y,z attributes.
    """
    # If landmarks are objects with x,y,z attributes, convert
    if len(landmarks) and hasattr(landmarks[0], 'x'):
        flat = []
        for lm in landmarks:
            flat.extend([lm.x, lm.y, lm.z])
        return flat

    # otherwise assume it's already a flat list/iterable
    return list(landmarks)
