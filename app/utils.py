import re
from pathlib import Path

def natural_key(path: Path):
    """
    Sort key that treats digit runs in filenames as integers, so
    _1, _2, ..., _10 instead of _1, _10, _11, _2.

    Works for names like 'X and IP Quiz (Week 10)_12.jpg'.
    """
    name = path.name
    # Split into digit and non-digit chunks: ["X and IP Quiz (Week ", "10", ")_", "12", ".jpg"]
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key
