import os
from pathlib import Path


def weight(name):
    root = os.environ.get("ACM_WEIGHTS_ROOT")
    if not root:
        raise RuntimeError("Set ACM_WEIGHTS_ROOT or pass --weights-root.")
    return str(Path(root).expanduser().resolve() / name)


def feature_root():
    root = os.environ.get("TWP_DATA_ROOT")
    if not root:
        raise RuntimeError("Set TWP_DATA_ROOT or pass --data-root.")
    return Path(root).expanduser().resolve()
