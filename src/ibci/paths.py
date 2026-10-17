"""Where the datasets are stored.

By default every dataset lives under ``data/`` in the repository. Two optional settings change this:

1. The environment variable ``IBCI_DATA`` points to another folder, for example a larger drive.
2. A file ``data_roots.json`` at the repository root (not tracked by git) gives a folder per dataset or participant,
   for setups where the data are spread over several drives, for example::

       {"default": "D:/ibci-data", "T8": "H:/ibci-data", "T9": "G:/ibci-data"}

An explicitly set ``IBCI_DATA`` wins; otherwise ``data_roots.json`` is used for the keys it names, then its
"default" entry, then ``data/``. Each folder has the layout described in
docs/REPRODUCE.md (``link/``, ``perich/``, ``braingate/decoding/<participant>/``, ``braingate/yield/``, ``mindful/``).
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO / "data_roots.json"


def _local_roots():
    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {}


def data_root(key=None):
    """Folder that holds the data for ``key`` (a participant such as "T9", or "yield", "mindful", ...)."""
    if os.environ.get("IBCI_DATA"):
        return os.environ["IBCI_DATA"]
    roots = _local_roots()
    if key is not None and key in roots:
        return roots[key]
    return roots.get("default") or str(REPO / "data")
