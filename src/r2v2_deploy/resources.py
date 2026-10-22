"""Installed resource lookup; never mutates sys.path."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
_checkout = PACKAGE_ROOT.parents[1]
PROJECT_ROOT = Path(
    os.environ.get(
        "R2V2_RESOURCE_ROOT",
        str(
            _checkout
            if (_checkout / "r2v2_description/source").is_dir()
            else PACKAGE_ROOT / "_resources"
        ),
    )
)
