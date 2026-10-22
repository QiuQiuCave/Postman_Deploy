"""Explicit relocation of immutable bundle references; never rewrites evidence."""

import os
from pathlib import Path


def bundle_path(value):
    path = Path(value).expanduser()
    source = os.environ.get("R2V2_BUNDLE_SOURCE_ROOT")
    target = os.environ.get("R2V2_BUNDLE_TARGET_ROOT")
    if bool(source) != bool(target):
        raise ValueError("Bundle relocation requires both explicit roots")
    if source:
        source, target = Path(source), Path(target)
        if (
            not source.is_absolute()
            or not target.is_absolute()
            or len(source.parts) < 3
            or len(target.parts) < 3
            or ".." in source.parts
            or ".." in target.parts
        ):
            raise ValueError("Invalid bundle relocation roots")
        if path.is_absolute() and path.is_relative_to(source):
            return target / path.relative_to(source)
    return path
