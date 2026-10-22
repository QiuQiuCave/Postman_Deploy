"""Small asset registry and resolved run configuration, not a plugin framework."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


class ConfigurationErrors:
    def error(self, message):
        raise ValueError(message)


def resolve_archived_path(value):
    """Relocate references in memory without rewriting original manifests."""
    path = Path(value).expanduser()
    old = os.environ.get("R2V2_ARCHIVE_ROOT")
    new = os.environ.get("R2V2_DATA_ROOT")
    if old and new and path.is_absolute() and path.is_relative_to(Path(old)):
        return Path(new) / path.relative_to(Path(old))
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


class AssetRegistry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.document = json.loads(self.path.read_text())
        if self.document.get("schema_version") != 1:
            raise ValueError("Unsupported asset registry")

    def resolve(self, name):
        item = self.document["assets"][name]
        value = os.path.expandvars(item["path"])
        if "$" in value:
            raise ValueError(f"Unresolved environment variable for asset {name}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        if item.get("sha256") and (not path.is_file() or digest(path) != item["sha256"]):
            raise ValueError(f"Asset hash mismatch: {name}")
        return path


def environment():
    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    )


def prepare_run(output, config, *, assets=(), seed=0):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    from r2v2_deploy.data.artifacts import write_json

    write_json(output / "resolved_config.json", config)
    repos = {}
    for label, variable in (("HALO-mj", "R2V2_HALO_REPO"), ("Postman_Deploy", "R2V2_DEPLOY_REPO")):
        root = os.environ.get(variable)
        if root:
            sha = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True
            )
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True
            )
            repos[label] = dict(
                sha=sha.stdout.strip() if sha.returncode == 0 else None, dirty=bool(dirty.stdout)
            )
        else:
            supplied = os.environ.get("R2V2_HALO_SHA" if label == "HALO-mj" else "R2V2_DEPLOY_SHA")
            if supplied and (
                len(supplied) != 40 or any(c not in "0123456789abcdef" for c in supplied)
            ):
                raise ValueError("Expected full40-character source SHA")
            repos[label] = dict(
                sha=supplied,
                source="explicit build SHA" if supplied else None,
                reason=None if supplied else f"{variable} not set; never guess a revision",
            )
    write_json(
        output / "provenance.json",
        dict(
            schema_version=1,
            repositories=repos,
            seed=int(seed),
            assets={str(p): digest(p) for p in assets},
            environment=environment(),
        ),
    )
    return output
