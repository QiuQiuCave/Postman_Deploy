#!/usr/bin/env python3
"""Export a successful hand-fixture trace for inspectable full-body replay."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.r2v2_crate_motion_recording import load_crate_motion, record_crate_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be new or empty; never overwritten")
    args = parser.parse_args()
    path = record_crate_motion(args.source_dir, args.output_dir)
    motion = load_crate_motion(path)
    print(f"Recorded {motion.manifest['samples']} samples / {motion.duration_s:.3f} s: {path}")
    print(f"NPZ SHA256: {motion.manifest['trajectory_sha256']}")


if __name__ == "__main__":
    main()
