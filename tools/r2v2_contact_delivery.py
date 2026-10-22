"""Checkout entry for the isolated native CONTACT runtime (not the Loco FSM).

The hash-pinned upstream package is unmodified. Use requirements-contact-delivery.txt
in a separate Python 3.12 environment, not the legacy root project environment.
"""

from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from r2v2_deploy.cli.delivery import main as delivery_main

    return delivery_main()


if __name__ == "__main__":
    raise SystemExit(main())
