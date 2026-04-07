from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    raise SystemExit(
        "Placeholder only: logistic-regression baseline will be added next using the same split and artifact pipeline."
    )


if __name__ == "__main__":
    main()
