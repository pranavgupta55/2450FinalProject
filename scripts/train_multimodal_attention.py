from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    raise SystemExit(
        "Placeholder only: the multimodal attention model will be added after FinBERT features are integrated."
    )


if __name__ == "__main__":
    main()
