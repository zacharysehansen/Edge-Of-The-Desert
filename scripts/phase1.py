from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge_of_the_desert.phase1.initial_cleanup import run_initial_cleanup


def resolve_config_path(argv: list[str]) -> Path:
    for index, arg in enumerate(argv):
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
        if arg == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1])
    return Path("config/phase1.example.json")


run_initial_cleanup(resolve_config_path(sys.argv[1:]))

from edge_of_the_desert.phase1.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
