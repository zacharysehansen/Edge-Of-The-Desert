from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import glob


def load_paths(config_path: Path) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    resolved_config = config_path if config_path.is_absolute() else repo_root / config_path
    config = json.loads(resolved_config.read_text())
    paths = config.get("paths", {})
    raw_dir = resolve_repo_path(repo_root, paths.get("raw_dir", "data/raw"))
    processed_dir = resolve_repo_path(repo_root, paths.get("processed_dir", "data/processed"))
    final_dir_value = paths.get("final_dir", "data/Final")
    final_dir = resolve_repo_path(repo_root, final_dir_value)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": repo_root,
        "config_path": resolved_config,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "final_dir": final_dir,
    }

def load_arizona_huc12_ids(raw_dir: Path) -> set[str]:
    ids: set[str] = set()
    pattern = str(raw_dir / "nwaa_data_*_of_6.csv")
    for filepath in glob.glob(pattern):
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                huc12_id = row.get("huc12_id", "").strip()
                if huc12_id:
                    ids.add(huc12_id)
    return ids

def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def progress(message: str) -> None:
    print(f"[phase1-prep] {message}", file=sys.stderr, flush=True)
