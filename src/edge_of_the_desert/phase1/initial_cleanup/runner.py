from __future__ import annotations

from pathlib import Path

from .common import load_paths, progress
from .prepare_nwaa_public_supply import prepare_nwaa_public_supply
from .filter_ir_huc12_total_to_az import ensure_ir_huc12_az_monthly
from .prepare_snotel_bundle import ensure_snotel_monthly
from .prepare_powell_data import ensure_powell_monthly
from .prepare_azpop import ensure_azpop_monthly

import csv
import glob


def run_initial_cleanup(config_path: Path) -> None:
    paths = load_paths(config_path)
    progress(f"starting initial cleanup using {paths['config_path'].relative_to(paths['repo_root'])}")
    prepare_nwaa_public_supply()
    ensure_ir_huc12_az_monthly(**paths)
    ensure_snotel_monthly(**paths)
    ensure_powell_monthly(**paths)
    ensure_azpop_monthly(**paths)
    progress("initial cleanup complete")

