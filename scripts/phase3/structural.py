"""Python mirror of frontend/structural.js — Layer 2 and Layer 3 arithmetic.

Both implementations read the same `frontend/structural_params.json`, so what is
duplicated here is roughly forty lines of formula and never a number.
`scripts/phase3/check_catalog_parity.py` compares the two on a fixed scenario set, so
the headless acceptance sweep cannot drift away from what the browser computes.

See frontend/structural.js for the reasoning; this file deliberately carries no
argument of its own, only the same operations in the same order.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "frontend" / "structural_params.json"

STRUCTURAL_OUTPUTS = ("groundwater", "grace", "ndvi", "surface_water", "wildlife")

SQ_M_PER_ACRE = 4046.8564224
CFS_PER_CMS = 35.3147


class Structural:
    def __init__(self, params: dict | None = None):
        self.params = params or json.loads(PARAMS_PATH.read_text())
        self.c = {k: v["value"] for k, v in self.params["constants"].items()}
        self.reversion = {
            k: v["value"] for k, v in self.params["reversion_per_month"].items()
        }
        self.pumping_per_unit = {
            k: v["value"] for k, v in self.params["pumping_mgd_per_slider_unit"].items()
        }

    # ── the pumping chain ────────────────────────────────────────────────────

    def pumping_af_per_month(self, slider: str, delta: float) -> float:
        per_unit = self.pumping_per_unit.get(slider)
        if per_unit is None:
            return 0.0
        return per_unit * delta * self.c["af_per_mgd_month"]

    def mead_reduction_kaf_per_year(self, elevation_ft: float) -> float:
        for tier in self.params["mead_tiers"]:
            if elevation_ft > tier["above"]:
                return float(tier["az_combined_kaf"])
        return float(self.params["mead_tiers"][-1]["az_combined_kaf"])

    def mead_substituted_pumping_af_per_month(self, delta: float) -> float:
        baseline = self.params["mead_baseline_elevation"]
        cut_kaf = self.mead_reduction_kaf_per_year(
            baseline + delta
        ) - self.mead_reduction_kaf_per_year(baseline)
        return (
            cut_kaf
            * 1000.0
            * self.c["region_share_of_az_reduction"]
            * self.c["groundwater_substitution_fraction"]
        ) / 12.0

    def flow_to_log_anomaly(self, delta_cfs: float) -> float:
        return math.log(max(1e-6, 1 + delta_cfs / self.c["regional_baseline_cfs"]))

    def runoff_cfs(self, delta_impervious_points: float) -> float:
        contrast = (
            self.c["runoff_coefficient_impervious"]
            - self.c["runoff_coefficient_natural"]
        )
        region_m2 = self.c["region_acres"] * SQ_M_PER_ACRE
        depth_m_per_s = self.c["mean_precipitation_mm_day"] / 1000 / 86400
        return (
            (delta_impervious_points / 100) * contrast * depth_m_per_s * region_m2
            * CFS_PER_CMS
        )

    # ── per-lever forcing ────────────────────────────────────────────────────

    def lever_contribution(self, lever: dict, deltas: dict) -> float:
        delta = deltas.get(lever["slider"], 0.0)
        if delta == 0:
            return 0.0
        path = lever["path"]

        if path == "pumping_to_depth":
            return (
                self.pumping_af_per_month(lever["slider"], delta)
                / self.c["storage_af_per_ft"]
            )
        if path == "mead_tier_to_depth":
            return (
                self.mead_substituted_pumping_af_per_month(delta)
                / self.c["storage_af_per_ft"]
            )
        if path == "pumping_to_storage":
            return (
                -self.pumping_af_per_month(lever["slider"], delta)
                * self.c["grace_units_per_af"]
            )
        if path == "mead_tier_to_storage":
            return (
                -self.mead_substituted_pumping_af_per_month(delta)
                * self.c["grace_units_per_af"]
            )
        if path == "effluent_to_flow":
            mgd = (
                delta
                * (self.c["gpcd_groundwater"] / 1e6)
                * self.c["effluent_return_fraction"]
            )
            return self.flow_to_log_anomaly(mgd * self.c["cfs_per_mgd"])
        if path == "runoff_to_flow":
            return self.flow_to_log_anomaly(self.runoff_cfs(delta))
        if path == "capture_to_flow":
            captured = (
                self.pumping_af_per_month(lever["slider"], delta)
                * self.c["stream_capture_fraction"]
            )
            return self.flow_to_log_anomaly(-captured * self.c["cfs_per_af_month"])
        if path == "mead_capture_to_flow":
            captured = (
                self.mead_substituted_pumping_af_per_month(delta)
                * self.c["stream_capture_fraction"]
            )
            return self.flow_to_log_anomaly(-captured * self.c["cfs_per_af_month"])
        if path == "landcover_impervious":
            return (delta / 100.0) * (self.c["ndvi_impervious"] - self.c["ndvi_natural"])
        if path == "landcover_irrigated":
            return (
                (delta / 100.0)
                * self.c["irrigated_fraction"]
                * (self.c["ndvi_irrigated_crop"] - self.c["ndvi_natural"])
            )
        raise ValueError(f"Unknown structural path '{path}'")

    # ── Layer 3 ──────────────────────────────────────────────────────────────

    @staticmethod
    def integrate_rate(per_month: float, lam: float, months: int) -> float:
        if months <= 0:
            return 0.0
        if lam <= 0:
            raise ValueError(
                "Refusing to integrate without a reversion rate: naive accumulation "
                "is unbounded (PHASE3_PLAN.md §4a)."
            )
        return (per_month / lam) * (1 - (1 - lam) ** months)

    def integrate_learned_residual(
        self, output: str, residual: float, months: int
    ) -> float:
        if not self.params["integrate_learned_residual"]["value"]:
            return residual
        lam = self.reversion.get(output, 0.0)
        if lam <= 0:
            return residual
        return self.integrate_rate(residual, lam, months)

    # ── public entry point ───────────────────────────────────────────────────

    def response(self, deltas: dict, month: int, duration_months: int) -> dict:
        result = {out: {"total": 0.0, "byLever": {}} for out in STRUCTURAL_OUTPUTS}
        for lever in self.params["levers"]:
            contribution = self.lever_contribution(lever, deltas)
            if contribution == 0:
                continue
            displacement = (
                self.integrate_rate(
                    contribution, self.reversion.get(lever["output"], 0.0), duration_months
                )
                if lever["kind"] == "rate"
                else contribution
            )
            bucket = result[lever["output"]]
            bucket["total"] += displacement
            bucket["byLever"][lever["id"]] = {
                "displacement": displacement,
                "perMonth": contribution if lever["kind"] == "rate" else None,
                "kind": lever["kind"],
                "tier": lever["tier"],
                "slider": lever["slider"],
            }

        for transfer in self.params["transfers"]:
            source = result.get(transfer["from"])
            if not source or source["total"] == 0:
                continue
            displacement = self.c[transfer["constant"]] * source["total"]
            bucket = result[transfer["to"]]
            bucket["total"] += displacement
            bucket["byLever"][transfer["id"]] = {
                "displacement": displacement,
                "perMonth": None,
                "kind": "transfer",
                "tier": transfer["tier"],
                "from": transfer["from"],
            }
        return result
