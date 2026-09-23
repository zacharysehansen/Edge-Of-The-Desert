# 18 — Fire-ignition measurement

**What to build:** A research script that measures the relationship between human ignition density and distance-to-development using `data/raw/wildfire/` (fire perimeters 1984–2023) and `data/raw/NLCD/` (annual impervious cover 1995–2024). Uses the project's standing three-part decision rule (Δ > 0, 4 of 5 folds, |t| ≥ 2), declared before the run. If the result is REAL: a human→wildfire coefficient ships with source and status tags. If NULL: wildfire is labelled climate-only and the null is recorded in PHASE3_PLAN.md alongside the other experiment nulls. Expects a possible sign tension: development adds ignitions (sprawl edge) but removes fuel (dense core). Writes `model/experiment_fire_ignition.json`.

**Blocked by:** None — can start immediately (independent research pass).

**Status:** ready-for-agent

- [ ] Ignition points are extracted from the fire perimeter dataset
- [ ] Distance-to-development is computed from NLCD impervious surface for each ignition
- [ ] The relationship is tested under the three-part rule: Δ > 0, 4/5 folds, |t| ≥ 2
- [ ] If REAL: coefficient, confidence interval, and source are written to the experiment JSON
- [ ] If NULL: the null is recorded and wildfire is flagged as climate-only
- [ ] Sign tension between core (fuel removal) and edge (ignition) is investigated and reported
- [ ] Decision rule is declared in the script's docstring before the measurement code
