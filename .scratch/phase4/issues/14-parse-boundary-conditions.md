# 14 — Parse and ingest boundary-condition data (OEO + CRSS)

**What to build:** Scripts that parse the acquired projection files into clean, version-controlled CSVs. (a) `data/raw/projections/AZ_OEO_All_Series_2025-2060.zip` → Pima County population by series (low/medium/high), annual, to `data/processed/projections/pima_population_oeo.csv`. (b) `data/raw/crss/FEIS_AppM_LowerBasin_DemandSchedule.pdf` → Arizona CAP and Mead depletion schedules by priority class, annual 2027–2060, to `data/processed/projections/az_depletions_crss.csv` (extracted via `pdftotext -layout`). These become the business-as-usual reference trajectories that the population and Lake Mead controls deviate from.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] OEO zip is extracted and Pima County series (low/medium/high) written to a clean CSV with columns: year, low, medium, high
- [ ] Medium series matches the values in PHASE4.md §4.6 (2025: 1,093,761 → 2060: 1,143,575)
- [ ] CRSS PDF is parsed via `pdftotext -layout` and Arizona depletion schedules written to CSV
- [ ] CSV columns include: year, user/priority class, depletion (KAF)
- [ ] Both CSVs are written to `data/processed/projections/`
- [ ] A verification script prints summary statistics matching the values cited in PHASE4.md
