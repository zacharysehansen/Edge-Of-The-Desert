# Phase 3 — Layer 2 parameter sourcing and D5 design notes

**Status: research complete, no code changed yet.** This is the input document for implementing
[PHASE3_PLAN.md](PHASE3_PLAN.md) §4 (Layer 2) and §2 D5 (slider reparameterization). It exists
because the plan tagged every Layer 2 coefficient `UNTESTED — parameters need sourcing`, and
sourcing them was the blocking step. Most are now sourced; the ones that are not are named as
such below, with the value actually assumed.

Claim tags follow [PROBLEMS.md](PROBLEMS.md).

---

## 1. Lake Mead → shortage tier → Arizona reduction `VERIFIED`

**This is the one the plan called "the app's most rigorously grounded" lever, and it is now
fully sourced from primary text.** Transcribed verbatim from Table 1 of the Lower Basin
Operating Agreement (LBOps), *Drought Contingency Plan Agreements — Final Review Draft,
2018-10-05*, Bureau of Reclamation, §C "Combined DCP Contributions and 2007 Interim Guidelines
Shortages", retrieved from
<https://www.usbr.gov/ColoradoRiverBasin/documents/dcp/DCP_Agreements_Final_Review_Draft.pdf>.

Volumes are **thousand acre-feet per year**. The Arizona "Combined" column is the one the model
needs — it is the 2007 Interim Guidelines shortage *plus* the DCP contribution.

| Projected Jan 1 Lake Mead elevation (ft msl) | AZ 2007 IG shortage | AZ DCP contribution | **AZ combined** |
|---|---|---|---|
| above 1,090 | 0 | 0 | **0** |
| at or below 1,090 and above 1,075 | 0 | 192 | **192** |
| at or below 1,075 and at or above 1,050 | 320 | 192 | **512** |
| below 1,050 and above 1,045 | 400 | 192 | **592** |
| at or below 1,045 and above 1,040 | 400 | 240 | **640** |
| at or below 1,040 and above 1,035 | 400 | 240 | **640** |
| at or below 1,035 and above 1,030 | 400 | 240 | **640** |
| at or below 1,030 and at or above 1,025 | 400 | 240 | **640** |
| below 1,025 | 480 | 240 | **720** |

Notes for implementation:

- The bands collapse to five distinct Arizona values: **0 / 192 / 512 / 592 / 640 / 720 kAF**.
  Encode as a descending-threshold step function on elevation, evaluated on the *annual mean*
  Mead elevation the slider implies (the agreement keys on the projected January 1 elevation;
  using the scenario's sustained elevation is the honest analogue and should be documented as
  such).
- The "above 1,090 → 0" row is not in Table 1; it is the complement of the table's coverage and
  is safe to assert — no shortage or contribution is required above 1,090.
- The full range of the step function (0 → 720 kAF) is only reachable if the Mead slider spans
  roughly **995 – 1,220 ft**. Today's `computed_stats.json` p5/p95 range is **1,058 – 1,195 ft**,
  which cannot reach the 1,045/1,025 tiers at all. **The Mead slider must be re-ranged as part of
  D5 or the lever stays half-dead.**

### Two parameters remain assumed on this path `UNTESTED`

The AZ reduction is state-wide; only part of it lands in the eight-county study region, and only
part of *that* is replaced by pumping rather than by fallowing or conservation.

| Parameter | Assumed | Band | Why |
|---|---|---|---|
| `region_share_of_az_reduction` | 0.60 | 0.4 – 0.8 | CAP serves Maricopa, Pinal and Pima. Pinal + Pima are in-region; Maricopa is not. The Tier-1 cut fell almost entirely on the CAP agricultural pool, which is predominantly Pinal. **Not sourced — should be replaced with CAP delivery data by county.** |
| `groundwater_substitution_fraction` | 0.50 | 0.3 – 0.7 | Share of lost CAP water replaced by groundwater pumping rather than fallowing. Pinal's DCP mitigation explicitly funded new wells. **Not sourced.** |

Both scale the lever linearly. State them in the UI with the band.

---

## 2. Groundwater storage balance `UNTESTED` — the weak link

> **Superseded outright, 2026-09-10 — `A_eff` is now MEASURED, and it is neither limit.**
> [PHASE3_PLAN.md §12a](PHASE3_PLAN.md) ran the spreading-cone test §12 called for and measured
> the storage coefficient at each horizon. The cone does spread (809,419 → 1,627,214 acres from
> 1 to 12 months) but it **plateaus at 1.23–1.93 M acres — two to three times the *irrigated*
> area — and never approaches the 12.5 M alluvial acres assumed below.** The long-run figure is
> 7.7× past the end of the evidence at any horizon out to 24 months, so it is **not** the
> "correct long-horizon anchor" the note below claims. What ships is the value measured at the
> scenario's own duration: **244,082 AF/ft at 12 months** (t = +1.78), band 184,023–289,336,
> `status: MEASURED`. The arithmetic below is retained only as the derivation of a bound that
> turned out not to bind.

> **Superseded in part, 2026-09-08.** `A_eff` below has been measured against the panel
> — see [PHASE3_PLAN.md §12](PHASE3_PLAN.md). A climate- and trend-controlled
> distributed-lag regression gives `S_y · A = 110,891 AF/ft` (t = +3.16, n = 246),
> against the 1.87 M AF/ft assumed here: **16.9× apart**. The resolution is that the
> two are the short-run and long-run limits of a spreading drawdown cone, not rival
> estimates — the measured `A_eff` (739,276 acres) is ~1.09× the region's *irrigated*
> area, because `depth_to_water_anomaly_ft` is an index over monitored wells in
> agricultural basins, not a regional water table. The reasoning below is left intact
> because it is still the correct long-horizon anchor; what changed is that the plan
> now knows which limit it is quoting.

```
ΔDepth (ft) = ΔPumping (AF/yr) × t (yr) / (S_y × A_eff)
1 MGD = 1,120 AF/yr        (1 Mgal = 3.06889 acre-ft × 365 d)
```

- **`S_y` = 0.15**, band 0.10 – 0.20. Southern Arizona basin-fill alluvium. USGS gravity
  estimates in Tucson Basin / Avra Valley give an average of **0.27** for wells away from major
  streams and **0.16 – 0.21** in comparable alluvial aquifers (Pool & Anderson, *Ground-Water
  Storage Change and Land Subsidence in Tucson Basin and Avra Valley, Southeastern Arizona,
  1998–2002*, USGS SIR 2007-5275, <https://pubs.usgs.gov/publication/sir20075275>). ADWR's
  Tucson AMA MODFLOW models use 0.10 – 0.20 for basin fill. **The exact ADWR value was not
  retrieved — azwater.gov returns HTTP 403 to WebFetch and curl is sandboxed here.** 0.15 is the
  midpoint of the ADWR modelling range and is stated as an assumption, not a citation.
- **`A_eff`** — the area the drawdown spreads over. **This is the least defensible number in the
  whole plan and should be treated as the dominant uncertainty.**
  - Region area, computed in-repo from the dissolved eight-county TIGER boundary in EPSG:5070:
    **112,423 km² = 43,406 mi² = 27,779,840 acres** `MEASURED`.
  - Alluvial basins are only part of that; the rest is mountain block. Assumed alluvial fraction
    **0.45** → `A_eff ≈ 19,530 mi² ≈ 12.5 M acres`, giving `S_y·A ≈ 1.87 M AF per foot`.
  - Sanity check: a sustained ±50% swing on 2,600 MGD irrigation ≈ ±1.46 MAF/yr → **±0.78 ft/yr**.
    (Measured short-run equivalent, PHASE3_PLAN.md §12: **±13.2 ft/yr**. The gap is the
    16.9× above, and is why Layer 2 cannot ship without the mean-reverting integration.
    §12a's measurement at a 12-month horizon lands at **±6.0 ft/yr**, between the two.)
    Against the `depth_to_water_anomaly_ft` output range (−8.87 … +10.36, i.e. 19.2 ft = 100
    score points) that is **~4 score points at 12 months, ~12 at 36 months.** So the acceptance
    criterion in §7 (≥5 points at 12 months) is **marginal for irrigation and will be missed
    outright by the weaker levers** — see §5 below. Do not tune `A_eff` to make the test pass;
    that would defeat the entire premise of the plan.

### Two dead ends, recorded so they are not retried

1. **Deriving `A_eff` from pumped HUC12s does not work** `MEASURED`. The intent was to take only
   the HUC12 watersheds carrying nonzero withdrawal. All **1,271** in-region HUC12s have nonzero
   irrigation in `IR_HUC12_Tot_WD_monthly_2000_2020.csv` (86,721 of 87,020 nationally), so the
   filter returns the whole region and adds nothing.
2. **Deriving effective `S_y` from GRACE ÷ well depth does not work** `MEASURED`. The idea was
   that `Δstorage(cm) = S_y × Δ(water table, cm)`, making the regression slope of
   `grace_groundwater_anomaly` on `−depth_to_water_anomaly_ft × 30.48` an in-repo measurement of
   `S_y` over the GRACE footprint. Over the 252-month overlap the correlation is **r = −0.752 with
   the wrong sign** — GRACE falls across the record while monitored well levels *rise*, because
   the well network sits in CAP-recharge-affected basins while GRACE integrates total storage
   over a much larger footprint. The two are not measuring the same thing. This is consistent
   with the GRACE null already in PROBLEMS.md.

---

## 3. Population → public supply → groundwater `MEASURED` (in-repo derivation)

No external citation needed — the per-capita groundwater draw is derivable from the project's own
data and is self-consistent by construction:

```
GPCD_groundwater = public_supply_groundwater_mgd × 1e6 / population
                 ≈ 804 × 1e6 / 6.4e6  ≈  126 gallons per capita per day
```

Chain: `Δpopulation × 126 gpcd → ΔMGD → ×1,120 AF/yr per MGD → the storage balance in §2`.
Compute it in the parameter script from the reference-year values rather than hard-coding 126.

---

## 4. Land-cover arithmetic for NDVI

### 4a. Irrigated area fraction `VERIFIED`

2017 Census of Agriculture, Table 10 "Irrigation: 2017 and 2012", row *Irrigated land … acres,
2017*, USDA NASS Volume 1 Chapter 2, Arizona county-level data
(<https://www.nass.usda.gov/Publications/AgCensus/2017/Full_Report/Volume_1,_Chapter_2_County_Level/Arizona/st04_2_0010_0010.pdf>):

| County | Irrigated acres, 2017 |
|---|---|
| Cochise | 86,008 |
| Graham | 46,682 |
| Greenlee | 5,136 |
| La Paz | 97,139 |
| Pima | 30,008 |
| Pinal | 232,224 |
| Santa Cruz | 2,551 |
| Yuma | 181,395 |
| **Eight-county total** | **681,143** |

Against the in-repo region area of 27,779,840 acres: **irrigated fraction = 2.452 %**.

### 4b. NDVI endpoints `UNTESTED` — `ndvi_impervious` now MEASURED

> **`ndvi_impervious` measured, 2026-09-11 — the assumption below was 5.5× too strong.**
> [PHASE3_PLAN.md §17](PHASE3_PLAN.md). OLS of MOD13A3 NDVI on NLCD impervious fraction, 287
> months, ~131,000 cells each, on the MODIS grid inside the eight-county cutline. Slope
> **−0.0250** (IQR −0.0457 … −0.0092); endpoint at 100% impervious **0.2046** (IQR 0.1982 …
> 0.2135), against the **0.08 (band 0.05–0.12)** assumed below.
>
> **The adopted value comes from a difference-in-differences**, not that cross-section: each MODIS
> cell differenced against its own past (2000–04 vs 2019–23, 130,833 cells), which removes the
> siting confound — cities are built on valley floors, not on a random sample of the region. Slope
> **−0.0356** (HC1 t = −14.7). Ships as `ndvi_impervious = 0.1811`, stated relative to
> `ndvi_natural` so the difference *is* the measured slope. urbanization → NDVI goes −3.73 →
> **−0.97**.
>
> **The assumption below was not absurd — it was describing the wrong half of the region.** Split by
> what each cell was before it was paved: dry desert **−0.0027** (indistinguishable from zero;
> desert NDVI is already about what pavement reads), cropland/riparian **−0.1792** (83% of the
> region's whole vegetation signal). A 45× difference. The 0.08 endpoint implies −0.1367, which is
> the cropland-conversion case. The shipped coefficient is the historical mix.
>
> `ndvi_irrigated_crop` is still assumed: its predictor (a cropland mask) is not in `data/raw/`.


The plan says both endpoints are "measurable in the project's own MOD13A3 pixels". They were not
measured — that needs a pass over `data/raw/modis_ndvi/` HDFs, which was out of budget. Assumed:

| Endpoint | Assumed | Band | Note |
|---|---|---|---|
| `ndvi_impervious` | 0.08 | 0.05 – 0.12 | dense urban surface |
| `ndvi_irrigated_crop` | 0.55 | 0.45 – 0.65 | irrigated cropland, growing season |
| `ndvi_natural` | **in-repo** | — | use the regional NDVI baseline, 0.2167 |

Arithmetic:

```
urbanization → NDVI :  ΔNDVI = (Δimpervious_points / 100) × (ndvi_impervious − ndvi_natural)
irrigation   → NDVI :  ΔNDVI = Δfraction_of_baseline_withdrawal × irrigated_fraction
                               × (ndvi_irrigated_crop − ndvi_natural)
```

**Both are physically tiny, and that is the correct answer, not a bug.** Urbanization at +2.0
points of impervious cover gives ΔNDVI ≈ −0.0027 ≈ **3.1 score points** against the NDVI output
range of 0.088. Irrigation at −50 % gives ΔNDVI ≈ −0.004 ≈ **4.5 score points**. A 1 %-of-area
land-cover change genuinely does not move an eight-county mean NDVI. The irrigation lever is
still worth building because it is *positive* on NDVI while *negative* on groundwater — the
tension the plan explicitly wants shown rather than hidden.

---

## 5. Consequence for the §7 acceptance criteria — read this before implementing

The plan's criterion is "every human slider moves at least one output by ≥ 5 score points across
its min→max range at a 12-month duration." With honestly-sourced coefficients:

| Lever | Best path | Est. 12-mo effect | Meets ≥5? |
|---|---|---|---|
| Irrigation | groundwater depth | ~4 pts estimated (±factor 2 on `A_eff`) — **measured: 60.1 pts, band 51–80**, at the coefficient measured for a 12-month horizon (PHASE3_PLAN.md §12a) | **yes, by 12×** |
| Lake Mead | groundwater depth via tiers | depends on re-ranging; 0 if the slider cannot reach 1,075 | only if re-ranged |
| Population | groundwater depth via GPCD | small | likely no |
| Public supply | groundwater depth | small | likely no |
| Urbanization | NDVI | ~3.1 pts | no |

`ESTIMATED`. **Implement the test at the stated threshold and let it fail where it fails.** The
right response to a lever that falls short is either a documented longer duration or an honest
statement that the physical effect is small — not a larger coefficient. The criterion as written
assumes every lever has a ≥5-point effect available; §4b shows at least one provably does not.
Suggest reporting per-lever pass/fail plus the sign-stability check, and treating the sign check
as the hard gate.

---

## 6. D5 slider reparameterization — design that was settled

Emit a `policy` block per slider in `computed_stats.json` alongside the existing raw
`min`/`max`/`default` (keep those; `models.js` and `slider_sensitivity.py` both read them):

```json
"irrigation_total_withdrawal_mgd": {
  "min": 256.24, "max": 4814.99, "default": 2720.56,
  "policy": {
    "mode": "scale",
    "baseline": 2715.9,
    "seasonal": [12 multiplicative factors, mean 1.0],
    "unit": "%", "min": -60, "max": 40, "default": 0, "step": 1
  }
}
```

Reconstruction in `models.js` / the Python mirror:

- `mode: "scale"`  → `raw(month, d) = baseline × (1 + d/100) × seasonal[month−1]`
- `mode: "offset"` → `raw(month, d) = baseline + d + seasonal[month−1]`

Definitions:

- `seasonal[m] = mean(month m) / mean(all)` for scale, `mean(month m) − mean(all)` for offset,
  over the full record of that variable.
- `baseline` = mean of the **deseasonalized** series over the **reference year 2020** — the last
  calendar year for which *all* human series exist (irrigation and public supply both stop at
  2020; population, impervious and Mead run to 2023). Using one common reference year keeps the
  default scenario from mixing epochs.
- Therefore `raw(m, 0)` is the climatological normal for month *m*, which is exactly what D5 asks
  for: a deseasonalized baseline with the seasonal shape supplied by the month selector.
- `sliderBaseline(key)` in `models.js` — used for every lag/roll blend — must become
  `baseline × seasonal[month]` (or `+`), not the flat p50. This makes the lag features seasonally
  correct, which is a real improvement independent of everything else.

Policy ranges settled:

| Slider | mode | unit | min | max | step |
|---|---|---|---|---|---|
| population | offset | people added | −500,000 | +3,000,000 | 50,000 |
| irrigation | scale | % of 2020 annual withdrawal | −60 | +40 | 1 |
| public supply | scale | % | −40 | +60 | 1 |
| impervious | offset | points of impervious cover | −0.4 | +2.0 | 0.05 |
| Lake Mead | offset | ft (display absolute elevation) | −95 | +130 | 1 |

Mead's 2020 deseasonalized baseline is ≈ **1,088.8 ft**, so that range spans ≈ 994 – 1,219 ft and
reaches every DCP tier in §1. Display the absolute elevation in the UI, since the tiers are
defined on absolute elevation.

### Climate sliders

Measured deseasonalized statistics `MEASURED` (2000– for nClimDiv):

| Slider | mode | mean | seasonal shape | residual p5 | p95 | 2020 deseasonalized level |
|---|---|---|---|---|---|---|
| `temperature_2m_c` | offset | 19.09 °C | −10.29 … +10.79 by month | −2.29 | +2.52 | 19.91 |
| `precipitation_mm_day` | scale | 0.7765 mm/d | ×0.239 (May) … ×2.483 (Jul) | −0.935 | +1.604 | 0.7749 |
| `nclimdiv_pdsi` | offset | −1.768 | −0.42 … +0.40 by month (small) | −2.64 | +3.64 | −0.753 |

Decision taken: **give the climate sliders the same `policy` machinery** so the month selector
actually drives their seasonality, but set their delta ranges from the deseasonalized residual
p5/p95 above rather than from a policy scenario, so climate response magnitudes stay comparable
to the numbers already reported in PHASE3_PLAN.md §1. This is a small extension beyond the letter
of D5 (whose table lists only the five human levers) and should be called out as such — without
it the app is half absolute and half anomaly, and the month dropdown does nothing to temperature.

---

## 7. Architecture decisions settled (not yet built)

- **Parameters live in one generated JSON**, `frontend/structural_params.json`, written by a new
  `scripts/phase3/structural_params.py`. That script derives the in-repo values (region area,
  GPCD, NDVI baseline, irrigated fraction) and merges them with the literature/policy constants
  declared inline with citation and status tag. Every coefficient carries `value`, `band`,
  `source`, `status`.
- **Layer 2 arithmetic is implemented twice** — `frontend/structural.js` for the browser and a
  mirror inside `scripts/phase3/slider_sensitivity.py` for the headless acceptance test — but
  both read the same JSON, so only ~40 lines of arithmetic are duplicated, never a number.
- **Guard the duplication**: make `frontend/structural.js` runnable under Node (v22.22.2 is
  installed; it supports `import … with { type: 'json' }`) with a `--dump` flag that prints
  structural contributions for a fixed scenario set, and have the Python side compare. Cheap, and
  it catches drift between the two implementations.
- **Layer 3 ships in the mean-reverting form only** (PHASE3_PLAN §4b), never the naive form, and
  not before Layer 2.
- **D4** (`TOP_INPUTS`): generate from `model/*_feature_importance.json` into a separate
  `frontend/top_inputs.json` rather than into `computed_stats.json`, so `generate_stats.py` stays
  the sole writer of the latter.

## 8. Environment facts worth keeping

- `azwater.gov`, `pubs.usgs.gov` and `congress.gov` all return **HTTP 403** to WebFetch; `curl`
  has no network in this sandbox. `usbr.gov` and `nass.usda.gov` work. PDFs fetched by WebFetch
  are saved to disk and are readable with `pdftotext -layout`, which is how the DCP table in §1
  and the NASS table in §4a were recovered — WebFetch's own summarizer could not read either.
- `node` v22.22.2, `onnxruntime` 1.24.4, `geopandas`, `pandas`, `numpy` all available.
- The eight-county boundary loads via `scripts/phase1/region.py::load_county_boundary()`;
  reproject to EPSG:5070 for areas.
