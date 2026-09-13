# Phase 2 Report — Model Training Results

## Overview

Phase 2 builds six independent regression models predicting environmental outcomes for the
eight-county southern Arizona study area. Each model uses human pressures (population,
irrigation, public supply, urbanization) and environmental conditions (temperature,
precipitation, drought, lake levels) as input features to predict a single environmental
response.

**Modeling window:** **per model** — see below. There is no longer a global window, and
no longer a global 2002-10 floor: two models now run on pre-satellite data.
**Panel:** 14 monthly CSVs merged into a **696-row × 31-column** panel spanning
**1968-01 → 2025-12**, plus a 58-row annual panel. Each model masks it down to its own window.

> **Five of the six models now beat both of their baselines.** That is a different report from
> the one this file used to be. Four models were repaired — every one of them **at the target**,
> not at the estimator — and two were extended past the satellite era onto NOAA nClimDiv.
> **[PROBLEMS.md](PROBLEMS.md) is the register** of what was wrong, how it was found, and what
> is measured versus estimated. This file reports what the models currently score.

---

## How to read these numbers

Four of these models are **residual-over-lag1**: they predict the *change* from last month
and reconstruct the level by adding last month's value back. That makes the headline R²
misleading, because most of it comes from the lag1 anchor — which the model does not
predict and gets for free. A model with *zero* skill still posts a high level R².

So every model is judged against two baselines, and must clear **both**:

| Metric | Question it answers | Fails when |
|--------|--------------------|-----------|
| **R² (level)** | Does it beat predicting the test-fold mean (a flat line)? | R² ≤ 0 |
| **Skill** = R² − persistence R² | Does it beat copying last month's value? | Skill ≤ 0 |

Neither alone is sufficient.

**R² (target)** is the R² on the thing the model was actually trained on — the residual, for
a residual model. It is the single most informative column: it is the part the model is
responsible for.

**⚠ Skill is a verdict, not a magnitude.** It is only comparable between models that share a
target *and* a window, because the persistence baseline moves underneath it. Surface water and
wildfire both got measurably **better** in their most recent change while their skill numbers went
**down**, purely because persistence rose on the longer window. Where a before/after comparison
matters below, it is stated as ΔR² **on identical test rows**. See
[the comparability trap](#the-comparability-trap-skill-scores-that-move-the-wrong-way).

All scores are **nested**: feature selection, model choice, and hyperparameter search all
happen inside each training fold. See [scripts/phase2/metrics.py](scripts/phase2/metrics.py).

---

## Model Results Summary

| Model | Target | Window | Rows | Feat. | Verdict | Skill | R² (level) | Persistence | R² (target) |
|-------|--------|--------|------|-------|---------|-------|-----------|-------------|-------------|
| **Surface Water** | Discharge log anomaly | **1980-02 → 2025-12** | **551** | 26 | **OK** | **+0.6833** | **0.7519** | 0.0686 | **0.7267** |
| **Wildfire** | Wildfire risk index | **1984-02 → 2023-12** | **479** | 26 | **OK** | **+0.4125** | 0.3227 | −0.0898 | 0.3227 |
| **NDVI** | Vegetation health | 2002-10 → 2023-12 | 255 | 46 | **OK** | **+0.2588** | 0.7862 | 0.5274 | **0.5453** |
| **Wildlife** | Bird abundance anomaly | **1968 → 2024** (annual) | **56** | 12 | **OK** | **+0.2167** | 0.2247 (LOO) | 0.0080 | — |
| **GRACE** | Groundwater anomaly | 2002-10 → 2023-12 | 204 | **4** | **OK** | **+0.2015** | 0.5664 | 0.3649 | **0.2845** |
| Groundwater | Well depth anomaly (ft) | 2002-10 → 2020-12 | 219 | 16 | **NO SKILL** | −0.0123 | 0.3872 | 0.3995 | −0.1075 |

Sorted by skill. **Retrained 2026-09-12** (PHASE3_PLAN.md §27); the table it replaces is in git
history at `e8d7df2`. Three things moved, for three different reasons:

1. **GRACE is a different model.** A standardised ridge on four physical inputs — GLDAS
   land-surface storage change, rain, last month's rain, temperature anomaly — replaced the
   45-feature XGBoost after a pre-declared model-class experiment (§26). Skill −0.0349 → **+0.2015**.
   The deployed model reproduces the experiment to the fourth decimal.
2. **The drought index was regenerated.** `water_stress.py` had weighted Maricopa's DSCI by
   Graham's area and Gila's by La Paz's (PROBLEMS.md P8); the corrected series correlates 0.9992
   with the old one. NDVI (six DSCI features) and groundwater (one) retrained on it. Groundwater
   went from +0.0086 to −0.0123 — the same "approximately zero" it has been at through four
   re-runs, and still no skill.
3. **Wildfire moved +0.04 on a byte-identical matrix.** Its inputs carry no DSCI and no GLDAS;
   the feature matrix was diffed against the committed one and is identical, and two trainings
   in one process agree with each other exactly. The committed artifact came from a different
   library environment (`requirements.txt` is unpinned), and GridSearch picked a different
   winner. Fold 0 went 0.215 → 0.423; the other four are within 0.02. `model_comparison.json`
   now records the xgboost / scikit-learn / numpy versions that produced every number.

Groundwater clears the level baseline arithmetically and loses to persistence by 0.01 — it is
listed as no skill because ±0.01 is not a result. It is, however, **no longer actively harmful**,
which is where it started.

---

## Model Details

### Surface Water (Discharge) — skill +0.6830 ✅

- **Formulation:** residual-over-lag1, **Ridge** (α = 1.0). No target transform — the log now
  lives *inside* the target.
- **Window 1980-02 → 2025-12 · 551 rows · 26 features (nClimDiv only)**
- **R² (level) 0.7516 · persistence 0.0686 · R² (residual) 0.7264**
- **Five folds: 0.774 / 0.703 / 0.755 / 0.756 / 0.771** — the tightest spread in the project.

**The best model in the project, and it was fixed at the target.** `discharge_cfs_mean` was a
mean of raw cfs across ~130 gages whose flows span four orders of magnitude, so it was not a
regional average — it was whichever gage was biggest. One dam-regulated gage (**09525503**)
supplied **45.3%** of it, which means the old model was being asked to predict a Bureau of
Reclamation release schedule from rainfall. Phase 1 now centers each gage on its own long-term
mean *in log space* before averaging, so every gage votes equally and a doubling on a small wash
counts the same as a doubling on the Gila.

Then the window was extended to the NWIS floor: **1980–2025, 551 rows** (was 219). Reaching past
2000 collapses the feature set to nClimDiv alone (63 → 26 features), losing `usdm_dsci` — the
panel's single highest-signal input. That trade was **measured, not assumed**, on identical test
rows (the common 2002-10 → 2020-12 period, same folds, same baseline):

| | rows | features | R² | skill |
|---|---|---|---|---|
| SHORT (2002–2020, full feature set) | 219 | 63 | +0.6777 | +0.6344 |
| **LONG (1980–2025, nClimDiv only)** | **551** | **26** | **+0.7955** | **+0.7522** |

**ΔR² = +0.1178.** The 332 extra months buy more than the lost features cost.

The winner is **linear** and train R² (0.824) sits only just above CV R² (0.752) — there is
essentially no overfitting. The top features are all drought and rain:
`nclimdiv_pdsi_roll6`, `nclimdiv_precipitation_mm_day_roll3`, `nclimdiv_precipitation_mm_day`,
`nclimdiv_pdsi_lag1`. Rain predicts flow; the old target could not see that because the dam was
drowning it out.

### Wildfire (Monthly) — skill +0.3716, R² = +0.2819 ✅

- **Formulation:** direct (fire is not autoregressive), **XGBoost** (squared-error)
- **Window 1984-02 → 2023-12 · 479 rows · 26 features (nClimDiv only)**
- **R² (level) 0.2819 · persistence −0.0898 · five folds 0.22 / 0.06 / 0.28 / 0.38 / 0.49**
- **68.1% of months are zero** (326 of 479) — MTBS maps only large fires.

This model works, and it took three interventions to establish that it was always a *target*
problem, never an estimator problem:

1. **The real fix — a true ignition date.** The old target's month came from `DATE_CUR`, a
   database-maintenance timestamp, not an ignition date; 57% of AZ fires landed on five ETL
   batch-load dates and the implied fire season peaked in February. Rebuilding on **MTBS `ig_date`**
   moved R² from −0.0338 (worse than a flat line) to **+0.3122** on the 2002–2023 window, loss
   function untouched. Every driver correlation flipped to its physical sign (temperature
   −0.27 → +0.46, precipitation +0.09 → −0.11).

2. **The 1984 extension paid.** With a real time axis, the window was pushed to the MTBS floor
   (1984), which — like surface water — collapses the feature set to nClimDiv alone (51 → 26
   features, dropping `usdm_dsci`/MERRA-2/NDVI/GRACE). Measured on **identical test rows**, varying
   only the training data: **R² +0.3618 → +0.4100 (ΔR² +0.0483), winning all five folds.** The 224
   extra months buy more than the lost features cost. The headline CV R² *looks* worse (0.3122 →
   0.2819) only because the pre-2000 folds are harder (68% zero months); do not compare skill across
   the window change — persistence moved from −0.25 to −0.09 underneath it.

3. **The Tweedie loss was tried, and lost.** At ~68% zeros the target is genuinely zero-inflated, so
   a `reg:tweedie` XGBoost was added as a fourth competition candidate. It scored **0.2895 vs
   squared-error XGBoost's 0.3381** out-of-fold and did not win. The zero-inflation was real but was
   never the binding constraint; reshaping the loss did not convert into R². It remains in the
   competition as a self-selecting option.

Top features are `month_cos`, `nclimdiv_temperature_c_lag1`, `nclimdiv_temperature_c_roll6`,
`nclimdiv_precipitation_mm_day_roll3` — seasonality and weather, which is what drives fire. The
old top two were `population` and `population_roll12`: a trend line.
**Hard limit:** ignition is stochastic; R² ~0.3 on monthly large-fire activity is a good result and
is not expected to climb much further.

### NDVI (Vegetation Health) — skill +0.2653 ✅

- **Formulation:** residual-over-lag1, XGBoost
- **Window 2002-10 → 2023-12 · 255 rows · 46 features**
- **R² (level) 0.7928 · persistence 0.5274 · R² (residual) 0.5584**
- **Five folds (residual): 0.49 / 0.55 / 0.78 / 0.49 / 0.47**

It explains **56%** of the variance in month-to-month NDVI *change*, which is real predictive
content, not an artifact of the anchor. Vegetation responds to temperature and precipitation on a
lag that the model captures well: the top feature is `precip_x_temperature` (0.337), then
`temperature_2m_c_lag1` (0.164).

**It is also the only target that was audited and came back clean the first time.** The target is
`mean(valid MODIS pixels)` — structurally the same "mean over whoever reported" shape that broke
three other models — but empirically null here: **369,305 valid pixels/month at CV 0.1%**, and
`corr(pixel turnover, |month-to-month NDVI jump|) = +0.000`, against ~+0.75 for the convicted
targets. MOD13A3 is a monthly composite that removes cloud churn upstream, so the roster is
effectively fixed.

Dropping the irrigation and public-supply features to reach 2023 cost almost nothing
(−0.0075 skill, measured); the 36 extra rows returned +0.056. That asymmetry is exactly what
[the variance decomposition](#the-binding-constraint-most-inputs-carry-no-monthly-information)
predicted: those inputs were a month-of-year clock that `month_sin`/`month_cos` already carry.

**Constraint:** MODIS Terra launched 2000-02. There is no earlier NDVI, so unlike surface water
and wildfire this model cannot be extended backward — only forward.

### Wildlife (Bird Abundance) — skill +0.2046 ✅

- **Formulation:** direct, **Ridge** (α = 100), nested LOO
- **Window 1968 → 2024 · 56 rows · 12 features**
- **LOO R² +0.2125 · persistence 0.0080 · Spearman ρ 0.5068 (p = 6.7e-05; permutation
  p = 0.0001 over 20,000 shuffles)**

**Fixed at the target, then made significant by the extension.** `total_abundance` was a *sum over
whichever routes were surveyed that year*, and the regional survey network runs 15–26 routes/year
(statewide, ~15/year in the 1970s–80s versus ~45 from 1990). Over the full record the old target
correlated **+0.942 with `route_count`**: the model was being asked to predict how many people went
birdwatching, from temperature and drought.

`abundance_anomaly` centers each route on its own long-term mean log abundance before averaging.
Effort correlation drops to **−0.118**, and a `_effort_confound_check()` guard now refuses to write
a target exceeding ±0.6.

**Sample size was never the binding constraint.** At n=20 and 22 features *unchanged*, the target
fix alone moved LOO R² from −0.2108 to **+0.2661** and ρ from 0.2159 (p = 0.36) to 0.5278
(p = 0.017). What the nClimDiv extension to n=56 bought was **significance**, not fit: LOO R² is
slightly *lower* (0.2125) because reaching 1968 forces dropping NDVI, GRACE and the USDM — losing
NDVI as a food-availability proxy, genuinely the mechanism you would most want for birds — but
ρ = 0.51 at p = 0.0001 is a real result where ρ = 0.53 at p = 0.017 was a suggestive one.

Behind the autoregressive lag, the top driver is **prior-year precipitation**
(`nclimdiv_precipitation_mm_day_annual_sum_lag1`), then annual and summer PDSI. Desert bird
productivity tracking *last* year's rain, through vegetation and insect abundance, is exactly the
known ecology — and not something a model reading a survey roster could ever have found.

**The NDVI trade-off was tested and the rows win decisively `MEASURED`.** Reaching 1968 costs NDVI
(MODIS Terra launched 2000-02, and one NDVI column deletes every pre-2000 row), so the model must
choose n=56 without it or n=23 with it. Scored on the identical 23 common years, nested LOO,
judged on a 20,000-shuffle permutation test:

| variant | n | feat | LOO R² | ρ | perm p |
|---|---|---|---|---|---|
| **A — shipped, 1968–2024** | 56 | 12 | **+0.0956** | **+0.409** | 0.054 |
| B — short 2000–2023, no NDVI | 23 | 12 | −0.1702 | +0.110 | 0.618 |
| C — short 2000–2023, **+ NDVI** | 23 | 15 | −0.2107 | +0.150 | 0.495 |

NDVI's own contribution (C vs B, same rows) is **Δρ +0.041, p = 0.495** — nothing. The decisive
result is B: cutting to 23 years *without changing a feature* drops R² below a flat line and puts ρ
under its own permutation null. **At n=23 there is no working model for NDVI to improve**, so the
mechanism was swamped rather than refuted — and the configuration that could test it properly, long
record *and* NDVI, cannot exist. Keep the 1968 window. The real
signal survives the correction: per-route abundance declines **~23% across the record (p < 0.0001)**.

### Groundwater (Well Depth) — skill −0.0123 ⚠ *(+0.0086 before the 2026-09-12 retrain)*

- **Formulation:** residual-over-lag1, in-fold competition of 5 candidates, winner **XGBoost**
- **Window 2002-10 → 2020-12 · 219 rows · 16 features** (47 dropped in-fold)
- **R² (level) 0.3872 · persistence 0.3995 · R² (residual) −0.1075** — after the drought-index
  regeneration (P8); the numbers in the text below are from the previous run and sit inside the
  same fold noise (+0.0107, +0.0091, +0.0058, +0.0086, −0.0123 across five re-runs)

**This model used to be worse than doing nothing** (residual R² −0.3179) and is now at
approximately zero. That is a **correctness fix, not a performance fix**, and it was always going
to be.

The cause was the target. `depth_to_water_ft_mean` was a mean over whatever wells reported that
month, and wells sit at wildly different depths (40 ft to 400 ft). Measured on the daily pull,
roster turnover correlated **+0.746** with the month-to-month jump in the mean, and the raw target's
standard deviation (**27.8 ft**) was nearly **five times** the anomaly's (5.9 ft). Roughly 80% of
what the column was "measuring" was its own roster. Phase 1 now emits `depth_to_water_anomaly_ft`
(each well centered on its own mean, 24-month minimum record) plus an `n_wells` diagnostic — which
is the artifact, not a predictor, and is excluded from `X`.

**It is also the only model that could see either feature fix**, and the results mirror each other.
`public_supply_groundwater_mgd` had the same national-scope bug as irrigation — 34,817 MGD, **5.4×
Arizona's entire water use across every sector**, summed over 87,020 watersheds instead of 1,133.
Corrected to **761 MGD** it moved groundwater skill +0.0058 → **+0.0086** and residual R² −0.0474 →
**−0.0364**, again within fold noise (per-fold Δ `[+0.064, −0.020, −0.001, +0.002, +0.010]`, mean
+0.011 against a spread of 0.064).

**The two fixes had opposite effects on selection, and both are informative.** Corrected irrigation
was *promoted* from never-selected to #2 by importance. Corrected public supply was **dropped
entirely** (it had been carrying 0.0691) — which is the right call, because a national public-supply
sum is a near-pure seasonal template and the model already has `month_sin`/`month_cos`. The corrupt
version was being selected; the correct one is not. Two features became more honest and the score
did not move either time.

**And the irrigation result specifically is worth recording.**
`irrigation_total_withdrawal_mgd` was **~99.9% nodata sentinel** summed over **87,020 watersheds
nationwide** — 5.43e7 MGD, four orders of magnitude too large, and near-constant. Corrected to
**2,560 MGD** it became physical (June peak, January trough) and, for the first time, in-fold
selection **started choosing it**: from never-selected to the **#2 feature by importance (0.0963)**.
The out-of-fold score did not follow — skill +0.0091 → +0.0058, well inside fold noise (per-fold Δ
`[−0.128, +0.043, +0.008, −0.002, −0.013]`, mean −0.018 against a fold-Δ sd of 0.065). **A feature
the model actively wants that still does not convert is about as direct a measurement of the
binding constraint as this project can make.** See PROBLEMS.md P7.

**Having landed at zero, it stops here.** What actually drives month-to-month aquifer change is
pumping, and the observable part of it is small
([the binding constraint](#the-binding-constraint-most-inputs-carry-no-monthly-information)).
Do not tune this model. ADWR pumpage — long named in this project as the fix — is **annual and
AMA-only**, and annual data cannot move a monthly residual model.

### GRACE (Groundwater Anomaly) — skill +0.2015 ✅ *(was −0.0349 as an XGBoost)*

- **Window 2002-10 → 2023-12 · 204 rows · 4 features** — a standardised ridge, alpha 10 by inner
  TimeSeriesSplit(3), on `gldas_tws_proxy_delta`, `precipitation_mm_day`,
  `precipitation_mm_day_lag1`, `temperature_2m_c_anomaly`
- **R² (level) 0.5664 · persistence 0.3649 · R² (residual) 0.2845**
- Folds: level R² 0.09 / 0.70 / 0.66 / 0.82 / 0.57 — still widest in fold 0, no longer negative.
- Standardised coefficients +0.0076 (GLDAS change), +0.0011, +0.0007 (rain, this and last month),
  −0.0040 (temperature anomaly): every sign physical, GLDAS carrying 7× rain's weight.

> **Replaced 2026-09-12 — PHASE3_PLAN.md §25–§27.** Everything below this note was true of the
> XGBoost and remains the record of why it failed. What changed is not the data and not a feature:
> GLDAS-2.1's land-surface storage change (soil + snow + canopy) correlates **+0.62** with GRACE's
> monthly change at a slope of **+0.8 m/m**, and a model with four degrees of freedom captures
> that where 60 draws of boosted trees on 200 rows could not — a one-coefficient OLS on that
> column alone scores +0.244 out of fold against the XGBoost's +0.021 on identical blocks. The
> pumping residual is still unobserved; the +0.28 residual R² is a measurement of how much of the
> month-to-month change is *not* pumping. The estimator was the floor.

*The original XGBoost account follows.*

- **Window 2002-10 → 2023-12 · 204 rows · 45 features** *(XGBoost, superseded)*
- **R² (level) 0.3300 · persistence 0.3649 · R² (residual) 0.0302**
- Folds are wild: level R² −0.87 / 0.61 / 0.59 / 0.75 / 0.58 (σ = 0.60).

**The only remaining failure, and the target is not the problem.** GRACE went through the same two
audit rules that convicted four other targets and passed both. Rule 1 (does the averaging
denominator move?): the anomaly is a spatial mean over the ~24 GRACE cells in the bbox; reprocessing
the nc4 gives ~20 valid cells/month, **constant within each mission** — the count changes once in
226 months, at the GRACE→GRACE-FO handover. Rule 2 (physics): a strong secular decline
(trend r = **−0.84**, p = 2e-61 — the Arizona depletion signal), co-moving correctly with drought
(−0.22) and Lake Mead (+0.75). **There is no GRACE target fix to find.**

Three separate corrections landed on this model and the score fell each time, because each previous
number was flattering:

1. **Tuning optimism.** The original 0.5200 was the best of 60 hyperparameter draws scored on the
   folds that were then reported. Nested CV → 0.4047.
2. **Fabricated targets.** **33 months inside the window were never measured** — `grace_available == 0`
   across **23 separate blocks** (2003-06, 2011-01, 2011-06, 2011-12, 2012-05, …), GRACE's routine
   missing-month dropouts, not just the mission changeover. Interpolated values are *smooth*,
   therefore easy to predict, and were propping up the score. Requiring both the target and the lag1
   anchor to be real measurements removes 51 rows and drops residual R² 0.1180 → **0.0302**.
3. **The window extension didn't help.** 36 real rows gained, 51 fill rows removed. Net −15.

**That is the model's actual skill; it was always ~zero.** GRACE anomaly is a *storage integral* —
it responds to cumulative water balance, and the monthly pumping that drives that balance is
unobserved. It is also at its instrument floor (2002-04), so it gains nothing from any window
extension. Two honest caveats on the target itself: at ~300 km resolution the "regional" mean is
**leakage-smeared**, and it runs **+0.71 with the well-depth anomaly where physics wants a negative
sign** — most likely a GRACE-vs-local-wells scale mismatch (managed recharge has raised water tables
in the monitored AMA wells while regional storage falls), not a defect.

**The irrigation trade was re-tested, and the answer is to leave the window alone `MEASURED`.**
GRACE gave up irrigation and the other 2020-capped features to reach 2023 — a decision made when
irrigation was 99.9% nodata sentinel. With the feature corrected it has a real, correctly-signed,
significant partial correlation with GRACE's monthly change (**−0.260**, p < 0.001, calendar
removed), so the trade was measured properly: identical test rows, the common 168 months, five
folds, training strictly before each test fold, varying only the feature set.

| variant | window | features | R² level | R² target | Δ target | folds won |
|---|---|---|---|---|---|---|
| **A — shipped** | 2002–2023 | 45 | +0.4479 | +0.0528 | — | — |
| B — all 2020-capped | 2002–2020 | 57 | +0.4646 | +0.0787 | +0.0259 (t=1.13) | 3/5 |
| C — irrigation only | 2002–2020 | 49 | +0.4541 | +0.0626 | +0.0098 (t=0.57) | 2/5 |

Persistence on those rows is +0.4369, so the shipped model's skill on this subset is only +0.011.
Neither variant is significant and irrigation alone does not win a majority of folds; the largest
effect is **0.09×** the fold-to-fold spread (sd 0.296). **And the design favoured the challengers** —
variant A's 36 extra months fall at the end of the record, so they never enter training for these
folds and A is denied its row advantage entirely. A null under those conditions is a strong null.

**A significant correlation is not a usable feature.** Two independent measurements now say the
same thing: GRACE cannot use corrected irrigation even when handed it, and groundwater *selects* it
(#2 by importance) and gains nothing.

**Stop tuning it.** Two live leads remain, both Phase 1 data-acquisition
(see [next steps](#recommended-next-steps)), both probes rather than rescues.

---

## The comparability trap: skill scores that move the wrong way

Two models in this report got **better** in their most recent change while their **skill score went
down**. This is not a paradox; it is what skill does when the window moves.

| Model | Change | Skill before → after | Persistence before → after | The honest number |
|---|---|---|---|---|
| Surface water | extend to 1980–2025 | +0.7075 → **+0.6830** | −0.0429 → **+0.0686** | ΔR² **+0.1178** on fixed test rows |
| Wildfire | extend to 1984–2023 | +0.5615 → **+0.3716** | −0.25 → **−0.09** | ΔR² **+0.0483** on fixed test rows |

Skill is R² minus persistence R². Both extensions reach into older, quieter data where copying last
month works *better*, so the baseline rises and the subtraction eats the gain. The same trap fires
across a **target** change: surface water's jump from +0.1469 to +0.7075 when the per-gage anomaly
shipped is real, but roughly half of it is persistence getting *worse* on a spikier target rather
than the model getting better.

**Rule: quote the R² that stands alone, and the fixed-test-row ΔR² for before/after. Treat skill as
a pass/fail verdict.**

---

## Per-model windows

Every model used to run on one global window, 2002-10 → 2020-12. The end date was inherited
from irrigation and public supply, which stop at 2020-12; the start was inherited from GRACE.
**Neither bound was real for most models**, and the cap was costing four of them a fifth of the
record.

The mechanism was subtle. All six models shared one feature list, and each `X` carried
irrigation, public supply, well depth *and* discharge as cross-features. The
`mask = x.notna().all(axis=1)` in [features.py](scripts/phase2/features.py) then dropped every
row where any of them was NaN — so a single dead column silently truncated models whose targets ran
well past it. Models are now declared in `_MONTHLY_MODEL_SPECS` with their own target, window, and
feature set, and two of them (`long_record: True`) reach past the satellite era onto nClimDiv alone.

**The 2002-10 floor still binds the models that use GRACE as an input**, because GRACE is
zero-filled before 2002-04 and pulling that in would feed fabricated values to every consumer.
Pushing the start to 2000 was tested and is **worse** (NDVI skill +0.2453 vs +0.2653) — it forces
dropping the GRACE features, which cost more than the 23 extra rows return. **Models carrying no
GRACE features are not bound by it at all**, which is what let surface water and wildfire reach
1980 and 1984.

| Model | Window | Rows | Bound by |
|-------|--------|------|----------|
| Surface water | 1980-02 → 2025-12 | 551 | NWIS gage record |
| Wildfire | 1984-02 → 2023-12 | 479 | MTBS floor (1984) |
| Wildlife | 1968 → 2024 | 56 | BBS floor (1968) |
| NDVI | 2002-10 → 2023-12 | 255 | MODIS Terra (2000-02) + the GRACE floor |
| Groundwater | 2002-10 → 2020-12 | 219 | **a hardcoded `YEAR_END = 2020`, not the data** |
| GRACE | 2002-10 → 2023-12 | 204 | GRACE mission start (2002-04) |

**Groundwater is the last model still bound by a constant rather than by the world.**
[`groundwater_levels.py:53-54`](scripts/phase1/groundwater_levels.py#L53-L54) still hardcodes
`YEAR_START = 2000` / `YEAR_END = 2020`; NWIS has these wells from ≤1980 to 2025+ today. Extending
it is now *safe* (the per-well anomaly target landed first, which was the prerequisite) but is not
expected to buy skill — more rows of an unobserved driver is still an unobserved driver. What it
would buy is **deployability**, which is the one thing this model currently lacks.

### The other payoff is deployability

Before the per-model windows, **NDVI, GRACE and wildfire could not produce a prediction for any
month after 2020-12.** Their exported ONNX signatures demanded `irrigation_total_withdrawal_mgd`
and `public_supply_groundwater_mgd`, and those series do not exist past 2020 — so scoring current
conditions would have meant fabricating the model's own inputs. Those three, plus surface water and
wildlife, now carry **zero** features that die at 2020-12 and run on live data.

**Groundwater is the sole exception**, and only because of the constant above.

> **A side effect worth recording.** Un-capping discharge from `MONTHLY_CAPPED_2020_BASES` (it now
> runs to 2025) silently handed NDVI, GRACE and wildfire a discharge feature block they had never
> had — they lacked it only as an *accident* of the old 2020 coverage gap. It cost wildfire 0.034
> skill. Discharge is a *response*, not a pressure, so the exclusion is now explicit in
> `_RESPONSE_FEATURE_BASES`. If feeding discharge into NDVI is worth trying on the merits, that is
> its own experiment; it should not arrive as a side effect of a Phase 1 window change.

---

## Pipeline Architecture

```
scripts/phase2/
├── merge.py                   — Build monthly (696×31) and annual (58×44) panels
├── features.py                — Per-model window, target and feature set (_MONTHLY_MODEL_SPECS)
├── metrics.py                 — Nested CV, persistence baseline, skill score
├── baselines.py               — Standalone lag1 / roll3 baselines
├── model_ndvi.py              — XGBoost residual-over-lag1
├── model_grace.py             — Ridge on 4 physical inputs (GLDAS storage change, rain, rain lag1,
│                                temperature anomaly), residual-over-lag1, real-target-only mask;
│                                the XGBoost search is kept for the experiment scripts only
├── model_groundwater.py       — In-fold competition (5 candidates), residual
├── model_surface_water.py     — In-fold competition (3 candidates), residual on a log anomaly
├── model_wildfire_monthly.py  — In-fold competition (4 candidates incl. Tweedie), direct
├── model_wildlife.py          — In-fold competition (3 candidates), nested LOO
├── export.py                  — Leaderboard + model_comparison.json
└── run_phase2.py              — Orchestrator with --models / --skip-models
```

Every model script routes its scoring through `metrics.nested_cv_evaluate`, which takes a
single `fit_predict(x_train, y_train, x_test)` callable and guarantees it never sees the fold
it is scored on. Adding a new model means writing that one function.

**Phase 1 dependency worth naming:** [`nclimdiv.py`](scripts/phase1/nclimdiv.py) (NOAA nClimDiv
PDSI/temperature/precipitation by Arizona climate division, 1895–2026, free and unauthenticated)
is what makes the two long-record models and the wildlife extension possible at all. It is
area-weighted to the region by true polygon intersection in an equal-area projection, and it
cross-checks against MERRA-2 on the 288-month overlap, refusing to write if they disagree:
**temperature r = +0.9992, precipitation r = +0.9202.**

**Three regression guards now ship with Phase 1**, each closing a bug class that actually bit:
`_validate_seasonality()` refuses to write the wildfire CSV if the peak ignition month falls
outside April–August; `_effort_confound_check()` refuses to write a wildlife target whose
correlation with survey effort exceeds ±0.6; and `irrigation.py`, `public_supply.py` and `nclimdiv.py`
refuse to write physically impossible values — a regional total above 20,000 / 5,000 MGD, a
temperature below −40 °C, negative rainfall — or to run at all without a HUC12 spatial filter. That last one replaced a
`log.warning` that had been silently emitting a national sum for the life of the project — **a
warning nobody reads is not a safeguard.**

---

## The binding constraint: most inputs carry no monthly information

Before proposing model changes, it is worth asking what the inputs actually contain. Decomposing
each input's variance into (a) the share that varies *within* a year at all, and (b) how much of
that within-year wiggle is just an identical month-of-year template repeating:

| Input | Within-year variance | …of which a repeating template | What it really is |
|-------|---------------------|-------------------------------|-------------------|
| `usdm_dsci` | 42.1% | 2.7% | **genuine monthly news** |
| `precipitation_mm_day` | 95.4% | 48.2% | **genuine monthly news** |
| `irrigation_total_withdrawal_mgd` | **99.3%** | **95.2%** | **mostly a calendar** (was: a corrupt time trend) |
| `mead_pool_elevation` | 7.7% | 42.4% | some genuine signal |
| `temperature_2m_c` | 98.8% | 96.8% | mostly a calendar |
| `public_supply_groundwater_mgd` | 96.8% | **97.6%** | **a calendar in disguise** (measured on the pre-fix national sum) |
| `population` | **0.4%** | — | **a time trend** |
| `impervious_pct` | **0.6%** | — | **a time trend** |

Public supply swings hard within each year, but it is the *same swing every year* — a month-of-year
clock, and `month_sin`/`month_cos` are already in the feature set. Population and impervious % are
smooth interpolations from annual data with essentially no monthly variation; they are proxies for
*time*.

**Two consequences, both of which bite.**

**1. Do not narrate these as causal drivers.** `population` and `population_roll12` were the top-2
wildfire features under the old target — the model reading a trend line, not a mechanism. Both are
gone from wildfire's feature set now, but `population` and `impervious_pct` remain in NDVI, GRACE
and groundwater, where the same caution applies.

**2. It is why the 2020 feature drop was nearly free** — dropping irrigation and public supply to
extend NDVI cost **−0.0075 skill**, measured. They were templates.

> **⚠ The irrigation row was re-measured on 2026-09-04 and changed completely.** Its original
> numbers (11.5% / 19.6% / "some genuine signal") were computed on a **corrupt** series: the feature
> was ~99.9% nodata sentinel, summed over 87,020 watersheds nationwide, and four orders of magnitude
> too large. Its 11.5% "within-year variance" was nodata churn. The corrected series is 99.3% /
> 95.2% — physically right, and on this table's own taxonomy a **calendar**. **The fix moved
> irrigation from a disguised time trend to a disguised calendar.**
>
> The lesson for this table: **a variance decomposition tells you how an input varies, not whether
> it is correct.** Check magnitude against a physical range, then sign, then variance.
>
> **The full feature audit (2026-09-04) found one more of these**, and it was the same bug in the
> same cloned function: `public_supply_groundwater_mgd` was also a national sum, at **34,817 MGD —
> 5.4× Arizona's entire water use across every sector**, and 44× its own Arizona source. Corrected
> to 761 MGD. Its row above is still the pre-fix measurement. Everything else in the panel passed:
> population validates against the Census to within 1.3%, Mead sits inside its physical envelope,
> and PDSI's apparent outliers are the real 1905/1941/1993 pluvials.

It also explains why groundwater and GRACE resist better modeling: the human pressures that drive
month-to-month aquifer change are **weakly** observed. That phrasing is deliberate and is a
correction. With the corrected irrigation series, the partial correlation with each target's
month-to-month change — *after* projecting out the month-of-year harmonics that `month_sin`/
`month_cos` already encode — is **+0.147 for groundwater (p = 0.021)** and **−0.260 for GRACE
(p < 0.001)**, both with the right physical sign, and roughly three-quarters of each raw
correlation survives the calendar control. On the corrupt series those same numbers were +0.065
and −0.060: noise.

So the driver is not *missing*; the observable part of it accounts for roughly **2%** (groundwater)
and **7%** (GRACE) of the variance in monthly change, and the large part — the year-to-year pumping
anomaly responding to drought, allocation cuts and crop prices — remains unmeasured. **Those two
models are data-limited, not model-limited**, but the limit is signal strength, not absence.

**That was tested, and the signal does not convert.** GRACE has the larger partial correlation
(−0.260) and could not see irrigation at all, having dropped it to extend past 2020. Handed the
feature back on identical test rows it gains **+0.0098** target R² and wins 2/5 folds — a null, in a
comparison structured in its favour. See the GRACE section above.

---

## Recommended next steps

> **See [PROBLEMS.md](PROBLEMS.md)** for the full register: every known defect, its root cause,
> its solution, and what is measured versus merely estimated.

Ordered by expected gain per unit of effort, and reflecting what was **empirically tested**, not
what sounds plausible.

1. ~~**Audit the remaining features.**~~ ✅ **Done.** All 31 panel columns checked for magnitude,
   sentinels, scope and sign. Two bugs: `public_supply_groundwater_mgd` had the same national-scope
   sum as irrigation (34,817 → **761 MGD**), and `nclimdiv.py`'s single `MISSING = -99.99` matched
   only one of its three elements, leaking −99.90 °F as **−73.28 °C** and −9.99 in/month as
   **−8.46 mm/day**. The nClimDiv leak never reached a model — confined to unpublished 2026
   months, six months beyond the panel ceiling, and verified by arithmetic rather than assumed
   (blending one missing division would move the regional mean ≥9.8 °C; the only in-panel shifts
   are ≤0.11 °C, which are NOAA's own revisions to recent months) — but it feeds three of the four
   working models and would have activated silently on any window advance. Everything else passed.

2. ~~**Wildlife: revisit the NDVI trade-off.**~~ ✅ **Done — null.** On identical rows NDVI adds
   Δρ +0.041 (p = 0.495). More decisively, the window cut it requires takes the model from
   R² +0.0956 to **−0.1702** on its own, with ρ falling below its permutation null. Keep the 1968
   window. See the wildlife section above.

3. **Groundwater: extend the NWIS pull to 1980–2025.** Not for skill — there is no reason to expect
   any. For **deployability**: it is the only model that cannot be scored against current
   conditions, and the only remaining hardcoded window in Phase 1
   ([`groundwater_levels.py:53-54`](scripts/phase1/groundwater_levels.py#L53-L54)). The per-well
   anomaly target already landed, which was the safety prerequisite.

4. **Probe CAP monthly deliveries** via Reclamation HydroData. The one remaining *monthly* pumping
   proxy: Tucson Water shifted off groundwater onto Colorado River water from 2001, so high CAP
   delivery means low pumping — and unlike a seasonal template, deliveries respond to shortage tiers
   and allocation cuts. Cheap, because [`lake_mead.py`](scripts/phase1/lake_mead.py) already ingests
   from that endpoint family. **Aimed at GRACE, not groundwater**, and a probe, not a rescue.

5. **Probe GLDAS soil moisture** as a GRACE feature. Monthly ΔTWS is a fast part (soil moisture
   responding to weather) plus a slow part (groundwater responding to pumping); GLDAS informs the
   fast part via the same `earthaccess` path GRACE already uses. Modest and uncertain — precip lags
   and rolls may already carry it. **Do not decompose TWS → GWS expecting skill**: that strips out
   exactly the component climate features *can* predict and leaves the pumping residual they
   cannot.

**Two hygiene items**, neither of which moves a score, both worth closing for parity:

- **`wildfire_risk_index` is min-max normalized over the full series**
  ([`wildfire_monthly.py:173-178`](scripts/phase1/wildfire_monthly.py#L173-L178)), so the target's
  scale depends on the series maximum — the same class of look-ahead as the `consecutive_dry_years`
  bug. Min-max is a linear transform, so no R² changes. The 50/50 `fire_count`/`log_acres` blend is
  also arbitrary.
- **NDVI has no availability diagnostic.** No `n_pixels` count, no `ndvi_available` flag, and a
  silent `limit=3` interpolation in `_check_and_fill_gaps`
  ([`ndvi.py:319-320`](scripts/phase1/ndvi.py#L319-L320)). GRACE and both anomaly targets emit
  availability/count columns; NDVI is the odd one out — and that diagnostic is precisely what made
  the roster-churn bug class visible everywhere else.

### Do not

- **Do not tune GRACE or groundwater.** Both are at their data-limited ceiling. Their constraint is
  what is missing from the panel, not what is wrong with the estimator.
- **Do not acquire ADWR pumpage.** Checked: it is **annual and AMA-only**, filed by March 31, with
  Yuma, Greenlee and Gila largely outside
  any AMA. An annual value is a step function whose within-year variance is ~0, so its contribution
  to a monthly difference is ~0 — the same diagnosis the variance table gives `population`.
- **Do not compare skill across a window or target change.** See
  [the comparability trap](#the-comparability-trap-skill-scores-that-move-the-wrong-way).

### Tried and failed — do not repeat

- **Blaming wildfire's loss function.** Ranked here for a long time as "the largest untested upside
  in the project." The zero-inflation was real but was never the binding problem: the month labels
  came from a database edit date. Fixing the target alone, loss untouched, moved R² −0.0338 →
  +0.3122; Tweedie was then tried on the working model and **lost** (0.2895 vs 0.3381).
- **Blaming wildlife's sample size.** This report said "with n=20 and 22 features, no search will
  rescue this." Voided: at n=20 and 22 features unchanged, the target fix moved LOO R² −0.2108 →
  +0.2661.
- **Giving GRACE the corrected irrigation feature at the cost of its 2023 window.** Motivated by a
  real, significant, correctly-signed partial correlation (−0.260, p < 0.001) and it still failed:
  +0.0098 target R², 2/5 folds, in a test rigged in its favour. **A significant correlation is not
  a usable feature.**
- **Fixing the irrigation feature, as a route to skill.** Worth doing and done — it was 99.9%
  nodata summed nationwide — but it moved nothing. Five of six models are unchanged to four decimal
  places (only groundwater can see irrigation at all), and groundwater moved within fold noise. The
  corrected feature is now the model's **#2** by importance, having previously never been selected.
  **Correctness fix, not a performance fix**, exactly like the groundwater target before it.
- **Long-window cumulative features for groundwater** (roll24, roll36, cum60 of irrigation and
  public supply, on the theory that an aquifer integrates pumping over years). No effect:
  skill +0.0050 → +0.0047 at a 1-month horizon, and actively harmful at 12 months. Reason: those
  inputs have no month-specific content to accumulate.
- **A 12-month prediction horizon for groundwater.** Looks like a win (+0.22 skill) and is not:
  persistence collapses to −0.83 over that horizon, so the model beats a strawman while its own
  R² is −0.61 — worse than a flat line.
- **In-fold feature selection as a score fix.** The leak was real and is fixed, but it was not
  inflating anything: selecting inside the fold scored identically for groundwater and slightly
  *better* for surface water.
- **Extending the window to rescue a model with a broken target.** Wildfire got 36 more rows on the
  fabricated `DATE_CUR` axis and fell from R² 0.0041 to −0.0338; GRACE's skill went −0.0072 →
  −0.0349 once its fabricated target months were removed. Once wildfire's target became a real
  ignition date, re-extending **helped** (+0.0483 ΔR²). **The lesson is diagnose the target
  first** — not that extension never pays.
- **Pushing the monthly start back to 2000-01 for GRACE-consuming models.** NDVI skill +0.2453 vs
  **+0.2653** for the 2002-10 start. Keep the floor.

---

## What changed in the audit

Six harness bugs, all fixed. Listed so the old numbers in git history are interpretable.

**1. Residual models were scored only on the reconstructed level.** The lag1 anchor dominates
that score, so a model with zero or negative skill still looked strong. Every model now also
reports R² on the target it was actually trained on, and the leaderboard prints both.

**2. Hyperparameters were selected on the folds that were then reported.** `RandomizedSearchCV`
searched 60 draws over the same `TimeSeriesSplit` used for the final score — a max over noisy
estimates. All scoring is now nested. Impact: NDVI −0.009 (it was fine), GRACE **−0.120** (it
was not).

**3. Model choice and feature selection also touched the test folds.** `_select_features` fit
XGBoost on the full panel before CV, and the competition scripts picked their winner with
`max(candidates, key=cv_r2)` on the reported folds. Both now happen inside `_fit_predict`.
(Measured separately, the feature-selection leak turned out *not* to be inflating scores.)

**4. `export.py` listed only four of the six models.** `MODEL_IDS` omitted `groundwater` and
`surface_water`, so their report rows were hand-written with a blank baseline column. That is
the specific reason a model that loses to persistence once shipped as a headline result.

**5. `consecutive_dry_years` had look-ahead.** It computed its dry/wet threshold from a
full-series median, so a given year's "is this a dry year" flag depended on precipitation that
had not happened yet. Now expanding.

**6. `generate_stats.py` dropped zeros before computing percentiles.** Sensible for population or
discharge, where a zero is a missing reading — but the corrected wildfire target is zero in most
months, and those zeros are the *most common real observation*. The frontend was about to show a
resting wildfire risk of **0.41** (the median of fire months only) instead of **0.0**.
Zero-dropping is now opt-out per variable via `zero_is_data`, set on the five series where zero is
a real reading: wildfire, PDSI, and the three signed anomaly targets (groundwater, surface water,
wildlife). *A latent bug, activated by the wildfire target fix — and the per-station anomaly
targets, which are signed and centered on zero, walked straight into the same trap.*

---

## Known limitations

- **Monthly pumping is only weakly observed.** The single biggest constraint in the project, and
  the whole reason GRACE and groundwater fail. The corrected irrigation series does carry real,
  correctly-signed information about both targets' monthly change (partial Δr +0.147 and −0.260
  after removing the calendar), but that is ~2% and ~7% of the variance. No estimator change
  touches the rest.
- **Groundwater cannot be deployed against current conditions.** Its target stops at 2020-12
  because of a hardcoded constant, not because the data stops.
- **The two long-record models run on nClimDiv alone.** Surface water and wildfire have no
  `usdm_dsci`, no MERRA-2, and no satellite cross-features — that is the price of reaching 1980 and
  1984, and in both cases it was measured to be worth paying.
- **Irrigation and public supply end 2020-12.** NDVI and GRACE therefore have **no human-pressure
  water inputs at all** after 2020; they run on climate, drought, population and reservoir state.
  Per the variance table this costs almost nothing, but it should be stated rather than discovered.
- **GRACE has 33 fill months** inside its window across 23 blocks. They are excluded from its
  target, but remain as *features* for NDVI (a smoothed input, not a fabricated answer);
  `grace_available` is retained there so the model can learn to discount them.
- **GRACE is leakage-smeared.** At ~300 km resolution, an eight-county "regional" mean is a coarse
  proxy, and it correlates +0.71 with well-depth anomaly where physics wants a negative sign.
- **Wildlife n=56 is still small**, and reaching it cost NDVI as a feature. Spearman ρ with a
  permutation test — not R² — is the reliability indicator at this sample size.
- **Wildfire ignition is stochastic.** No model predicts *when* a fire starts, only the seasonal
  and anthropogenic propensity for one. R² ~0.28 is near the ceiling.
