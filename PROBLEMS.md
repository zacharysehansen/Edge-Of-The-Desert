# Known Problems and Their Solutions

A single register of everything currently wrong with the six models, why, and what to do
about it. Companion to [PHASE2_REPORT.md](PHASE2_REPORT.md) (what the models score) and
[data/Final/DATA.md](data/Final/DATA.md) (what the data is).

**Every claim here is tagged:**

- **`MEASURED`** — run, with the number. Trust it.
- **`VERIFIED`** — the data was inspected or the endpoint queried. The fact is confirmed; the
  *payoff estimate* attached to it is not.
- **`ESTIMATED`** — a projection. Could be wrong. Do not quote it as a result.
- **`UNTESTED`** — reasoning only.

---

## Status at a glance

| Model | Skill | Verdict | Binding problem | Fixable? |
|-------|-------|---------|-----------------|----------|
| Surface water | **+0.6830** | **works** ✅ | ~~target was 45% one dam-regulated gage~~ **FIXED**; extended to 1980–2025 | **done — see [M2](#m2-surface-water--the-target-was-mostly-one-regulated-river-fixed)** |
| Wildfire | **+0.3716** | **works** ✅ | ~~month came from a DB edit date~~ **FIXED**; extended to 1984–2023 | **done — see [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed)** |
| NDVI | **+0.2653** | **works** | none — at MODIS instrument floor | already healthy |
| Groundwater | **+0.0086** | no skill, but no longer harmful ✅ | ~~compositional artifact~~ **FIXED**; now [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured) | target fixed — **now stop** |
| GRACE | −0.0349 | no skill | the observable pumping signal is too weak | **stop tuning.** Irrigation trade re-tested 2026-09-04: **null**; the full deseasonalized human block re-tested under the real nested tuner 2026-09-10: **null** (t=1.21, 83% of the gain is one 28-row fold — PHASE3_PLAN.md §11.5); **CAP monthly deliveries tested 2026-09-12: null** (5/5 folds, t=1.52, 72% of the gain the same fold — §23; on the full 2002–2023 window the gain shrinks to a quarter, t=1.27 — §24, CAP closed); **GLDAS tested 2026-09-12: null as a feature block** (t=1.34), but a one-coefficient OLS on GLDAS storage change scores **+0.244** out of fold vs the shipped **+0.021** — the estimator, not the data, is the floor (§25). **Confirmed by step 2b (§26): a ridge on four physical inputs scores +0.2845 target R², +0.5664 level R², skill +0.2015, 4/5 folds, t = 2.52 — REAL. Shipping it is an architecture change, not yet taken.** Only acquisitions left: [CAP](#option-b--cap-deliveries-as-a-monthly-pumping-proxy-untested--worth-one-probe) / [GLDAS](#option-c--gldas-land-surface-state-as-features-not-a-new-target-untested) / OpenET |
| Wildlife | **+0.2046** | **works** ✅ | ~~the target was a survey-effort index~~ **FIXED** | **done — see [M6](#m6-wildlife--the-target-was-a-survey-effort-index-fixed)** |

> **2026-09-12 — the region is not the one the documents named, and it is being kept.** See
> [P8](#p8-the-region-was-never-the-one-the-documents-named-measured--kept-on-purpose).

**Five of six models now predict something, and no model is actively harmful anymore.** GRACE is
the only remaining failure, and its cause ([P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured))
is missing data, not a missing model.

> **2026-09-04 — the first *feature* bug, and it moves P5.** `irrigation_total_withdrawal_mgd` was
> **~99.9% nodata sentinel**. The source matrix encodes missing data as literal numbers — `999`
> ("no data for this HUC12", 73.2% of Arizona cells) and `888` ("no data this month", the file's
> most common non-zero value *and* its maximum, in a variable with 189,456 distinct values) — and
> `irrigation.py` summed them as data. Worse, `data/raw/wbd/` did not exist, so the spatial filter
> fell through to a fallback that summed **all 87,020 HUC12 columns, nationwide** (the file spans
> HUC regions 01–18, Maine to Oregon) and labelled it an eight-county total. The shipped feature was
> **5.43e7 MGD** — four orders of magnitude above Arizona's entire water use — and near-constant
> (std/mean **0.014**), i.e. a disguised time trend. **Fixed:** both sentinels masked, WBD shapefile
> built (1,133 of 7,558 HUC12 centroids inside the eight counties), fallback now **raises**, and
> magnitude/variability guards added. The series is now **2,560 MGD, std/mean 0.622**, peaking in
> June and troughing in January. **This is the first bug found on the feature side rather than the
> target side** — see [P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed)
> — and it forces a revision of both [P4](#p4-most-inputs-carry-no-monthly-information-measured)
> (whose irrigation row was measuring the bug) and
> [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured).

> **2026-07-15 — the GRACE target was audited too, and it is also sound: the failure is data, not
> the target.** Same two rules. Rule 1: the anomaly is a spatial mean over the ~24 GRACE cells in
> the bbox; reprocessing the nc4 gives ~20 valid cells/month, **constant within each mission** (the
> count changes once in 226 months, at the GRACE→GRACE-FO handover) — no roster churn. Rule 2: a
> **strong, highly significant secular decline** (trend r = **−0.84**, p = 2e-61 — the Arizona
> depletion signal), co-moving correctly with drought (usdm −0.22) and Lake Mead (+0.75). So GRACE
> measures regional storage; it is **not** artifactual like the four fixed targets. **This confirms
> [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured) rather than overturning it**
> — GRACE fails because monthly change in a storage integral is driven by *unobserved pumping*, not
> because the target is wrong. Two caveats: at ~300 km resolution the regional mean is
> **leakage-smeared** (a coarse proxy), and it runs **+0.71 with well-depth anomaly where physics
> wants a negative sign** — most likely the GRACE-vs-local-wells scale mismatch, not a target defect.
> **There is no GRACE target fix to find.** See [M4](#m4-grace--data-limited-at-its-instrument-floor).

> **2026-07-15 — the NDVI target was audited, and for the first time the audit came back clean.**
> Both project rules were applied. Rule 1 (does the averaging denominator move?): the target is
> `mean(valid MODIS pixels)`, structurally the same "mean over whoever reported" shape that broke
> groundwater, surface water and wildlife — but empirically null here. Reprocessing all 574 HDFs to
> recover the pixel count the script never recorded: **369,305 valid pixels/month, CV 0.1%**, and
> **corr(pixel turnover, |month-to-month NDVI jump|) = +0.000** (the convicted targets ran ~+0.75).
> MOD13A3 is a monthly *composite* that removes cloud churn upstream, so the roster is effectively
> fixed. Rule 2 (physical signs): `precip.shift(1)` **+0.562** (the desert green-up lag), drought
> **−0.251**, monsoon seasonality peaking September — all correct. Fill is one backfilled month
> (2000-01), outside the model window. **NDVI measures vegetation.** The target-bug streak was four
> for four; NDVI breaks it in the good direction. Three hygiene gaps remain (no `n_pixels`
> diagnostic, no availability flag, a silent 3-month interpolation limit) — none affects the score.
> See [M1](#m1-ndvi--healthy-at-its-instrument-floor).

> **2026-07-15 — wildfire extended to 1984, and it paid; the Tweedie loss was tried and lost.**
> With the MTBS target real ([M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed)), the
> window was pushed back to the MTBS floor: **479 months (was 255)**, 1984–2023. Reaching before
> 2000 collapses the feature set to nClimDiv alone (**26 features, was 51**) — no `usdm_dsci`, no
> MERRA-2, no NDVI/GRACE — the same trade surface water made. It was *measured*, not assumed:
> holding the test rows fixed at the common 2006–2023 folds and varying only the training data,
> **R² +0.3618 → +0.4100 (ΔR² = +0.0483, five folds for five).**
> **⚠ The headline CV R² went *down* (0.3122 → 0.2819) and skill fell (+0.5615 → +0.3716)** while
> the model got better, because the pre-2000 folds are harder (68% zero months) and persistence
> rose from −0.25 to −0.09 — the not-comparable-across-windows trap again. Quote the fixed-test
> ΔR². Separately, a **Tweedie objective** (work item #1) now competes for this zero-inflated
> target and **lost**: 0.2895 vs squared-error XGBoost's 0.3381 out-of-fold. It stays in the
> competition as a self-selecting candidate. **Both changes together are the cleanest possible
> confirmation of the M3 thesis: the target was the whole story, not the loss and not the sample
> size.** See [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed).

> **2026-07-13 (d) — surface water extended to 1980–2025, and the extra rows paid.** NWIS now
> yields **551 months** (was 219). Reaching back past 2000 costs the entire non-nClimDiv feature
> block — including `usdm_dsci`, the panel's highest-signal input — so the trade was *measured*,
> not assumed. Holding the test rows fixed at the common 2002–2020 period and varying only the
> training data: **R² +0.6777 → +0.7955 (ΔR² = +0.1178)**. Level R² on its own window is now
> **0.7516** with all five folds in 0.70–0.77.
> **⚠ The headline skill number went *down* (+0.7075 → +0.6830) while the model got better** —
> persistence rises from −0.0429 to +0.0686 on the longer window, so skill is not comparable
> across the change. This is exactly the trap
> [the rules](#the-three-rules-worth-keeping) warn about.
> See [M2](#m2-surface-water--the-target-was-mostly-one-regulated-river-fixed).

> **2026-07-13 (c) — wildlife works, and it was a target bug too.** `total_abundance` was a *sum
> over whichever routes were surveyed that year*, and route counts swing 15-26 regionally (and
> triple statewide around 1990). The old target therefore correlated **+0.94 with `route_count`**
> over the full record: it measured how many people went birdwatching. Effort-correcting it to a
> per-route anomaly moved LOO R² from **−0.2108 to +0.2661 at n=20 unchanged** — the sample size
> was never the binding constraint. Then `nclimdiv.py` (NOAA drought/temp/precip, 1895+) unlocked
> the BBS record back to 1968: **n = 20 → 56**, Spearman ρ **0.51, permutation p = 0.0001**.
> See [M6](#m6-wildlife--the-target-was-a-survey-effort-index-fixed).

> **2026-07-13 (b) — the per-station anomaly targets shipped, and surface water was hiding a
> second bug.** Both `depth_to_water_ft_mean` and `discharge_cfs_mean` were means over *whichever
> stations reported that month*. Phase 1 now centers each station on its own long-term mean before
> averaging ([P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed)).
> **Groundwater** went from skill −0.0791 to **+0.0107** — it is no longer worse than doing
> nothing, landing at ~zero exactly as predicted. **Surface water** was expected to be unaffected
> (its gage roster is stable). Instead it went from skill +0.1469 to **+0.7075**, because the raw
> mean turned out to be **45% a single dam-regulated gage** whose flow is a Reclamation release
> schedule, not weather. See [M2](#m2-surface-water--the-target-was-mostly-one-regulated-river-fixed).

> **2026-07-13 (a) — wildfire is fixed, and the old diagnosis in this document was wrong.**
> It blamed the loss function. The real cause was that the monthly target's *time axis was
> fabricated*: the month was parsed from `DATE_CUR`, a database-maintenance timestamp. Phase 1
> now ingests **MTBS ignition dates** instead. Wildfire went from R² **−0.0338** (worse than a
> flat line) to R² **+0.3122**, skill **+0.5615**, with the loss function untouched.
> See [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed).

> **Four of the five fixes were target bugs, found the same way, and none needed a paid or
> credentialed download.** The fifth was a *feature* bug
> ([P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed)) and went unfound for
> far longer, because the audit discipline was only ever pointed at targets.
> See [the rules](#the-three-rules-worth-keeping).

---

# Part 1 — Cross-cutting problems

These affect multiple models. Fix these first; several per-model problems dissolve when they
are addressed.

---

## P1. Phase 1 throws away decades of available data `VERIFIED`

**Symptom.** The monthly models train on 219–255 rows and the annual models on 20. Sample
size is blamed for the annual models' failure.

**Root cause.** Every Phase 1 acquisition script hardcodes a `2000` start — and in two cases a
`2020` end — as a *project convention*. None of these are source limits. The sources have far
more, and in one case we have already downloaded it and are discarding it at parse time.

| Source | Phase 1 takes | Actually available | Evidence |
|---|---|---|---|
| ~~**BBS wildlife**~~ | ✅ **1968–2024, n=56** | 1968–2024 | **DONE** — see [M6](#m6-wildlife--the-target-was-a-survey-effort-index-fixed) |
| ~~**NWIS surface water**~~ | ✅ **1980–2025, n=551** | ≤1970 → 2025+ | **DONE** — 1.64M daily rows, 73–110 gages/month |
| **NWIS groundwater** | 2000–2020 | **≤1980 → 2025+** | probed: 12 active wells in 2024–25 |
| ~~**MTBS wildfire**~~ | ✅ **1984–2023, n=479** | 1984 → 2024 | **DONE** — see [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed) |
| MERRA-2 temp/precip | 2000–2023 | **1980 → now** | product spec |
| NLCD impervious | 2000–2023 | 1985 → 2024 | per DATA.md |
| Lake Mead | 2000–2023 | 1935 → now | per DATA.md |

The offending constants:

- ~~[`water_surface.py`](scripts/phase1/water_surface.py) — `YEAR_START = 2000`, `YEAR_END = 2020`~~ ✅ **now 1980–2025**
- [`groundwater_levels.py:28-29`](scripts/phase1/groundwater_levels.py#L28-L29) — `YEAR_START = 2000`, `YEAR_END = 2020`
- ~~[`wildlife.py`](scripts/phase1/wildlife.py) — `START_YEAR = 2000`~~ ✅ **now 1968**
- ~~[`wildfire_monthly.py`](scripts/phase1/wildfire_monthly.py) — `PROJECT_START = 2000`~~ ✅ **now 1984**
- plus the equivalents in `ndvi.py`, `water_stress.py`, `temperature.py`, `precipitation.py`

**Correction to the record.** [PHASE2_REPORT.md](PHASE2_REPORT.md) previously stated that
groundwater and surface water "cannot be extended by any means short of new data" because
their targets end 2020-12. **That is wrong.** The data exists in NWIS today; Phase 1 simply
never requested it. The 2020 wall is an artifact of a constant, not of the world.

**Solution.** Re-run Phase 1 with per-source ranges driven by each source's *actual* floor
rather than a shared convention. See [P2](#p2-the-usdm-is-a-hard-floor-at-2000-01--and-it-is-your-best-feature-verified),
which is the prerequisite, and [P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed),
which is the danger.

---

## P2. The USDM is a hard floor at 2000-01 — and it is your best feature `VERIFIED`

**Symptom.** Extending any model before 2000 appears to require giving up `usdm_dsci`.

**Root cause.** The U.S. Drought Monitor *began* in January 2000. There is no earlier data and
never will be. This matters more than it sounds, because the variance decomposition in
[PHASE2_REPORT.md](PHASE2_REPORT.md) identifies `usdm_dsci` as the **single highest-signal
input in the entire panel** — 42.1% within-year variance, of which only 2.7% is a repeating
month-of-year template. Nothing else is close. Losing it to buy history would be a bad trade.

Two other sources are also at permanent instrument floors:

- **MODIS NDVI — 2000-02.** Terra launch. The NDVI model is already sitting on it.
- **GRACE — 2002-04.** Mission start. The GRACE model is already sitting on it.

**Solution — this is the unlock. ✅ SHIPPED as [`nclimdiv.py`](scripts/phase1/nclimdiv.py) `MEASURED`**

> Built and running. Area-weighted to the region by true polygon intersection in an equal-area
> projection (Southeast 42.6%, South Central 33.6%, Southwest 12.7%, East Central 11.1%) — not a
> county-to-division lookup. It cross-checks against MERRA-2 on the 288-month overlap and refuses
> to write if they disagree: **temperature r = +0.9992, precipitation r = +0.9202**, which is what
> confirms the Fahrenheit→Celsius and inches/month→mm/day conversions and the area weights.
> It carries **PDSI, temperature and precipitation**, so it replaces the MERRA-2 floor too, not
> just the USDM's. First consumer: [M6](#m6-wildlife--the-target-was-a-survey-effort-index-fixed)
> (n = 20 → 56).
>
> **Gotcha, recorded so nobody loses an hour to it:** NCEI advertises an AAAA record whose endpoint
> resets the connection. `curl -4` works, `curl -6` does not, so on any IPv6-preferring host
> `requests` fails with a bare ConnectionError that looks like an outage. `nclimdiv.py` pins
> urllib3 to IPv4 (`_force_ipv4`). Also note nClimDiv uses **its own state numbering, not FIPS** —
> Arizona is `02` there and `04` in FIPS.


Replace the USDM with **NOAA nClimDiv**, which publishes monthly drought indices by climate
division back to **1895**:

```
https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-pdsidv-v1.0.0-<YYYYMMDD>
```

Fetched and parsed during this investigation: **Arizona, all 7 climate divisions, monthly,
1895–2026, free, no authentication.** PDSI, PHDI, ZNDX and SPI ship in the same series
(`climdiv-{pdsi,phdi,zndx,sp01,...}dv-*`). Area-weight the divisions to the eight-county
region exactly as `water_stress.py` already area-weights counties.

Keep `usdm_dsci` where it is available (2000+) — it is a different and arguably better
construct. Add PDSI as the long-record backbone so pre-2000 models have a drought signal at
all.

**With the USDM replaced, the next floor is MERRA-2 at 1980.** That is a good place to stop.
Going earlier means substituting nClimDiv/PRISM temperature and precipitation too, which is
possible (1895) but only buys wildlife ~11 extra rows.

**One Phase 1 script — `nclimdiv.py` — is the prerequisite for P1 paying off at all.**

---

## P3. Compositional churn in station-averaged targets `MEASURED` — *now fixed*

**Symptom.** The groundwater model was *actively worse than doing nothing*: residual
R² = **−0.3179**, meaning it predicted month-to-month change worse than predicting no change.

**Root cause.** `depth_to_water_ft_mean` was a mean over **whatever wells happened to report
that month**, and wells sit at wildly different depths (one well's water table is 40 ft down,
another's is 400). The reporting set churns, so the target moved for reasons that were
compositional, not hydrological — and therefore physically unlearnable. The optimizer's
discovery that "extreme regularization is key" was it learning to predict ~zero change:
persistence wearing a costume.

**Re-measured during the fix, and it was worse than this document claimed.** The old text said
turnover correlates 0.37 with the jump in the mean. Measured directly on the daily pull:

| | old raw mean | per-well anomaly |
|---|---|---|
| turnover ↔ \|month-to-month jump\| | **+0.746** | +0.333 |
| standard deviation | **27.8 ft** | 5.9 ft |

The raw target's standard deviation was nearly **five times** the anomaly's. Roughly 80% of what
`depth_to_water_ft_mean` was "measuring" was its own roster.

**Solution — applied.** Subtract each station's own long-term mean *before* averaging (the
standard construction for a changing station network), with a 24-month minimum record per station
so a "long-term mean" is actually long-term. Phase 1 now emits `depth_to_water_anomaly_ft` and
`discharge_log_anomaly` alongside the old raw means, plus `n_wells` / `n_gages` — **the diagnostic
that makes this class of bug visible at all.** Neither count is exposed as a model feature; they
are the artifact, not a predictor.

**Measured on groundwater:** skill −0.0791 → **+0.0107**; residual R² −0.3179 → **−0.0259**.
That is a *correctness* fix — it stops the model being actively harmful — but it lands at
approximately **zero**, not at a good model, exactly as predicted. **Groundwater is now done.**

**On surface water this document was wrong in an interesting way.** It predicted the same
construction had "so far gotten away with it, because 20 years of a mostly-stable gage network is
survivable." The roster half of that is right — gage turnover correlates **−0.02** with the jump,
i.e. no churn at all. But the fix helped enormously anyway, for a reason nobody had looked for.
See [M2](#m2-surface-water--the-target-was-mostly-one-regulated-river-fixed).

**The forward-looking rule still stands, and is now cheap to honour.** The gage and well networks
of 1980 are *not* the networks of 2020. Stretching a mean-over-reporting-stations target across 45
years would not dilute the artifact — it would **amplify** it. The anomaly target is now in place
*before* any NWIS extension, which is the order this required.

---

## P4. Most inputs carry no monthly information `MEASURED`

**Symptom.** Feature-importance tables look meaningful. They are substantially not.

**Root cause.** Decomposing each input's variance into (a) the share that varies *within* a
year, and (b) how much of that is just an identical month-of-year template repeating:

| Input | Within-year variance | …of which a repeating template | What it really is |
|-------|---------------------|-------------------------------|-------------------|
| `usdm_dsci` | 42.1% | 2.7% | **genuine monthly news** |
| `precipitation_mm_day` | 95.4% | 48.2% | **genuine monthly news** |
| `irrigation_total_withdrawal_mgd` | **99.3%** | **95.2%** | **mostly a calendar** (was: a corrupt time trend) |
| `mead_pool_elevation` | 7.7% | 42.4% | some genuine signal |
| `temperature_2m_c` | 98.8% | 96.8% | mostly a calendar |
| `public_supply_groundwater_mgd` | 96.8% | **97.6%** | **a calendar in disguise** |
| `population` | **0.4%** | — | **a time trend** |
| `impervious_pct` | **0.6%** | — | **a time trend** |

**Consequences, both of which bite:**

1. **Do not narrate these as causal drivers.** `public_supply_groundwater_mgd_lag1` ranking #2
   for NDVI does not mean municipal pumping drives vegetation — it is the model reading a
   calendar that `month_sin`/`month_cos` already encode. `population` and `population_roll12`
   being the top-2 wildfire features is the model reading a trend line, not a mechanism.

2. **It is *why* the 2020 feature drop was nearly free.** Dropping irrigation and public supply
   to extend NDVI to 2023 cost **−0.0075 skill** `MEASURED`. They were templates.

> **⚠ The irrigation row above was re-measured on 2026-09-04 and it changed completely.** The
> original numbers (11.5% within-year, 19.6% template, "some genuine signal") were computed on the
> **sentinel-corrupted** series described in
> [P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed) — a near-constant
> whose 11.5% "within-year variance" was nodata churn, not irrigation. The corrected series is
> **99.3% within-year, 95.2% of it a repeating template**: physically right (June peak, January
> trough) and, on this table's own taxonomy, a calendar. **The fix moved irrigation from a disguised
> time trend to a disguised calendar.** Neither is monthly news — but see
> [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured), because the residual 4.8% is
> not nothing.

**Solution.** Already applied where it matters (those features are dropped from the extended
models). The remaining action is editorial: never present these importances as mechanism.

**A caution this table earned the hard way.** A variance decomposition cannot tell you whether the
input is *correct* — only how it varies. Irrigation sat in this table for the life of the project
labelled "some genuine signal" while being 99.9% nodata. Run the decomposition, but check the
magnitude and the sign first.

---

## P5. Monthly pumping is *weakly observed*, not unobserved `MEASURED`

> **Revised 2026-09-04.** This section used to be titled "Monthly pumping is unobserved" and
> tagged `UNTESTED`. It rested substantially on the P4 table's irrigation row — and that row was
> **measuring a bug**, not the data
> ([P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed)). With a correct
> irrigation series the claim needs weakening in one specific way, and it survives everywhere else.

**Symptom.** GRACE (skill −0.0349) and groundwater (+0.0086) both fail, and neither responds
to modeling effort.

**What is now measured.** The corrected monthly irrigation series
([P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed)) *does* carry
information about both targets' month-to-month change, with the correct physical sign, and it
**survives controlling for the calendar**. Projecting out month-of-year harmonics (1st–3rd, i.e.
what `month_sin`/`month_cos` already encode) from both differenced series:

| | raw Δr | **partial Δr** (calendar removed) | p | variance retained |
|---|---|---|---|---|
| irrigation vs **groundwater** Δ | +0.175 | **+0.147** | 0.021 | 71% |
| irrigation vs **GRACE** Δ | −0.301 | **−0.260** | <0.001 | 75% |

More irrigation → water table deeper, regional storage down. Both signs correct, both significant,
and roughly three-quarters of each correlation is *not* seasonal. **So "unobserved" was too strong.**
The corrupt series' equivalents were +0.065 and −0.060 — indistinguishable from noise, which is
exactly why this was invisible until the feature was fixed.

**The claim that survives, stated precisely.** Monthly pumping is *observed weakly*: the partial
correlations correspond to about **2% (groundwater) and 7% (GRACE) of the variance in monthly
change**. That is real signal and it is nowhere near enough to carry a model. The honest framing is
no longer "the driver is missing from the panel" but **"the observable part of the driver is small,
and the large part — the year-to-year pumping anomaly that responds to drought, allocation cuts and
crop prices — is what remains unmeasured."**

**Evidence that it is genuinely weak, not merely under-used.** Groundwater is the one model that
can see irrigation, and after the fix its in-fold selection **started choosing the feature** — it
rose from never-selected to the **#2 importance (0.0963)**. The out-of-fold score did not improve
(skill +0.0091 → +0.0058, within fold noise). A feature the model actively wants, that still does
not convert, is close to a direct measurement of this section's thesis.

**Solution.** Acquire **better monthly pumping data**, or accept the ceiling. This is a **Phase 1
data-acquisition task, not a Phase 2 modeling task**.

**The live experiment was run, and it came back null `MEASURED`.** GRACE has the larger partial
correlation (**−0.260**) and could not see irrigation at all, having dropped the feature to extend
past 2020. Measured on identical test rows (the common 2002-10 → 2020-12 period, 168 rows, five
folds, training strictly before each test fold):

| variant | window | features | R² target | Δ vs shipped | folds won |
|---|---|---|---|---|---|
| **A — shipped** | 2002–2023 | 45 | +0.0528 | — | — |
| B — all 2020-capped features | 2002–2020 | 57 | +0.0787 | **+0.0259** (t=1.13) | 3/5 |
| C — **irrigation only** | 2002–2020 | 49 | +0.0626 | **+0.0098** (t=0.57) | 2/5 |

**Neither is significant, and irrigation alone does not even win a majority of folds.** The largest
effect is **0.09×** the fold-to-fold spread of the metric itself (sd 0.296).

**The design was tilted in favour of B and C, and they still lost.** Because the fixed test rows all
sit at or before 2020-12 and each fold trains strictly before its test window, variant A's 36 extra
months — which fall at the *end* of the record — never enter training. The comparison therefore
gives A none of its row advantage and isolates the feature difference alone. A null under those
conditions is a strong null.

**So the ~7% of differenced variance is real and does not convert.** That is the same result the
groundwater model gave from the other direction (it *selects* corrected irrigation at #2 importance
and gains no skill). Two independent measurements, one conclusion: **the observable pumping signal
is genuine, correctly signed, and too weak to carry a model.** P5's last no-new-data lead is closed;
what remains are acquisitions ([Option B](#option-b--cap-deliveries-as-a-monthly-pumping-proxy-untested--worth-one-probe),
[Option C](#option-c--gldas-land-surface-state-as-features-not-a-new-target-untested), OpenET).

**Until then: stop tuning GRACE and groundwater.** Their ceiling is set by what is weak in the
panel, not by what is wrong with the estimator.

### ⚠ ADWR is not the answer. This document used to say it was. `VERIFIED`

The old text named "ADWR Active Management Area reporting" as *the* obvious candidate and ranked
acquiring it as a top work item. **Checked, and it fails on the one axis that matters:**

1. **The grain is annual, not monthly.** Non-exempt wells inside an AMA report *annual* pumpage,
   filed by March 31. There is no monthly reporting requirement anywhere in the program.
2. **The coverage is partial.** AMA reporting only applies *inside* an AMA. Of the eight counties:
   Maricopa (Phoenix AMA), Pima (Tucson AMA), Pinal (Pinal AMA), Santa Cruz (Santa Cruz AMA) and
   parts of Cochise (Douglas, Willcox AMAs) are covered. **Yuma, Greenlee and Gila are largely
   outside any AMA**, where pumping is essentially unmetered. *(County list corrected 2026-09-12 —
   [P8](#p8-the-region-was-never-the-one-the-documents-named-measured--kept-on-purpose); it used to say Graham and La Paz.)*
3. **Annual data cannot fix a monthly residual model, and [P4](#p4-most-inputs-carry-no-monthly-information-measured)
   already proves it.** The groundwater and GRACE models predict *month-to-month change*. An annual
   value is a step function: constant for eleven months, then it jumps. Its within-year variance is
   ~0, so its contribution to a monthly difference is ~0. That is exactly P4's diagnosis of
   `population` (0.4% within-year variance) and `impervious_pct` (0.6%) — **"a time trend."**
   Acquiring ADWR pumpage would buy one more trend line, not monthly news. It is also consistent
   with the [already-recorded failure](#part-4--tried-and-failed-do-not-repeat) of long-window
   cumulative pumping features: there was no month-specific content to accumulate.

**So the ADWR acquisition task, as previously written, would have cost weeks and delivered
nothing.** Two real options remain.

### Option A — change the model's grain, not the data `UNTESTED`

If pumping is only observed annually, **model the response annually.** ADWR AMA annual pumpage
against the annual change in the per-well anomaly index is physically coherent, and it is the
model the data actually supports (~25–40 rows). The per-well anomaly target from
[P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed) aggregates up cleanly. This follows the
project's own rule: fit the model the data can support instead of demanding the data serve a grain
it was never collected at.

### Option B — CAP deliveries as a *monthly* pumping proxy `MEASURED — probed 2026-09-12: NULL`

> **Probed, and it is a null with a lesson.** `scripts/phase1/cap_deliveries.py` acquired CAP's
> monthly deliveries 1999-01..2026-07 from CAP's own reports (331 months, every year checked
> against its printed total). Under `model_grace.py`'s real nested tuner on identical rows the
> deseasonalized CAP block lifts target R² **+0.0394 → +0.1059, 5 of 5 folds — and t = +1.52
> against the ≥ 2.0 rule fixed before the run**, because 72 % of the gain is the 28-row first
> fold, the same signature as the human-block null (PHASE3_PLAN.md §11.5). Full record in
> [PHASE3_PLAN.md §23](PHASE3_PLAN.md). Two corrections to the text below: **the Reclamation
> Havasu diversion is the wrong series** — it runs at r = −0.2 to deliveries within a year because
> CAP fills Lake Pleasant in winter — and the acquisition *did* pay elsewhere:
> `region_share_of_az_reduction` is now `MEASURED` at 1.0 from the delivery drops of 2022–2025.
> **Re-run on GRACE's full 2002–2023 window as the declared last CAP run (PHASE3_PLAN.md §24):
> the gain shrank to +0.0151, 4 of 5 folds, t = 1.27 — NULL. The fold-0 signature was a
> sample-size artifact. CAP is closed for GRACE.**

**CAP is the substitute for pumping.** Tucson Water began shifting off groundwater onto Colorado
River water in 2001; when CAP delivery is high, pumping is low. So CAP delivery is a genuine
monthly signal for the thing that is otherwise unobserved — and unlike ADWR pumpage, it is *not*
a calendar template, because deliveries respond to shortage tiers and allocation cuts.

**The plumbing already exists.** [`lake_mead.py`](scripts/phase1/lake_mead.py) already pulls
monthly series from **Reclamation HydroData** via direct CSV URLs
(`https://www.usbr.gov/uc/water/hydrodata/reservoir_data/<site>/csv/<param>.csv`). A CAP-diversion
series from the same source would drop straight into the existing Phase 1 pattern.

Leads to check, in order:

| Source | What to look for | Status |
|---|---|---|
| **Reclamation HydroData / RISE** | A monthly CAP diversion or Mark Wilmer Pumping Plant series. Same endpoint family `lake_mead.py` already uses. | **start here** — plumbing exists |
| **Reclamation decree accounting report** — [`usbr.gov/lc/region/g4000/4200Rpts/DecreeRpt/`](https://www.usbr.gov/lc/region/g4000/4200Rpts/DecreeRpt/) | Monthly diversion tables by contractor (CAP is a contractor). Believed to carry month-by-month tables; **the PDF was too large to fetch, so this is unverified.** | verify |
| **Kyl Center CAP deliveries** — [azwaterblueprint.asu.edu](https://azwaterblueprint.asu.edu/news/central-arizona-project-cap-deliveries) | Every CAP delivery **1985–2024, categorized by use**. Grain not stated on the landing page — could be annual. | verify grain first |
| **Municipal monthly production** (Tucson Water) | Monthly groundwater production by utility. Probably a records request, not a download. | last resort |

**Honest expectation.** CAP is more likely to help **GRACE** than groundwater. GRACE anomaly is a
*storage integral* — it responds to cumulative regional water balance, which a monthly
delivery/substitution series speaks to directly. Groundwater's well-depth target is local and
noisy and has other problems ([M5](#m5-groundwater--target-fixed-now-at-its-data-limited-ceiling-fixed-correctness)).
**Do not treat this as a rescue for groundwater.** It is a probe, and its main value is that it is
cheap: the Reclamation ingest pattern is already written.

### Option C — GLDAS land-surface state as *features*, not a new target `MEASURED — probed 2026-09-12: NULL as a feature; the estimator is the constraint`

> **Probed.** `scripts/phase1/gldas.py` acquired GLDAS-2.1 Noah monthly state 2000–2023 over the
> study box. Its same-month storage change correlates **+0.62** with GRACE's change (slope +0.8 m/m).
> Added as an 8-column block to the shipped model under the real nested tuner: **+0.0210 → +0.0891
> target R², 5 of 5 folds, t = 1.34 — NULL** by the pre-declared rule (PHASE3_PLAN.md §25). But a
> **one-coefficient OLS on GLDAS's storage change alone scores +0.244 out of fold on the same
> blocks** — ten times the shipped model. "Modest and uncertain" below was right about the feature
> test and wrong about why: the signal is large; the 45-feature tree ensemble on 200 rows cannot
> use it. M4's "no solution available at Phase 2" was measured with one estimator class. The open
> item is a model-class experiment (§25), which is an architecture decision.

GRACE measures total water storage (TWS): the monthly *change* is a **fast** part — soil moisture
responding to recent weather — plus a **slow** part — groundwater responding to pumping and
recharge. Option B chases the driver of the slow part. **GLDAS/NLDAS** (NASA land-surface models;
monthly **soil moisture, snow water equivalent, evapotranspiration**, free via the same
`earthaccess` path GRACE already uses) gives an observationally-forced estimate of the *fast* part,
so it is a legitimate **feature** for predicting monthly ΔTWS.

- **This is a new Phase 1 feature script, not an edit to `grace_groundwater.py`** — that script
  only builds the target. GLDAS would join the monthly panel like any other input.
- **Honest expectation: modest and uncertain.** GLDAS soil moisture is partly a smoothed integral
  of precipitation, and precip lags/rolls are already in the panel, so the marginal signal over
  what already exists is the open question. Worth a timeboxed probe; not a sure thing. It informs
  the soil-moisture component of the change, **not** the pumping component — so like Option B it is
  more plausible for GRACE than for the well-depth groundwater model.

**⚠ The related-looking move is a trap: do NOT decompose TWS into "true" groundwater storage in
order to improve prediction.** GWS = TWS − soil moisture − snow − surface water is the standard way
to make `grace_groundwater_anomaly` finally *mean* groundwater (right now it is misnamed — the raw
variable is `lwe_thickness`, total storage). But it cuts against prediction: it strips out the
soil-moisture component, which is exactly the part climate features *can* predict, and leaves the
pure pumping-driven residual, which they cannot. **A better-named target and a better-predicting
model point in opposite directions here.** Decompose only for a correctly-labelled descriptive
layer, never expecting skill.

---

## P6. Documentation understated the GRACE fill `MEASURED` — *now fixed*

**Symptom.** [DATA.md](data/Final/DATA.md) documented two GRACE gaps (pre-mission zeros; the
2017–18 mission changeover). The model was quietly training and scoring on fabricated targets.

**Root cause.** GRACE powers down its accelerometers in low-solar-cycle months, so months are
simply *missing*. The real fill extent: **62 of 288 months are not measurements** — 27
pre-mission zeros plus **33 interpolated months in 23 separate blocks** (2003-06, 2011-01,
2011-06, 2011-12, 2012-05, 2012-10, 2013-03, 2013-08/09, …). Interpolated values are *smooth*,
therefore easy to predict, and were **propping up the score**.

**Solution — applied.** `_MONTHLY_MODEL_SPECS["grace"]["require_real_target"]` now requires
`grace_available == 1` on **both** the target and the lag1 anchor (otherwise the residual is a
change measured against a filled value). Cost: 51 rows. Effect: residual R² **0.1180 → 0.0302**.

**The score went down and that is the correct outcome.** The model's skill was always ~zero;
the fill was hiding it. DATA.md now documents all 23 blocks.

---

## P7. Feature-side data was unaudited `MEASURED` — *audit complete, two bugs fixed*

**Symptom.** None. That is the point — a corrupt feature produces no failing test, no implausible
score, and no warning. `irrigation_total_withdrawal_mgd` shipped for the life of the project as a
number four orders of magnitude too large, and the only trace was a P4 table row describing it as
"some genuine signal."

**Root cause, in two independent parts.**

**(a) Two nodata sentinels were summed as data.** The source matrix
`IR_HUC12_Tot_WD_monthly_2000_2020.csv` encodes missing values as literal numbers:

| sentinel | meaning | prevalence (Arizona regions 14/15) |
|---|---|---|
| `999` | no data for this HUC12 | **73.2% of cells**; 5,179 of 7,558 columns are 999 in every month |
| `888` | no data for this month | 7,133 cells (1.4%), 1,163 columns, never a whole column |

`888` is the subtle one: it is simultaneously the **most common non-zero value** in the file and its
**maximum**, in a variable with 189,456 distinct values. A continuous physical quantity does not
land on exactly 888.000 seven thousand times. Masking both drops the maximum from 888 to 810.8 and
leaves a continuous distribution.

**(b) The spatial filter silently fell through to a national sum.** `irrigation.py` prefers a WBD
shapefile, then a crosswalk, then a fallback that returned **every** column with a `log.warning`.
`data/raw/wbd/` did not exist, so the fallback fired on every run. The fallback's docstring claimed
"the source file is already Arizona-scoped"; it is not — it spans **HUC regions 01–18**, first
column in Maine, last in Oregon. The shipped "eight-county irrigation withdrawal" was a sum over
**87,020 watersheds nationwide**, three-quarters of them nodata sentinels.

**The result was worse than a missing feature.** At 5.43e7 MGD with std/mean **0.014**, it was a
near-constant with a slow drift — i.e. **a disguised time trend**. Both failing targets trend hard,
so it correlated **−0.373** with well depth and **−0.523** with GRACE by trend-matching alone, while
its differenced correlations (+0.065, −0.060) were noise. It looked like the most informative human
pressure in the panel and was measuring nothing.

**Solution — applied.**

1. Both sentinels masked before any arithmetic (`NODATA_SENTINELS = (999, 888)`).
2. `data/raw/wbd/WBDHU12.shp` built from the WBD geodatabases for HUC regions 14 and 15 — 7,558
   polygons, of which **1,133** have centroids inside the eight counties. Centroids are computed in
   an equal-area projection (EPSG:5070), matching the `nclimdiv.py` convention.
3. **The fallback now raises.** A warning nobody reads is not a safeguard; this is the same
   discipline as `_validate_seasonality()` and `_effort_confound_check()`.
4. Magnitude and variability guards: refuse to write above 20,000 MGD (Arizona's *entire* water use
   is ~6,000–7,000 MGD), warn below std/mean 0.10.

| | before | after |
|---|---|---|
| mean | **5.43e7 MGD** | **2,560 MGD** |
| std/mean | 0.014 | **0.622** |
| HUC12 columns summed | 87,020 (national) | **1,133 (eight-county)** |
| cells masked as nodata | 0 | **70.4%** |
| seasonality | none | **June peak, January trough** |

**Downstream effect: almost none, and that is informative.** Only the groundwater model can see
irrigation — NDVI and GRACE dropped it to run past 2020, and surface water and wildfire are
`long_record` (nClimDiv only). Five of six models are **unchanged to four decimal places**.
Groundwater moved skill +0.0091 → **+0.0058** and residual R² −0.0290 → **−0.0474**, which is
**within fold noise** (per-fold Δ `[−0.128, +0.043, +0.008, −0.002, −0.013]`, mean −0.018 against a
fold-Δ sd of 0.065). The meaningful change is in *selection*: the corrupt feature was **never
chosen** by in-fold selection, and the corrected one is now the **#2 feature by importance**. The
model rejected the bug and accepts the fix — and still gets no skill from it, which is the cleanest
available evidence for
[P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured).

**The generalisable problem.** Every audit this project had run was pointed at **targets**. The two
rules in [the rules](#the-three-rules-worth-keeping) — "what is the target a mean *of*?" and "does
the target correlate with physics?" — are equally valid questions about features, and nobody had
asked them.

## The full feature audit `MEASURED` — 2026-09-04

All 31 panel columns across 14 sources, checked for magnitude against a published physical
reference, nodata sentinels, spatial scope, and sign. **Two bugs, one of them live.**

### (1) `public_supply_groundwater_mgd` — the same bug in the same cloned function `FIXED`

`public_supply.py` is a near-clone of `irrigation.py`: same `_filter_huc12s_bbox_fallback`, same
missing `data/raw/wbd/`, so it was also summing **all 87,020 national HUC12 columns** instead of
1,133. It shipped **34,817 MGD — 5.4× Arizona's entire water use across every sector**, and 44× its
own Arizona source.

Unlike irrigation this source is sentinel-free: all 7,558 Arizona columns scan clean (min 0, max
141.2, no 999/888). One bug, not two. Corrected to **761 MGD**, and the script's pre-existing
seasonal check now passes meaningfully (summer 946 > winter 536 MGD).

**Effect: groundwater skill +0.0058 → +0.0086, residual R² −0.0474 → −0.0364 — within fold noise**
(per-fold Δ `[+0.064, −0.020, −0.001, +0.002, +0.010]`, mean +0.011 against a spread of 0.064). All
five other models unchanged to four decimal places.

**The selection result mirrors irrigation's, in the opposite direction.** Corrected irrigation was
*promoted* from never-selected to #2 by importance; corrected public supply was **dropped entirely**
(it had been carrying 0.0691). That is the right call — a national public-supply sum is a near-pure
seasonal template ([P4](#p4-most-inputs-carry-no-monthly-information-measured): 96.8% within-year,
97.6% of it a repeating clock) and the model already has `month_sin`/`month_cos`. **The corrupt
version was being selected and the correct one is not.**

### (2) nClimDiv's missing-value constant was wrong for two of three elements `FIXED (latent)`

`nclimdiv.py` had a single `MISSING = -99.99`. The codes are not uniform:

| element | real code | caught? | what it became |
|---|---|---|---|
| `pdsi` | −99.99 | ✅ | — |
| `tmpc` | **−99.90** | ❌ | (−99.90−32)×5/9 = **−73.28 °C** |
| `pcpn` | **−9.99** | ❌ | −9.99×25.4/30 = **−8.46 mm/day** |

The constant was correct for exactly one of the three, which is why PDSI passed the audit and the
other two did not.

**No model was ever contaminated.** All six affected months were 2026-07 onward and the panel
ceiling is 2025-12 — six months of margin. This was checked rather than assumed: re-running the
script also shifted twelve *in-panel* 2025 months, which looked alarming until the arithmetic ruled
it out. A single missing division carries at least an 11.1% area weight, so blending one sentinel
into the regional mean would move it by **≥9.8 °C**; the observed 2025 shifts are **≤0.11 °C**, i.e.
90× too small. They are nClimDiv's own revisions to recent months, picked up by the re-download —
NOAA revises the last year or so as late station reports arrive. The surface-water model's score
moved by 1.1e-05 as a result, which is that revision and nothing else. But nClimDiv feeds **three of the four working models**
(surface water, wildfire, wildlife) and this would have activated silently the first time any window
advanced. Fixed with per-element codes plus `_validate_physical_bounds()`, which refuses to write
−73 °C or negative rainfall regardless of what the source's convention does next. The MERRA-2
cross-check still passes (r = +0.9992 / +0.9202).

### What passed

| Check | Result |
|---|---|
| `population` vs Census 8-county sums | **0.996× (2000), 1.013× (2020)** — strong validation |
| `mead_pool_elevation` / `mead_storage` | 1042–1214 ft, ≤25.0M af — inside the 895–1229 ft / 26.1M af envelope |
| `nclimdiv_pdsi` outliers >7 | **real** — 1905, 1915, 1941, 1979, 1993, every one a known AZ pluvial |
| temperature, precipitation, NDVI, impervious %, DSCI, water stress | all in physical range |
| Spatial scope of every other script | sound — `region.py` API, or county FIPS in the query (`groundwater_levels`, `water_surface`); `lake_mead` correctly needs no filter |

**Only the two HUC12 clones ever had the silent fallback**, and both are now fixed and fail loud.

---

# Part 2 — Per-model problems

---

## M1. NDVI — healthy, at its instrument floor

**Status: the one unambiguously good model.** Skill **+0.2653**, residual R² **0.5584** — it
explains 56% of the variance in month-to-month NDVI *change*, which is real predictive content,
not an artifact of the lag1 anchor.

**Audited 2026-07-15 — clean `MEASURED`.** The target is `mean(valid MODIS pixels)`, the same
"mean over whoever reported" construction that broke three other models, so it was audited the same
way. Rule 1: reprocessing all 574 HDFs to recover the never-recorded pixel count gives **369,305
valid pixels/month at CV 0.1%** and **corr(pixel turnover, |NDVI jump|) = +0.000** — the roster does
not move, because MOD13A3's monthly compositing removes cloud churn before this script averages.
Rule 2: `precip.shift(1)` **+0.562** (green-up lag), drought **−0.251**, September monsoon peak — all
physical. This is the first audited target that measures what its name claims. Hygiene gaps (no
`n_pixels` diagnostic, no `ndvi_available` flag, a silent 3-month interpolation limit in
`_check_and_fill_gaps`) remain and are worth closing for parity, but none moves the score.

**Remaining constraint.** MODIS Terra launched in **2000-02**. There is no earlier NDVI. The
model cannot be extended backward, only forward (to 2025, ~+18 months `ESTIMATED`).

**Note:** post-2020 it runs on climate, drought, population and reservoir state only — it has
**no human-pressure water inputs at all** out there, because irrigation and public supply end
in 2020. Per [P4](#p4-most-inputs-carry-no-monthly-information-measured) this costs almost nothing, but
it should be stated rather than discovered.

**Action:** this and surface water are the only two models where tuning effort converts into
genuine skill. Optimize these.

---

## M2. Surface water — the target was mostly one regulated river `FIXED`

**Status: fixed, and it is now the best model in the project.** Skill **+0.7075**, residual
R² **0.6822**, level R² **0.6646**. All five folds positive and tight: 0.68, 0.81, 0.68, 0.58,
0.58. Winner is **ElasticNet** — a *linear* model — and train R² (0.672) sits *below* CV R²
(0.682), so there is not merely no overfitting, there is negative overfitting.

**Root cause: `discharge_cfs_mean` was 45% one gage.** Discharge spans four orders of magnitude
across the 130 gages in the region, so a mean of raw cfs is not a regional average — it is
whichever gage is biggest. Gage **09525503** alone supplied **45.3%** of it. That gage is
dam-regulated: its flow is a Bureau of Reclamation release schedule, not weather.

**So the old model was being asked to predict a release schedule from rainfall.** It is a small
wonder it managed skill +0.1469 at all.

**The fix.** Phase 1 now centers each gage on its own long-term mean **in log space** before
averaging ([P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed)). Log, because discharge is
multiplicative — a doubling on a small wash should count the same as a doubling on the Gila. The
result, `discharge_log_anomaly`, is a genuine regional index in which every gage votes equally.

**The feature importances became physical.** The top five are now *all precipitation*:

| Feature | Importance |
|---|---|
| `precipitation_mm_day_anomaly` | 0.328 |
| `precipitation_mm_day` | 0.238 |
| `precipitation_mm_day_roll3` | 0.090 |
| `ndvi_lag1` | 0.089 |
| `precip_x_impervious` | 0.065 |

Rain predicts flow. That is how desert hydrology works, and the old target could not see it
because the dam was drowning it out.

**⚠ Read the skill number carefully — it is not comparable to the old +0.1469.** The *target
changed*, so both R² and the persistence baseline moved. Persistence on the new index scores
**−0.0429** (a log anomaly of flashy desert streams is spiky month-to-month, where the old
dam-smoothed series was easy to copy), and skill is R² minus that. The defensible claims are the
ones that stand on their own: **level R² 0.6646 > 0** and **residual R² 0.6822 > 0**, both
baselines cleared, five for five folds. This is not "surface water got 4.8× better" — it is
**"the old target was largely a dam schedule; the new one is regional streamflow, and regional
streamflow is genuinely predictable from rain."** A better question, better answered.

## The window extension shipped, and it paid `MEASURED`

`water_surface.py` now pulls **1980–2025**: 1.64M daily rows, 73–110 gages/month, **551 monthly
rows** (was 219). This was safe to do *only because the per-gage anomaly target landed first* —
the gage roster of 1980 is not the roster of 2020, and a raw mean across that span would have
amplified the compositional artifact rather than diluted it.

**The trade is real and was measured, not assumed.** Reaching back past 2000 collapses the feature
set to nClimDiv alone: **63 features → 26**, losing `usdm_dsci` (the panel's single highest-signal
input per [P2](#p2-the-usdm-is-a-hard-floor-at-2000-01--and-it-is-your-best-feature-verified)) and
`precip_x_impervious` (a top-5 feature). That is exactly the trade that *failed* for GRACE and
wildfire ([Part 4](#part-4--tried-and-failed-do-not-repeat)), so it could not be assumed to work.

The valid experiment holds the test rows fixed and varies only the training data — comparing the
two models' skill scores directly would be meaningless, because the windows differ and the
persistence baseline moves underneath both. Scored on the **identical 180 test months**
(2002-10 → 2020-12), same target, same folds, same baseline (+0.0433):

| | rows | features | R² | skill |
|---|---|---|---|---|
| SHORT (2002–2020, full feature set) | 219 | 63 | +0.6777 | +0.6344 |
| **LONG (1980–2025, nClimDiv only)** | **551** | **26** | **+0.7955** | **+0.7522** |

**ΔR² = +0.1178.** The 332 extra months buy more than the lost features cost.

**On its own window** the shipped model posts level R² **0.7516**, residual R² **0.7264**, skill
**+0.6830**, winner **Ridge** (still linear), with all five folds tight in 0.70–0.77. The top
features are PDSI drought and precipitation — `nclimdiv_pdsi_roll6`, `nclimdiv_precipitation_mm_day_roll3`,
`nclimdiv_precipitation_mm_day`, `nclimdiv_pdsi_lag1`.

**⚠ The headline skill number went DOWN while the model got better.** Skill fell +0.7075 →
**+0.6830** because persistence on the longer window rises from −0.0429 to **+0.0686**. Skill is
*not comparable across a window change*, for the same reason it is not comparable across a target
change. Quote the R² (0.6646 → **0.7516**) and the fixed-test-row ΔR² (**+0.1178**); treat skill
as a pass/fail verdict, not a magnitude.

**A side effect worth recording.** Un-capping discharge from `MONTHLY_CAPPED_2020_BASES` (it now
runs to 2025, so it is no longer a short series) silently handed NDVI, GRACE and wildfire a
discharge feature block they had never had — they only ever lacked it as an *accident* of the 2020
coverage gap, not by design. It cost wildfire 0.034 skill (+0.5615 → **+0.5271**). Discharge is a
*response*, not a pressure, so the exclusion is now explicit in `_RESPONSE_FEATURE_BASES` and all
three models are back to their prior scores. If feeding discharge into NDVI is worth trying on the
merits, that is its own experiment — it should not arrive as a side effect of a Phase 1 window
change.

---

## M3. Wildfire — the target was not measuring wildfire `FIXED`

**Status: fixed, and the model now works.** On the shipped 1984–2023 window: R² **+0.2819**,
skill **+0.3716**, five folds 0.22 / 0.06 / 0.28 / 0.38 / 0.49. What made it work was the
*target*, not the estimator. Giving the model a real MTBS ignition-date time axis moved R² from
**−0.0338** (worse than a flat line) to **+0.3122** on the original 2002–2023 window; extending to
the 1984 MTBS floor then added **+0.0483 R² on fixed test rows**; and reshaping the loss to Tweedie
for the zero-inflated target changed nothing (it lost the competition). Three interventions, one
lesson, restated below.

The diagnosis below is preserved because the *reasoning* is the reusable part — and because
this document previously got it wrong in a way worth remembering.

**Root cause: the monthly time axis is fabricated.**
[`wildfire_monthly.py:76-79`](scripts/phase1/wildfire_monthly.py#L76-L79) parses the month out of
`DATE_CUR` and takes the year from `FIRE_YEAR` — two different fields stitched into one date.
**`DATE_CUR` is a record-maintenance timestamp, not an ignition date.**

Measured on `data/raw/wildfire/InterAgencyFirePerimeterHistory_*.csv` (4,395 AZ fires):

- **57% of all Arizona fires fall on five calendar days**: Feb 1 (**1,221 fires**), Mar 14 (434),
  Dec 31 (335), Jan 24 (327), Jan 21 (173). No physical process does this. These are ETL
  batch-load dates.
- **The implied fire season is midwinter.** February is the peak month (1,328 fires); January
  (602), March (587) and December (528) follow. The actual AZ fire season — May (184), June
  (138), September (119) — sits at the *bottom*.

**Every correlation with a physical driver has the wrong sign:**

| Driver | Corr. with `wildfire_risk_index` | Expected if this were real fire |
|---|---|---|
| `temperature_2m_c` | **−0.271** | strongly **positive** — fires burn in the heat |
| `precipitation_mm_day` | **+0.087** | strongly **negative** — the monsoon ends fire season |
| `usdm_dsci` (drought) | **−0.025** | strongly **positive** |
| `ndvi` | −0.139 | — |

The target is *anti*-correlated with heat and *positively* correlated with rain. It is a
database-maintenance calendar wearing a fire costume.

**This was documented in the repo the whole time.** [`wildfire.py:22`](scripts/phase1/wildfire.py#L22) —
the *annual* script — states in its own docstring: *"The raw file does not have a monthly date
field, only FIRE_YEAR."* The monthly script manufactured one anyway.

**This voids the previous diagnosis.** This document used to blame a zero-inflated count forced
through squared-error loss, and ranked Tweedie/classification as "the largest untested upside in
the project." **That was wrong.** Tweedie and the classification reframe both fix the *shape of
the target's distribution*; neither can fix its *time axis*. The model is being asked to predict
which month a database row was last edited, and no loss function makes that answerable from
temperature and drought.

It also explains the result that looked paradoxical: **extending the model to 255 rows made it
worse** (0.0041 → −0.0338). More rows of a scrambled time axis is more noise. Sample size was
never the constraint, and neither was the loss.

**R² ≈ 0 is not "the ceiling of the wrong loss." It is the correct score for a target with no
recoverable monthly signal.**

## The fix that shipped `MEASURED`

There is **no ignition date anywhere in `data/raw/wildfire/`** — verified exhaustively. The
perimeter history has no such column; the Operational Data Archives carry only `Create Date` /
`Date Current` / `Polygon Date Time` (all maintenance fields, 2022/2023/2025 only); none of the
three `Public_EventDataArchive_*.gdb` geodatabases has a discovery field in any layer; `IRWINID`
is 53% populated, too sparse to join on. **The parser could not be fixed in place — there was
nothing correct to parse.**

[`wildfire_monthly.py`](scripts/phase1/wildfire_monthly.py) now ingests **MTBS** (Monitoring
Trends in Burn Severity) fire-occurrence points, which publish a true **`ig_date`**:

- 3.5 MB download, no auth: `mtbs_fod_pts_data.zip` from the USGS MTBS composite-data endpoint.
- Filters out `Prescribed Fire` / `Other` (9,462 + 4,310 of 30,613 national records are *not*
  wildfire), clips to the eight-county boundary with the existing `region.filter_points`.
- 326 fires in-region for 2000–2023; **105 more available back to 1984** for a future extension.

**The seasonality is now physically real** — June peak (97 fires), May/July shoulders (65 each),
near-zero November–January. And every driver correlation flipped to its expected sign:

| Driver | Old (`DATE_CUR`) | New (MTBS `ig_date`) | Expected |
|---|---|---|---|
| `temperature_2m_c` | −0.271 | **+0.463** | positive ✅ |
| `precipitation_mm_day` | +0.087 | **−0.114** | negative ✅ |
| `usdm_dsci` | −0.025 | **+0.105** | positive ✅ |

**The feature importances became physical too.** Top features are now `month_cos`,
`temperature_2m_c_roll6`, `precipitation_mm_day_anomaly_lag1`, `temperature_2m_c` — seasonality
and weather, which is what drives fire. The old top two were `population` and `population_roll12`:
a trend line ([P4](#p4-most-inputs-carry-no-monthly-information-measured)).

**A regression guard now ships with the ingest.** `_validate_seasonality()` refuses to write the
CSV if the peak ignition month falls outside April–August. This is the check that would have
caught the bug on day one, and it makes the class of error unrepeatable.

**Trade-off, stated honestly.** MTBS only maps fires above a size threshold (≥1000 acres in the
West), so `fire_count` now means *large* fires, not all ignitions. That is the quantity a
climate-driven index can actually speak to — but it is why **177 of 288 months (61%) are now
zero**, up from 83.

**The loss function — tried, and it lost `MEASURED`.** At ~68% zeros the target is genuinely
zero-inflated, so a Tweedie objective was the obvious next step — the recommendation this document
originally made for the wrong reason, now made for the right one. It was added as a fourth
candidate in the competition (`reg:tweedie`, `tweedie_variance_power` tuned inside each fold) and
**it lost on out-of-fold R²: 0.2895 vs squared-error XGBoost's 0.3381.** The better-shaped loss did
not convert into R². It stays in the competition as a self-selecting candidate — it will win only
if it earns it on a future data change. So the "largest untested upside" this document once claimed
for wildfire is now tested from both directions (wrong reason, then right reason) and is not there:
**the target was the whole story.**

## The 1984 extension shipped, and it paid `MEASURED`

`wildfire_monthly.py` now ingests MTBS back to its **1984** floor, and the Phase 2 spec became a
`long_record` model like surface water: **479 monthly rows (was 255)**, 1984–2023. This was safe
*only because the target is a real ignition date* — the reason the previous window extension
([Part 4](#part-4--tried-and-failed-do-not-repeat)) made wildfire worse was a fabricated `DATE_CUR`
time axis, so more rows were more noise. With a real target, more rows are more signal.

**The trade is the surface-water trade, and it was measured, not assumed.** Reaching before 2000
collapses the feature set to nClimDiv alone: **51 features → 26**, losing `usdm_dsci` and every
MERRA-2 / NDVI / GRACE cross-feature. That is exactly the trade that *failed* for GRACE and for the
old fabricated wildfire target, so it could not be assumed to work. Holding the test rows fixed at
the common 2006–2023 folds and varying only the training data:

| | rows | features | R² |
|---|---|---|---|
| SHORT (2002–2023, full feature set) | 255 | 51 | +0.3618 |
| **LONG (1984–2023, nClimDiv only)** | **479** | **26** | **+0.4100** |

**ΔR² = +0.0483, winning all five folds.** The 224 extra months buy more than the lost features
cost — fire responds to drought, heat and precipitation, which is precisely what nClimDiv
(PDSI/temp/precip, 1895+) carries.

**On its own window** the shipped model posts R² **0.2819**, skill **+0.3716**, winner **XGBoost**
(squared-error), five folds 0.22 / 0.06 / 0.28 / 0.38 / 0.49. The top features stayed physical:
`month_cos`, then `nclimdiv_temperature_c_lag1`, `nclimdiv_temperature_c_roll6`,
`nclimdiv_precipitation_mm_day_roll3` — seasonality and weather, which is what drives fire.

**⚠ The headline CV R² went DOWN while the model got better.** R² fell 0.3122 → 0.2819 and skill
fell +0.5615 → +0.3716, both because the pre-2000 folds are harder (68% zero months, and MTBS maps
fewer large fires in 1984–99) and persistence rose from −0.25 to −0.09. Skill is *not comparable
across a window change*. Quote the fixed-test-row ΔR² (**+0.0483**); treat skill as a pass/fail
verdict, not a magnitude — the same rule surface water established.

**Unaffected: the annual model.** [`wildfire.py`](scripts/phase1/wildfire.py) keys off
`FIRE_YEAR`, a real field, and never touched `DATE_CUR`. Only the monthly grain was compromised.

**Minor, still open `MEASURED`.** `wildfire_risk_index = 0.5·minmax(fire_count) +
0.5·minmax(log_acres)`, min-max normalized **over the full series**, so the target's scale depends
on the series maximum — the same class of look-ahead as the `consecutive_dry_years` bug. Min-max
is a linear transform, so it does not change any R²; this is hygiene, not a score bug. The 50/50
blend is also arbitrary. Both were left alone to keep the timestamp fix isolated.

**Hard limit, still true.** Ignition is stochastic. No model predicts *when* a fire starts — only
the seasonal and anthropogenic propensity for one. R² 0.31 on monthly large-fire activity is a
good result; do not expect it to climb much further.

---

## M4. GRACE — data-limited, at its instrument floor

**Status: no skill.** Skill **−0.0349**, residual R² **0.0302**.

**Audited 2026-07-15 — the target is sound; the failure is data, not a target bug `MEASURED`.**
GRACE went through the same two rules that convicted four other targets, and it passed. Rule 1 (does
the averaging denominator move?): the anomaly is a spatial mean over the ~24 GRACE cells in the bbox;
reprocessing the nc4 gives ~20 valid cells/month, **constant within each mission** (the count changes
once in 226 months, at the GRACE→GRACE-FO handover) — no roster churn. Rule 2: the series carries a
**strong, highly significant secular decline** (trend r = −0.84, p = 2e-61 — the Arizona depletion
signal) and co-moves correctly with drought (usdm −0.22) and Lake Mead (+0.75). So the target
measures regional storage; it is **not** artifactual the way `DATE_CUR` or the roster means were.
**This confirms [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured) rather than
overturning it:** GRACE fails because monthly change in a storage integral is driven by unobserved
pumping, not because the target is wrong. Two honest caveats: at ~300 km resolution the "regional"
mean is **leakage-smeared** (a coarse proxy, not a clean eight-county measurement), and it correlates
**+0.71 with the well-depth anomaly where physics wants a negative sign** — most likely the
GRACE-vs-local-wells scale mismatch (managed recharge has raised water tables in the monitored AMA
wells while regional storage falls), not a target defect. **Do not keep hunting for a GRACE target
fix; there is not one to find.**

**Three separate corrections landed on this model, and the score fell each time** — each time
because the previous number was flattering:

1. **Tuning optimism.** The original 0.5200 came from picking the best of 60 hyperparameter
   draws scored on the folds that were then reported. Nested CV → 0.4047.
2. **Fabricated targets.** 33 fill months were being trained and scored on
   ([P6](#p6-documentation-understated-the-grace-fill-measured--now-fixed)). Removing them → 0.3300.
3. **The window extension didn't help.** 36 real rows gained, 51 fill rows removed. Net −15.

**Root cause: [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured).** GRACE anomaly is a *storage integral* —
it responds to cumulative water balance, and the monthly pumping that drives that balance is not
observed. It is also at its instrument floor (2002-04), so it gains nothing from
[P1](#p1-phase-1-throws-away-decades-of-available-data-verified).

**The irrigation trade was re-tested on 2026-09-04 and the answer is: keep the 2023 window
`MEASURED`.** GRACE dropped irrigation and the other 2020-capped features to reach 2023, and that
decision was made when irrigation was
[99.9% nodata sentinel](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed). With
a correct feature the trade was measured properly — identical test rows, 168 common months, five
folds, training strictly before each test fold, varying only the feature set:

| variant | window | features | R² level | R² target | folds won vs shipped |
|---|---|---|---|---|---|
| **A — shipped** | 2002–2023 | 45 | +0.4479 | +0.0528 | — |
| B — all 2020-capped features | 2002–2020 | 57 | +0.4646 | +0.0787 (**+0.0259**, t=1.13) | 3/5 |
| C — irrigation only | 2002–2020 | 49 | +0.4541 | +0.0626 (**+0.0098**, t=0.57) | 2/5 |

Persistence on those identical rows is **+0.4369**, so even the shipped model's skill on this subset
is only **+0.011**. Neither variant is significant; the biggest effect is a tenth of the
fold-to-fold spread. **Do not move GRACE's window, and do not add irrigation to it.**

**Solution: none available at Phase 2.** Do not keep searching for hyperparameters. Either
acquire monthly pumping data or ship this as a descriptive/diagnostic layer rather than a
predictive model.

> **Superseded 2026-09-12 — there is a solution at Phase 2, and it is not tuning.** The sentence
> above was measured with one estimator class. PHASE3_PLAN.md §25–§26: GLDAS's land-surface
> storage change explains most of GRACE's monthly change (r = +0.62, slope +0.8 m/m), and a **ridge
> regression on four physical inputs** scores **+0.2845 target R² / +0.5664 level R² / skill +0.2015**
> under the same nested folds where the shipped XGBoost scores +0.0210 / +0.3372 / −0.0277 — 4 of 5
> folds, t = 2.52, REAL by the pre-declared rule. The tree ensemble on ~200 rows was the floor.
> The pumping residual is still unobserved; what the linear model captures is the part of the
> change that is *not* pumping. Deployment is an architecture change and is listed in §26.

**Two live leads, both Phase 1 data-acquisition, neither inside this model's script.** (1) **CAP
deliveries** (work item **#5**) — a monthly proxy for the Colorado-River-water *substitution* that
offsets pumping, which a storage integral can use directly; the Reclamation HydroData ingest pattern
already exists in [`lake_mead.py`](scripts/phase1/lake_mead.py). (2) **GLDAS soil moisture** as a
feature (work item **#7**) — informs the fast, weather-driven component of monthly ΔTWS. Both are
probes with modest, honest expectations, not rescues. See
[P5 Option B](#option-b--cap-deliveries-as-a-monthly-pumping-proxy-untested--worth-one-probe) and
[Option C](#option-c--gldas-land-surface-state-as-features-not-a-new-target-untested).

---

## M5. Groundwater — target fixed; now at its data-limited ceiling `FIXED (correctness)`

**Status: no skill, but no longer harmful.** Skill **+0.0086** (was −0.0791), residual
R² **−0.0364** (was −0.3179). *(The +0.0107 / −0.0259 first reported for the anomaly-target fix
became +0.0091 / −0.0290 on re-run, then +0.0058 / −0.0474 after the irrigation fix and these
after the public-supply fix
([P7](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed)); all four sit inside
fold noise of each other.)* It no longer predicts month-to-month change *worse than predicting
no change*, which is where it started.

**Two stacked problems, in order. The first is now fixed; the second is not fixable here.**

1. ~~**The target is a compositional artifact**~~ **FIXED.** Per-well anomaly index shipped
   ([P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed)). The churn was worse than this
   document estimated — turnover correlated **+0.746** with the month-to-month jump, and the raw
   target's standard deviation (27.8 ft) was ~5× the anomaly's (5.9 ft).
2. **It then hits [P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured)** and lands at ~zero, exactly as
   predicted. This was a *correctness* fix, not a performance fix, and it was always going to be.

**This is done. Stop tuning it.** The prediction in this document — "lands at approximately zero
skill and stays there until monthly pumping data exists" — was correct, and the measurement
confirms it. Extending to 1980–2025 via [P1](#p1-phase-1-throws-away-decades-of-available-data-verified)
is now *safe* (the target fix landed first, which was the requirement) but there is no reason to
expect it to buy skill: more rows of an unobserved driver is still an unobserved driver.

**Nothing available moves this model.** ADWR pumpage — which this document used to name as the fix — is **annual and AMA-only**, and annual data cannot move a monthly residual model ([P5](#adwr-is-not-the-answer-this-document-used-to-say-it-was-verified)). The one remaining monthly proxy, CAP deliveries (work item **#6**), is aimed at GRACE and is **not** expected to rescue groundwater. Leave this model where it is.

---

## M6. Wildlife — the target was a survey-effort index `FIXED`

**Status: fixed, and it works.** Nested LOO R² **+0.2125** (was −0.2108 — *worse than a flat
line*), skill **+0.2046**, Spearman ρ **0.5068**. Under a 20,000-shuffle **permutation test**,
p = **0.0001** against a null whose 95th-percentile |ρ| is only 0.266. n went from **20 to 56**.

**Root cause: `total_abundance` was never a measure of bird abundance.** It is a *sum over
whichever routes were surveyed that year*, and the survey network is not constant — 15 to 26
routes/year in the region, and statewide the BBS ran ~15 routes/year in the 1970s-80s versus ~45
from 1990 onward. So the target rose and fell with **survey effort**.

Measured on the old 2000–2024 output:

| | correlation with `route_count` |
|---|---|
| `total_abundance` | **+0.840** (p = 2.8e-07) |
| `abundance_index` (the target) | **+0.861** (p = 6.6e-08) |
| `species_richness` | **+0.866** (p = 4.8e-08) |

Over the full 1968–2024 record it is **+0.942**. The model was being asked to predict *how many
people went birdwatching* from temperature and drought. Like the wildfire target, it is
physically unlearnable — and for the same underlying reason as groundwater
([P3](#p3-compositional-churn-in-station-averaged-targets-measured--now-fixed)): a mean or sum
over a churning station roster measures the roster.

**The fix.** `abundance_anomaly` — each route centered on its *own* long-term mean log abundance
before averaging, so a year with more routes is not automatically a bigger year, and a productive
riparian route and a sparse desert route count equally. Effort correlation drops to **−0.118**. A
`_effort_confound_check()` guard now refuses to write a target that exceeds ±0.6, making this
class of error unrepeatable — the same role `_validate_seasonality()` plays for wildfire.

**The real signal survives the correction.** Per-route bird abundance declines **~23% across the
record (p < 0.0001)**. The decline was always genuine; the old target simply conflated it with
routes being dropped.

## The diagnosis in this document was wrong `MEASURED`

It said: *"Root cause: p ≫ n. 22 features, 20 rows,"* and PHASE2_REPORT.md said *"No search will
rescue this."* **Both are voided.** Fixing the target alone, at **n=20 unchanged and 22 features
unchanged**, moved LOO R² from −0.2108 to **+0.2661** and Spearman ρ from 0.2159 (p = 0.36) to
0.5278 (**p = 0.017**). Sample size was never the binding constraint. The target was.

That is now **four for four**: every model this project has fixed was fixed by repairing what the
target measured, not by changing the estimator.

## Then the extension, which is what makes it *significant* `MEASURED`

`nclimdiv.py` now ingests NOAA nClimDiv — PDSI, temperature and precipitation by Arizona climate
division, **1895–2026**, free and unauthenticated, area-weighted to the region by true polygon
intersection in an equal-area projection (Southeast 42.6%, South Central 33.6%, Southwest 12.7%,
East Central 11.1%). It cross-checks itself against MERRA-2 on the 288-month overlap and refuses
to write if it disagrees: **temperature r = +0.9992, precipitation r = +0.9202.**

That unlocked the BBS record back to **1968**, its true floor:

| | before | after |
|---|---|---|
| rows | 20 | **56** |
| features | 22 | **12** |
| LOO R² | −0.2108 | **+0.2125** |
| Spearman ρ | 0.2159 (p = 0.36) | **0.5068 (permutation p = 0.0001)** |

Note LOO R² is *slightly lower* than the target fix alone at n=20 (0.2661), and that is the
honest outcome: n=56 forces dropping NDVI, GRACE, USDM and every other 2000-floored input, so the
model loses NDVI as a food-availability proxy — genuinely the mechanism you would most want for
birds. **What it buys is significance.** ρ = 0.53 at n=20 (p = 0.017) is a suggestive result;
ρ = 0.51 at n=56 (p = 0.0001) is a real one.

**The feature importances are physically right.** Behind the autoregressive lag, the top driver is
**prior-year precipitation**, then PDSI drought and summer PDSI. Desert bird productivity tracking
*last* year's rain — through vegetation and insect abundance — is exactly the known ecology, and
it is not something a model reading a survey roster could ever have found.

## The NDVI trade-off was tested, and the rows win decisively `MEASURED`

Reaching 1968 costs NDVI: MODIS Terra launched 2000-02, and one NDVI column in `X` makes the
`notna().all()` mask delete every pre-2000 row. So the model must choose **n=56 without NDVI** or
**n=23 with it**. The mechanism argued for NDVI — the top non-autoregressive driver is prior-year
precipitation, but the causal chain is rain → vegetation → food → birds, and NDVI *is* the middle
link measured directly.

Pre-registered design, all three variants scored on the **identical 23 common years** (2000–2023,
2020 dropped), nested LOO reusing `model_wildlife._fit_predict`, decision rule the permutation test
on Spearman ρ rather than R²:

| variant | window | n | feat | LOO R² | ρ | perm p | null 95th |ρ| |
|---|---|---|---|---|---|---|---|
| **A — shipped** | 1968–2024 | 56 | 12 | **+0.0956** | **+0.409** | 0.054 | 0.414 |
| B — short, no NDVI | 2000–2023 | 23 | 12 | −0.1702 | +0.110 | 0.618 | 0.416 |
| C — short, **+ NDVI** | 2000–2023 | 23 | 15 | −0.2107 | +0.150 | 0.495 | 0.416 |

**Primary test (C vs B — NDVI's contribution, same rows and window): ΔR² −0.041, Δρ +0.041,
p = 0.495.** Nothing.

**But the real finding is B.** Cutting to 23 years *without changing a single feature* takes the
model from R² +0.0956 to **−0.1702** — below a flat line — and ρ from 0.409 to 0.110, well under its
own null 95th percentile of 0.416. **Both short-window variants are indistinguishable from noise.**

So the question "does NDVI help?" turns out to be unanswerable rather than answered: at n=23 there
is no working model for NDVI to improve. The 33 rows are doing all of the work, and no feature
recovers them. **Keep the 1968 window; do not add NDVI.** The mechanism was never refuted — it was
swamped, and the only configuration that could test it properly (long record *and* NDVI) cannot
exist, because the satellite does not go back to 1968.

*(A scores ρ = 0.409 / p = 0.054 here against 0.5068 / p = 0.0001 in the shipped model. That is not
a discrepancy: this is a 23-point subset of its 56, and 23 points carry far less power. The shipped
number remains the headline.)*

---

# Part 3 — Recommended order of work

Ranked by expected gain per unit of effort. `nclimdiv.py` and the per-station anomaly targets have
landed, which unblocked everything below them.

| # | Action | Unblocks | Expected gain |
|---|--------|----------|---------------|
| ~~**0**~~ | ~~**Wildfire: rebuild the target on a real ignition date**~~ | M3 | ✅ **DONE** — MTBS `ig_date`. R² −0.0338 → **+0.3122**, skill **+0.5615** |
| ~~**0b**~~ | ~~**Per-station anomaly targets (groundwater + surface water)**~~ | P3, M2, M5 | ✅ **DONE** — groundwater skill −0.0791 → **+0.0107** (no longer harmful); surface water → **+0.7075**, residual R² **0.6822** |
| ~~**0c**~~ | ~~**Wildlife: effort-corrected target + extend BBS to 1968**~~ | M6 | ✅ **DONE** — LOO R² −0.2108 → **+0.2125**, n 20 → **56**, ρ 0.51 (permutation p = 0.0001) |
| ~~**0d**~~ | ~~**`nclimdiv.py`** — PDSI/temp/precip, AZ climate divisions, 1895–2026~~ | P1, P2, M2, M6 | ✅ **DONE** — validated against MERRA-2 (r = 0.999 / 0.920) |
| ~~**0e**~~ | ~~**Surface water: extend to 1980–2025**~~ | M2, P1 | ✅ **DONE** — n 219 → **551**. On identical test rows, R² **+0.6777 → +0.7955** |
| ~~**1**~~ | ~~**Wildfire: Tweedie/hurdle loss**~~ | M3 | ✅ **DONE — and it lost.** Tweedie competes for the zero-inflated target but scores 0.2895 vs squared-error XGBoost's 0.3381 out-of-fold. Kept as a self-selecting candidate |
| ~~**2**~~ | ~~**Wildfire: extend to 1984**~~ | M3, P1 | ✅ **DONE** — n 255 → **479**. On identical test rows, R² **+0.3618 → +0.4100 (ΔR² +0.0483)** |
| ~~**2b**~~ | ~~**Fix the irrigation feature** (sentinels + national-scope sum)~~ | P7, P4, P5 | ✅ **DONE** — 5.43e7 → **2,560 MGD**. Five models unchanged; groundwater within noise. Revised P4 and P5 |
| ~~**3**~~ | ~~**GRACE: re-test the irrigation trade now the feature is correct**~~ | P5, M4 | ✅ **DONE — null.** On identical test rows, irrigation adds **+0.0098** target R² (2/5 folds, t=0.57); the whole 2020-capped block adds **+0.0259** (3/5 folds, t=1.13). Both ≈0.1× the fold-to-fold spread. **Keep the 2023 window.** See [M4](#m4-grace--data-limited-at-its-instrument-floor) |
| ~~**4**~~ | ~~**Wildlife: revisit the NDVI trade-off**~~ | M6 | ✅ **DONE — null, and the rows win decisively.** On identical rows NDVI adds Δρ +0.041 (p=0.495); but cutting to n=23 alone drops R² **+0.0956 → −0.1702**, below a flat line. **Keep the 1968 window.** See [M6](#m6-wildlife--the-target-was-a-survey-effort-index-fixed) |
| ~~**4b**~~ | ~~**Audit the remaining features** the way targets were audited~~ | P7 | ✅ **DONE — all 31 columns.** Two bugs: public supply was the same national sum (34,817 → **761 MGD**); nClimDiv's `MISSING` constant matched 1 of 3 elements, leaking **−73.28 °C** and **−8.46 mm/day** (latent, outside the panel). Everything else passed — population validates to Census within 1.3% |
| ~~**4c**~~ | ~~Audit the NDVI and GRACE targets~~ | — | ✅ **DONE — both clean.** NDVI: pixel turnover↔jump **0.000**, signs all right. GRACE: no cell churn, strong depletion trend (r **−0.84**), right drought/Mead signs — target sound, failure is data ([P5](#p5-monthly-pumping-is-weakly-observed-not-unobserved-measured)), not a target bug |
| ~~**5**~~ | ~~**Probe CAP monthly deliveries via Reclamation HydroData**~~ | P5, M4 | ✅ **DONE 2026-09-12 — NULL for GRACE** (5/5 folds, t = 1.52; PHASE3_PLAN.md §23), but the series measured the Lake Mead lever's first link at 1.0. Not via HydroData: CAP's own reports; Reclamation's diversion is winter-weighted by Lake Pleasant. Original text: the one remaining *monthly* pumping proxy. Cheap — `lake_mead.py` already ingests from this endpoint. Aimed at **GRACE**, not groundwater |
| ~~**6**~~ | ~~**Acquire ADWR monthly pumping**~~ | — | ❌ **DEAD.** ADWR pumpage is **annual and AMA-only**; annual data cannot move a monthly residual model. See [P5](#adwr-is-not-the-answer-this-document-used-to-say-it-was-verified) |
| ~~**7**~~ | ~~**Probe GLDAS soil moisture as a GRACE feature**~~ | P5, M4 | ✅ **DONE 2026-09-12 — NULL as a feature block (t=1.34), and the important result is elsewhere:** a one-coefficient OLS on GLDAS storage change out-scores the shipped 45-feature model ten-fold out of fold. PHASE3_PLAN.md §25. Original text: informs the fast (weather-driven) part of monthly ΔTWS; new `earthaccess` feature script. Modest, uncertain — precip lags may already carry it. Do **not** decompose TWS→GWS expecting skill ([P5 Option C](#option-c--gldas-land-surface-state-as-features-not-a-new-target-untested)) |

**Do not:** tune GRACE. Tune groundwater — its target is fixed and it is at its data-limited
ceiling. Chase surface water's skill number as if it were comparable to the old one
([M2](#m2-surface-water--the-target-was-mostly-one-regulated-river-fixed) explains why it is not).

**Audit the remaining targets the way wildfire and surface water got audited.** Both bugs were
found by asking whether the target measured the thing it was named after, and both times the
answer was no — a database edit date, and a dam release schedule. **That question has now paid for
itself twice; when finally asked of NDVI and GRACE (2026-07-15) both came back clean — the targets
were sound.** So the score is four target bugs found, two targets audited-clean: the check is cheap
and worth running, but it is no longer true that it "has never once come back clean."

---

# Part 4 — Tried and failed. Do not repeat.

Recorded so nobody spends the effort twice.

- **Trading wildlife's 1968 record for NDVI.** Mechanistically well-motivated — NDVI is the
  vegetation link that prior-year precipitation only proxies — and it failed twice over. NDVI adds
  Δρ +0.041 (p = 0.495) on identical rows, and the window cut it would require takes the model from
  R² +0.0956 to **−0.1702** on its own. At n=23 there is no model left for a feature to improve.
  **Sample size is the binding constraint here, which is the opposite of what was true for the
  target bug** — the target fix worked at n=20 unchanged, but the *feature* question needs rows.
- **Giving GRACE the corrected irrigation feature, at the cost of its 2023 window.** Motivated by a
  real, correctly-signed, significant partial correlation (**−0.260**, p<0.001, calendar removed)
  and it still failed: on identical test rows irrigation adds **+0.0098** target R² and wins 2/5
  folds; the entire 2020-capped block adds **+0.0259** and wins 3/5. Neither is distinguishable from
  noise. The design gave variant A *none* of its 36-row advantage (those months fall at the end of
  the record and never enter training for the common folds), so this is a null under conditions
  that favoured the challenger. **A significant correlation is not a usable feature.**
- **Retraining GRACE on the whole deseasonalized human block, under its own nested tuner.** The
  strongest-looking case in the panel: on one fixed XGBoost config the block took GRACE's target R²
  from −0.0119 to +0.0539 and its level R² from +0.4121 to +0.4462. Re-run with
  `model_grace.py`'s real `RandomizedSearchCV(n_iter=60)` inside every fold, on identical rows, it
  is **+0.0394 → +0.0960 target R², a +0.0566 gain that wins 4 of 5 folds at t = +1.21** — below
  the t ≥ 2.0 bar declared before the run. **Tuning is not what kills it, and that is the part
  worth remembering:** the real search lifted the shipped arm +0.0513 and the human arm +0.0420, so
  the gap survived nearly intact. What kills it is that **83% of the difference is fold 0**, the
  28-training-row fold where both arms are worse than predicting zero and the block is merely less
  catastrophic. Across the other four folds it is +0.0119 at t = +0.69. The block buys
  *robustness when starved of rows*, not skill — worthless for a model that deploys on all 204 —
  and buying it means re-accepting the 2020-12 cap GRACE deliberately traded away for 36 months.
  **A gain carried by one small fold is a sample-size effect wearing a feature-set costume.**
  See `scripts/phase2/experiment_grace_nested.py`, PHASE3_PLAN.md §11.5.
- **Fixing the irrigation feature as a route to skill.** Correct and worth doing — it was 99.9%
  nodata summed over the entire United States — but it moved no model. Five of six are unchanged to
  four decimal places; groundwater moved within fold noise while *promoting* the feature from
  never-selected to #2 by importance. **Correctness fix, not a performance fix.**
- **Long-window cumulative features for groundwater** (roll24, roll36, cum60 of irrigation and
  public supply, on the theory that an aquifer integrates pumping over years). No effect: skill
  +0.0050 → +0.0047 at a 1-month horizon, and actively harmful at 12 months. Reason:
  [P4](#p4-most-inputs-carry-no-monthly-information-measured) — those inputs have no month-specific
  content to accumulate.
- **A 12-month prediction horizon for groundwater.** Looks like a win (+0.22 skill) and is not:
  persistence collapses to −0.83 over that horizon, so the model beats a strawman while its own
  R² is −0.61 — worse than a flat line.
- **In-fold feature selection as a score fix.** The leak was real and is fixed, but it was not
  inflating anything: selecting inside the fold scored identically for groundwater and slightly
  *better* for surface water.
- **Extending the window to rescue GRACE or wildfire.** Done, and it did not. Wildfire got 36
  more rows and fell from R² 0.0041 to −0.0338; GRACE's skill went −0.0072 → −0.0349 once its
  fabricated targets were removed. **Sample size was never the binding constraint on either.**
  The extension helped NDVI, which was already the only healthy monthly model.
  *(For wildfire we now know why the extra rows actively hurt **then**: the month labels were
  meaningless, so more rows were more noise. Once the target became a real MTBS ignition date,
  re-extending to 1984 **helped** — +0.0483 R² on fixed test rows — which is the cleanest possible
  confirmation that the target, not the sample size, was always the issue. See
  [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed).)*
- **Blaming wildfire's loss function.** This document once ranked a Tweedie/classification
  reframe as "the largest untested upside in the project." The zero-inflation was real, but it
  was not the binding problem: the month labels came from a database edit date. Fixing the
  *target* alone — loss function untouched — moved R² from **−0.0338 to +0.3122**. Tweedie was
  then tried on the working model and **lost** (0.2895 vs 0.3381 out-of-fold, see
  [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed)): the zero-inflation was real but
  it was never the binding problem, and reshaping the loss did not convert into R².
  **Diagnose the target before the estimator.** The correlation signs (temperature **−0.27**,
  precipitation **+0.09**) would have caught this at any point and cost one query.
- **Pushing the monthly start back to 2000-01.** NDVI skill +0.2453 vs **+0.2653** for the
  2002-10 start. It forces dropping the GRACE features, which are worth more than the 23 extra
  rows. **Keep the 2002-10 floor.**

---

# Appendix — Evaluation problems, already fixed

These were harness bugs, not model bugs. All are fixed; listed so the old numbers in git history
are interpretable.

1. **Residual models were scored only on the reconstructed level.** The lag1 anchor dominates
   that score, so a model with zero or negative skill still looked strong. Every model now also
   reports R² on the target it was actually trained on.
2. **Hyperparameters were selected on the folds that were then reported.** `RandomizedSearchCV`
   searched 60 draws over the same `TimeSeriesSplit` used for the final score — a max over noisy
   estimates. All scoring is now nested. Impact: NDVI −0.009 (it was fine), GRACE **−0.120**
   (it was not).
3. **Model choice and feature selection also touched the test folds.** `_select_features` fit
   XGBoost on the full panel before CV, and the competition scripts picked their winner with
   `max(candidates, key=cv_r2)` on the reported folds. Both now happen inside `_fit_predict`.
4. **`export.py` listed only four of the six models.** `MODEL_IDS` omitted `groundwater` and
   `surface_water`, so their report rows were hand-written with a blank baseline column. That is
   the specific reason a model that loses to persistence shipped as a headline result.
5. **`consecutive_dry_years` had look-ahead.** It computed its dry/wet threshold from a
   full-series median, so a given year's "is this a dry year" flag depended on precipitation that
   had not happened yet. Now expanding.
6. **One global modeling window.** All six models shared a 2002-10 → 2020-12 window and one
   feature list, so a single dead column truncated models whose targets ran to 2023-12. Models
   now declare their own window and features in `_MONTHLY_MODEL_SPECS`.
7. **`generate_stats.py` dropped zeros before computing percentiles.** Sensible for population or
   discharge, where a zero is a missing reading — but the corrected wildfire target is zero in
   **177 of 288 months**, and those zeros are the *most common real observation*. The frontend was
   therefore about to show a resting wildfire risk of **0.41** (the median of fire months only)
   instead of **0.0**. Zero-dropping is now opt-out per variable via `zero_is_data`; only wildfire
   sets it, so every other series is unchanged. *Latent bug, activated by the [M3](#m3-wildfire--the-target-was-not-measuring-wildfire-fixed) fix.*

---

## P8. The region was never the one the documents named `MEASURED` — *kept, on purpose*

**Symptom.** None, again. `scripts/phase1/region.py` declares `COUNTIES` by name and `COUNTY_FIPS`
by code, and every filter in the pipeline runs on the code. Two codes did not match the names beside
them:

```
"04013",  # Graham    -> 04013 is MARICOPA. Graham is 04009.
"04007",  # La Paz    -> 04007 is GILA.     La Paz is 04012.
```

So every county-filtered series — population, irrigation and public-supply HUC12s, wells, gages,
DSCI, nClimDiv, BBS routes, the region area, the NDVI-endpoint cutline — was built over **Pima,
Pinal, Santa Cruz, Cochise, Maricopa, Greenlee, Yuma and Gila**, Phoenix included, while every
document said Graham and La Paz. Found 2026-09-11 while adding a boundary argument to
`ndvi_endpoints.py`; measured by
[`scripts/phase3/region_variants.py`](scripts/phase3/region_variants.py) → `model/region_variants.json`,
which rebuilds what can be rebuilt from disk under three county sets and changes nothing:

| quantity | as shipped (Maricopa + Gila) | minus Maricopa | documented (Graham + La Paz) |
|---|---|---|---|
| county area, acres | 27,780,270 | 21,875,628 | 24,663,508 |
| population, 2020 | **6,362,198** | 1,917,139 | **1,919,003** |
| HUC12s with data | 1,131 | 899 | 992 |
| irrigation, MGD | 2,526 | 1,676 | 2,361 |
| mean NDVI | 0.2437 | 0.2518 | 0.2285 |
| mean impervious, % | 3.173 | 1.772 | 1.594 |
| `ndvi_impervious` DiD slope | −0.0356 | −0.0217 | −0.0189 |
| `ndvi_irrigated` DiD slope | +0.3295 | +0.2100 | +0.3140 |

Population is the number that matters: the shipped series is **3.3× the documented region's**,
roughly 70% Phoenix. Irrigation is nearly a wash, because La Paz and Graham are irrigation-heavy
(97,139 and 46,682 irrigated acres in 2017) and between them roughly offset Maricopa's 180,214.

**Decision: the as-shipped region is kept, and the documents are corrected to it.** Not because
re-filtering is expensive — though it is: wells and gages are pulled from NWIS per county code and
the cached pulls cover only these eight, so the documented set needs new acquisitions and a full
Phase 1 → 2 → 3 rerun — but because the as-shipped set is the better-posed region for what the
application models. With Maricopa in, the region contains **all three CAP Active Management Areas**
(Phoenix, Pinal, Tucson), which is the unit the Lake Mead shortage tiers act on. The ADWR–CAP joint
shortage statement puts Arizona's reduction as *"borne almost entirely by the CAP system"*, and CAP
delivers only to Maricopa, Pinal and Pima. A region that excludes Maricopa excludes CAP's largest
customer and then has to guess what share of the cut lands inside it — which is exactly what
`region_share_of_az_reduction = 0.60` was doing.

Two things are recorded against the decision so it does not read as a rationalisation:

- **The region was selected by a typo, not by design**, and Gila County in particular is incidental
  — mountainous, 54,000 people, 1,296 irrigated acres, no AMA. It stays because removing it is a
  rerun for no modelling gain; it contributes forest NDVI and Tonto-basin fires to the means.
- **The name "Edge of the Desert" and the "southern Arizona" framing now overstate the border
  focus.** The region is central *and* southern Arizona: the CAP counties plus the border counties.

**What was actually wrong, and is now fixed** (2026-09-12):

1. **`irrigated_fraction` mixed the two regions.** Its numerator summed the NASS 2017 irrigated
   acres for the eight *named* counties (681,143) and divided by the area of the eight *selected*
   ones. Same NASS table, right counties: **718,832 acres, fraction 2.588 %** (was 2.452 %). It is
   also the calibration that turns HUC12 withdrawal into irrigated area in `ndvi_endpoints.py`, so
   `ndvi_irrigated_crop` was re-measured — see PHASE3_PLAN.md §22. To first order the irrigation →
   NDVI lever does not move: the lever is `irrigated_fraction × slope`, the fraction rose 5.5 % and
   the slope, calibrated on the same acres, fell by the same factor.
2. **`region_share_of_az_reduction` was reasoned from the wrong map.** Its source string said
   "Pinal and Pima are in-region and Maricopa is not". Re-derived at **0.95, band 0.85–1.00**, still
   `UNTESTED`: the residual is 4th-priority on-river water outside the CAP system. Every Lake Mead
   path scales by 0.95 / 0.60 = 1.58×.
3. **`water_stress.py` weighted Maricopa's DSCI by Graham's area and Gila's by La Paz's.** The
   drought index fetched the right counties (it filters on the code) but weighted 04013 at 4,641 sq
   mi instead of 9,226 and 04007 at 4,513 instead of 4,795. Re-fetching all eight counties from the
   USDM API and re-weighting **reproduces the shipped column exactly** under the wrong weights, and
   under the right ones moves it by **r = 0.99916** on levels, **0.9984** on month-to-month change,
   mean +0.42 DSCI on a series with sd 112. The weights are corrected in the script;
   `data/Final/water_stress_monthly.csv` is **not regenerated**, because `usdm_dsci` is a feature in
   GRACE (6 features), NDVI (6) and groundwater (1) and regenerating it means retraining all three
   for a 0.08 % change in correlation. `FIXED (latent)`, same status as the nClimDiv constant in P7.
   Whoever next retrains Phase 2 should regenerate it first.
4. **Names, everywhere.** `region.py`, the config mirror, README, PHASE1_SETUP, DATA.md,
   PHASE3_PARAMS and the AMA-coverage list in P5 above.

**What is not changed.** No model is retrained, no Phase 1 series is regenerated, and no Phase 3
calibration other than the two constants above and the NDVI endpoint that depends on one of them.
Every measured number in PHASE2_REPORT.md and PHASE3_PLAN.md was always a measurement on *this*
region; only the caption was wrong.

**One caveat that survives either decision.** The raster series — NDVI, impervious cover, MERRA-2
temperature and precipitation, GRACE — clip to the bounding box, not to the counties. The box
(−114.81, 31.33, −109.05, 34.5) covers all ten counties in play plus a sliver of Yavapai, so the
climate inputs and the NDVI target were always a slightly larger footprint than the human inputs,
under both readings of the county list.

**Rule it adds.** A list of names and a parallel list of codes are two sources of truth.
`load_county_boundary()` now asserts that the shapefile's `NAME` for each FIPS matches the name
listed beside it, positionally, and raises if not — verified to fire by putting "Graham" back. The
case for the guard is the same as for every guard in P7: this defect produced no failing test.

## The three rules worth keeping

**Judge every model against both baselines.** R² > 0 (beats a flat line) *and* skill > 0 (beats
copying last month). Neither alone is sufficient — wildfire posts a +0.94 skill against a −0.97
persistence strawman while sitting **below the mean**.

**A skill score is only comparable across models that share a target.** When the target changes,
the persistence baseline changes underneath it, and skill silently changes meaning. Surface water's
jump from +0.1469 to +0.7075 is real, but roughly half of it is persistence getting *worse* on a
spikier target rather than the model getting better. Quote the R² that stands alone; treat skill as
a verdict, not a magnitude.

**Check what the target is actually a mean *of*.** Three of this project's four fixed bugs were
targets that did not measure what their column name claimed: a database edit date, a mean over a
churning well roster, and a mean that was 45% one dam-regulated gage. None was a modeling problem
and none needed new data. Before tuning anything, decompose the target and ask which stations,
which rows, and which mechanism are actually driving it.

**When a fix makes the score go down, that is usually the fix working.** GRACE fell three times
under three corrections, and each lower number was truer than the one before it.

**Ask the target questions of the features too.** Four target bugs were found by asking what a
column was a mean *of* and whether it correlated with physics. Nobody pointed those questions at the
*inputs* until 2026-09-04, and the first one examined —
[irrigation](#p7-feature-side-data-was-unaudited-measured--audit-complete-two-bugs-fixed) — turned out to
be 99.9% nodata sentinel, summed over the entire United States, and four orders of magnitude too
large. It had sat in the variance table labelled "some genuine signal" the whole time. **A variance
decomposition tells you how an input varies, not whether it is correct. Check the magnitude against
a physical range first, then the sign, then the variance.**

**Check the magnitude against a physical range before anything else.** It is the cheapest audit
available and it caught both feature bugs in one pass. 5.43e7 MGD of irrigation and 34,817 MGD of
public supply are, respectively, four orders of magnitude and 5.4x Arizona's entire water budget —
neither needed a model, a correlation, or a variance decomposition to spot, only the question "is
this number possible?" Nobody had asked it in the life of the project.

**A constant that is right once is not right.** `nclimdiv.py` had a single `MISSING = -99.99` while
the source uses three different codes; it happened to match PDSI, so PDSI was clean and temperature
and precipitation silently carried -73 C and negative rainfall. Per-element conventions need
per-element handling, with a physical-bounds backstop underneath in case the convention changes.

**A silent fallback is a bug, not a safeguard.** Both national sums shipped because
`_filter_huc12s_bbox_fallback` logged a warning and carried on. Every guard this project has added
since — `_validate_seasonality()`, `_effort_confound_check()`, and now irrigation's magnitude
check — **refuses to write** instead. Prefer a crash to a plausible-looking number.

**Every model this project has fixed was fixed at the target, not the estimator.** Four for four
on targets — and the fifth bug, irrigation, was a *feature*, which is why it survived four rounds
of target auditing:

| Model | What the target actually measured | Fix | Result |
|---|---|---|---|
| Wildfire | a database edit date (`DATE_CUR`) | MTBS ignition dates | R² −0.034 → **+0.312** |
| Groundwater | which wells reported that month | per-well anomaly | skill −0.079 → **+0.011** |
| Surface water | a dam release schedule (45% one gage) | per-gage log anomaly | skill +0.147 → **+0.708** |
| Wildlife | how many people went birdwatching | per-route anomaly | LOO R² −0.211 → **+0.213** |
| *(feature)* **Irrigation** | *how many watersheds were missing, nationwide* | mask 999/888, filter to 8 counties | 5.43e7 → **2,560 MGD**; no model moved |
| *(feature)* **Public supply** | *national public-supply pumping* | filter to 8 counties | 34,817 → **761 MGD**; no model moved |
| *(feature)* **nClimDiv temp/precip** | *unmasked missing codes* | per-element codes + bounds guard | −73 °C and negative rain removed; latent, no model affected |

Not one was a modeling problem. Not one needed new data that had to be bought or credentialed. In
every case the *name of the column* was a claim about the world that nobody had checked, and in
every case the check was one correlation and cost about a minute. **Suspect the data first** — and
specifically, ask what the target is a mean or a sum *of*, and whether that denominator moves.

**The irrigation row is the one that should worry you most.** The four target bugs were found
because a *model was failing* and someone went looking. Irrigation was not attached to a failing
model — it was a feature in six of them — so nothing ever prompted the check, and it survived four
rounds of increasingly careful auditing. It was found only because someone finally asked whether
5.43e7 MGD was a physically possible number. **They have now all been asked** — see the audit above; two more bugs, both magnitude-detectable.

**Check that the target correlates with physics *before* touching the estimator.** Every driver
of the wildfire target had the wrong sign — it was anti-correlated with heat and positively
correlated with rain. That is one query, and it would have saved this project from ranking a
loss-function change as its top modeling priority for a model whose target was never fire. A bad
score tells you *something* is wrong; it does not tell you it is the model. Suspect the data
first.
