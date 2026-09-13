# Phase 3 Plan — The Sliders Do Not Work

**Status: complete. All seven steps of §6 are done — the sliders work, every sign is stable,
and each output card states where its number came from. The §7 gate passes on all 20
lever/output pairs.**

**Addenda since, each closing an item this document left open:**
[§11.3](#113-a-double-counting-hazard-the-plan-does-not-cover-verified) (the double-count was eight
paths, not one; now gated), [§11.5](#115-one-open-question-and-the-experiment-that-settles-it-measured)
(GRACE's human-block retrain is a null — a one-fold sample-size effect), and
[§12a](#12a-the-spreading-cone-test-measured) (the aquifer storage coefficient was shipping at the
long-run limit, 7.7× past the evidence; measuring it at the scenario horizon multiplies every
groundwater lever by 7.7), and
[§15a](#15a-parameter-bands-on-the-cards-measured) (the cards now show each lever's range over the
declared parameter bands, and which constants drive its width), and
[§22](#22-the-region-was-never-the-one-this-document-named-measured) (the county list selected
Maricopa and Gila, not Graham and La Paz, for the life of the project; the region is kept and two
constants that had been reasoned from the wrong map are re-derived), and
[§23](#23-cap-deliveries-acquired-the-lake-mead-lever-measured-grace-still-null-measured) (§16 item 2:
CAP monthly deliveries acquired 1999-2026; `region_share_of_az_reduction` measured at 1.0 from the
delivery record; as a GRACE feature the series wins 5 of 5 folds and still fails the pre-declared
t-test — NULL), and [§24](#24-cap-on-graces-full-window-null-and-smaller-cap-is-closed-measured)
(the same test on GRACE's own 2002–2023 window, declared as the last CAP run: the gain shrinks to a
quarter of §23's, t = 1.27 — NULL, and CAP is closed for GRACE), and
[§25](#25-gldas-acquired-null-by-the-rule-and-the-rule-tested-the-wrong-thing-measured) (GLDAS
land-surface state acquired; as a feature block for the shipped GRACE model it is NULL by the same
rule — and a one-coefficient linear model on GLDAS's storage change alone scores ten times the
shipped model out of fold, which says the estimator is the constraint, not the data), and
[§26](#26-the-estimator-was-the-floor-a-four-input-linear-model-gives-grace-its-first-skill-measured)
(step 2b: a ridge residual model on four physical inputs scores +0.28 target R² and +0.20 skill
against the shipped +0.02 and −0.03, 4 of 5 folds, t = 2.52 — **REAL**, the first positive verdict
this document has recorded for GRACE; deployment is an architecture change and is not done here),
and [§27](#27-grace-ships-as-the-linear-model-the-retrain-and-what-moved-measured) (it is done:
GRACE ships as the ridge, the drought index is regenerated, all six models are retrained, every
gate passes, and the GRACE card's rain response is now +7.6 points and correctly signed), and
[§28](#28-the-groundwater-index-is-two-aquifers-and-only-one-of-them-is-predictable-measured)
(groundwater: the linear model is NULL — the estimator is not its floor; the target is: 44 of 66
wells are in Cochise County, 3 are in the Phoenix and Pinal AMAs, the Cochise and Pima sub-indices
are uncorrelated, and the Cochise index alone forecasts at +0.35 target R² where the blend forecasts
at +0.01. What to do about it is a target decision, left open), and
[§29](#29-the-groundwater-output-is-the-cochise-index-and-it-forecasts-measured) (option A taken:
the output is the Cochise well index, shipped as a four-input ridge with skill +0.23 and every
climate sign physical; the Lake Mead lever comes off that card; the municipal levers are scaled to
Cochise's share; the storage coefficient is recalibrated; and the irrigation lever now saturates a
score scale that is two feet wide), and
[§30](#30-a-climate-sign-gate-and-what-it-found-measured) (the learned climate responses now have
declared physical signs checked in every month; both ridge models pass everywhere, and five
XGBoost responses in NDVI and surface water fail, one of them the streamflow-versus-rain response
that is wrong-signed in six months of the year), and
[§31](#31-the-constraint-experiment-and-what-the-streamflow-model-turned-out-to-be) (the monotone
constraint ships for streamflow at no cost and does nothing, because the deployed streamflow model
is a ridge; it does not ship for NDVI, which loses 0.037; the ridge has learned a derivative, which
is right for a fast river and wrong for a sustained slider, and passes the sign test 12 of 12 as a
one-month pulse; PDSI → streamflow is a genuine wrong sign either way), and
[§32](#32-13-revisited-integrating-the-learned-residual-does-not-ship-and-cannot-fix-a-sign) (§13's
switch flipped under declared criteria: it multiplies every climate response by a positive factor
of 2.8 to 11, sends groundwater to −187 and streamflow to +114 points, and leaves every sign
exactly where it was, because a positive multiplier cannot change one. Off it stays).

For the readable version of what the finished model *does* — every path from a slider to an
output, and the reasoning behind each — see [DISCUSSION.md](DISCUSSION.md). This document is the
record of what was measured and when.

This document is about a defect that is invisible in [PHASE2_REPORT.md](PHASE2_REPORT.md) and
[PROBLEMS.md](PROBLEMS.md) because both are scored on forecast accuracy, and the defect does not
touch forecast accuracy. It touches the entire point of the project.

> **The premise of this tool is that you move a human-pressure slider and the environmental
> outputs respond.** They do not. Every human lever in the app — population, irrigation, public
> supply, urbanization, Lake Mead — swung across its full historical range moves every output by
> at most **1.7 points out of 100**. Public supply moves nothing anywhere. Climate moves the same
> outputs by 13–44 points. **The app is a precipitation tracker with five decorative controls.**

Claim tags follow [PROBLEMS.md](PROBLEMS.md): **`MEASURED`** (run, with the number),
**`VERIFIED`** (inspected/confirmed, payoff not), **`ESTIMATED`** (a projection), **`UNTESTED`**
(reasoning only).

---

## 1. The measurement

Each of the six exported ONNX models was loaded directly and run through the same feature-catalog
construction the browser uses ([`models.js:303` `buildFeatureCatalog`](frontend/models.js#L303)),
at `durationMonths = 12`, July, every other slider at its default. Each slider was then swung from
its `computed_stats.json` minimum to its maximum. The number reported is the change in the **0–100
score the UI actually renders**, via `normalizeOutput` in [state.js](frontend/state.js).

Reproduce with `python scripts/phase3/slider_sensitivity.py --mode sweep`
([source](scripts/phase3/slider_sensitivity.py)).

| slider (min → max) | grace | ndvi | groundwater | surface water | wildfire | wildlife |
|---|---|---|---|---|---|---|
| Population | −0.64 | +0.14 | −0.06 | **0.00** | **0.00** | **0.00** |
| Irrigation withdrawal | **0.00** | **0.00** | +0.57 | **0.00** | **0.00** | **0.00** |
| Public supply groundwater | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| Urbanization (impervious %) | −0.05 | +0.94 | 0.00 | **0.00** | **0.00** | **0.00** |
| Lake Mead level | −0.12 | −1.67 | 0.00 | **0.00** | **0.00** | **0.00** |
| Precipitation | +13.3 | +16.9 | −0.97 | +40.1 | −40.7 | +43.6 |
| Temperature | −6.9 | −2.4 | −0.55 | −22.5 | +33.4 | +15.5 |
| Drought (PDSI) | −1.5 | −1.1 | −0.16 | −2.5 | +0.25 | +13.9 |

`MEASURED`. **The largest human effect in the entire application is 1.7 points.** It is
`mead_pool_elevation → ndvi`, and it is backwards.

---

## 2. Three causes, each independently fatal

### D1 — Four of six models contain no human features at all `MEASURED`

Not "small coefficients." The columns are absent from the ONNX input signature:

| model | human features in `X` |
|---|---|
| `surface_water` | **0** |
| `wildfire_monthly` | **0** |
| `wildlife` | **0** |
| `groundwater` | 3 — `irrigation_total_withdrawal_mgd`, `population_lag1`, `precip_x_impervious` |
| `ndvi` | 12 — population, impervious, Mead (+ lags). **No irrigation, no public supply.** |
| `grace` | 12 — same set as NDVI |

Two mechanisms in [features.py](scripts/phase2/features.py) produce this, and both were introduced
as *improvements*:

1. **`long_record: True`** ([features.py:553-566](scripts/phase2/features.py#L553-L566)) restricts
   the feature set to `_LONG_RECORD_MONTHLY_FEATURES`
   ([features.py:412-440](scripts/phase2/features.py#L412-L440)) — 26 nClimDiv columns which by
   construction contain **zero** human inputs. This is what took surface water to 1980 and wildfire
   to 1984, and it is why both now have no human features whatsoever. Wildlife's
   `_WILDLIFE_FEATURES` ([features.py:372-385](scripts/phase2/features.py#L372-L385)) has the same
   property for the same reason — reaching 1968.
2. **`extended: True`** ([features.py:571-579](scripts/phase2/features.py#L571-L579)) drops every
   base in `MONTHLY_CAPPED_2020_BASES` ([merge.py:43](scripts/phase2/merge.py#L43)) — which is
   exactly where irrigation and public supply live. This is why NDVI and GRACE, which *do* carry a
   human block, carry it without the two water-withdrawal levers.

**The R² gains and the dead sliders came from the same commits.** PHASE2_REPORT.md records each of
these trades as measured and worth paying, and on forecast accuracy that is true. It was never
scored against the thing the trade actually cost.

### D2 — Where the features do exist, the fitted signs are noise `MEASURED`

Sweeping the same min→max range across every month of the year:

```
population  → ndvi          by month:  - - - + + + + + + + - -     sign flips
population  → groundwater   by month:  + + + - - - - - - - + +     sign flips
impervious  → ndvi          +0.60 … +0.94    more pavement → greener vegetation
mead_pool   → ndvi          −0.90 … −1.67    fuller reservoir → browner vegetation
```

A coefficient whose sign depends on the month-of-year encoding is not a mechanism.

The cause is already documented in this project — PHASE2_REPORT.md's own variance decomposition
("The binding constraint: most inputs carry no monthly information") and
[PROBLEMS.md P4](PROBLEMS.md). Population has **0.4%** within-year variance and impervious % has
**0.6%**: they are proxies for *time*, so their fitted coefficient is whatever else drifted across
2002–2020. Irrigation is **99.3%** within-year of which **95.2%** is an identical repeating
month-of-year template, and public supply is **97.6%** template: their coefficients are a *summer*
effect wearing an irrigation label, collinear with `month_sin`/`month_cos` and with temperature.

**This is the part that matters for the salvage:** the report already established that these inputs
carry no independent monthly information. That finding was filed as an explanation for why two
models fail. It is also, unavoidably, a proof that **no retraining can make these sliders work**,
which is a much larger consequence and was not drawn at the time.

### D3 — The frontend runs residual models exactly one step, regardless of duration `VERIFIED`

[`runPipeline`](frontend/models.js#L466) executes each model **once** and
[`finalizePrediction`](frontend/models.js) adds the single predicted residual to a fixed
`SEED_BASELINES` anchor ([models.js:69](frontend/models.js#L69)). `scenarioDurationMonths` is only
consumed by `lagByDuration` / `rollByDuration` / `annualSumByDuration` to blend the *input*
lag and rolling features — it never advances the model in time.

So "irrigation raised for 3 years" renders **one month** of response against a baseline anchored in
the historical past. For a depletion process — where the whole physical story is that a deficit
*accumulates* — this discards the entire effect. Even a perfectly-signed monthly coefficient would
display at 1/36 scale.

This is a genuine product bug independent of D1 and D2, and it is the cheapest of the three to fix.

### D4 — The UI misreports its own drivers `VERIFIED` ✅ **FIXED**

`TOP_INPUTS` ([state.js:55-64](frontend/state.js#L55-L64)), rendered on every output card by
[ui.js:234](frontend/ui.js#L234), still says wildfire "responds mainly to: population, urbanization
(lag), temperature (lag)" and NDVI to "temperature (lag), public supply groundwater, precipitation."
Wildfire has had **no** population or urbanization feature since the MTBS target fix; NDVI has never
had public supply since it went `extended`. The app was actively telling users that human pressures
drive outputs that cannot see them.

Now generated from `model/*_feature_importance.json` by
[`scripts/phase3/top_inputs.py`](scripts/phase3/top_inputs.py) into
`frontend/top_inputs.json`, so it is derived from the deployed models and cannot drift again. Each
card also states how many of that model's inputs are human levers at all — which is the D1 number,
put where a user can see it.

### D5 — the slider *ranges* are not policy scenarios `MEASURED` ✅ **FIXED**

[`generate_stats.py:140-142`](scripts/phase3/generate_stats.py#L140-L142) sets every slider's
min/max to the **p5/p95 of the raw monthly series, pooled across all months and all years**. Read
that against the variance decomposition in PHASE2_REPORT.md:

| slider | within-year variance | what its min→max axis actually is |
|---|---|---|
| Irrigation | 99.3% (95.2% a repeating template) | **a month-of-year dial.** "Max" means June. |
| Public supply | 96.8% (97.6% template) | **a month-of-year dial.** |
| Population | 0.4% | **a time-travel dial.** "Min" means 2002. |
| Urbanization | 0.6% | **a time-travel dial.** |
| Lake Mead | 7.7% | mostly the 2000s drawdown — largely a time dial |

So even with correct coefficients, the control is measured in the wrong units. Dragging irrigation
to maximum does not ask "what if we pumped hard?" — it asks "what if it were June?", which is a
question `month_sin`/`month_cos` already answer and which explains, independently of D1 and D2, why
the learned models find nothing there to use.

**This blocked Layer 2**, because β must be defined per unit of *sustained annual* withdrawal, not
per unit of seasonal swing. Sliders are now reparameterized as policy deltas around a
deseasonalized baseline — % change in annual withdrawal, people added, points of impervious cover,
feet of reservoir elevation — with the seasonal shape supplied by the month selector, where it
belongs. See [§9](#9-what-d5-and-d4-changed-measured).

---

## 3. Why retraining alone cannot fix this

The obvious response to D1 is "put the human features back and retrain." That is necessary but
**not sufficient**, and shipping it alone would be worse than the current state — it would replace
visibly-dead sliders with visibly-wrong ones.

The reason is identification, not sample size. `irrigation_total_withdrawal_mgd` is 95.2% a repeating
seasonal template. Its variance *is* the calendar. Any estimator handed that column alongside
`month_sin`/`month_cos` and temperature is being asked to separate three collinear encodings of
"it is July" using 219 monthly rows, and it will apportion the summer signal arbitrarily between them.
The evidence that this is what happens is already in the repo:

- Groundwater **selects** corrected irrigation at #2 by importance and gains **+0.003 skill**
  (`MEASURED`, PHASE2_REPORT.md) — the model wants the column and cannot convert it.
- GRACE has a real, significant, correctly-signed partial correlation with irrigation
  (**−0.260**, p < 0.001, calendar removed) and handed the feature back on identical test rows gains
  **+0.0098** target R² and wins 2/5 folds (`MEASURED`) — a null in a test rigged in its favour.
- PROBLEMS.md's own conclusion: **"a significant correlation is not a usable feature."**

The project has already run this experiment twice and got a null twice. Running it a third time with
more features forced in will produce coefficients, but there is no basis to believe their sign or
magnitude, and D2 shows what those coefficients look like when you do it: month-dependent sign flips.

**The deeper framing:** this project built a *forecaster* and shipped it as a *simulator*. A model
optimized for out-of-fold R² will always route around low-variance causal inputs and lean on the
lag1 anchor plus climate — that is correct behaviour for the objective, and fatal for the product.
Prediction and intervention are different questions and need different machinery. Nothing here is a
mistake in the modeling; it is a mismatch between the objective that was optimized and the claim the
interface makes.

---

## 4. The architecture

Split the two jobs explicitly, and label which is which in the UI.

```
output(t) = ML_climate( climate, season, lag1 )          ← learned, unchanged
          + Σ_j  β_j · ( lever_j − baseline_j ) · r_j(t) ← structural, signed by physics
```

### Layer 1 — the learned models stay exactly as they are

No retraining, no re-scoring, no retraction. Surface water at R² 0.75, wildfire at 0.28, NDVI at
0.79, wildlife at ρ 0.51 — these are the credibility of the project and they are all climate models.
PHASE2_REPORT.md remains true as written, because it never claimed slider response.

### Layer 2 — a structural response layer for the human levers

Signs and magnitudes fixed by water balance and published policy, **not** fitted from a collinear
219-row panel. Each lever gets a documented coefficient, a citation, and an uncertainty band.

| Lever | Path to output | Basis | Status |
|---|---|---|---|
| Irrigation, public supply → **groundwater depth** | `ΔDepth = ΔPumping · Δt / (S_y · A)` | Aquifer storage balance. Two parameters: specific yield `S_y` and effective area `A`. | `UNTESTED` — parameters need sourcing from ADWR AMA hydrologic reports |
| Lake Mead → **groundwater depth** | elevation → shortage tier → CAP reduction (AF/yr) → substituted groundwater pumping → storage balance above | **Published step function.** 2007 Interim Guidelines + 2019 DCP define elevation thresholds (~1,090 / 1,075 / 1,050 / 1,025 ft) each with a stated Arizona reduction. | `UNTESTED` — thresholds and per-tier volumes must be read out of the DCP text, not recalled |
| Urbanization → **NDVI** | area converted × (NDVI_desert − NDVI_impervious) | Land-cover arithmetic. Both endpoints are measurable in the project's own MOD13A3 pixels. | `UNTESTED` — endpoints derivable in-repo |
| Population → **public supply → groundwater** | per-capita municipal demand × population, chained into the storage balance | Closes the loop so population stops being a bare trend line. | `UNTESTED` |
| Irrigation → **NDVI** | irrigated-area greenness contribution | Physically *positive* on irrigated pixels and negative for groundwater — the tool should show that tension, not hide it. | `UNTESTED` — sign is the point; magnitude needs the irrigated-area fraction |

**This is the most defensible part of the plan and the part that will feel least like modeling.**
That instinct is worth resisting: a coefficient derived from a published policy document with a
stated uncertainty is strictly more honest than an XGBoost coefficient whose sign flips with the
month. The Lake Mead lever in particular goes from being the app's most dead control to its most
rigorously grounded one, because the shortage tiers are *law*, not inference.

### Layer 3 — integrate over the scenario duration ⚠ **RUN. The naive version is invalid.**

**Original specification, now superseded:** roll the residual models forward month by month for
`scenarioDurationMonths`, feeding each model's own output back into its `lag1` feature.

**This was implemented and measured, and it does not work.** See
[§4a](#4a-the-layer-3-experiment--measured) for the full result. The short version: the exported
models are not dynamical systems, so there is nothing to integrate. The revised specification is
in [§4b](#4b-layer-3-revised--mean-reverting-integration).

Reproduce with [`scripts/phase3/slider_sensitivity.py`](scripts/phase3/slider_sensitivity.py).

---

## 4a. The Layer 3 experiment `MEASURED`

### The finding: none of the residual models carries a feedback term

Every residual model deletes its own lagged target before fitting —
`x_train = x.drop(columns=[lag1_col])`, in [model_ndvi.py:84-86](scripts/phase2/model_ndvi.py#L84-L86),
[model_surface_water.py:222-225](scripts/phase2/model_surface_water.py#L222-L225), and the
groundwater and GRACE scripts. The lag1 column is carried through `features.py` only to serve as the
reconstruction anchor; it is never a predictor. Confirmed against the exported artifacts:

```
ndvi           own-target features in X: NONE
grace          own-target features in X: NONE
groundwater    own-target features in X: NONE
surface_water  own-target features in X: NONE
```

That is a **defensible choice for one-step forecasting** — it stops the anchor doubling as a
predictor and keeps the residual R² honest. But it means the model is

```
dy_t = f(exogenous_t)        with no dependence on y_t
```

There is no recession, no restoring force, no equilibrium. **Integrating it D steps is arithmetically
identical to multiplying the one-step answer by D.** The per-step residual under a fixed scenario
varies only with month-of-year (sd 0.090 for surface water, entirely seasonal), so the rollout is a
pure linear ramp.

### What that produces

`MEASURED`, `--mode rollout --months 36`:

| slider (min→max) | grace | ndvi | groundwater | surface water |
|---|---|---|---|---|
| Precipitation | +334 | +683 | −45 | **+1371** |
| Temperature | −246 | −55 | −14 | **−783** |

Scores are supposed to be bounded 0–100. Under sustained maximum precipitation, surface water
reaches a log anomaly of **+17.3** by month 36 against a historical p95 of **+1.06** — about
**3 × 10⁷ times normal flow**. The rollout also drifts with *nothing perturbed*: at default sliders,
36 steps moves GRACE **0.73×** and surface water **0.58×** of their entire historical range, because
each model has a small nonzero mean residual at the (jointly implausible, all-p50) default input and
integration turns any bias into a ramp.

**Naive forward integration must not ship.** It is not a conservative approximation; it is unbounded.

### It does light up the human sliders — including the wrong-signed ones

At 36 months, irrigation → groundwater goes from **+0.57 → +19.69** score points, correctly signed
(more pumping, deeper water). That is the first visible, correctly-signed human response in this
project. But the same 36× multiplier hits everything in [D2](#d2--where-the-features-do-exist-the-fitted-signs-are-noise-measured):
Lake Mead full → NDVI **−82.9**, urbanization → NDVI **+38.9**. Integration is an amplifier. It
amplifies the bug.

### And it cannot touch three of the six outputs

`surface_water`, `wildfire` and `wildlife` remain **exactly 0.00** for all five human levers at
every duration tested (1, 12, 36 months). This is the decisive result: **D1 is structural, and no
amount of integration fixes it.** Multiplying zero by 36 is zero. Those three models have no human
features, so the only way they will ever respond to a human lever is Layer 2.

## 4b. Layer 3, revised — mean-reverting integration `MEASURED`

The missing restoring force can be estimated from the observed panel rather than assumed. Regressing
`dy_t` on `(y_{t-1} − ȳ)` gives a per-target monthly reversion rate λ (`--mode lambda`):

| target | λ | e-folding time | t | reads as |
|---|---|---|---|---|
| surface water | 0.3511 | 2.8 months | −10.81 | flow recedes fast ✅ |
| NDVI | 0.2221 | 4.5 months | −5.99 | vegetation recovers in a season ✅ |
| GRACE | 0.0386 | 25.9 months | −2.28 | a storage integral, slow ✅ |
| groundwater | 0.0157 | 63.8 months | −1.24 | an aquifer barely reverts ✅ (n.s.) |

**All four have the right sign and physically sensible time constants.** Integrating with

```
y_t = y_{t-1} + f(x_t) − λ·(y_{t-1} − y_eq)
```

converges to a finite displacement `f/λ` under sustained forcing. Measured (`--revert`):

- **NDVI and surface water become stable** — NDVI settles at +0.2147 by month 36 and +0.2209 at
  month 120; surface water plateaus at −0.2959 and stays there.
- **GRACE and groundwater still creep**, because λ is small enough (0.039, 0.016) that the model's
  own bias, divided by λ, is a large number. Arguably correct for an aquifer, but it means the
  displayed value depends on the horizon rather than converging.
- Climate responses remain too large (precipitation → GRACE +183 at 36 months, still 1.8× the
  historical range). Dividing an off-manifold residual by a small λ is not a trustworthy operation.
- **Human levers move and stay bounded:** irrigation → groundwater **+15.1**, population → GRACE
  −10.8. The wrong signs survive: Mead → NDVI **−9.35**, urbanization → NDVI **+3.86**.

**Verdict:** mean-reverting integration is the correct form of Layer 3 and should be what ships —
but it is **necessary, not sufficient, and must not ship before Layer 2**. On its own it makes the
app's wrong answers larger and more confident. λ should also be re-derived per model rather than
from the raw panel once Layer 2 exists, since part of what λ currently absorbs is the response the
structural layer will supply explicitly.

### Layer 4 — provenance in the UI

Each output card shows the split: *climate (learned model)* vs *human levers (water balance)*, with
the Layer-2 term's basis available on hover or in a footnote. Replace the stale `TOP_INPUTS` strings
(D4) with values generated from the actual `_feature_importance.json` files at build time so they
cannot drift again.

This is not a disclaimer. It is a better product than the black box: it answers *why* the number
moved, which XGBoost never could.

---

## 5. The cross-check that should still be run ✅ **RUN — see [§10](#10-the-5-cross-check-result-measured)**

Independently of Layer 2, run **one** retrain as a measurement, not as a fix:

- force the human block back into all six models on the 2002–2020 window (drop `long_record` and
  `extended` for this variant only — it is not the shipped model),
- **deseasonalize** each human feature by subtracting its month-of-year mean, so the estimator sees
  pumping *anomalies* rather than the calendar — this is the step that gives the coefficient a
  chance to mean something,
- constrain the signs: XGBoost `monotone_constraints` per feature, or a sign-constrained linear fit,
- report the out-of-fold R² cost against the shipped models on identical test rows.

Three possible outcomes, all useful:

| result | interpretation | action |
|---|---|---|
| Constrained fit agrees in sign with Layer 2 and costs little R² | Corroboration for the structural coefficients | Cite it in the write-up; consider blending |
| Constrained fit agrees in sign but costs real R² | The panel can see it, weakly, at a price | Keep Layer 2; report the trade |
| Constrained fit cannot find the sign at all | The panel genuinely cannot identify these effects | **Publishable finding**; Layer 2 is the only route |

Given the two existing nulls (§3), the third outcome is the most likely `ESTIMATED`. That is a
result worth having in writing, because it is the justification for Layer 2 existing.

**It was run.** `ESTIMATED` was right for 18 of 21 lever/target pairs and wrong for three — and
the three are the ones Layer 2 was going to build first. Full result in
[§10](#10-the-5-cross-check-result-measured).

---

## 6. Order of work

**Revised after the Layer 3 experiment.** Layer 3 was originally first on the grounds that it was
the cheapest change with the largest information gain. It delivered the information — the exported
models are not integrable, and three of six outputs can never respond to a human lever without
Layer 2 — and in doing so it demoted itself. It cannot ship before the signs are fixed.

1. ~~**Layer 3 — forward integration.**~~ ✅ **Done as an experiment; result is a qualified
   negative.** Naive integration is unbounded and must not ship; the mean-reverting form
   ([§4b](#4b-layer-3-revised--mean-reverting-integration)) is correct but must wait for Layer 2.
   Harness lives at [`scripts/phase3/slider_sensitivity.py`](scripts/phase3/slider_sensitivity.py).
2. ~~**D5 — reparameterize the sliders** as policy deltas around a deseasonalized baseline.~~
   ✅ **Done.** `generate_stats.py` emits a `policy` block per slider; `frontend/catalog.js`
   reconstructs raw model inputs from delta + month climatology. Layer 2 is unblocked.
   See [§9](#9-what-d5-and-d4-changed-measured).
3. ~~**D4 — regenerate `TOP_INPUTS`** from the importance sidecars.~~ ✅ **Done.**
   [`scripts/phase3/top_inputs.py`](scripts/phase3/top_inputs.py) → `frontend/top_inputs.json`.
4. ~~**The cross-check (§5)**, as a Phase 2 experiment script.~~ ✅ **Done.**
   [`scripts/phase2/experiment_human_block.py`](scripts/phase2/experiment_human_block.py).
   Outcome 3 for 18 of 21 pairs; outcome 1 for the three that Layer 2 needs most.
   See [§10](#10-the-5-cross-check-result-measured). Produces the number that justifies
   everything after it — and it is now the *only* remaining route to making `surface_water`,
   `wildfire` and `wildlife` respond to a human lever through the learned layer.
   **Except wildlife, where the answer is already in.** PHASE2_REPORT.md measured the short-window
   variant: n=23 drops LOO R² from +0.0956 to **−0.1702** with ρ falling below its own permutation
   null. Wildlife cannot carry human features through the learned layer at any price, so it needs
   no experiment — it goes straight to Layer 2.
5. ~~**Layer 2 — the structural layer**~~ ✅ **Done** — [§13](#13-layer-2-and-layer-3-built-measured). Starting with Lake Mead (published tiers, least parameter
   guessing) then the groundwater storage balance, then urbanization → NDVI. **Now load-bearing:**
   the experiment showed it is the only mechanism that can reach half the outputs.
6. ~~**Layer 3 (mean-reverting form) ships alongside Layer 2**~~ ✅ **Done**, in closed form, and
   applied to the STRUCTURAL forcing only — [§13](#13-layer-2-and-layer-3-built-measured) measured
   that integrating the learned residual too amplifies D2's wrong signs until they beat the physics.
7. ~~**Layer 4 — provenance UI.**~~ ✅ **Done** — [§15](#15-layer-4--provenance-in-the-ui).

## 7. Acceptance criteria

In the spirit of the Phase 1 regression guards
(`_validate_seasonality`, `_effort_confound_check`), this defect should not be able to return
silently. Add a test that fails the build when it does:

- **Every human slider moves at least one output by ≥ 5 score points** across its min→max range at
  a 12-month duration. Currently the maximum is 1.7 and four sliders score 0.00 everywhere.
- **Every lever's sign is stable across all 12 months** and matches a declared expected sign. This
  is what D2 violates today.
- The sensitivity sweep runs headlessly against the exported ONNX files, so it catches a feature-set
  change in Phase 2 as well as a bug in the frontend.

A warning nobody reads is not a safeguard — the same lesson this project already learned from the
national-sum irrigation bug.

## 8. What this does not change

- **PHASE2_REPORT.md stays valid.** Every score in it is a forecast score, honestly obtained, and
  none of them are affected. No number is retracted.
- **PROBLEMS.md stays valid**, and its P4/P5 findings become *load-bearing* rather than
  explanatory — the variance decomposition is the proof that Layer 2 is necessary.
- **No model is retrained for deployment** under this plan. The learned layer is the part that
  works. This is now a measurement rather than a policy: the one model with a reason to revisit
  it was GRACE, and [§11.5](#115-one-open-question-and-the-experiment-that-settles-it-measured)
  ran its real nested pipeline on the human-block feature set and returned a null.

The one thing that changes is the claim the interface makes. Right now it implies the six outputs
respond to human pressure and they do not. After this, the response is real, correctly signed,
accumulates over time, and is labelled with where it came from.

---

## 9. What D5 and D4 changed `MEASURED`

### The sliders are now policy deltas

`generate_stats.py` emits a `policy` block per slider alongside the raw p5/p95, and
[`frontend/catalog.js`](frontend/catalog.js) reconstructs the raw model input from it:

```
mode "scale"    raw(month, d) = baseline * (1 + d/100) * seasonal[month]
mode "offset"   raw(month, d) = baseline +  d          + seasonal[month]
```

`baseline` is the deseasonalized 2020 mean (the last year every human series covers), so `d = 0`
is the climatological normal for whichever month is selected, and the month dropdown — not the
slider — supplies seasonality. Ranges are in [PHASE3_PARAMS.md §6](PHASE3_PARAMS.md).

**Departure from PHASE3_PARAMS.md §6, recorded because it changes numbers:** the climate sliders
are ranged on the p5/p95 of deseasonalized **annual means**, not monthly ones. §6 specified
monthly, to keep climate magnitudes comparable to §1 above; measured, monthly does the opposite.
A slider is held for the whole scenario duration, and a sustained monthly extreme is not a year:

| slider | monthly p5/p95 | annual p5/p95 | wettest/most extreme year on record |
|---|---|---|---|
| precipitation | −93% … +160% | −32% … +45% | +73.5% |
| temperature | −2.29 … +2.52 °C | −0.71 … +0.79 °C | +1.07 °C |
| PDSI | −2.64 … +3.64 | −1.96 … +2.42 | +2.86 |

Held for 12 months, the monthly range put precipitation → surface water at **+121 points on a
0–100 scale**. The annual range puts it at **+36.7**, against §1's +40.1 — which is what §6 said
it wanted.

### Two frontend bugs the rewrite exposed and fixed `MEASURED`

Both were invisible in §1 because the old table was measured against the same buggy catalog.

1. **The five `usdm_dsci` lag/rolling features never responded to the drought slider.** The
   catalog derived `usdm_dsci` from PDSI but built `usdm_dsci_lag1/lag3/roll3/roll6/roll12` from
   `sliderValues.usdm_dsci` — a control the UI does not expose, pinned at its p50 forever. Those
   five features feed `grace`, `ndvi` and `groundwater`.
2. **The rolling and anomaly features did not match what was fitted.**
   [features.py](scripts/phase2/features.py#L74-L107) defines `X_rollW = X.shift(1).rolling(W).mean()`
   — offsets 1..W, excluding the current month — and `X_anomaly = X − X_roll12`, a departure from
   the trailing year. The old catalog's rolling means included the current month, and its
   "anomaly" was the distance from the pooled p50, which is a different quantity entirely. Fixed;
   the anomaly features now reproduce the seasonal departure the models were trained on (+10.79 °C
   in July at `d = 0`, against a July seasonal factor of +10.79).

### The default scenario is no longer off-manifold `MEASURED`

§4a reported that a 36-month rollout drifts with *nothing perturbed* — GRACE by 0.73× and surface
water by 0.58× of their full historical range — because the all-p50 default is a jointly
implausible input the models were never fitted near. With `d = 0` now meaning "the climatological
normal for this month", that drift falls to:

| target | drift at default, before | after |
|---|---|---|
| GRACE | 0.73× | **0.04×** |
| surface water | 0.58× | **0.15×** |
| NDVI | — | 0.31× |
| groundwater | — | 0.14× |

This does not make naive integration valid — §4a's verdict stands — but it removes one of the two
reasons the mean-reverting form still creeps.

### The sweep, re-measured on policy ranges

`python scripts/phase3/slider_sensitivity.py --mode sweep` (July, 12-month duration):

| slider (policy min → max) | grace | ndvi | groundwater | surface water | wildfire | wildlife |
|---|---|---|---|---|---|---|
| Population (−0.5M … +3M) | −0.24 | +0.19 | **0.00** | **0.00** | **0.00** | **0.00** |
| Irrigation (−60% … +40%) | **0.00** | **0.00** | +0.36 | **0.00** | **0.00** | **0.00** |
| Public supply (−40% … +60%) | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| Urbanization (−0.4 … +2.0 pts) | −0.68 | +1.37 | **0.00** | **0.00** | **0.00** | **0.00** |
| Lake Mead (−95 … +130 ft) | +0.39 | −3.27 | **0.00** | **0.00** | **0.00** | **0.00** |
| Precipitation (−32% … +45%) | −3.38 | +4.96 | −0.20 | +36.70 | −14.50 | +10.37 |
| Temperature (−0.71 … +0.79 °C) | +0.18 | +0.47 | −0.31 | −0.28 | +9.56 | +1.05 |
| Drought (−1.96 … +2.42 PDSI) | −1.73 | −4.80 | −0.15 | −1.06 | +0.01 | +7.35 |

**The human levers are still dead, and that is the expected result.** D5 was never a fix for D1
or D2 — it was the prerequisite for Layer 2, because β has to be defined per unit of sustained
annual withdrawal. The largest human effect is now 3.27 points (Mead → NDVI) and it is still
backwards. Four sliders still score 0.00 everywhere.

### Guard against the duplication

The scenario→feature arithmetic exists twice: `frontend/catalog.js` for the browser, and the
`Catalog` class in `slider_sensitivity.py` for the headless sweep. If they drift, the acceptance
test starts measuring a different application from the one that ships — the exact class of failure
this document exists to record.

`python scripts/phase3/check_catalog_parity.py` runs `node frontend/catalog.js --dump` over eight
fixed scenarios and compares all 111 driver features against the Python mirror. They agree to
1e-9 on every one.


---

## 10. The §5 cross-check result `MEASURED`

Run with [`scripts/phase2/experiment_human_block.py`](scripts/phase2/experiment_human_block.py);
full output in `model/experiment_human_block.json`. Five monthly models (wildlife excluded — §6
step 4 explains why its answer was already in), four to five feature-set variants each, **all
scored on identical test rows** in the common 2002-10..2020-12 window with identical fixed
hyperparameters, so the feature set and the sign constraint are the only things that vary.

### Headline

| | count |
|---|---|
| lever/target pairs with a declared physical sign | 21 |
| **sign found in every fold** | **3** |
| sign unstable across folds | 4 |
| sign never found | 2 |
| effect smaller than 0.05 sd — no effect to have a sign | 12 |

> **Qualification added after [§12](#12-layer-2-step-1--the-aquifer-calibration-measured).**
> "Folds agreeing" is weaker evidence than it looks for a slow-moving lever. TimeSeriesSplit
> folds all share the same secular trend, so a regressor that is mostly a proxy for the calendar
> year gets the same sign in every fold for a reason that has nothing to do with mechanism —
> five folds agreeing is one piece of evidence, not five. The test is sound for irrigation
> (99.3% within-year variance) and weak for Mead (7.7%), population (0.4%) and impervious (0.6%)
> — the same D5 variance decomposition that motivated the policy sliders. §12 re-tests all three
> surviving pairs with an explicit time control; two survive and one does not.

**This is outcome 3 for 18 of 21 pairs**, as §5 predicted: the panel cannot identify these effects,
and no retraining will make those sliders work. Twelve of the eighteen are not even wrong — the
p10→p90 response is under 5% of the target's own standard deviation, so there is no effect for a
sign to be attached to.

### The three exceptions, and why they matter more than the eighteen

| lever → target | folds agreeing | effect | expected |
|---|---|---|---|
| irrigation withdrawal → **groundwater depth** | **5/5** | +0.600 sd (deeper) | + |
| irrigation withdrawal → **GRACE storage** | **5/5** | −0.401 sd (depleted) | − |
| Lake Mead elevation → **groundwater depth** | **5/5** | −0.171 sd (shallower) | − |

Two of these three survive a climate- and trend-controlled regression
([§12](#12-layer-2-step-1--the-aquifer-calibration-measured)); Mead does not. **The two that
survive are the aquifer storage balance [§4](#layer-2--a-structural-response-layer-for-the-human-levers)
listed first for Layer 2** — the aquifer storage balance and the Mead → shortage-tier → substituted
pumping chain. Deseasonalized and left unconstrained, the panel finds all three, at full fold
stability, in the direction water balance says they must go. The GRACE result independently
reproduces the partial correlation already in PROBLEMS.md (−0.260, p < 0.001).

That is corroboration for Layer 2's two most important coefficients from a completely separate
method, and it is a stronger result than §5 expected. It does not make the learned layer a
substitute for Layer 2 — a direction is not a magnitude in feet per acre-foot — but it means the
structural coefficients are not being written into a vacuum.

### What deseasonalizing bought `MEASURED`

The §5 step that "gives the coefficient a chance to mean something" is the one that did the work.
Target R², identical test rows:

| model | shipped | + human block | + deseasonalized |
|---|---|---|---|
| grace | −0.0119 | +0.0368 | **+0.0539** |
| groundwater | −0.7172 | −0.8673 | **−0.6884** |
| ndvi | +0.3425 | +0.4008 | +0.3108 |
| surface water | +0.6855 | +0.5819 | +0.5767 |
| wildfire | +0.1908 | +0.0689 | +0.0812 |

**GRACE and groundwater are better with the deseasonalized human block than shipped** — GRACE on
both the residual it is trained on and the reconstructed level (+0.4121 → +0.4462), groundwater on
the level (+0.0646 → +0.1254). Those are the two models where the water-balance signs were found.
The two long-record models go the other way, decisively.

### What forcing the human block costs the long-record models `MEASURED`

Surface water and wildfire have to give up the pre-2000 training rows to carry human features,
because the human series do not exist before 2000. Scored on the same test rows:

| model | shipped (2002–2020 rows) | shipped + pre-2000 rows | human block forced in |
|---|---|---|---|
| surface water | +0.6855 | **+0.7671** | +0.5767 |
| wildfire | +0.1908 | **+0.2329** | +0.0812 |

So the true cost of a human block in surface water is **0.767 → 0.577 = −0.190 target R²**, and it
buys five levers of which none has a stable sign and two have no effect at all. For wildfire,
−0.152 for two levers, both with no effect. **These two models should not be retrained. Their human
response has to come from Layer 2 or not at all** — which is what §4a already concluded from the
other direction.

### Sign-constraining is not free, and it is not the answer either

`monotone_constraints` forcing every declared sign, target R² against shipped:

| model | shipped | sign-constrained | cost |
|---|---|---|---|
| grace | −0.0119 | −0.0055 | **−0.006 (improves)** |
| ndvi | +0.3425 | +0.3291 | +0.013 |
| wildfire | +0.1908 | +0.0789 | +0.112 |
| surface water | +0.6855 | +0.5687 | +0.117 |
| groundwater | −0.7172 | −0.9322 | +0.215 |

Groundwater is the instructive one: the model whose two headline signs the panel *did* find is
also the one that pays most to have all five imposed. Constraining `population`,
`public_supply_groundwater_mgd` and `impervious_pct` against a panel that has no information about
them costs 0.215 R² to buy three coefficients with no evidence behind them. **A constrained
retrain is not a cheaper Layer 2; it is Layer 2 with the citations removed.**

### Consequence for the plan

- §4's Layer 2 table stands, and its first two rows now have independent directional support.
- `surface_water` and `wildfire` are confirmed closed to the learned layer. With wildlife already
  closed (§6 step 4), **three of six outputs can only ever respond to a human lever through
  Layer 2.**
- Nothing here changes a shipped model. No retrain is recommended. The experiment is a
  measurement, exactly as §5 specified.


---

## 11. What the cross-check changes about the rest of the plan

[§10](#10-the-5-cross-check-result-measured) is a measurement. This is what follows from it for
the work that is left. Four things change; one substantially.

### 11.1 Layer 2 is two tiers, not one — and Layer 4 needs three provenance categories

[§4](#layer-2--a-structural-response-layer-for-the-human-levers)'s table treated all five
structural coefficients alike. §10 splits them:

| tier | levers | what the interface may claim |
|---|---|---|
| **Corroborated** | irrigation → groundwater, irrigation → GRACE | published parameters **and** an independent regression estimate that survives climate and trend controls ([§12](#12-layer-2-step-1--the-aquifer-calibration-measured)) |
| **Structural-only** | population → groundwater, public supply → groundwater, **Mead → groundwater**, urbanization → NDVI, irrigation → NDVI | published parameters alone; the panel measured no effect, or an effect that dissolved under controls |

> **Correction.** This table first listed Mead → groundwater as corroborated, on the
> strength of §10's 5/5-fold sign agreement. [§12](#12-layer-2-step-1--the-aquifer-calibration-measured)
> shows that result is a trend artifact and does not survive a time control (t falls from
> +0.96 to +0.12, with the wrong sign throughout). Mead belongs in the structural-only tier.
> The DCP tier chain is still the best-grounded lever in the plan — its thresholds are law —
> but the panel does not independently confirm it, and the UI must not imply otherwise.

[Layer 4](#layer-4--provenance-in-the-ui) was specified with two categories — *climate (learned)*
vs *human (water balance)*. It needs **three**. "Two independent methods agree on the direction"
and "one method, and the other found nothing" are different epistemic claims, and distinguishing
them is the entire pitch of this architecture over the black box it replaces.

### 11.2 The plan's dominant uncertainty now has a second, disagreeing estimate ⚠

[PHASE3_PARAMS.md §2](PHASE3_PARAMS.md) calls `A_eff` "the least defensible number in the whole
plan" and assumes `S_y · A ≈ 1.87 M AF/ft`, from 45% of the eight-county area as alluvial basin.
§10's measured slope is an independent estimate of that same quantity:

```
917 MGD sustained irrigation anomaly (p10→p90)  =  85,587 AF/month
measured Δdepth                                 =  +0.710 ft/month
implied  S_y · A                                =  120,500 AF/ft     vs 1.87 M assumed  (15.6x)
implied  A_eff at S_y = 0.15                    =  804,000 acres
2017 Census irrigated acres, eight counties     =  681,000 acres     (1.18x)
```

`ESTIMATED`. The empirical `A_eff` lands within ~20% of the region's **irrigated** area, not its
alluvial area. That reads as a diagnosis rather than a coincidence: `depth_to_water_anomaly_ft` is
a per-well anomaly over the **monitored well network**, which sits in agricultural basins, so
drawdown from irrigation is concentrated near the fields and not spread over 12.5 M acres.
**PHASE3_PARAMS.md §2 sized a regional water table when the output is a monitored-well index — it
was answering the wrong question.**

The caveat has to travel with the number: an XGBoost partial-dependence slope absorbs everything
correlated with pumping anomalies, drought above all, so 120,500 AF/ft is a *lower* bound on
`S_y · A` and the implied lever an *upper* bound. The honest response is a band spanning both
estimates — roughly an order of magnitude — not adopting whichever end is convenient. That matters
here more than usual, because the empirical value makes the lever ~15x stronger and would sail
through the §7 acceptance test. PHASE3_PARAMS.md §2 already anticipated exactly this temptation:
*"Do not tune `A_eff` to make the test pass; that would defeat the entire premise of the plan."*

**This was the first Layer 2 task, before any β was written.** ✅ **Done — see
[§12](#12-layer-2-step-1--the-aquifer-calibration-measured).** The answer is that the two numbers
are not competing estimates of one quantity: they are the short-run and long-run limits of a
spreading drawdown cone, and their 16.9x ratio is exactly the ratio of the two areas.

### 11.3 A double-counting hazard the plan does not cover `VERIFIED`

✅ **RESOLVED — and the hazard was wider than this section described.** Gated by
`slider_sensitivity.py --mode no-double-count`.

**The original hazard.** `groundwater` is the one model whose Layer 1 **already contains
irrigation** — it is one of its three human features — and it already responds: **+0.36 score
points** across the policy range (§9). Layer 2 adds a structural irrigation term on top of a
learned model that already has one, so for that single path the effect would be counted twice.

This section proposed to *state it and accept it* at an estimated ~9% learned share, and required
that the outcome be a decision rather than an oversight. **The decision is recorded here, and it
is not the one proposed: there is nothing left to accept.** §13's climate-only fix removed the
overlap entirely, and the three things this section asked for — a measurement, a decision, and the
owed Mead → groundwater check — are each answered below.

#### The double-count is gone, measured to exact equality `MEASURED`

`climateOnly()` holds all five human levers at their climatological normal before the ONNX models
run, so Layer 1 cannot see a human delta at all. Swinging each human lever across its full policy
range, in all 12 months, changes the raw learned output of all six models by **exactly zero** —
`0.0e+00` across 5 × 6 × 12 = **360 comparisons**, not a small number but bit-identical output.
Every shipped lever/output pair's learned share is **0.0%**.

Gate: `python scripts/phase3/slider_sensitivity.py --mode no-double-count`. It asserts exact
equality rather than a tolerance, because Layer 1 does not consume these values at all — any
nonzero difference is a wiring bug, not a numerical one. Reverting `climate_only()` to a no-op
makes it fail 8 pairs and exit nonzero, so it is a gate and not a decoration.

#### The exposure was eight paths, not one — this section undercounted it

Auditing the deployed `*_feature_names.json` sidecars directly, rather than reasoning from which
lever each model was *meant* to carry:

| model | human features it actually carries |
|---|---|
| `grace` | population ×3, impervious ×4, **mead ×8** |
| `ndvi` | population ×3, impervious ×4, **mead ×8** |
| `groundwater` | **irrigation ×1**, population_lag1, `precip_x_impervious`, mead_total_release ×2 |
| `surface_water`, `wildfire_monthly`, `wildlife` | none |

So **"GRACE is clean" was true only of irrigation.** `extended: True` did drop irrigation from its
feature set, but GRACE still carries fifteen other human features, and so does NDVI. Had the human
deltas kept flowing into Layer 1, the double-count would have been:

| removed learned term (policy min → max, July) | score points | sign across 12 months |
|---|---|---|
| mead → ndvi | **−3.27** | **flips** (+2 / −10) |
| impervious → ndvi | +1.37 | **flips** (+11 / −1) |
| impervious → grace | −0.68 | **flips** (+10 / −2) |
| mead → grace | +0.39 | **flips** (+4 / −8) |
| **irrigation → groundwater** | **+0.36** | stable |
| population → grace | −0.24 | stable |
| population → ndvi | +0.19 | **flips** (+7 / −5) |
| impervious → groundwater | +0.00 | stable (below rounding) |

`MEASURED`. The path this section was written about is the **fifth largest** of the eight, and the
largest — mead → ndvi at −3.27 — is nine times bigger and has the wrong sign in ten of twelve
months. **Five of the eight flip sign with the month**, which is [D2](#d2--where-the-features-do-exist-the-fitted-signs-are-noise-measured)
measured one more time, on the deployed artifacts, in the units the UI renders. That is the real
finding: the overlap was not a 9% accounting rounding error on one well-grounded lever, it was
mostly unstable coefficients on levers this section did not list.

#### The owed Mead → groundwater check `MEASURED`

**There was never a path, even before the fix.** `groundwater`'s two Mead features are
`mead_total_release` and `mead_total_release_roll3` — *release volume*, not pool elevation — and
the model does not carry `mead_pool_elevation` at all. Release is a `CONSTANT_DRIVERS` entry in
[`catalog.js`](frontend/catalog.js), pinned at 12,657 with no slider, so the Mead control could
not move it. Measured deviation: **+0.00**. The check is discharged with a null, and the null has
a mechanism rather than being an absence of evidence.

#### The 9% estimate was high by about half

| | score points |
|---|---|
| learned irrigation → groundwater (counterfactual, pre-fix) | +0.36 |
| structural irrigation → groundwater (shipped) | **+60.11** |
| combined, had both been counted | +60.47 |
| **learned share** | **0.6%** |

`MEASURED`. The ~9% figure assumed ~4 structural points from
[PHASE3_PARAMS.md §2](PHASE3_PARAMS.md); after
[§12a](#12a-the-spreading-cone-test-measured) recalibrated the lever to 60.11 the learned share is
**0.6%**. Both terms carry the same sign, so it would have been a genuine inflation rather than a
cancellation. The share is now small enough that magnitude is no longer the argument for the gate —
the argument is the wiring, which is why the gate asserts exact equality rather than a threshold.

#### What the decision actually is

**Layer 1 is climate-only, permanently, and it is now gated rather than asserted.** That is the
decision this section asked for. It is recorded as a measurement for the reason §13 gives — the
same structural choice fixes the double-count, removes D2's wrong signs from the interface, and
makes every lever month-invariant, so it should not be revisited for one of those three reasons in
isolation.

One residual fragility, stated rather than fixed: `climateOnly()` derives its key list from
`panel: 'human'` in `SLIDER_DEFS`. That is the right default — a new human lever is covered
automatically — but a lever placed in the climate panel for layout reasons would silently begin
double-counting. The `--mode no-double-count` gate exists to catch exactly that, and it reads the
same `panel` field, so it is a live check on the wiring and not a restatement of it.

### 11.4 §7's acceptance criteria need restructuring

Twelve of the eighteen failed pairs in §10 had an effect below **0.05 sd** — not a wrong sign, no
effect. A flat *"every human slider moves at least one output by ≥ 5 score points"* therefore asks
the application to display something the panel says is absent.
[PHASE3_PARAMS.md §5](PHASE3_PARAMS.md) predicted several levers would miss that bar with honestly
sourced coefficients; §10 upgrades the prediction to a measurement.

Restructure [§7](#7-acceptance-criteria) as:

- **Hard gate: sign stability across all 12 months**, matching a declared expected sign. This is
  what [D2](#d2--where-the-features-do-exist-the-fitted-signs-are-noise-measured) violates today
  and what Layer 2 satisfies by construction, so it is the criterion that can actually hold.
- **Reported, not gated: magnitude per lever**, with *"this lever's physical effect is genuinely
  small"* an allowed pass rather than a failure — provided the number and its basis are shown.

### 11.5 One open question, and the experiment that settles it `MEASURED`

✅ **RUN — the verdict is NULL, but this section's hypothesis was wrong about why.**
[`scripts/phase2/experiment_grace_nested.py`](scripts/phase2/experiment_grace_nested.py) →
`model/experiment_grace_nested.json`.

**The question.** GRACE scored **better** with the deseasonalized human block than with its
shipped feature set — on both the residual it is trained on (−0.0119 → +0.0539) and the
reconstructed level (+0.4121 → +0.4462), on identical test rows. That could not be claimed against
the deployed model, because every §10 variant used one fixed XGBoost configuration rather than each
model's own nested tuning, so the comparison is internally valid but its "shipped" column is not
the shipped model's real score. This is the one place where
[§8](#8-what-this-does-not-change)'s "no model is retrained for deployment" might be leaving
something on the table.

**The rule was fixed before the run**, in the script's docstring, because PROBLEMS.md Part 3 says
in bold *do not tune GRACE* and a probe must not be allowed to become a search. The human block
counts as REAL only on all three of: Δ target R² > 0, wins in ≥ 4 of 5 folds, and |paired t| ≥ 2.0
across folds. The t-bar is the standard PROBLEMS.md item #3 already used to call the irrigation
trade null.

#### The result: two of three criteria pass, so it is a NULL

Both arms re-run under `model_grace.py`'s real pipeline — `RandomizedSearchCV(n_iter=60, inner
TimeSeriesSplit(3))` tuned inside every training fold — on the 168 rows `build_variants` guarantees
are shared, with GRACE's `require_real_target` filter applied so neither the target nor its anchor
is zero-fill:

| arm | features | human | target R² | level R² | skill vs persistence |
|---|---|---|---|---|---|
| shipped | 45 | 15 | +0.0394 | +0.4389 | +0.0020 |
| + deseasonalized human block | 58 | 22 | **+0.0960** | **+0.4744** | **+0.0375** |

| criterion | result | |
|---|---|---|
| Δ target R² > 0 | **+0.0566** | PASS |
| wins ≥ 4 of 5 folds | **4/5** | PASS |
| \|paired t\| ≥ 2.0 | **t = +1.21** | **FAIL** |

`MEASURED`. **VERDICT: NULL.** The shipped feature set stands and nothing is retrained.

#### This section's hypothesis was that tuning would close the gap. It does not

The prediction was that the advantage was an artifact of an undertuned shipped arm. The real search
lifts **both** arms by a similar amount, so the gap survives almost intact:

| | §10 fixed config | real nested tuner | lift |
|---|---|---|---|
| shipped | −0.0119 | +0.0394 | +0.0513 |
| + human block | +0.0539 | +0.0960 | +0.0420 |
| **the gap** | **+0.0658** | **+0.0566** | −0.0092 |

So the fixed config *was* understating the shipped arm, by 0.051 — but it understated the human arm
by almost as much, and the difference between the two is not where the answer lives. **The null is
on significance, not on tuning.** Recorded because this section predicted the opposite and the run
refuted it.

#### Where the advantage actually comes from: one fold, and it is the starved one

| fold | test span | n_train | shipped | + human block | Δ |
|---|---|---|---|---|---|
| 0 | 2005-04..2007-07 | **28** | **−0.4433** | **−0.2079** | **+0.2354** |
| 1 | 2007-08..2009-11 | 56 | +0.0789 | +0.0432 | −0.0356 |
| 2 | 2009-12..2013-01 | 84 | +0.2501 | +0.2859 | +0.0358 |
| 3 | 2013-02..2017-05 | 112 | +0.0169 | +0.0548 | +0.0379 |
| 4 | 2017-06..2020-12 | 140 | +0.2945 | +0.3039 | +0.0094 |

**Fold 0 is 83% of the summed difference.** It is the earliest fold, with 28 training rows, and it
is the one fold where *both* arms are worse than predicting zero — the human block is merely less
catastrophic. Across the other four folds the difference is +0.0119 with t = +0.69 (a post-hoc
diagnostic, not a second test).

That reframes the finding. The human block does not add skill to GRACE; it adds **robustness when
the model is starved of training data**, which is a real property and a useless one for a deployed
model that fits on all 204 rows. §10's headline was reporting a small-sample effect as a feature-set
effect, and the fold table is the only view in which that is visible.

#### What it would have cost anyway

Adopting the human block means re-accepting the 2020-12 cap that the human series impose, which is
the trade PHASE2_REPORT.md records GRACE already making deliberately in the other direction: it
gave up irrigation and the other 2020-capped features to reach 2023-12, buying 36 months — 18% of
the record. So even a REAL verdict would have had to clear that cost, and a +0.0566 target R²
at t = 1.21 does not come close.

**§8 stands as measured rather than assumed: no model is retrained for deployment.** GRACE's
remaining routes are the data acquisitions PROBLEMS.md Part 3 already lists — CAP monthly
deliveries (#5) and GLDAS soil moisture (#7) — not a feature-set rearrangement of the panel it
already has.

### 11.6 What does not change

`surface_water`, `wildfire` and `wildlife` are confirmed closed to the learned layer, now with a
price attached (−0.190 and −0.152 target R² to force the block in; wildlife measured separately in
PHASE2_REPORT.md). **Three of six outputs can reach a human lever only through Layer 2**, which
makes Layer 2 load-bearing rather than supplementary. PHASE2_REPORT.md and PROBLEMS.md remain
valid as written.


---

## 12. Layer 2, step 1 — the aquifer calibration `MEASURED`

Run with [`scripts/phase3/aquifer_calibration.py`](scripts/phase3/aquifer_calibration.py); output
in `model/aquifer_calibration.json`. This is [§11.2](#112-the-plans-dominant-uncertainty-now-has-a-second-disagreeing-estimate-),
the task that had to happen before any β was written.

### Method

`S_y · A` is the denominator of every groundwater lever in Layer 2. The §10 estimate came from an
XGBoost partial-dependence slope, which absorbs anything correlated with a pumping anomaly. This
replaces it with a distributed-lag regression that can be controlled:

```
Δdepth_t = a + Σ_{j=0..3} b_j · Q_anom_{t-j} + climate + season (+ trend) + e
S_y · A  = 1 / Σ_j b_j
```

`Q_anom` is the **deseasonalized** withdrawal anomaly in AF/month, so the coefficient is
identified off excess pumping rather than off the calendar. Climate controls are PDSI,
precipitation and temperature at lags 0/1/3/6 — everything pumping could be standing in for,
drought above all. Standard errors are Newey-West at 6 lags, because monthly water-level
residuals are autocorrelated and OLS errors would overstate every result here.

### Result: irrigation is real, and it strengthens under controls

| specification | β (ft per AF/month) | t | S_y·A (AF/ft) | R² |
|---|---|---|---|---|
| raw | 7.133e−06 | +2.60 | 140,184 | 0.088 |
| + climate | 8.006e−06 | +2.99 | 124,900 | 0.145 |
| + climate + trend | **9.018e−06** | **+3.16** | **110,891** | 0.148 |

n = 246. The coefficient *grows* and the t-statistic *rises monotonically* as controls are added,
which is the opposite of what a confounded association does. Four independent estimates — XGBoost
partial dependence (120,500), and the three rows above — span 110,891 to 140,184 AF/ft, a ±12%
spread across two entirely different estimators.

`A_eff` at `S_y = 0.15` is **739,276 acres = 1.09× the 2017 Census irrigated acreage** for the
eight counties (681,143). That is the physical reading of the number:
`depth_to_water_anomaly_ft` is a per-well anomaly over the **monitored well network**, which sits
in agricultural basins, so a pumping anomaly draws down the area around the fields — not
12.5 M acres of regional alluvium.

### The 16.9× discrepancy is not a disagreement

| | S_y·A | −60% irrigation, 12 months |
|---|---|---|
| measured, short run | 110,891 AF/ft | 1.386 ft/month = 16.6 ft/yr = **86.5 score points** |
| physical, long run ([PHASE3_PARAMS.md §2](PHASE3_PARAMS.md)) | 1,875,000 AF/ft | 0.082 ft/month = 0.98 ft/yr = **5.1 score points** |

**A drawdown cone spreads with time, so the area it draws from grows.** The regression identifies
the response to a *monthly* pumping anomaly, which is the short-run limit; the alluvial-basin
figure is the long-run limit. Their ratio is 16.9×, which is exactly the ratio of the two areas —
the two numbers are consistent, and PHASE3_PARAMS.md §2 was not wrong so much as answering a
different question from the one the regression asks.

**The consequence for Layer 2 is a constraint, not a coefficient choice.** Multiplying the
short-run β by a 12-month duration gives 86.5 score points, which is the same linear-ramp error
[§4a](#4a-the-layer-3-experiment-measured) already measured and rejected for the learned models.
So:

- Layer 2 uses the **measured short-run coefficient**, which is the one with evidence behind it.
- It **must** be integrated with the mean-reverting form of [§4b](#4b-layer-3-revised--mean-reverting-integration),
  not multiplied by duration. Layer 3 stops being an enhancement and becomes a correctness
  requirement for Layer 2.
- The physical estimate becomes the **long-horizon anchor** the integration should converge
  toward, which is a testable property rather than an assumption.
- The reported band spans both limits — roughly an order of magnitude — because the honest
  statement is that the answer depends on how long the scenario runs.

> ⚠ **The third bullet was tested and it fails — see
> [§12a](#12a-the-spreading-cone-test-measured) for what ships.** The first implementation also did
> not follow the first bullet: it shipped the long-run limit, not the short-run one. Both are now
> superseded by a measurement at the scenario's own horizon.

This also settles [PHASE3_PARAMS.md §5](PHASE3_PARAMS.md)'s worry that irrigation would be
"marginal" against the §7 ≥5-point threshold. At the long-run limit it is 5.1 points, right at the
line; at the short-run limit it saturates. Neither was tuned to pass, and both are reported.

### Public supply and Lake Mead: no empirical support

| lever | raw | + climate | + climate + trend | verdict |
|---|---|---|---|---|
| public supply → depth | t = +0.17 | t = −0.11 | t = +0.91 | nothing, at any specification |
| Mead elevation → depth | t = +0.96 | t = +0.66 | **t = +0.12** | wrong sign throughout, dissolves under a time control |
| irrigation → GRACE | t = **−3.61** | t = **−2.53** | t = **−1.93** | correct sign throughout, attenuates but survives |

Mead is the instructive failure and the reason [§11.1](#111-layer-2-is-two-tiers-not-one--and-layer-4-needs-three-provenance-categories)
was corrected. Its coefficient is positive in every specification — a *fuller* reservoir
associated with a *deeper* water table, the opposite of the mechanism — and adding a linear time
control drops t from +0.96 to +0.12. Mead elevation is 7.7% within-year variance (D5): across
2002–2020 it is very largely a proxy for the calendar, and so is much of what happened to
groundwater. §10's 5/5-fold agreement was five folds sharing one trend.

**This does not weaken the Mead lever, it relocates its justification.** The DCP thresholds are
published law ([PHASE3_PARAMS.md §1](PHASE3_PARAMS.md)) and remain the most rigorously grounded
input in the plan. What changes is that the panel gives it no independent support, so it ships as
structural-only and the UI says so.

Public supply has no support from any method — §10 found 1/5 folds and no effect, and all three
specifications here are indistinguishable from zero. It is structural-only, from the per-capita
derivation in [PHASE3_PARAMS.md §3](PHASE3_PARAMS.md), with a wide band.


---

## 12a. The spreading-cone test `MEASURED`

[§12](#12-layer-2-step-1--the-aquifer-calibration-measured) called the long-run figure *"the
long-horizon anchor the integration should converge toward, which is a testable property rather
than an assumption."* Nobody ran it. Running it changes the most consequential number in Layer 2.

**Why it had to be run.** §12's bullets say Layer 2 uses the short-run coefficient; the shipped
`structural_params.py` used `storage_long_run` instead, justified in its own source string as *"the
LONG-RUN limit, which is the right one for a sustained policy slider."* Nothing in the plan recorded
that reversal, and the two differ by **16.9×** on every pumping→groundwater path. Since
`storage_af_per_ft` is a divisor, shipping the long-run end made every groundwater lever the
weakest it could possibly be.

### The test

If a drawdown cone spreads, the storage coefficient implied by a *sustained* pumping change must
**rise with the horizon it is measured over**. So measure it at each horizon: total drawdown
`depth(t) − depth(t−h)` on the total volume pumped across those h months, climate and trend
controlled, Newey-West at `h + 6` lags. One estimator, varied only by `h` — the same discipline
[§10](#10-the-5-cross-check-result-measured) applies by holding its XGBoost config fixed.
`horizon_sweep()` in [`aquifer_calibration.py`](scripts/phase3/aquifer_calibration.py).

| horizon | β (ft/AF) | t | S_y·A (AF/ft) | implied A_eff (acres) |
|---|---|---|---|---|
| 1 mo | 8.24e−06 | **+3.86** | 121,413 | 809,419 |
| 2 mo | 5.96e−06 | +3.43 | 167,780 | 1,118,536 |
| 3 mo | 5.26e−06 | +3.21 | 189,987 | 1,266,581 |
| 6 mo | 5.43e−06 | +2.87 | 184,023 | 1,226,821 |
| 9 mo | 4.88e−06 | +2.26 | 204,925 | 1,366,165 |
| **12 mo** | **4.10e−06** | **+1.78** | **244,082** | **1,627,214** |
| 18 mo | 4.26e−06 | +1.96 | 234,956 | 1,566,376 |
| 24 mo | 3.46e−06 | +1.66 | 289,336 | 1,928,910 |

### Three findings, and only the first was expected

**1. The cone does spread.** 809,419 → 1,627,214 acres from 1 to 12 months, monotone. §12's physics
is confirmed, and the 1-month value reproduces §12's 110,891 AF/ft to within 9% under a different
control specification — the two estimators agree.

**2. It plateaus at two to three times the *irrigated* area, not at the alluvial basin.** From 3
months on the coefficient sits between 184,023 and 289,336 AF/ft — 1.23M to 1.93M acres, against
681,143 irrigated acres. **The physical long-run figure of 12.5M acres is 7.7× past anything the
record supports at any horizon out to 24 months.** It is not an anchor the data converges toward;
it is an extrapolation past the end of the evidence. §11.2 had already guessed this — it noted the
empirical `A_eff` lands near the region's *irrigated* area because the target is a monitored-well
index sitting in agricultural basins — but it priced that as a short-run artifact, and the sweep
shows it holds out to two years.

**3. Both of the plan's candidate values are wrong for the slider's actual duration.** §12's
short-run figure understates a 12-month scenario by 2.2×; the shipped long-run figure overstates
the storage — and so understates the lever — by 7.7×.

### What ships

`storage_af_per_ft` is now **measured at the scenario horizon**: 244,082 AF/ft, `status: MEASURED`,
band 184,023–289,336 (the 3–24 month plateau). Neither limit is adopted, because the test says the
answer is neither. The groundwater column of the sweep moves accordingly:

| lever → groundwater | before (long-run) | after (measured) |
|---|---|---|
| Irrigation | +7.82 | **+60.11** |
| Public supply | +2.35 | **+18.03** |
| Population | +1.29 | **+9.90** |
| Lake Mead | −0.55 | **−4.22** |

Groundwater is now the most human-responsive output in the application, which is what it should be
— it is the only output that measures pumping directly. At the band's ends, irrigation → groundwater
spans roughly **51 to 80 points**, so the width is as load-bearing as the point estimate and
belongs on the card.

### This is a measurement that made a lever bigger, not a lever tuned bigger

[PHASE3_PARAMS.md §2](PHASE3_PARAMS.md) says *"Do not tune `A_eff` to make the test pass; that
would defeat the entire premise of the plan,"* and §11.2 repeats it. The distinction that keeps this
on the right side of that line: the horizon sweep was specified to answer *"what is the coefficient
at the duration the slider runs?"*, the §7 gate does not test magnitude at all
([§11.4](#114-7s-acceptance-criteria-need-restructuring)), and the result was whatever it was — it
happens to sit between two numbers that were both already in the plan. The t at 12 months is
**+1.78**, which is not conventionally significant, and that is exactly why the band ships with it.

**One caveat that travels with the number.** The estimate is weakest where it matters most: t falls
from +3.86 at 1 month to +1.78 at 12, because overlapping windows leave progressively less
independent variation. The plateau is identified by the horizons either side of 12 as much as by 12
itself. A shorter-horizon slider would rest on firmer evidence than a longer one.

---

## 13. Layer 2 and Layer 3, built `MEASURED`

Parameters: [`scripts/phase3/structural_params.py`](scripts/phase3/structural_params.py) →
`frontend/structural_params.json`. Arithmetic: [`frontend/structural.js`](frontend/structural.js)
with a Python mirror in [`scripts/phase3/structural.py`](scripts/phase3/structural.py), guarded by
`check_catalog_parity.py`. Ten levers, each with `value`, `band`, `source`, `status` and the tier
[§11.1](#111-layer-2-is-two-tiers-not-one--and-layer-4-needs-three-provenance-categories) assigns.

### The architecture error the §7 gate caught

§4 writes the split as `output = ML_climate(climate, season, lag1) + Σ β_j (lever_j − baseline_j)`.
The first implementation still fed the human slider deltas into the learned models as well, and
the acceptance gate failed two levers because of it: `impervious → ndvi` and `mead → grace` flipped
sign in some months, because D2's month-dependent learned coefficients were large enough to
overwhelm a small structural term.

**Layer 1 now sees climate only** — the five human levers are held at their climatological normal
when the ONNX models run, and Layer 2 owns the whole human response. That is what §4's equation
says, and it fixes three things at once:

- the D2 wrong signs disappear from the interface entirely,
- the [§11.3](#113-a-double-counting-hazard-the-plan-does-not-cover-verified) double-count is gone
  by construction rather than by accounting — and measurably so: it covered **eight** lever/output
  paths, not just the irrigation → groundwater one §11.3 named,
- every lever's effect becomes **month-invariant**, which is what a structural coefficient should
  be and what D2 said the fitted ones were not.

Nothing measurable is lost: §10 established the panel cannot identify those coefficients.

### Layer 3 ships in closed form, on the structural forcing only

For a constant forcing the mean-reverting recursion has an exact solution, so there is no rollout
loop and no accumulated inference drift:

```
z_t = (1 − λ)·z_{t−1} + s      ⟹      z_n = (s/λ)·(1 − (1−λ)^n)
```

At `n = 1` this returns exactly `s`, so a one-month scenario is unchanged from the old behaviour;
it converges to `s/λ` instead of ramping. Rate levers (the storage balance) integrate; **level**
levers do not — converting desert to pavement is a persistent offset, not a rate.

**Whether to also integrate the learned residual was measured, not assumed:**

| lever → output | integration OFF | integration ON |
|---|---|---|
| precipitation → surface water | +36.70 | **+103.93** (off a 0–100 scale) |
| Mead → NDVI (a D2 wrong sign) | −3.27 | **−14.01** |
| urbanization → NDVI | **−3.73 (correct)** | **+2.13 (wrong sign wins)** |

The last row decides it. With integration off, Layer 2's physics dominates and the rendered sign
is correct; with it on, the amplified learned coefficient beats the structural term. §4a already
established the exported models carry no own-target feedback, so there are no dynamics in them to
integrate — an externally fitted λ only multiplies what is already there, including the errors.
`integrate_learned_residual` is therefore **false**, recorded as a parameter with its measurement
rather than as a silent choice.

Integrating the *deviation from the default-scenario residual* rather than the raw residual is
what fixes §4b's remaining creep: at the climatological normal the system is at rest by
definition, so a nonzero residual there is model bias, and dividing a bias by a small λ is exactly
how the old rollout drifted.

### The sweep, with all three layers

| slider (policy min → max) | grace | ndvi | groundwater | surface water | wildfire | wildlife |
|---|---|---|---|---|---|---|
| Population | −3.09 | 0.00 | +1.29 | 0.00 | 0.00 | 0.00 |
| Irrigation | **−18.76** | **+9.30** | **+7.82** | 0.00 | 0.00 | 0.00 |
| Public supply | **−5.63** | 0.00 | +2.35 | 0.00 | 0.00 | 0.00 |
| Urbanization | 0.00 | −3.73 | 0.00 | 0.00 | 0.00 | 0.00 |
| Lake Mead | +1.32 | 0.00 | −0.55 | 0.00 | 0.00 | 0.00 |

> This is the sweep **as §13 measured it**, kept as the record of that step. The groundwater
> column was later superseded by [§12a](#12a-the-spreading-cone-test-measured)'s recalibration
> (irrigation +7.82 → +60.11); see [§14](#where-the-app-now-stands) for the current numbers.
| Precipitation | −3.38 | +4.96 | −0.20 | +36.70 | −14.50 | +10.37 |
| Temperature | +0.18 | +0.47 | −0.31 | −0.28 | +9.56 | +1.05 |
| Drought (PDSI) | −1.73 | −4.80 | −0.15 | −1.06 | +0.01 | +7.35 |

Compare §1: every human lever was ≤ 1.7 points and four were 0.00 everywhere, with the largest
effect backwards. **Every human lever now moves at least one output, every sign is correct, and
no sign depends on the month.** The climate columns are unchanged from
[§9](#9-what-d5-and-d4-changed-measured) — Layer 1 was not touched, exactly as §8 promised.

`surface_water`, `wildfire` and `wildlife` remain 0.00 for every human lever. That is honest
rather than unfinished: §4's table has no structural path to them, §10 measured that the panel
has none either, and inventing one would be the fabrication this layer exists to replace.

### The §7 gate, restructured per §11.4

`python scripts/phase3/slider_sensitivity.py --mode acceptance` — **all 10 levers pass**, holding
their declared sign in all 12 months, with magnitude reported and not gated:

| lever → output | tier | points at 12 months |
|---|---|---|
| irrigation → groundwater | corroborated | 7.82 → **60.11** (§12a) |
| irrigation → grace | corroborated | 18.76 |
| public supply → groundwater | structural-only | 2.35 → **18.03** (§12a) |
| population → groundwater | structural-only | 1.29 → **9.90** (§12a) |
| irrigation → ndvi | structural-only | 9.30 |
| public supply → grace | structural-only | 5.63 |
| Lake Mead → groundwater | structural-only | 0.55 → **4.22** (§12a) |
| urbanization → ndvi | structural-only | 3.73 |
| population → grace | structural-only | 3.09 |
| Lake Mead → grace | structural-only | 1.32 |
| population → groundwater | structural-only | 1.29 |
| Lake Mead → groundwater | structural-only | 0.55 |

Two of the five sliders clear the old ≥5-point bar; all five hold their sign. That is the outcome
[PHASE3_PARAMS.md §5](PHASE3_PARAMS.md) predicted, and no coefficient was moved to change it.

Lake Mead is the smallest lever in the app despite being the best-*grounded* one, and the reason
is worth stating on the card rather than hiding: the full 1,090 → below-1,025 swing is a 528 kAF/yr
Arizona reduction, but only an assumed 60% lands in-region and only an assumed 50% of that is
replaced by pumping rather than fallowing. Both fractions are `UNTESTED`, both scale the lever
linearly, and together they cut it to 30% of the headline number.


---

## 14. Reaching surface water and wildlife `MEASURED`

[§13](#13-layer-2-and-layer-3-built-measured) left three outputs with no human response at all.
[§10](#10-the-5-cross-check-result-measured) had closed the learned route to them, and §4's Layer 2
table had no row for them either — but that was an omission in the plan, not a physical fact. Two
of the three can be reached; the third should not be.

### The structural idea: transfer edges, not new lever sets

The outputs form a chain, and `groundwater` and `ndvi` already carry working levers. So one
coefficient per **edge** propagates every lever upstream of it, instead of a new lever set invented
per output:

```
lever → pumping        → stream capture   → surface water
lever → population     → effluent         → surface water
lever → impervious     → storm runoff     → surface water
                         surface water    → riparian habitat → wildlife
```

That is three new numbers rather than a dozen, and it means the DCP shortage tiers now propagate
all the way to bird abundance.

### Surface water — three paths, with opposing signs

The mechanism is not inferred. A primary source that was already sitting in the repository states
it: *"The Santa Cruz River is largely dependent on discharges from water reclamation facilities"*
and *"runoff after storms depends on the amount of impervious surface"* — A Living River: Santa
Cruz River 2024, Downtown Tucson to Marana, Supplementary Report, Sonoran Institute.

| path | sign | basis |
|---|---|---|
| population → effluent → flow | **+** | the perennial reaches *are* treated wastewater |
| urbanization → storm runoff | **+** | same source; `features.py` already calls `precip_x_impervious` the dominant urban-desert discharge term |
| pumping → stream capture | **−** | stream–aquifer depletion, the documented cause of lost perennial reach on the Santa Cruz and San Pedro |

**Population carries two of these at once, and they disagree.** More people means more effluent
(+) and more municipal pumping (−) on the same streams. The net is positive at the shipped
parameters (+1.74 score points), but that is an *output* of the model rather than an assumption,
and the §7 gate was changed to say so: where a slider's levers conflict, the gate checks sign
stability only and reports which way it resolved. Getting this wrong the first time is what caught
it — the gate failed `population → surface water` because the test had assumed one lever per pair.

A flow change converts to the target exactly, because the target is `mean_g ln(Q_g / Q_normal)`:
`Δanomaly = ln(1 + ΔQ / Q_regional)`, with `Q_regional` derived in-repo from the gage record. That
assumes the change is distributed in proportion to existing flow, which effluent is not — it lands
on a few reaches — so this **overstates** the effluent lever. Stated rather than damped by an
invented factor.

### Wildlife — one edge, and the one that was measured

| edge | specification | β | t | p |
|---|---|---|---|---|
| **surface water → wildlife** | + climate | +0.1953 | **+2.19** | **0.029** |
| | + climate + trend | +0.0536 | +0.42 | 0.675 |
| NDVI → wildlife | + climate | −0.5933 | −0.19 | 0.845 |
| | + climate + trend | +0.2948 | +0.17 | 0.867 |

`scripts/phase3/transfer_calibration.py`, HAC(2) errors, n = 44 (1980–2024) for the riparian edge.

**The riparian edge ships**, at the conservative trend-controlled value with the significant
climate-only fit as the band's upper end. Its sign is positive in both specifications — unlike
Mead in [§12](#12-layer-2-step-1--the-aquifer-calibration-measured), which was wrong-signed in
every one. Riparian corridors carrying disproportionate bird abundance is the documented Southwest
mechanism, and this is the only route wildlife has to any human lever.

**The NDVI edge does not ship.** It is a null: sign flips between specifications, p ≈ 0.85 either
way, effect +0.029 sd. With no panel support *and* no citable published elasticity, building it
would be inventing a number — the exact thing Layer 2 exists to avoid.

### A phantom link the chain exposed `VERIFIED`

`OUTPUT_DEFS.ndvi.feedsInto` claimed `"wildlife"`, and `runPipeline` passed `{ ndvi: ndviRaw }`
into the wildlife model. **The wildlife model has no `ndvi` feature**, so
`createInputFloat32Array` silently dropped it. The app declared a dependency it did not have —
the same class of defect as [D4](#d4--the-ui-misreports-its-own-drivers-verified), in the wiring
rather than in a caption. Both are corrected. (`grace → ndvi` is real; NDVI does carry the whole
GRACE block.)

Also corrected while here: the GRACE output was labelled **cm** but the series is `lwe_thickness`
straight from the source `.nc4` with no conversion, and its sd of 0.0479 is 4.8 cm, not 0.05 mm.
It is metres.

### Wildfire gets nothing, and that is the finding

Human ignitions are the majority of US fire *counts*, but ignition is not the limiting factor for
large-fire *extent* in the Southwest — fuel and weather are — and the MTBS target measures extent.
No coefficient from population or impervious cover to large-fire risk could be sourced, and §10
measured both at under 0.05 sd. It ships as **climate-only**, declared in
`structural_params.json` so the interface states it rather than leaving a row of zeros unexplained.

### Where the app now stands

| slider (policy min → max) | grace | ndvi | groundwater | surface water | wildfire | wildlife |
|---|---|---|---|---|---|---|
| Population | −3.09 | 0.00 | **+9.90** | +1.74 | — | +0.43 |
| Irrigation | −18.76 | +9.30 | **+60.11** | −2.37 | — | −0.59 |
| Public supply | −5.63 | 0.00 | **+18.03** | −0.72 | — | −0.18 |
| Urbanization | 0.00 | −3.73 | 0.00 | +3.30 | — | +0.82 |
| Lake Mead | +1.32 | 0.00 | **−4.22** | +0.17 | — | +0.04 |

**Every human slider now reaches four of the six outputs**, against §1's baseline where the
largest human effect anywhere was 1.7 points and four sliders were 0.00 everywhere. All 20
lever/output pairs hold their sign in all 12 months.

The groundwater column is bold because
[§12a](#12a-the-spreading-cone-test-measured) recalibrated it after §14 was first written: every
number in it is 7.7× what it was, because `storage_af_per_ft` had shipped at the long-run limit and
is now measured at the scenario's horizon. Groundwater is now the application's most
human-responsive output, which is the right ordering — it is the only output that measures pumping
directly. Irrigation → groundwater spans **51 to 80 points** across that coefficient's band, so
the band is as load-bearing as the point estimate.

The wildlife column is small — 0.04 to 0.82 points — because it is a second-order effect reached
through one attenuating edge, and the conservative end of that edge's band shipped. At the band's
upper end (+0.1953, the significant fit) it is 3.6x larger. That range belongs on the card, which
is Layer 4 — ✅ **done, see [§15a](#15a-parameter-bands-on-the-cards-measured)**, and it turned out
to be the least uncertain of the ranges that needed showing.


---

## 15. Layer 4 — provenance in the UI

Each output card now shows how much of its movement came from the learned climate model and how
much from the structural levers, with the human part broken out per lever and tagged with its
basis:

```
Groundwater Depth vs Normal
  climate (learned model)                              −1.4
  human levers (structural)                            −3.2
      Irrigation Withdrawal          CORROBORATED      −4.7
      Population (municipal pumping) STRUCTURAL        +1.1
      Lake Mead Level (DCP tier)     STRUCTURAL        +0.4

Streamflow vs Normal
  climate (learned model)                              −1.4
  human levers (structural)                            +5.5
      Urbanization (storm runoff)    STRUCTURAL        +2.7
      Population (effluent)          STRUCTURAL        +1.8
      Irrigation (stream capture)    STRUCTURAL        +1.4
      Population (stream capture)    STRUCTURAL        −0.3

Wildfire Risk Index
  climate (learned model)                              −1.4
  no structural path to this output
```

Three design points, each following from something measured earlier:

- **Three tiers, not two.** [§11.1](#111-layer-2-is-two-tiers-not-one--and-layer-4-needs-three-provenance-categories)
  established that *climate*, *corroborated* and *structural-only* are different epistemic
  claims. The badge is the only place the interface can say which one it is, and hovering any
  row gives the mechanism, the evidence, and what the tier means.
- **`via` labels are load-bearing, not decoration.** Population reaches streamflow twice — through
  effluent (+) and through the pumping it drives (−). Listing "Population" twice with opposite
  numbers and no explanation would be worse than saying nothing.
- **The split is computed from raw values, not from the rendered score**, because
  `normalizeOutput` clamps to 0–100 and a clamped total would not equal the sum of its parts.

The `TOP_INPUTS` line above it is generated from the deployed models' importance sidecars
([D4](#d4--the-ui-misreports-its-own-drivers-verified)), so nothing on the card is hand-written
prose that can drift from the artifacts.

**This is not a disclaimer.** It answers *why* the number moved, which the black box it replaces
never could — and it is what makes the difference between a coefficient with a citation and an
XGBoost coefficient whose sign flipped with the month legible to whoever is using the tool.

---

## 15a. Parameter bands on the cards `MEASURED`

Layer 4 answered *where did this number come from*. It did not answer *how well is it known*, and
after [§12a](#12a-the-spreading-cone-test-measured) that gap was no longer defensible: every Layer 2
number is a product of constants, **twelve of which ship with a declared band**. Ten of those still
feed a lever path: six still `UNTESTED`, and the dominant one now `MEASURED` but at t = +1.78.
(`specific_yield` and `alluvial_fraction` are the other two, vestigial since §12a measured
`storage_af_per_ft` directly rather than deriving it from their product — the relevance scan finds
them irrelevant to every scenario rather than being told to skip them.) A card printing `−36.1` and
nothing else claims a precision the parameters do not have, which is this document's own opening
complaint in miniature.

Two open items collapsed into this one. [§14](#where-the-app-now-stands) asked for the wildlife
edge's range on the card; the Lake Mead lever is weak mostly because
`region_share_of_az_reduction` (0.4–0.8) and `groundwater_substitution_fraction` (0.3–0.7) are both
guesses that multiply. Neither needed a better point estimate. Both needed the width shown.

### Method

`structuralResponseBand` / `response_band` take the envelope over the **corners** of the relevant
bands. Corners rather than propagated derivatives because every path here is a product, quotient or
difference of positive quantities composed with a log, so each is monotone in each constant across
its band and the corner extremes are the true extremes. Corners are also the only version that
stays correct when levers **share** a constant: all three pumping levers divide by
`storage_af_per_ft`, so they move together, and summing independent per-lever minima would
understate the width. Which constants are "relevant" is found by perturbation, not from a
path → constant map — the same reason `climateOnly()` keys off `panel` instead of a list, so it
cannot drift when a formula gains a factor.

Each lever also reports its **drivers**: the banded constants that actually move it. That is the
difference between *"this number is uncertain"* and *"this number is uncertain because the aquifer
storage coefficient is only known to a factor of 1.6"* — the second is actionable.

### What the cards now say

A scenario with irrigation −60%, +1.5 M people, +1.0 pt impervious, Mead −60 ft:

```
Groundwater Depth vs Normal
  human levers (structural)              −29.2      −40.8 to −22.7
      Irrigation Withdrawal  CORROBORATED −36.1      −47.8 to −30.4
      Population (municipal pumping)       +4.2       +3.6 to +5.6
      Lake Mead Level (DCP tier)           +2.6       +0.9 to +6.5   ← 7× wide

Bird Abundance vs Normal
  human levers (structural)               +0.9       +0.6 to +5.3   ← 6× wide
      via Streamflow vs Normal             +0.9       +0.6 to +5.3
```

Three things that only became visible once the widths were on screen:

- **Wildlife is the least certain number in the application** — +0.6 to +5.3, a 6× range, because
  it compounds the transfer edge's own band with every upstream surface-water parameter. §14 asked
  for the edge's 3.6× range; the honest figure is wider than that, since the edge is not the only
  uncertain factor on the path.
- **Lake Mead → groundwater spans 7×** (+0.9 to +6.5) on two `UNTESTED` multipliers. The lever is
  not *small*, it is *unknown* — a materially different statement from the one the point estimate
  was making, and the reason that item closes here rather than with better guesses.
- **Irrigation → GRACE carries no band at all.** Its only non-unit factor is `grace_units_per_af`,
  a geometric conversion with nothing to be uncertain about. An empty range is information too: it
  says the width comes from the parameters, not from the rendering.

### What the range is not

It is **Layer 2 parameter uncertainty only.** The learned climate term on the same card carries its
own error — PHASE2_REPORT.md's CV spread — and that is not in this interval; nor is structural error
in the water balance itself, nor the band on `λ`. The tooltip says so rather than letting the range
imply a total error bar it is not.

The point estimate stays, and stays first. A card showing only a range could not be reconciled with
the bar above it, and the arithmetic elsewhere in the card uses the point value. A range narrower
than the one-decimal precision the number is printed to is dropped, because below that it implies a
distinction the display cannot support.

Guarded by `check_catalog_parity.py`, which now compares both envelopes, every lever's band, the
driver attribution, and asserts each band contains its own point estimate — a band that excludes
its point estimate is a bug in the envelope, not a wide uncertainty. Reverting the JS relevance
threshold from `1e-12` to `1e-3` makes it fail on three scenarios.

---

## 16. Paths forward

Every item this document opened is now closed
([§11.3](#113-a-double-counting-hazard-the-plan-does-not-cover-verified),
[§11.5](#115-one-open-question-and-the-experiment-that-settles-it-measured),
[§12a](#12a-the-spreading-cone-test-measured),
[§15a](#15a-parameter-bands-on-the-cards-measured)). What remains is ranked below, and the ranking
follows from one observation that took the whole of Layers 2-4 to make legible:

**Layer 1 is finished and Layer 2 is not.** Four of six models forecast honestly and the two that
do not (GRACE, groundwater) fail on missing data rather than on modelling — PROBLEMS.md says *stop
tuning* both, and §11.5 confirmed it for the last plausible feature-set change. Layer 2 is the
opposite: it works, every sign holds, and **12 of its 16 levers are `structural-only`** with six
live parameter bands still `UNTESTED`. So the remaining leverage is in *narrowing what Layer 2
claims*, not in improving what Layer 1 predicts.

### 1. Measure `ndvi_impervious` and `ndvi_irrigated_crop` from data already on disk

✅ **DONE, both of them.** `ndvi_impervious` — [§17](#17-ndvi_impervious-measured-measured) — was
5.5× too strong. `ndvi_irrigated_crop` — [§20](#20-ndvi_irrigated_crop-measured--and-the-assumption-was-right-measured) —
was very nearly exact, measured through the HUC12-withdrawal route because there is no cropland
mask on disk. **Both NDVI endpoints are now `MEASURED`, and both NDVI levers `corroborated`.**

[PHASE3_PARAMS.md §4b](PHASE3_PARAMS.md) records both as *"derivable from `data/raw/modis_ndvi/`
but not measured"* — the pass "was out of budget". The budget is the only thing that was missing:
**574 MOD13A3 granules (11 GB) and 30 annual NLCD fractional-impervious rasters (26 GB) are sitting
in `data/raw/`.** No acquisition, no network, no new source.

It is also better posed than §4b assumed. The urbanization lever is
`ΔNDVI = (Δimpervious/100) × (ndvi_impervious − ndvi_natural)`, so the quantity the interface
multiplies **is the slope of NDVI on impervious fraction** — directly regressable across pixels,
with the endpoint recovered as `ndvi_natural + slope` rather than assumed. Two `UNTESTED` bands
become `MEASURED`, and the NDVI column of the sweep stops resting on two guesses.

### 2. Probe CAP monthly deliveries (Reclamation HydroData)

The best-value acquisition, because **one pull serves two unrelated problems.** It is
[PROBLEMS.md](PROBLEMS.md) Part 3 item 5 — the last remaining *monthly* pumping proxy, aimed at
GRACE, the only model with negative skill — and it is independently what
`region_share_of_az_reduction`'s own note asks for: *"Should be replaced with CAP delivery data by
county."* That constant and `groundwater_substitution_fraction` are why Lake Mead → groundwater
spans 7× ([§15a](#15a-parameter-bands-on-the-cards-measured)); CAP deliveries by county would
retire the first of them. Cheap: `lake_mead.py` already ingests from that endpoint.

### 3. ~~Corroborate more levers with the method that already worked~~ ❌ **tried, and it fails**

**Superseded by [§21](#21-the-streamflow-constants-cannot-be-identified-from-this-panel-measured).**
All three estimable streamflow quantities return nulls, including `stream_capture_fraction`, which
was the one with usable anomaly variance. Streamflow's 2.8-month memory makes discharge nearly a
weather variable, and a pumping signal does not survive climate explaining half its variance.
The replacement is a **per-gage** design — 204 gages and 1.6 M daily records are already on disk,
and gages below heavy-pumping HUC12s can be differenced against gages that are not, which is the
design that worked twice for NDVI. The original text follows.

#### Original recommendation, kept for the record

[`aquifer_calibration.py`](scripts/phase3/aquifer_calibration.py) is a general pattern — regress the
lever against the project's own panel with climate and trend controls, Newey-West — and it is what
earned irrigation its `corroborated` badge and what §12a extended to horizons. The obvious next
candidates are `stream_capture_fraction` (0.05–0.25, and the widest band on the surface-water card)
and `effluent_return_fraction` (0.45–0.70). Not all six live UNTESTED bands are estimable from
this panel — some are land-cover constants with no time variation to regress — but these two are,
and the harness exists.

### 4. An urban-footprint NDVI output `UNTESTED` — proposed

[§17](#17-ndvi_impervious-measured-measured) established that paving costs **16% of the vegetation
signal on the land actually paved** and **83% where it replaces cropland**, while the eight-county
mean NDVI moves about a point. Both numbers are correct; they are answers to different questions.
The regional mean is the wrong observable for a small-area, high-intensity change, and no
re-parameterisation fixes that — it is what a regional mean is *for*.

The proposal is a **seventh output**: NDVI within the urban footprint, rather than across the eight
counties. It would move hard under the urbanization slider, because the dilution that flattens the
regional number is exactly what it removes.

What makes it plausible rather than merely appealing:

- **The data is already on disk and already co-registered.**
  [`ndvi_endpoints.py`](scripts/phase3/ndvi_endpoints.py) puts MOD13A3 and NLCD on one grid, and the
  footprint mask is a threshold on the impervious raster it already builds.
- **The measurement exists.** The DiD slope and its strata are the coefficient; no new estimate is
  needed for Layer 2.
- **It is an output, not a lever**, so §7's gate and the tier system apply to it unchanged.

What has to be decided before building it, and none of it is obvious:

- **Layer 1 has nothing to say about it.** No exported model predicts urban-subset NDVI, so the new
  output would be structural-only for its entire value — the first output in the application with
  no learned component at all. That is a real departure from §4's architecture and should be a
  decision, not a side effect.
- **A fixed mask or a moving one?** If the footprint is defined by present-day impervious cover, the
  output measures "NDVI inside today's cities" and urbanizing *new* land does not enter it. If the
  mask moves with the slider, the denominator changes as the lever moves, and a changing denominator
  is how §10's compositional artifacts got in ([P3](PROBLEMS.md)).
- **Climate would still dominate it.** Urban NDVI responds to monsoon too. The gain over the
  regional output is the removal of area dilution, not the removal of weather.

`UNTESTED`, and deliberately left so: it is a scope decision about what the application is for, not
a modelling gap.

### 5. GLDAS soil moisture as a GRACE feature

[PROBLEMS.md](PROBLEMS.md) Part 3 item 7. Ranked last on its own description: modest and uncertain,
since precipitation lags may already carry the fast weather-driven part of monthly ΔTWS. Do **not**
decompose TWS→GWS expecting skill.

### What not to do

- **Do not tune GRACE or groundwater.** Both are at data-limited ceilings, PROBLEMS.md says so in
  bold, and §11.5 measured the last plausible exception and found a null.
- **Do not reach for parameters to make human levers bigger.**
  [PHASE3_PARAMS.md §2](PHASE3_PARAMS.md) and [§11.2](#112-the-plans-dominant-uncertainty-now-has-a-second-disagreeing-estimate-)
  both prohibit it, and [§11.4](#114-7s-acceptance-criteria-need-restructuring) made magnitude
  *reported, not gated* precisely so a genuinely small lever can pass honestly. §12a was legitimate
  because it measured the coefficient at the duration the slider runs, with the specification fixed
  before the result was known — not because the answer came out larger.
- **Do not add a lever without a source.** Wildfire stays climate-only; §14 measured that the
  ndvi → wildlife edge is a null and it is not to be re-added.

## 17. `ndvi_impervious`, measured `MEASURED`

[§16](#16-paths-forward) item 1, done. [PHASE3_PARAMS.md §4b](PHASE3_PARAMS.md) recorded both NDVI
endpoints as assumptions and said the pass over the HDFs "was out of budget". The budget was the
only thing missing: [`scripts/phase3/ndvi_endpoints.py`](scripts/phase3/ndvi_endpoints.py) →
`model/ndvi_endpoints.json`.

### What was measured

The lever is `ΔNDVI = (Δimpervious/100) × (ndvi_impervious − ndvi_natural)`, so the quantity the
interface multiplies is the **difference**, which is exactly the slope of NDVI on impervious
fraction. That is directly regressable. MOD13A3 NDVI and NLCD fractional impervious were put on one
grid — MODIS sinusoidal at the native 926.6 m, clipped to the same dissolved eight-county boundary
every Phase 1 script uses — and OLS run across ~131,000 cells in each of **287 months
(2000-02 … 2023-12)**, each month paired with its own year's NLCD.

| | |
|---|---|
| slope, median across months | **−0.0250** (IQR −0.0457 … −0.0092) |
| negative in | 251 of 287 months |
| `ndvi_impervious` at 100% impervious | **0.2046** (IQR 0.1982 … 0.2135) |
| **assumed in §4b** | **0.08** (band 0.05 … 0.12) |

**The assumption was 5.5× too strong, and its band does not overlap the measurement at all.**

### Why 0.08 was never plausible here

The binned means say it plainly. Even cells that are **90–100% impervious read NDVI ≈ 0.20**, against
a desert background of 0.23:

| impervious | mean NDVI |
|---|---|
| 0–1% | 0.2326 |
| 1–25% | 0.239 – 0.242 ← *greener than desert* |
| 25–50% | 0.2200 |
| 50–75% | 0.2140 |
| 90–100% | 0.2022 |

Two things cause it. A 926 m cell that is mostly pavement still carries lawns, parks and street
trees; and the desert being paved over was only at 0.23 to begin with, so there is very little room
to fall. The 1–25% band being *greener* than untouched desert is the urban fringe — irrigated
landscaping beats creosote. A value of 0.08 describes asphalt, not a kilometre of Tucson.

### Three checks that the co-registration is right

- **The intercept is a free falsification test.** At zero impervious the fit must reproduce the
  region's own natural NDVI, and it gives **0.2310** against `ndvi_natural` 0.2167.
- **The urban-core-only fit agrees.** Restricted to cells ≥25% impervious — where the extrapolation
  to 100% is actually anchored rather than leaning on cells that barely exist — the slope is
  **−0.0326** against the whole-region −0.0250, and the endpoint 0.2030 against 0.2046.
- **Region area from the reprojected cutline** is 27,795,794 acres against the repo's
  `region_acres` of 27,779,840 — 0.06% apart.

### The cross-section was the wrong design, and the right one says something else

`MEASURED`. Everything above is a **cross-section**: it compares Tucson to the desert around it. That
cannot separate *"this land is paved"* from *"this land was always different"* — cities in this
region sit on valley floors, alluvial fans and along washes, which were never a random sample of
the eight counties. So the cross-sectional slope is confounded by siting.

The record contains the better design. Impervious cover genuinely moved over the period —
**4,879 cells gained more than 10 points and 1,728 gained more than 40** — so each cell can be
differenced against **its own past** rather than against its neighbours, which removes every
time-invariant characteristic at once: soil, elevation, aspect, drainage, and whatever grew there
before. Mean NDVI over 2000–2004 against 2019–2023, mean impervious in 2001 against 2021, across
130,833 cells:

| design | slope | |
|---|---|---|
| cross-section, 287 months | −0.0250 | confounded by siting |
| **difference-in-differences** | **−0.0356** | HC1 t = **−14.7** |
| + controlling for 2001 impervious | −0.0382 | t = −15.9 |
| matched urbanised-vs-control contrast | −0.0419 | 3,896 vs 124,380 cells |

Four designs, −0.025 to −0.042. The regional drift common to every cell — unurbanised cells greened
by **+0.0166** over the same window — lands in the intercept, which is what a DiD is for.

### The finding: it is not the concrete, it is what the concrete replaced

Splitting the DiD by each cell's own pre-2005 NDVI is where the answer actually lives:

| what the cell was before | baseline NDVI | urbanising cells | slope | t |
|---|---|---|---|---|
| dry desert | 0.153 | 1,877 | **−0.0027** | −1.9 |
| typical | 0.238 | 1,238 | −0.0221 | −6.9 |
| green | 0.360 | 677 | −0.1224 | −15.9 |
| **cropland / riparian** | 0.520 | 104 | **−0.1792** | −7.5 |

**A 45× difference, and it resolves the whole question.** Paving dry desert costs essentially
nothing — desert NDVI is already about what pavement reads, so there is nothing to lose. Paving
cropland costs −0.179, which is **83% of the region's entire vegetation signal** for that cell.

It also rehabilitates the assumption this section started by overturning. §4b's 0.08 endpoint
implies a slope of −0.1367, which is close to the cropland-conversion row. **The assumption was not
absurd; it was describing the wrong half of the region** — the farmland that suburbs eat, not the
creosote flats they also eat.

### Why the lever stays small anyway, and why that is not a defect

The coefficient that ships is the **historical mix** (−0.0356), because the slider adds impervious
cover the way this region actually adds it — some onto farmland, much more onto desert. The lever
is small for two multiplied reasons, and neither is a modelling failure:

```
  local effect of fully paving a cell        −0.0356 NDVI   = −16.4% of the vegetation signal
  × the slider's whole range (+2.0 points)   = 2% of the region
  = region-wide                              −0.81 score points
```

The same coefficient, if the **entire eight counties** were paved, is **−40.5 score points**. So the
physics is not weak — the *instrument* is. An eight-county mean NDVI is a poor detector of
urbanization, because urbanization is a small-area, high-intensity change and a regional mean is
built to average exactly that away.

**That is worth stating plainly on the card rather than hiding behind a small number.** It is also
why urbanization's honest signature in this application is not NDVI at all: the same slider moves
**surface water +3.30** through storm runoff, four times its NDVI effect, because runoff integrates
over the same small area without diluting it.

### The slider range and the score scale were checked too, and neither is the problem

Before concluding that a one-point lever is correct, the two things that could have made it
artificially small were checked:

- **The slider range is already generous.** Regional impervious cover moved from 0.797% to 1.292%
  across the whole 2000–2023 record — **0.495 points in 24 years**. The slider's range is −0.4 to
  **+2.0** points, so its maximum is **four times the entire observed 24-year change**. Widening it
  would not be more honest, it would be less.
- **The score scale is climate's yardstick.** The NDVI bar spans 0.0879 NDVI — the p5–p95 of the
  regional monthly mean, which is the distance from a drought month to a monsoon month. Urbanization
  at four decades' worth of growth moves it 0.00071.

```
  climate : urbanization  on eight-county mean NDVI  ≈  123 : 1
```

Three independent measurements (cross-section, difference-in-differences, matched contrast) and two
calibration checks all say the same thing, so the coefficient is not what is wrong.

### What was actually wrong was the card

A row reading `Urbanization −0.97` states something true and leaves the reader with something false.
It is the regional mean's answer to a 2%-of-area question, and it says nothing about the land that
was actually paved — which is what a person moving an urbanization slider is picturing.

So the lever now declares a **`local_effect`**, and the card states both scales:

```
NDVI Vegetation Health
  human levers (structural)                              −0.8
      Urbanization              CORROBORATED             −0.8    −0.8 to −0.6
          On the land actually paved, NDVI falls 16%.
```

with the full explanation — the area dilution, the 40-point whole-region equivalent, and the
desert-versus-cropland split — on hover. The local figure is deliberately **not** added into any
total: it is context for the regional number, not a competing one.

The lever also moves from `structural-only` to **`corroborated`**, taking the application from two
corroborated levers to three (and §20 later makes it four). Its `evidence` field used to read *"None. §10 finds an effect of
−0.010 sd — no effect to have a sign."* That was the panel's verdict and the panel could not see
this: an eight-county monthly mean has no way to separate a 2%-of-area land-cover change from
weather. The rasters can, because they resolve the change where it happens.
[§11.1](#111-layer-2-is-two-tiers-not-one--and-layer-4-needs-three-provenance-categories) defines
`corroborated` as a structural mechanism plus an independent empirical check agreeing on the sign
after climate and trend controls, and a DiD across 130,833 cells with regional drift in the
intercept is that check.

### One caveat on the cropland row

Pinal County fallowed farmland under CAP cuts over the same period in which it urbanized. Those two
are correlated, so the −0.1792 stratum may carry some fallowing that is not attributable to
pavement. The DiD design absorbs fallowing that happened *without* urbanization — those cells are
controls — but not a correlated shock. The stratum is reported, not adopted; the shipped coefficient
is the whole-region mix.

### How it is adopted, and the one subtlety

`ndvi_impervious` ships as **0.1811 = ndvi_natural + the DiD slope**, `status: MEASURED`, band
0.1811 … 0.1917 spanning the two designs — *not* as the directly measured 0.2046. The difference is
deliberate. `ndvi_natural` (0.2167) is a p50 over months of the whole-region mean, while the
quantity the lever needs is the NDVI of *the land actually being paved*, which is the regression's
zero-impervious intercept (0.2310). Those differ by +0.0143. Stating the endpoint relative to
`ndvi_natural` makes `(ndvi_impervious − ndvi_natural)` equal the measured slope, which removes
that mismatch instead of inheriting it. The directly measured 0.2046 is recorded in the constant's
own note and in `ndvi_endpoints.json`.

The band spans the two designs — the DiD at one end and the cross-section at the other — rather
than either one's standard error. 131,000 MODIS cells are nowhere near independent, so a per-pixel
OLS error would manufacture a precision this has no claim to; and the spread *between* designs is
the larger and more honest uncertainty anyway.

The DiD was chosen on identification, **before its magnitude was known**. It happens to come out 42%
larger than the cross-section, which is the direction that requires care given
[PHASE3_PARAMS.md §2](PHASE3_PARAMS.md)'s prohibition on tuning levers upward — so the reason is
recorded here and in the constant's own note: differencing a cell against itself removes a confound
the cross-section cannot, and that was true before the number came back.

### Consequence, and a caveat that has to travel with it

| | before | after |
|---|---|---|
| urbanization → NDVI | −3.73 | **−0.97** |

The lever gets 3.8× weaker, and that is the correct direction: §4b's own text already said *"both
are physically tiny, and that is the correct answer, not a bug"* — it just had the wrong endpoint.
[§11.4](#114-7s-acceptance-criteria-need-restructuring) is why this is a pass and not a failure:
magnitude is reported, not gated.

**The caveat is seasonality.** The measured slope is strongly seasonal — most negative in Aug–Sep
(−0.053, −0.062) when the monsoon greens the desert but not the pavement, and around **zero or
slightly positive in May–June** (+0.002, −0.001) when pre-monsoon desert is bare and irrigated
urban land is the greener of the two. That is why the sign holds in 251 of 287 months rather than
all of them. Layer 2 applies a month-invariant coefficient **by design** —
[§13](#13-layer-2-and-layer-3-built-measured) made month-invariance the property that distinguishes
a structural coefficient from D2's fitted ones — so the median ships and the seasonality is
recorded here rather than being pushed into the lever. Making this one lever seasonal would
reintroduce exactly the month-dependent sign flipping the whole layer exists to remove.

### `ndvi_irrigated_crop` is not done, and why

It needs the same treatment but **its predictor is not on disk.** The impervious endpoint was
measurable because NLCD gives a per-pixel impervious *fraction* to regress against; there is no
equivalent irrigated-cropland raster in `data/raw/` — the NLCD holdings are fractional-impervious
only, with no land-cover class layer. The in-repo route that does exist is
`IR_HUC12_Tot_WD_monthly_2000_2020.csv` with the WBD HUC12 polygons: regress HUC12-mean NDVI on
HUC12 irrigation-withdrawal density. That is a real measurement and a coarser one — HUC12s are
large relative to fields — so it is worth doing and worth labelling as weaker evidence than the
impervious fit. Until then `ndvi_irrigated_crop` stays `UNTESTED` at 0.55 (band 0.45–0.65) and
irrigation → NDVI stays at +9.30.

---

## 18. The frontend could not start in a browser `VERIFIED`

Every gate passed and the application did not run. `frontend/structural.js` and
`frontend/catalog.js` both ended with

```js
if (process?.argv?.includes('--dump')) { ... }
```

**Optional chaining short-circuits a property whose value is null or undefined. It does not protect
an identifier that was never declared** — and in a browser `process` is undeclared, so that line
throws `ReferenceError` at module load. `structural.js` died, `models.js` and `ui.js` died importing
it, and the page drew its three panels and then never loaded a single model. Reported from the
browser console as:

```
Uncaught ReferenceError: process is not defined    structural.js:283
```

### Why five green gates said nothing

Because all five ran under Node, where `process` exists. The sweep, the acceptance gate, the
double-count gate and the parity check all exercise the *arithmetic* — much of it through the Python
mirror, which never imports the JS at all. The fifth, the `ui.js` DOM shim, did import it, and still
passed, because Node defines the global the browser does not.

That is the same shape as the defect this whole document opens with: everything in the repo scored
the thing that was easy to score. [§1](#1-the-measurement) was *nothing measured slider response*;
this is *nothing measured whether the page starts*.

### The fix, and the gate

Both guards become `typeof process !== 'undefined' && process.argv?.includes('--dump')`, which is
safe on an undeclared identifier. `--dump` still works, so `check_catalog_parity.py` is unaffected.

The DOM shim is now a committed file rather than one rewritten from scratch each time —
[`scripts/phase3/check_frontend.mjs`](scripts/phase3/check_frontend.mjs) — and it does three things
the other gates cannot:

1. **deletes the Node-only globals** (`process`, `require`, `__dirname`, `__filename`, `Buffer`,
   `global`) before importing anything, which reproduces the browser exactly. Verified: after
   `delete globalThis.process`, a bare `process` throws `ReferenceError` while `typeof process`
   stays safe — that difference is the entire bug;
2. **stubs the DOM** so `ui.js` builds its panels and can be read back — it renders 90 text nodes
   across 18 panels, and a blank page is a failure;
3. **stubs `onnxruntime-web`**, which the browser supplies through the importmap in `index.html`,
   so `models.js` and `ui.js` are actually checked rather than skipped over an unresolvable
   specifier.

Reinstating the `process?.` form makes it fail on `structural.js` with that exact `ReferenceError`
and exit non-zero, so it is a gate and not a decoration.

**It does not replace opening the page. It replaces *not* opening the page** — which is what the
suite had been doing, deliberately, since the Chrome extension is not connected in the development
environment and every check was built to run headless. The lesson is narrower than "test in a
browser": a check that runs only in the environment the code does *not* ship to will certify code
that cannot run in the one it does.

---

## 19. The signs that look wrong, and which one actually was `VERIFIED`

Raised from using the app: *"Increases in population, irrigation withdrawal, urbanization make the
streamflow increase, which is the opposite of what it should be. Public supply groundwater and
population increased the well depth."*

Every path was re-derived and checked against its band. **One of the five readings was a
misreading, three are correct and counterintuitive, and one was a genuine interface failure.**

### Streamflow, every path, at each slider's maximum

| lever | sign | points |
|---|---|---|
| population — effluent | + | **+1.82** |
| population — stream capture | − | −0.34 |
| **→ net for population** | | **+1.49** |
| irrigation — stream capture | − | **−0.96** |
| public supply — stream capture | − | **−0.43** |
| urbanization — storm runoff | + | **+2.74** |
| Lake Mead — stream capture | + | +0.04 |

**Irrigation does not raise streamflow — it lowers it, −0.96.** The sweep table reports a
`min → max` swing and irrigation's slider runs −60% → +40%, so its −2.37 there *is* "more irrigation,
less flow". Same for public supply. That reading was of the table, not of the model.

### The two that really are positive, and why both are right

**Urbanization → +2.74 is textbook.** Impervious surface prevents infiltration, so a larger share of
each storm becomes direct runoff. Higher runoff volume and much higher peak flow from urban
catchments is among the most robust results in hydrology. `features.py` already carries
`precip × impervious` as the dominant urban-desert discharge term for the same reason.

**Population → +1.49 is regionally specific, documented, and in this repo.** The perennial reaches of
the Santa Cruz through Tucson are treated wastewater. From
[Living-River-Downtown-Tucson-to-Marana-2024.pdf](Living-River-Downtown-Tucson-to-Marana-2024.pdf),
which the project already carries: Pima County's reclamation system *"continues to produce
high-quality effluent"*, and *"releasing effluent into the river provides habitat and helps
replenish the aquifer."* More people means more effluent means more flow past the gages. The lever's
own `mechanism` field has said so all along — *"counterintuitive and correct"*.

It is a tug-of-war, not an assumption: population reaches streamflow twice, through effluent (+) and
through the municipal pumping it drives (−), and **the net sign is an output of the model.** It
survives the whole parameter band — **+0.65 to +2.14** across every corner, and still +0.65 at the
worst case for it (stream capture at its maximum 0.25, effluent return at its minimum 0.45). So the
sign is not an artefact of a convenient parameter choice.

### Groundwater was the interface's fault, not the model's

`depth_to_water_anomaly_ft` is **depth to water**. More pumping lowers the water table, which means a
*larger* depth. So public supply `+18.03` and population `+9.90` are the correct sign, and
`higherIsBetter: false` already colours those deltas red.

But a bar labelled **"Groundwater Depth vs Normal"** that rises when you pump more reads as *more
groundwater* to anyone who does not stop to parse "Depth" — and colour is not a label. Three of the
six outputs are named for a quantity that rises when things get worse.

So every output card now states which way is up, under its name:

```
Groundwater Depth vs Normal                              ft
bar rises → water table DEEPER — less water
```

`grace` → *more water stored*; `ndvi` → *greener*; `surface_water` → *more flow past the gages*;
`wildfire` → *more burned area*; `wildlife` → *more birds*. It is a unit label, not a warning, and
it is set quietly.

### One real weakness found while checking, recorded not fixed

The effluent lever converts added discharge into a share of `regional_baseline_cfs` — the median
gage flow times the median gage count — which spreads it evenly across a **103-gage** network. But
the target is the *mean over gages of* `ln(Q/Q_normal)`, and effluent does not arrive evenly: it
enters a few Santa Cruz reaches. Concentrated flow into a handful of gages moves a mean-of-logs
differently from the same volume spread across all of them. The direction is unaffected and the
magnitude is the right order, but the normalisation is an approximation, and a per-gage version
would be the honest improvement. `UNTESTED`.

---

## 20. `ndvi_irrigated_crop`, measured — and the assumption was right `MEASURED`

[§16](#16-paths-forward) item 1, finished. The other half of §4b, and the last NDVI endpoint.

### The predictor had to be built, because it is not on disk

The impervious endpoint was measurable because NLCD hands you a per-pixel impervious *fraction*.
There is no cropland mask anywhere in `data/raw/` — the NLCD holdings are fractional-impervious
only. So the predictor is constructed from the HUC12 irrigation withdrawal matrix, reusing the
eight-county HUC12 selection `scripts/phase1/irrigation.py` already performs, sentinels and all
(999/888 are 70% of the regional cells and were once summed as data).

Withdrawal becomes irrigated *area* through **one calibration constant fitted on the whole record** —
`681,143 acres ÷ total withdrawal` — rather than by assuming an application depth. Because the
constant is fixed across windows, both where the irrigation is and how much of it there is are free
to move. Two checks that the construction is sound: the 1,133 HUC12 polygons total 27,769,412 acres
against the repo's `region_acres` of 27,779,840 (0.04% apart), and the mean per-cell irrigated
fraction comes out **0.0246** against the independently computed `irrigated_fraction` constant of
**0.0245**.

### The cross-section is unusable, and it says so out loud

| implied irrigated fraction | n | mean NDVI |
|---|---|---|
| 0.000–0.005 | 99,092 | **0.2437** |
| 0.005–0.020 | 6,371 | 0.2087 |
| 0.020–0.050 | 4,122 | **0.1933** |
| 0.050–0.100 | 7,573 | 0.2132 |
| 0.100–0.200 | 3,947 | 0.2380 |
| 0.200–0.700 | 5,550 | 0.2639 |

**NDVI dips before it rises.** The zero-irrigation cells include the mountains, which are greener
than any farm; the barely-irrigated cells are low desert valley. A cross-sectional fit here is
measuring *elevation*, and returns +0.0486 — an endpoint of 0.27, which would be absurd for
irrigated cropland. This is the same siting confound as [§17](#17-ndvi_impervious-measured-measured)
and much worse.

### The difference-in-differences

Each cell against its own past, 2000–2004 versus 2016–2020 (the HUC12 matrix stops at 2020),
126,655 cells:

| | slope | t |
|---|---|---|
| uncontrolled | +0.3542 | +34.8 |
| **baseline-controlled** (what ships) | **+0.3353** | **+33.4** |

The dose-response is monotone, and the sign flips on the correct side of the regional drift:

| Δ irrigated fraction | n | mean ΔNDVI |
|---|---|---|
| lost > 0.02 | 3,919 | **−0.0085** |
| −0.02 … −0.002 | 6,679 | +0.0078 |
| no change | 98,174 | **+0.0106** ← regional drift |
| +0.002 … +0.02 | 10,820 | +0.0177 |
| +0.02 … +0.05 | 3,883 | +0.0236 |
| gained > 0.05 | 3,180 | **+0.0354** |

**Cells that lost irrigation fell below the drift; cells that gained rose above it, in order.** That
is what a dose-response looks like when it is real, and it is a stronger argument than the t.

### The result: 0.5520 against 0.55 assumed

| | |
|---|---|
| measured `ndvi_irrigated_crop` | **0.5520** |
| assumed in §4b | 0.55 (band 0.45–0.65) |
| discrepancy | **0.4% from the assumed midpoint, well inside the assumed band** |

**This one was right.** §4b guessed two NDVI endpoints from the literature; one
([§17](#17-ndvi_impervious-measured-measured)) was 5.5× too strong and one was very nearly exact.
Worth recording as a pair, because it is the difference between "the assumptions were sloppy" and
"one assumption was describing the wrong half of the region" — and only measurement told them
apart.

Consequence for the app: irrigation → NDVI moves **+9.30 → +9.35**. Nothing visible changes, which
is the point.

### How the band is stated, and what it is not

Band **0.5520 … 0.5709**, spanning the baseline-controlled and uncontrolled fits. It is
**narrow because two designs agree, not because the quantity is precisely known**, and it is
deliberately *not* a confidence interval — 126,655 MODIS cells are nowhere near independent, and an
HC1 interval over them would manufacture precision, the same objection recorded in §15a and §17.

The honest weaknesses, none of which the band expresses:

- **Withdrawal is known per HUC12, not per field**, so every cell in a polygon carries the same
  value and the estimate is identified off ~339 irrigated HUC12s. This does *not* attenuate the
  slope — with `x` constant within a polygon the cell-level OLS slope equals the polygon-level one,
  and a polygon's mean NDVI is exactly `natural + fraction × (crop − natural)` — but it is coarser
  evidence than the impervious fit, which resolves to the pixel.
- **Allocation assumes withdrawal is proportional to irrigated area**, i.e. a uniform application
  depth across HUC12s. Crop mix varies; alfalfa is not cotton.
- **The stratified slopes range +0.18 (dry) to +0.62 (green)**, and the upper end is not credible as
  a crop endpoint — +0.62 on a 0.358 baseline implies an NDVI near 0.97. The strata are a
  diagnostic here, not band endpoints: `Δfraction` and baseline greenness interact.

The lever also moves `structural-only` → **`corroborated`**, taking the application to **four**
corroborated levers. Its `evidence` used to read *"None. §10 finds 1/5 folds, wrong-signed"* — the
panel's verdict, and the panel could not see a 2.5%-of-area land-cover effect against weather.

---

## 21. The streamflow constants cannot be identified from this panel `MEASURED`

[§16](#16-paths-forward) item 3, attempted and **failed — all three estimable quantities return
nulls.** [`scripts/phase3/streamflow_calibration.py`](scripts/phase3/streamflow_calibration.py) →
`model/streamflow_calibration.json`. This section exists because the recommendation was mine, and a
recommendation that does not survive contact with the data should say so where it was made.

### What was tried

The four widest remaining bands, inverted from Layer 2's own arithmetic. Since Layer 2 converts an
added flow as `ln(1 + Δcfs / baseline_cfs)`, a regression on a physically-scaled regressor inverts
straight back to the constant:

| constant | regressor | inversion |
|---|---|---|
| `stream_capture_fraction` | pumping anomaly, AF/month | `β = −capture × cfs_per_af_month / baseline_cfs` |
| `effluent_return_fraction` | population anomaly | `β = +return × (gpcd/1e6) × cfs_per_mgd / baseline_cfs` |
| `runoff_coefficient_*` | impervious × rain depth × area, in cfs | `β = +(c_imp − c_nat) / baseline_cfs` |

Only the runoff **contrast** is ever identifiable, never the two coefficients separately — they
enter the physics solely as a difference, which is a property of the model rather than a limit of
the data.

### Two of the three were predicted to fail, before any fit

Decomposing each regressor's variance on the panel:

| regressor | variance surviving deseasonalising | |
|---|---|---|
| irrigation withdrawal | **5.4%** | has usable anomaly variance |
| population | **99.8%** | a pure trend — no seasonal cycle at all |
| impervious cover | **99.9%** | likewise |

A regressor that is a pure trend cannot be separated from a trend control, and the trend control is
not optional: **the gage network itself moved from 73 to 110 gages** over the record, so anything
identified off slow secular change is identified off a changing denominator as much as off the
lever. That is why `n_gages` is a control here and is not elsewhere.

### The results, including the one that was supposed to work

| constant | + climate | + trend | + trend + gages | verdict |
|---|---|---|---|---|
| `stream_capture_fraction` | 0.6150 (t = −0.57) | −0.1616 (t = 0.15) | **−0.0025 (t = 0.00)** | NULL |
| `effluent_return_fraction` | −3.03 (t = −1.36) | 15.44 (t = 0.98) | **1.24 (t = 0.07)** | NULL — trend |
| `runoff_contrast` | −8.81 (t = −1.57) | 9.13 (t = 1.23) | **4.86 (t = 0.56)** | NULL — trend |

**The implied values swinging from 0.615 to −0.162 to −0.0025 across specifications is the
diagnosis, not a detail.** A quantity that is actually identified does not move like that when a
control is added. Two of them imply values outside [0, 1] entirely, which is impossible for a
fraction.

**`stream_capture_fraction` is the informative failure**, because it was the one with usable
anomaly variance and it still returned t = 0.00. The reason is streamflow's memory: λ = 0.3511, a
2.8-month half-life, which makes discharge very nearly a weather variable. Climate alone already
explains R² ≈ 0.47–0.50 of it, and a pumping signal worth a couple of score points does not survive
that. The aquifer could be calibrated ([§12](#12-layer-2-step-1--the-aquifer-calibration-measured))
precisely because it integrates for 64 months; the river forgets.

### What a null does and does not mean here

It does **not** mean stream capture is zero, effluent does not reach the river, or pavement does not
shed water. All three mechanisms are real and two are documented in sources this repo carries. It
means **the eight-county monthly panel cannot see them**, which is the same verdict
[§10](#10-the-5-cross-check-result-measured) reached for human coefficients generally, now
established specifically for the four constants someone would most want to narrow.

So the four stay `UNTESTED` at their sourced values, and their bands stay wide. That is the correct
outcome: a band that is wide because the quantity is genuinely unknown is honest, and narrowing it
on an unidentified regression would have been the worst available option.

### What would actually work, and it is not another panel regression

**The panel throws away the spatial variation that identifies these.** `discharge_log_anomaly` is a
mean over ~103 gages; the repo also holds **1,643,528 daily records from 204 distinct gages across
all eight counties** (`data/Final/water_surface_daily_8county_1980_2025.csv`). Gages downstream of
heavy-pumping HUC12s can be differenced against gages that are not — the same cell-differenced
design that worked twice for NDVI ([§17](#17-ndvi_impervious-measured-measured),
[§20](#20-ndvi_irrigated_crop-measured--and-the-assumption-was-right-measured)), applied to gages
instead of pixels. That is a real route and it is in the repo already.

For the runoff contrast specifically, the identification wants **event scale, not monthly means** —
the rainfall-runoff response to a storm, which is where impervious cover actually shows. Daily
discharge is on disk; **daily precipitation is not**, only monthly. That one needs an acquisition.

---

## 22. The region was never the one this document named `MEASURED`

Every "eight-county" in this document, and every county name in every other, was wrong in the same
way. `scripts/phase1/region.py` listed `04013` as Graham and `04007` as La Paz; those codes are
**Maricopa** and **Gila**, and every filter runs on the code. The full record — the measurement,
the decision to keep the region, and what it changes — is
[PROBLEMS.md P8](PROBLEMS.md#p8-the-region-was-never-the-one-the-documents-named-measured--kept-on-purpose).
This section is the Phase 3 consequence.

### Nothing measured here was measured on the wrong region

Every number in §1–§21 was computed on the region the code selects, which is the region the app
runs on. The population series that §10 found to be a pure trend is Phoenix's; the aquifer §12
calibrated is the one under Phoenix, Pinal and Tucson; the 130,833 MODIS cells §17 differenced sit
inside the Maricopa-and-Gila cutline. The captions were wrong, not the data, and no result in this
document is retracted.

### The region is kept, because it is the right one for the Lake Mead lever

With Maricopa in, the region holds **all three CAP Active Management Areas** — Phoenix, Pinal,
Tucson — and CAP delivers to no county outside those three. The ADWR–CAP joint shortage statement
(2021) puts Arizona's reduction as *"borne almost entirely by the CAP system"*. So the region is,
to within on-river 4th-priority water, the whole footprint of the shortage tiers §1 of
PHASE3_PARAMS.md transcribed. A region without Maricopa would exclude CAP's largest customer and
have to guess what share of the cut lands inside it — which is exactly the guess
`region_share_of_az_reduction = 0.60` was.

### Two constants had been reasoned from the wrong map

| constant | was | now | why |
|---|---|---|---|
| `region_share_of_az_reduction` | 0.60, band 0.40–0.80 | **0.95, band 0.85–1.00** | its source string said "Maricopa is not in-region"; it always was. Still `UNTESTED` — the residual is on-river 4th-priority water, and CAP deliveries by county would replace it |
| `irrigated_acres` | 681,143 | **718,832** | NASS 2017 Table 10 summed over the eight *named* counties (Graham 46,682 + La Paz 97,139) divided by the area of the eight *selected* ones. Maricopa (180,214) and Gila (1,296) in, the other two out; `irrigated_fraction` 2.452 % → **2.588 %** |

`irrigated_acres` is also the calibration `ndvi_endpoints.py` uses to turn HUC12 withdrawal into
irrigated area, so §20 was re-run. The baseline-controlled slope went **+0.3353 → +0.3177**, the
t-statistic did not move (33.38 → 33.38: the regressor was rescaled, nothing else changed), and
`ndvi_irrigated_crop` ships at **0.5344** (was 0.5520), still inside the assumed 0.45–0.65 and
2.8 % from its midpoint rather than 0.4 %. **The irrigation → NDVI lever does not move**: it is
`irrigated_fraction × slope`, the fraction rose 5.5 % and the slope fell by the same factor, because
both are calibrated on the same acres. The sweep reads +9.35 before and +9.35 after.

### What moved in the app

Every Lake Mead path scales by 0.95 / 0.60 = 1.58×. Nothing else does.

| slider (policy min → max) | grace | ndvi | groundwater | surface water | wildlife |
|---|---|---|---|---|---|
| Lake Mead, before | +1.32 | 0.00 | −4.22 | +0.17 | +0.04 |
| **Lake Mead, after** | **+2.09** | 0.00 | **−6.69** | **+0.27** | **+0.07** |

The Lake Mead → groundwater band narrows from **7×** ([§15a](#15a-parameter-bands-on-the-cards-measured))
to **4.3×**: at −64 ft for 12 months, +0.94 ft of depth, band +0.43 to +1.84. It is still the
widest human band on that card, and for the same reason — `groundwater_substitution_fraction`
(0.30–0.70) is untouched by any of this. All six gates pass; every lever holds its sign in all 12
months; Layer 1 is bit-identical across every human lever.

### One latent fix, deliberately not propagated

`water_stress.py` weighted Maricopa's DSCI by Graham's area and Gila's by La Paz's. Re-fetching all
eight counties and re-weighting moves the index by **r = 0.99916** (levels) and **0.9984**
(month-to-month change). The weights are corrected in the script; the CSV is not regenerated,
because `usdm_dsci` is a feature in three learned models and regenerating it is a retrain for a
0.08 % change in correlation. Whoever next retrains Phase 2 should regenerate it first
(PROBLEMS.md P8, item 3).

### What this does to §16

Item 2 — CAP monthly deliveries — gets better, not worse: with all three CAP counties in-region,
deliveries by county would turn `region_share_of_az_reduction` into a near-identity rather than an
apportionment, and the acquisition's whole value concentrates on the GRACE pumping proxy. The rest
of the ranking stands. And the guard: `load_county_boundary()` now asserts that each FIPS code's
shapefile `NAME` matches the name listed beside it, and was verified to fire by putting "Graham"
back.

## 23. CAP deliveries acquired; the Lake Mead lever measured; GRACE still null `MEASURED`

[§16](#16-paths-forward) item 2, done. One acquisition was supposed to serve two problems — the
last monthly pumping proxy for GRACE, and the county-level data
`region_share_of_az_reduction` asked for — and it served exactly one of them.

### The data

`scripts/phase1/cap_deliveries.py` → `data/Final/cap_deliveries_monthly.csv`: **331 months,
1999-01 to 2026-07**, total CAP deliveries in acre-feet with the M&I / agricultural / federal split.
Source is CAP's own published delivery reports — monthly reports by classification for 1999–2018,
year-to-date reports by contract type for 2018–2026 — not Reclamation, because Reclamation's
series turned out to be the wrong one (below). Every year passes a guard: the twelve months must
sum to the printed annual total within 0.2 %. Two years (2008, 2009) are scanned images; OCR
mis-read two cells of one and could not find the other, so their four totals rows were transcribed
from the page images and pass the guard exactly. The script fetches the PDFs if they are absent
(`data/raw/` is not committed).

**Reclamation's Havasu diversion is not a delivery series, and it took the cross-check to see it.**
The decree accounting reports carry "Central Arizona Project, pumped from Lake Havasu" monthly;
the row is machine-readable for 19 of 26 years and agrees with CAP's totals on the year
(deliveries / diversion = 0.968). Month by month the two run at **r = −0.2**: CAP pumps at Havasu
in winter to fill Lake Pleasant and delivers out of it in summer. PROBLEMS.md Option B named the
diversion as the thing to fetch. Had it been used, the "pumping proxy" would have peaked in
January.

The series does what Option B hoped: it is 91 % a summer template, the rest is policy. Annual
totals run 1.4–1.7 MAF through 2021, then **984 kAF (2022, Tier 1), 774 (2023, Tier 2a), 859
and 872 (2024–25, Tier 1)** — the shortage tiers, visible in the region's water.

### The Lake Mead lever's first link, measured

`scripts/phase3/cap_calibration.py` → `model/cap_calibration.json`. Every acre-foot CAP does not
deliver is undelivered inside the region (all three CAP counties are in-region,
[§22](#22-the-region-was-never-the-one-this-document-named-measured)), so the share of a declared
Arizona cut that reaches the region is
`(baseline − actual annual deliveries) / declared cut`, specified before the numbers were looked at:
Tier ≥ 1 years judged, 2015–2019 baseline (1,406 kAF/yr), band over years × two baselines.

| year | tier | declared cut | delivered | drop | ratio |
|---|---|---|---|---|---|
| 2020 | 0 | 192 | 1,425 | −19 | −0.10 |
| 2021 | 0 | 192 | 1,323 | 83 | 0.43 |
| **2022** | **1** | **512** | **984** | **422** | **0.82** |
| 2023 | 2a | 592 | 774 | 632 | 1.07 |
| 2024 | 1 | 512 | 859 | 548 | 1.07 |
| 2025 | 1 | 512 | 872 | 534 | 1.04 |

**`region_share_of_az_reduction` ships at 1.0, band 0.82–1.0, `MEASURED`** (mean 1.001 over
2022–2025; raw band 0.824–1.208, capped at 1.0 because the lever multiplies a *declared* cut and
the excess in 2023–2025 is compensated system conservation running alongside the tier). The two
Tier 0 years are reported and not judged: a 192 kAF DCP contribution that the agreement allowed to
be met from ICS and conservation credits did not show up as lost deliveries, and the ratios say so.
The constant has now been 0.60 (assumed, Maricopa thought out-of-region), 0.95 (reasoned, §22) and
1.0 (measured). Every Lake Mead path scales by 1.0 / 0.95:

| Lake Mead, policy min → max | grace | groundwater | surface water | wildlife |
|---|---|---|---|---|
| §22 | +2.09 | −6.69 | +0.27 | +0.07 |
| **§23** | **+2.20** | **−7.04** | **+0.28** | **+0.07** |

The band on Lake Mead → groundwater is 4.45× (at −64 ft for 12 months: +0.99 ft, +0.41 to +1.84),
and it is now almost entirely `groundwater_substitution_fraction` (0.30–0.70) plus the storage
coefficient. That is the honest picture: the tier table is published, the delivery loss is
measured, and what remains unknown is how much of the lost water is pumped instead of fallowed.

### As a GRACE feature: NULL, by the rule fixed before the run

`scripts/phase2/experiment_grace_cap.py` → `model/experiment_grace_cap.json`. The same design as
[§11.5](#115-one-open-question-and-the-experiment-that-settles-it-measured): both arms under
`model_grace.py`'s real nested tuner, identical rows (168, 2002-10..2020-12), the same five test
blocks, the same three-part rule declared in the docstring. The CAP block is seven columns
(deliveries, lag1, lag3, roll3, roll6, roll12, and a 12-month trailing sum, because a storage
integral responds to cumulative delivery), deseasonalized inside each fold like §10's human block.

| arm | features | target R² | level R² | skill vs persistence |
|---|---|---|---|---|
| shipped | 45 | +0.0394 | +0.4389 | +0.0020 |
| **shipped + CAP, deseasonalized** | 52 | **+0.1059** | +0.4826 | +0.0457 |
| shipped + CAP, raw | 52 | +0.0309 | +0.4355 | −0.0014 |

The shipped arm reproduces §11.5's shipped arm to the fourth decimal, so the comparison is on the
same footing. Then:

```
  Δ target R²  +0.0665       PASS
  wins         5 of 5        PASS
  paired t     +1.52         FAIL  (rule: ≥ 2.0)
```

**NULL. The shipped feature set stands.** And it fails for the reason §11.5 failed: fold 0 — the
28-training-row fold, 2005–07 — supplies **72 %** of the gain (+0.24 of +0.33 summed), which is
what a one-fold sample-size effect looks like. Two observations that do not change the verdict,
recorded because the next reader will make them:

- The raw arm is *worse* than shipped. The signal, if there is one, is the anomaly from the summer
  template, not the template. That is consistent with a pumping proxy and inconsistent with
  "month_sin by another name".
- Post hoc, folds 1–4 alone are +0.023 mean, 4 of 4, t ≈ 3.3. That is not a verdict — the rule was
  fixed with five folds and the first fold was known to be the weak one when it was fixed — but it
  is the reason a longer record would be worth re-running this on. The series runs to 2026; the
  panel stops at 2020-12 because irrigation and public supply do. Extending those two is what
  would give this test the rows it needs.

The proxy-sanity numbers are in the JSON and are mostly a caution: deseasonalized CAP deliveries
correlate **+0.46** with the HUC12 irrigation withdrawal, not negatively, because that matrix counts
surface-water deliveries as withdrawals; it cannot be used to check substitution. The correlation
with GRACE's monthly change is −0.19 for the month and +0.07 for the 12-month sum.

### What it settles

- **§16 item 2 is closed.** The acquisition is done, the first link of the Mead lever is
  `MEASURED`, and GRACE's answer is a second independent NULL with the same fold-0 signature.
  Five `UNTESTED` constants remain: `groundwater_substitution_fraction` and the four streamflow
  constants ([§21](#21-the-streamflow-constants-cannot-be-identified-from-this-panel-measured)).
- **Nothing is retrained**, so the DSCI reweighting from [§22](#22-the-region-was-never-the-one-this-document-named-measured)
  stays latent. It rides with the next retrain, whenever something earns one.
- **PROBLEMS.md P5 moves from "weakly observed" toward "unobserved at monthly grain."** The one
  monthly series that responds to policy rather than the calendar lifts GRACE by a sixth of its
  fold spread. GLDAS (§16 item 5) is the last feature route on the list, and it targets the fast
  part of the change, not pumping.

## 24. CAP on GRACE's full window: null, and smaller. CAP is closed `MEASURED`

The roadmap's step 1. [§23](#23-cap-deliveries-acquired-the-lake-mead-lever-measured-grace-still-null-measured)
was a null whose gain sat 72 % in a 28-training-row fold, and its window stopped at 2020-12 only
because the harness inherits the human-block window where irrigation and public supply end.
GRACE's shipped features and the CAP series both run to 2023-12 and beyond, so the honest next
question was whether the fold-0 signature was a sample-size artifact or a real effect that more
rows would confirm. `experiment_grace_cap.py --window full` scores the same three arms, the same
tuner, the same block and the same three-part rule on GRACE's own window — 2002-10..2023-12, the
204 rows `grace_cv_results.json` reports, minus the GRACE-to-GRACE-FO gap — and its docstring
declares it the second and last CAP run before the numbers were seen.

| arm | features | target R² | level R² | skill vs persistence |
|---|---|---|---|---|
| shipped | 45 | +0.0210 | +0.3372 | −0.0277 |
| **shipped + CAP, deseasonalized** | 52 | **+0.0361** | +0.3473 | −0.0176 |
| shipped + CAP, raw | 52 | +0.0286 | +0.3363 | −0.0286 |

```
  Δ target R²  +0.0151       PASS
  wins         4 of 5        PASS
  paired t     +1.27         FAIL  (rule: ≥ 2.0)
```

**NULL, and the direction of the change is the finding.** With 36 more months and a first fold
that trains on more rows, the gain fell from +0.0665 to **+0.0151** — a quarter — and the first
fold's contribution fell from +0.24 to +0.01. That is what a small-sample artifact does when the
sample grows; a real effect would have held its size and gained significance. The post-hoc
"folds 1–4" observation §23 recorded does not survive either: on this window the fold that loses is
fold 3, in the middle of the record, not the first.

Two things stay true and are worth keeping:

- **The block is not noise.** It wins 4 of 5 here and 5 of 5 in §23, and in both runs the
  deseasonalized arm beats the raw arm. Whatever CAP carries about pumping is real, and it is
  worth about 0.015 of target R² on a residual whose fold-to-fold spread is 0.30. That is the
  "weakly observed" of [PROBLEMS.md P5](PROBLEMS.md) measured with the best monthly proxy there is.
- **The shipped arm scores lower on the full window** (+0.0210 vs +0.0394): the 2021–2023 block
  is a shortage era the training folds never saw. GRACE's difficulty is not the feature set.

**CAP is closed for GRACE.** No third window, rule or block will be tried; the series stays in
`data/Final/` for the Lake Mead calibration ([§23](#23-cap-deliveries-acquired-the-lake-mead-lever-measured-grace-still-null-measured))
and for the per-well groundwater design, where deliveries by customer put the proxy at the scale
it acts on. The roadmap moves to step 2, GLDAS as GRACE features, and step 3, the per-well panel.

## 25. GLDAS acquired; null by the rule; and the rule tested the wrong thing `MEASURED`

The roadmap's step 2 ([§16](#16-paths-forward) item 5, PROBLEMS.md Option C). GLDAS-2.1 Noah
land-surface state — soil moisture in four layers, snow water equivalent, canopy storage,
evapotranspiration — is an observationally-forced estimate of the fast, weather-driven part of the
storage change GRACE measures. It was the last feature route on the list for GRACE.

### The data

`scripts/phase1/gldas.py` → `data/Final/gldas_monthly.csv`: **288 months, 2000-01 to 2023-12**,
bounding-box means over 293 land cells, fetched as DAP4 subsets (~90 KB each) from Earthdata's
cloud OPeNDAP service rather than as 288 global 24 MB granules. Auth is an Earthdata bearer token
outside the repo; the GES DISC application had to be approved on the account first, which the
script's 403 message now says. Guards: complete months, constant cell count, soil moisture in
range (322–464 mm over 0–200 cm), snow peaking in January, rain peaking in August, and the GLDAS
rainfall forcing at **r = 0.89** with the MERRA-2 precipitation series already in the panel.

### The physics, before any model

On the 204 scored rows, the same-month change in GLDAS's storage proxy (soil + snow + canopy)
correlates **+0.62** with GRACE's monthly change, **+0.54** after deseasonalizing both. The
in-sample slope is **+0.8 m of GRACE per m of GLDAS**, which is what it should be if GLDAS's
storage sits inside GRACE's and the remaining fifth is groundwater. (That slope also exposed a
labelling error: `grace_groundwater.py`'s docstring says centimetres, and the raw `lwe_thickness`
attribute says metres. The values were always metres; nothing numerical changes.)

Then the number that reframes the model: **a one-coefficient OLS on that single GLDAS column, fit
inside each fold and scored on the same five blocks, gets out-of-fold target R² of +0.244**
(folds +0.41, +0.43, −0.24, +0.46, +0.16). The shipped 45-feature model scores **+0.021** on those
rows.

### The declared test: NULL

`scripts/phase2/experiment_grace_gldas.py` → `model/experiment_grace_gldas.json`. Eight-column
GLDAS block added to the shipped feature set, judged raw (GLDAS is a physical state, not a
template), same nested tuner, same five blocks, same three-part rule, one run on GRACE's own window.

| arm | features | target R² | level R² | skill vs persistence |
|---|---|---|---|---|
| shipped | 45 | +0.0210 | +0.3372 | −0.0277 |
| **shipped + GLDAS, raw** | 53 | **+0.0891** | +0.3616 | −0.0033 |
| shipped + GLDAS, deseasonalized | 53 | +0.0596 | +0.3602 | −0.0047 |

```
  Δ target R²  +0.0682       PASS
  wins         5 of 5        PASS
  paired t     +1.34         FAIL  (rule: ≥ 2.0)
```

**NULL, and the shipped feature set stands** — the rule was fixed before the run and it holds.
Fold 2 (2011-10..2016-08, where the shipped model scores −0.38) carries 79 % of the gain, which is
the same one-fold signature as §11.5, §23 and §24.

### What the rule did and did not test

The rule asks one question: does adding a block to the shipped XGBoost, under its own 60-draw
tuner, on ~200 rows, produce a significant gain? Three blocks have now answered NULL. But the OLS
line above is a different question with a different answer: **a single physical coefficient
captures a signal the tuned tree ensemble cannot**, by a factor of ten out of fold, on identical
rows. The estimator is the constraint. A 45-feature gradient-boosted model on a 200-row panel is
free to fit anything, and fold by fold it fits the wrong thing; a model with one degree of freedom
and the right physics cannot.

This is not a result about GLDAS as a feature and it is not a licence to tune. It is the
measurement that separates the two remaining explanations for GRACE's floor — "the data cannot
carry a model" and "this model cannot carry the data" — and it lands on the second. PROBLEMS.md
M4 says *"no solution available at Phase 2"*; that was written when every estimator tried was the
same estimator.

### What follows, and what is decided here

Nothing ships from this section. GLDAS stays in `data/Final/`, unused by any exported model. The
roadmap's step 2 is closed as written. What it opens is a **step 2b — a model-class experiment,
not a feature experiment**: a linear (or ridge) residual model for GRACE on a small, physically
chosen feature set — GLDAS storage change, precipitation, temperature, the lag-1 anchor — evaluated
under exactly the same nested folds and the same three-part rule against the shipped model's
+0.021. If it passes, GRACE ships as a different kind of model from the other five, and that is an
architecture decision, not a modelling one. It is left for the owner to make; the harness is one
script away. If it is not taken, the roadmap's step 5 — formal demotion of GRACE to climatology
plus Layer 2 — remains the honest shipping fix.

One practical note for a future deployment: GLDAS's storage change is 46 % explained by the
panel's precipitation alone and 69 % with lags, temperature and season, so a frontend driver from
the precipitation slider (the way DSCI is driven from PDSI) is feasible; GLDAS would not need a
control of its own.

## 26. The estimator was the floor: a four-input linear model gives GRACE its first skill `MEASURED`

The roadmap's step 2b, and the answer to the question
[§25](#25-gldas-acquired-null-by-the-rule-and-the-rule-tested-the-wrong-thing-measured) left:
is GRACE's floor the data or the model class? `scripts/phase2/experiment_grace_linear.py` →
`model/experiment_grace_linear.json`.

### The test, declared before the run

Same harness as §23–§25: GRACE's own window, 204 rows, the same five test blocks, training
strictly before each, residual over the lag-1 anchor. The shipped arm is re-run under
`model_grace.py`'s real nested tuner so the comparison is self-contained (it reproduces §24 and
§25's shipped arm to the fourth decimal). The judged arm is a **ridge regression on four inputs**,
standardised, with the penalty chosen by an inner `TimeSeriesSplit(3)` inside every fold:

| input | why it is there |
|---|---|
| `gldas_tws_proxy_delta` | the physics — soil + snow + canopy change, slope +0.8 m/m against GRACE (§25) |
| `precipitation_mm_day` | this month's rain |
| `precipitation_mm_day_lag1` | last month's rain, the recharge lag |
| `temperature_2m_c_anomaly` | the evapotranspiration and drought departure |

The set was fixed before the run as the smallest that names a driver for each part of the
monthly storage change, and the docstring forbids editing it after the numbers are seen. Two
arms are reported and not judged: the one-column OLS from §25, and a ridge on the shipped 45
features plus the 8-column GLDAS block, to separate "linearity" from "small feature set".

### The result

| arm | features | target R² | level R² | skill vs persistence |
|---|---|---|---|---|
| shipped (XGBoost, nested) | 45 | +0.0210 | +0.3372 | −0.0277 |
| **linear_physical (ridge)** | **4** | **+0.2845** | **+0.5664** | **+0.2015** |
| linear_gldas1 (OLS) | 1 | +0.2442 | +0.5614 | +0.1965 |
| linear_all (ridge) | 53 | +0.2396 | +0.5338 | +0.1688 |

Persistence on these blocks is +0.3649.

```
  Δ target R²  +0.2635       PASS
  wins         4 of 5        PASS
  paired t     +2.52         PASS  (rule: ≥ 2.0)
```

**REAL.** The first positive verdict this document has recorded for GRACE, after five nulls
(§10, §11.5, §23, §24, §25). Fold by fold the shipped model went −0.29 → +0.34, +0.25 → +0.48,
−0.38 → −0.11, +0.27 → +0.47, and lost one: +0.26 → +0.24 in 2021–2023, the shortage era, where
the shipped model was strongest and the two are within noise. The level R² of +0.57 against
persistence +0.36 gives GRACE **+0.20 skill**; PROBLEMS.md's status table has carried −0.03 since
the nested-CV correction.

### What the two reported arms say

- `linear_all` (+0.24) scores like `linear_physical`, not like the XGBoost arms. **The win is
  linearity**, or more exactly the absence of a tree ensemble's freedom: ridge with 53 inputs still
  cannot fit the wrong thing the way 60 draws of boosted trees can on 200 rows. The small feature
  set adds +0.04 on top, which is the four inputs being the right four.
- `linear_gldas1` (+0.24) is 86 % of the judged arm. Most of what GRACE's monthly change contains
  is GLDAS's storage change, and the shipped model could not see it through 44 other columns.

The full-window fit lands on alpha = 10 with standardised coefficients of +0.0076 (GLDAS
change), +0.0011 and +0.0007 (rain, this month and last) and −0.0040 (temperature anomaly): every
sign is the physical one, and GLDAS carries seven times the weight of rain, which is what the
+0.8 slope implied.

### What this changes in the record

PROBLEMS.md M4 says *"no solution available at Phase 2"* and P5 says the driver is *"weakly
observed"*. Both were measured with one estimator class. The correct statement is now: **the
monthly change in GRACE storage is largely the land-surface storage change, which GLDAS observes
well, and a model with four degrees of freedom captures it where a model with hundreds could
not.** The pumping part — the slow residual — is still unobserved, and the +0.28 target R² is a
measurement of how much of the month-to-month signal is *not* pumping.

### What is decided here, and what is not

Nothing ships from this section either, and that is deliberate. The docstring declared that REAL
means GRACE's shipping model becomes this linear model, and it does — but shipping it is an
architecture change with a defined cost, and it is listed rather than taken:

1. `scripts/phase2/model_grace.py`: estimator becomes the standardised ridge on the four inputs,
   export via `skl2onnx` instead of `onnxmltools`; the sidecars and `historical_grace.csv` follow.
2. `scripts/phase2/merge.py` and `features.py`: the GLDAS series joins the panel and the GLDAS
   change is engineered as a feature.
3. `frontend/catalog.js` and its Python mirror: GLDAS's storage change needs a driver. It is 46 %
   explained by the precipitation slider alone and 69 % with lags, temperature and season (§25),
   so a `gldasFromPrecip` regression in the style of `dsciFromPdsi` is the design; GLDAS gets no
   control of its own. This is the piece that makes the GRACE card respond to the rain slider
   through the physics rather than through a tree's memory of it.
4. The drought-index regeneration that has waited since
   [§22](#22-the-region-was-never-the-one-this-document-named-measured) rides with this retrain,
   which means all six models are retrained and PHASE2_REPORT.md's leaderboard is re-measured.
5. The six gates, `top_inputs.py`, and `generate_stats.py` run again; GRACE's card provenance
   changes from "learned, no skill" to "learned, skill +0.20".

That is one working session, and it changes what the application's GRACE output *is*. It should
be a decision, taken with this table in front of whoever takes it, not a side effect of a probe.

## 27. GRACE ships as the linear model; the retrain, and what moved `MEASURED`

The decision [§26](#26-the-estimator-was-the-floor-a-four-input-linear-model-gives-grace-its-first-skill-measured)
listed was taken, and this is the record of doing it.

### What changed in the code

- `scripts/phase2/model_grace.py`: the estimator is a standardised ridge (`StandardScaler` +
  `RidgeCV`, alpha by inner `TimeSeriesSplit(3)`), exported through `skl2onnx` as one graph with
  the scaler inside, because `models.js` feeds raw values. The export is round-tripped through
  onnxruntime before it is written. The XGBoost search is kept under its old name so §11.5, §23–§26
  can still reproduce their "shipped" arm; it trains nothing.
- `scripts/phase2/features.py`: GRACE's spec carries `fixed_features`, the four inputs, with a
  comment that editing the list is a new pre-declared experiment, not a tweak. `gldas_tws_proxy_delta`
  is engineered from the GLDAS series `merge.py` now joins.
- `frontend/catalog.js` and the Python mirror: `gldas_tws_proxy_delta` is derived from the rain and
  temperature sliders by an OLS `generate_stats.py` writes into `computed_stats.json["DERIVED"]`
  (R² 0.61, n 287) — the same pattern as `dsciFromPdsi`. GLDAS gets no control; it is a physical
  state, and it carries no human input, so Layer 1 stays climate-only.
- `scripts/phase1/water_stress.py`'s corrected county weights were finally applied: the drought
  index was regenerated (r = 0.9992 with the old series) and every model retrained on it, which is
  the bundle [§22](#22-the-region-was-never-the-one-this-document-named-measured) promised.
- `scripts/phase2/export.py` records the xgboost / scikit-learn / numpy versions in
  `model_comparison.json`, for the reason below.

### The leaderboard, before and after

| model | skill before | **skill after** | level R² after | target R² after | why it moved |
|---|---|---|---|---|---|
| surface water | +0.6830 | **+0.6833** | 0.7519 | 0.7267 | noise |
| wildfire | +0.3716 | **+0.4125** | 0.3227 | 0.3227 | environment, see below |
| NDVI | +0.2653 | **+0.2588** | 0.7862 | 0.5453 | drought index regenerated |
| wildlife | +0.2046 | **+0.2167** | 0.2247 | — | noise |
| **GRACE** | **−0.0349** | **+0.2015** | **0.5664** | **0.2845** | **the ridge** |
| groundwater | +0.0086 | **−0.0123** | 0.3872 | −0.1075 | drought index regenerated; fold noise |

The deployed GRACE reproduces §26's judged arm to the fourth decimal, and its card now says
"responds mainly to land-surface storage change (GLDAS), temperature, precipitation" — written by
`top_inputs.py` from the ridge's standardised coefficients, not by hand.

**Wildfire moved +0.04 with nothing changed, and that was checked rather than accepted.** Its
inputs carry neither DSCI nor GLDAS. Its feature matrix was rebuilt from the previous commit in a
worktree and diffed cell by cell against the current one: identical. Two trainings in one process
agree to every decimal. So the committed artifact and the current one differ only in the library
environment — `requirements.txt` is unpinned — and GridSearch picked a different winner
(fold 0 went 0.215 → 0.423, the rest within 0.02). The leaderboard is reproducible within an
environment and not across them, which is now written into `model_comparison.json` with the
versions that produced it. Pinning is the fix and is left as a note, not done here.

### What moved in the app

The human rows of the sweep did not move at all: Layer 2 was not touched, and the no-double-count
gate still reports 360 bit-identical comparisons. The climate rows did:

| slider (policy min → max) | grace | ndvi | groundwater | surface water | wildfire | wildlife |
|---|---|---|---|---|---|---|
| precipitation, before | −3.38 | +4.96 | −0.20 | +36.70 | −14.50 | +10.37 |
| **precipitation, after** | **+7.61** | +5.16 | −0.40 | +36.70 | −16.59 | +10.37 |
| temperature, before | +0.18 | +0.47 | −0.31 | −0.28 | +9.56 | +1.05 |
| **temperature, after** | −0.47 | **−2.68** | −0.51 | −0.28 | +8.77 | +1.05 |
| drought (PDSI), before | −1.73 | −4.80 | −0.15 | −1.06 | +0.01 | +7.35 |
| **drought (PDSI), after** | 0.00 | −1.39 | 0.00 | −1.06 | −1.12 | +7.35 |

Two of those are corrections of sign. **GRACE's rain response was −3.38 under the XGBoost: more
rain, less stored water.** It is +7.61 under the ridge, through GLDAS's storage change, which is
the direction water goes. **NDVI's temperature response was +0.47: hotter, greener.** It is −2.68
after the retrain. Neither sign had been flagged before because [§19](#19-the-signs-that-look-wrong-and-which-one-actually-was-verified)
checked the *human* levers' signs; the climate term's signs were the learned models' business.
GRACE no longer responds to the PDSI slider because it no longer carries the drought index, and
its drought response now arrives through rain and temperature instead.

All six gates pass: every structural lever holds its sign in all 12 months, Layer 1 is
bit-identical across the human levers, the JS and Python catalogs agree on every scenario including
the new derived feature, and the frontend starts under the browser-conditions check.

### What is still true

The pumping residual is unobserved at monthly grain. GRACE's +0.28 residual R² is the land-surface
part of the change; the slow part still belongs to Layer 2, where it always did. Groundwater is at
zero skill for the fifth consecutive re-run and stays there until the per-well design (roadmap step
3) is tried. And the four GRACE inputs are fixed: a fifth is a new experiment with a declared rule,
not an edit.

## 28. The groundwater index is two aquifers, and only one of them is predictable `MEASURED`

The roadmap's steps 1 and 2 for the groundwater model, run together. Step 2 is a null by its
declared rule. Step 1 explains the null, and is the result.

### Step 1: what the target is an index of

`scripts/phase3/groundwater_diagnosis.py` → `model/groundwater_diagnosis.json`. The target
`depth_to_water_anomaly_ft` is the mean of per-well anomalies over whichever USGS daily-value
wells report each month — 10 to 42 of 66. The rebuild reproduces the shipped series exactly
(r = 1.000000). By county:

| county | wells | median reporting per month | with 10+ years |
|---|---|---|---|
| Cochise | 44 | 20 | 14 |
| Pima | 14 | 10 | 11 |
| Yuma | 5 | 1 | 1 |
| Maricopa | 2 | 0 | 0 |
| Pinal | 1 | 1 | 1 |

The USGS daily-value service does not monitor the Phoenix AMA; ADWR does. So the region's pumping
heartland, where all of its CAP water goes and where Layer 2's Lake Mead mechanism acts, has three
wells in the index. Cochise County — the Willcox and Douglas basins, agricultural pumping outside
any AMA for most of the record, no CAP water at all — has 44, and is 59 % of the roster in a typical
month (25 % to 76 %).

**The two halves do not move together.** Over 252 shared months the Cochise and Pima sub-indices
correlate **−0.13** in level and **+0.07** in month-to-month change. Their trends are +0.01 ft/yr
(Cochise, flat) and **−2.39 ft/yr** (Pima — the water table *rising* 2.4 ft a year under the Tucson
AMA's managed recharge). Their month-to-month volatility is 0.47 ft (Cochise) against **2.98 ft**
(Pima). So although Cochise supplies six wells in ten, the shipped index's monthly change is
**81 % Pima** (r = +0.81 with the Pima change, +0.31 with Cochise): ten wells in a managed, injected,
recovered aquifer set the noise, and thirty wells in a climate-and-pumping aquifer are averaged
into it. Roster churn, the artifact P3 fixed, is no longer the problem: a fixed roster of the 27
wells with ten or more years reproduces the shipped monthly change at r = +0.96.

**And the Cochise half is the one the panel can see.** Monthly change of each index against the
drivers, Pearson r:

| index | GLDAS storage change | rain | irrigation (deseasonalized) | CAP (deseasonalized) |
|---|---|---|---|---|
| shipped | −0.20 | −0.14 | +0.14 | +0.07 |
| **Cochise** | **−0.59** | **−0.44** | **+0.25** | +0.12 |
| Pima | −0.08 | −0.05 | +0.12 | +0.06 |

Every Cochise sign is physical (wet month → water table up → depth down; more pumping → depth
down), and the GLDAS correlation is as strong as GRACE's. The Pima wells respond to nothing in the
panel, because what moves them is Tucson Water's recharge and recovery schedule, which is an
operations decision, not weather and not the regional irrigation total.

### Step 2: the declared test, and its null

`scripts/phase2/experiment_groundwater_linear.py` → `model/experiment_groundwater_linear.json`.
The §26 design on groundwater: a ridge on five physical inputs (irrigation deseasonalized in-fold,
GLDAS storage change, rain, rain lag 1, temperature anomaly) against the shipped in-fold
competition, re-run, same five blocks, same rule.

| arm | features | target R² | level R² | skill |
|---|---|---|---|---|
| shipped (in-fold competition) | 16 | −0.0048 | +0.4141 | +0.0146 |
| linear_physical (judged) | 5 | +0.0128 | +0.4245 | +0.0250 |
| linear_climate | 4 | +0.0115 | +0.4198 | +0.0204 |
| linear_gldas1 | 1 | +0.0248 | +0.4305 | +0.0311 |

```
  Δ target R²  +0.0175       PASS
  wins         4 of 5        PASS
  paired t     +0.83         FAIL  (rule: ≥ 2.0)
```

**NULL.** Every arm, linear or not, sits at ±0.02 on this target. For groundwater the estimator
was never the floor; there is nothing in the blended index for any model class to learn.

### The exploratory number that says what to do — labelled as such

Not a declared test and not a verdict: the same five-input ridge, the same folds, applied to each
sub-index as its own residual target.

| target | target R² (five folds) | level R² | persistence | skill |
|---|---|---|---|---|
| shipped index | +0.013 (−0.05, 0.00, 0.04, 0.06, 0.02) | 0.42 | 0.40 | +0.03 |
| **Cochise sub-index** | **+0.353 (0.10, 0.26, 0.54, 0.51, 0.35)** | 0.56 | 0.33 | **+0.23** |
| Cochise, 10-year+ wells only | +0.329 | 0.50 | 0.27 | +0.24 |
| Pima sub-index | −0.005 | 0.35 | 0.34 | +0.01 |

Five positive folds on the Cochise index, a skill of +0.23 that matches the deployed GRACE, and a
Pima index that is exactly as unpredictable as the blend. **The groundwater model has been at zero
for five re-runs because its target averages a forecastable aquifer with an unforecastable one,
and the unforecastable one carries the variance.**

### What this means, and what is not decided here

Nothing ships from this section. The choice is about what the groundwater output *is*, and it
belongs to the owner:

- **A. Make the output the Cochise index.** "Well depth vs normal, Willcox and Douglas basins." It
  is 44 of the 66 wells, it is the region's unregulated agricultural pumping, the irrigation lever
  is physically right for it (and the §12 calibration would need re-running on it, with the
  storage coefficient likely to change), and it forecasts. The Lake Mead lever would have to come
  **off** that card: no CAP water reaches Cochise County, and a CAP-substitution mechanism applied
  to those wells is wrong in a way the current blend merely hides.
- **B. Two groundwater outputs.** The Cochise index as above, plus a Tucson AMA index that carries
  the Lake Mead and CAP mechanism and is honest about having no forecast skill from ten wells. The
  second becomes a real model only when ADWR's GWSI wells for the three AMAs are acquired (roadmap
  step 4), which puts the target where Layer 2's mechanism already is.
- **C. Keep the blend and relabel it.** Honest, and it forecasts nothing.

The per-well panel (roadmap step 3) is still worth building, but this section changes what it is
for: not to rescue the blend, but to give the Cochise model its wells' own basins and, once ADWR
data exists, to hold the AMA wells beside them.

## 29. The groundwater output is the Cochise index, and it forecasts `MEASURED`

Option A of [§28](#28-the-groundwater-index-is-two-aquifers-and-only-one-of-them-is-predictable-measured),
taken. `depth_to_water_anomaly_ft` is now the per-well anomaly index over the 44 Cochise County
wells (`groundwater_levels.py`, `TARGET_COUNTY_FIPS = 4003`); the old blend over every county is kept
beside it as `depth_to_water_anomaly_ft_allwells`, and reproduces the previous target to the last
digit. The card is "Well Depth vs Normal (Cochise basins)". The index's standard deviation is
0.65 ft where the blend's was 1.18; its 5th-to-95th-percentile range, which is the card's 0–100
scale, is **2.06 ft** where the blend's was 19.2.

### The model, and how it was chosen

The declared test (`experiment_groundwater_linear.py`, second run, rule and tie-break in its
docstring) compared the five-input ridge against "the shipped arm" on the Cochise target. The
ridge lost every fold (t = −2.54): **NULL**, and by the declared consequence the in-fold
competition was retrained on the new target. Then two things surfaced, and both are recorded
because they changed what shipped:

1. **The experiment's "shipped" arm was not the shipped pipeline.** It ran the in-fold competition
   on the 16 features the old model had exported, where the real pipeline selects in-fold from 63.
   That arm scored +0.456 target R²; the real pipeline, retrained, scored **+0.156** (skill +0.07).
   Against the real pipeline, on the same five blocks (the fold boundaries were checked to be
   identical), the linear arms score: five-input ridge Δ +0.197, 4 of 5, t = 1.84; climate-only
   ridge Δ +0.200, 4 of 5, t = 1.81; one-column OLS on GLDAS change Δ +0.244, 4 of 5, t = 2.08.
   The declared comparison, executed correctly, is a null by 0.16 of a t-statistic. It does not
   license the linear model on forecast grounds.
2. **The retrained pipeline's climate responses have the wrong physical sign.** In the sweep, more
   rain deepened the Cochise water table (+2.97 points) and hotter months made it shallower
   (−5.01). The climate-only ridge on the same target has every sign physical: GLDAS storage
   change −0.19, rain −0.06, rain lag −0.05, temperature anomaly +0.06 (standardised; positive
   is deeper). And the pipeline's winning candidate is the XGBoost-plus-elastic-net "blend", whose
   ONNX export drops the elastic-net half, so its shipped file is not the model that was scored.

So the decision was made on signs, not on the score: the declared forecast test says the two are
not distinguishable, and between forecast-equivalent candidates the one whose responses carry the
physical sign ships — the same standard [§19](#19-the-signs-that-look-wrong-and-which-one-actually-was-verified)
applies to the human levers and [§27](#27-grace-ships-as-the-linear-model-the-retrain-and-what-moved-measured)
found violated in two learned climate terms. **Groundwater ships as a standardised ridge on four
climate inputs** — GLDAS storage change, rain, rain lag 1, temperature anomaly — pinned in
`features.py` as `fixed_features`, exported through skl2onnx with the scaler inside and
round-tripped through onnxruntime, exactly as GRACE is. The in-fold competition stays in
`model_groundwater.py` for the experiment scripts and trains nothing.

| groundwater | skill | level R² | target R² | persistence |
|---|---|---|---|---|
| blend target, competition (pre-§28) | −0.0123 | 0.3872 | −0.1075 | 0.3995 |
| Cochise target, competition retrained | +0.0695 | 0.4009 | +0.1563 | 0.3313 |
| **Cochise target, four-input ridge (ships)** | **+0.2269** | **0.5582** | **+0.3565** | 0.3313 |

Folds of the shipped model, target R²: 0.08, 0.26, 0.56, 0.51, 0.37. This is the first
groundwater model in the project's history with skill, and it took a target change plus a sign
test, not a feature or a tuner.

### What changed in Layer 2

- **`storage_af_per_ft` recalibrated on the Cochise index** (`aquifer_calibration.py`): 707,463
  AF/ft at the 12-month horizon (t = 1.44), band 466,032–707,463; the 1-month coefficient is
  321,539 (t = 2.84) and the irrigation regression holds its sign under climate (t = 1.78) and
  trend (t = 1.70) controls. The first run of the calibration after the target change came back
  identical to the old one and was caught: it had read the processed panel before the retrain
  rewrote it. Regeneration order matters and is now in the checks note.
- **The Lake Mead → groundwater lever is removed.** No CAP water reaches Cochise County; a
  CAP-substitution mechanism has nothing to act on there. It stays on GRACE, which integrates the
  CAP basins, and returns to a groundwater card only when a Tucson-AMA index exists (ADWR wells,
  roadmap step 4).
- **The population and public-supply levers are scaled by `cochise_municipal_to_irrigation_share`
  = 0.191** (band 0.128–0.270), `MEASURED` by `cochise_share.py` from the HUC12 matrices assigned
  to counties by representative point: Cochise has 8.7 % of regional irrigation withdrawal and
  1.7 % of regional public supply (75 % of which is Maricopa). The storage coefficient is
  calibrated on regional irrigation, so it already carries the irrigation share; a municipal lever
  built from a regional withdrawal has to be scaled by the ratio of the two shares, or Phoenix's
  taps would be credited to Willcox's water table. Implemented as an optional `scale` field on a
  lever, in `structural.js` and the Python mirror alike, and the band machinery picks it up.

### The sweep

| slider (policy min → max) | groundwater before | **groundwater after** |
|---|---|---|
| population | +6.09 | +6.09 |
| irrigation | +60.11 | **+193.22** |
| public supply | +18.03 | +11.09 |
| urbanization | 0.00 | 0.00 |
| Lake Mead | −7.04 | **0.00** (lever removed) |
| precipitation | −0.40 | **−16.99** |
| temperature | −0.51 | +0.85 |
| drought (PDSI) | 0.00 | 0.00 |

All six gates pass. The climate rows are now physical and large: rain matters to the Cochise water
table, which the blend had hidden.

**The irrigation row saturates the card, and that is the yardstick, not the physics.** +193 points
is 4.0 ft for the full −60 % to +40 % swing of regional irrigation held for a year, on an index whose
entire historical range is 2.06 ft. The feet are credible: Cochise's own irrigation is ~9 % of the
regional total, and the Willcox basin declines 2–5 ft a year under actual pumping. What is small
is the index's historical variation, because a per-well anomaly over a slowly declining basin is
smooth. The output score is defined as the 5th–95th percentile of the historical series for every
output; on this one that makes the bar clamp at 100 for most of the irrigation slider's range. That
is reported here rather than fixed, because the fix is a decision about the score definition, not
about the model: a physical scale for this card (±5 ft, say) would keep the arithmetic and stop the
clamping, at the cost of the one rule every card shares.

### Two things this leaves open

- **A climate-sign gate.** The acceptance gate checks every human lever's sign in all 12 months. It
  has no declared signs for the learned climate responses, and §27 and §29 have now found four
  wrong-signed ones in two models. The table is short — rain lowers depth, raises GRACE, streamflow
  and NDVI, lowers fire; heat does the reverse — and it belongs in `slider_sensitivity.py`.
- **Out-of-sample rows.** The daily well pull stops at 2020-12; NWIS runs to 2025. Sixty new months
  would be the first genuinely held-out test of the Cochise model, and the way to judge the
  one-column GLDAS model (t = 2.08 against the pipeline here) without another post-hoc comparison.

## 30. A climate-sign gate, and what it found `MEASURED`

[§7](#7-acceptance-criteria) gates every human lever's sign in all 12 months and has since §13. The
learned climate responses had no such gate, and [§27](#27-grace-ships-as-the-linear-model-the-retrain-and-what-moved-measured)
and [§29](#29-the-groundwater-output-is-the-cochise-index-and-it-forecasts-measured) found four
wrong-signed ones by reading sweep tables. `slider_sensitivity.py --mode climate-signs` now declares
the sign each climate slider must have on each output — rain raises storage, greenness and flow and
lowers depth and fire; heat does the reverse; no expectation for birds under heat — swings each
slider across its policy range in every month, and fails any pair with a wrong-signed month above
0.5 points (below that the model is saying "no response", which is not a sign). It exits nonzero.

**One declaration was corrected after the first run, and this is the record of it.** PDSI →
wildfire was declared "−" and failed 11 of 12 months at +1.6 to +5.5 points: sustained wetness
*raising* fire. `features.py` documents why that is not wrong: "wet winter → dry summer (fuel growth
then ignition)" and "prior 2-year precipitation total (fuel load accumulation)". The PDSI slider is
a wetness index held for the whole scenario, reaching the fire model through 12-month rolls, so
both mechanisms are in play and no single sign follows from physics. The entry is now `None`.
Same-month rain → wildfire keeps its "−" and passes in all 12 months at 4.6 to 26.2 points, which is
the suppression mechanism showing where it should.

### What passes and what fails

| climate slider → output | expected | months ok | verdict |
|---|---|---|---|
| rain → GRACE | + | 12/12 | pass |
| rain → NDVI | + | 8/12 | **fail** |
| rain → groundwater depth | − | 12/12 | pass |
| rain → streamflow | + | **6/12** | **fail** |
| rain → wildfire | − | 12/12 | pass |
| rain → wildlife | + | 12/12 | pass |
| heat → GRACE | − | 12/12 | pass |
| heat → NDVI | − | 10/12 | **fail** |
| heat → groundwater depth | + | 12/12 | pass |
| heat → streamflow | − | 12/12 | pass |
| heat → wildfire | + | 12/12 | pass |
| PDSI → GRACE, groundwater | (no PDSI input) | — | pass |
| PDSI → NDVI | + | 7/12 | **fail** |
| PDSI → streamflow | + | **0/12** | **fail** |
| PDSI → wildfire | none | — | pass |
| PDSI → wildlife | + | 12/12 | pass |

**The two ridge models pass everywhere. Every failure is an XGBoost.** Month by month, in score
points for the slider's full swing:

```
                              Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
rain  -> NDVI          [+]   -2.1  +5.9  +6.9  +3.2  +1.0  -0.1  +5.2 +17.9  +9.4  -0.8  -2.6  -1.4
rain  -> streamflow    [+]   +4.0  +2.3  -1.0  -6.9  -3.5  -0.9 +36.7 +14.5  -3.3  -5.3  -0.4  +8.9
heat  -> NDVI          [-]   +1.2  -3.5  -0.5  +0.2  +0.1  +0.4  -2.7  +0.2  -5.1  +2.4  -0.4  -0.2
PDSI  -> NDVI          [+]   +2.0  +1.4  +1.5  +1.7  +1.8  +0.7  -1.4  -3.3  -1.9  -4.0  -2.9  -0.3
PDSI  -> streamflow    [+]   -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1  -1.1
```

The one that matters is **rain → streamflow**. The +36.7 in July is the number every sweep table in
this document has shown, and it is real. But the same model says more rain means *less* flow in
March through June and September through November, by up to 6.9 points, in months where the rain
slider's whole swing is 0.14 to 0.5 mm/day. ~~A tree ensemble fit to a monsoon-dominated record has
learned July and does something else the rest of the year.~~ **Corrected in §31: the deployed
streamflow model is the ridge candidate, not a tree, and the mechanism is different and more
interesting — it has learned a derivative, and a sustained slider asks it a level question.**
Streamflow is the project's best model (skill +0.68), so this is not a reason to replace it. PDSI →
streamflow is a genuine wrong sign: −1.1 in every month, and it fails as a one-month pulse too. The
three NDVI failures are small (mostly 1–4 points) and month-scattered; NDVI *is* an XGBoost, and
that is the signature of a tree splitting on month.

### What to do about it, not done here

XGBoost accepts `monotone_constraints`: a declaration that the response must be non-decreasing in
a named feature. Declaring rain and PDSI monotone-positive for streamflow and NDVI (and the lag and
rolling families with them) is the principled fix, and it is a modelling change to two shipped
models with skill, so it is an experiment with a rule — does the constrained model keep its
out-of-fold skill? — and a decision, not a patch. **Taken: §31.** Until then the gate is in the
README as a known failure with a count, and a change in that count is what a regression looks like.

## 31. The constraint experiment, and what the streamflow model turned out to be `MEASURED`

[§30](#30-a-climate-sign-gate-and-what-it-found-measured)'s proposal, run:
`scripts/phase2/experiment_monotone.py` → `model/experiment_monotone.json`. Each model's real
training procedure, on its own window and its own five outer folds (reproducing the shipped scores
to the fourth decimal), run twice: as shipped, and with every XGBoost it builds carrying
`monotone_constraints` — non-decreasing in the nClimDiv rain and PDSI families for streamflow;
non-decreasing in rain and non-increasing in the DSCI drought index for NDVI; interactions and
temperature free. The rule was declared in reverse, the burden on the loss: the constrained model
ships unless it loses more than 0.03 of target R², or loses significantly (|t| ≥ 2).

| model | arm | target R² | level R² | skill | Δ | wins | t | verdict |
|---|---|---|---|---|---|---|---|---|
| streamflow | shipped | +0.7267 | +0.7519 | +0.6833 | | | | |
| streamflow | constrained | +0.7249 | +0.7497 | +0.6811 | −0.0018 | 4/5 | −0.69 | **ships** |
| NDVI | shipped | +0.5453 | +0.7862 | +0.2588 | | | | |
| NDVI | constrained | +0.5085 | +0.7699 | +0.2424 | −0.0368 | 3/5 | −0.62 | **does not ship** |

NDVI's loss is one fold (2015–2017: 0.462 → 0.195) and it crosses the material line; the
unconstrained NDVI stands and its small, month-scattered sign failures stay on the record. Streamflow
ships constrained, at a cost of nothing — and then the retrained model's sign gate did not improve,
which was the declared trigger for stopping and reporting rather than iterating.

### The constraint was inert, because the streamflow model is not a tree

`surface_water_cv_results.json` says `winner: ridge`, before and after. The in-fold competition
has been picking the ridge candidate for this model all along; the XGBoost only ever runs the
feature selection. So a constraint on XGBoost changed the selected set (26 → 23 features, the
retrained model is kept as the declared outcome) and never touched the estimator the app runs.
§30's sentence about a tree ensemble learning July was wrong, and is struck there.

What the ridge actually learned, read off the exported graph's coefficients on the rain features
(per raw unit):

```
nclimdiv_precipitation_mm_day                +0.21      this month's rain
nclimdiv_precipitation_mm_day_lag1           -0.19      last month's rain
nclimdiv_precipitation_mm_day_anomaly        +0.20      this month vs the trailing year
nclimdiv_precipitation_mm_day_anomaly_roll3  -0.19      the last three months vs the trailing year
```

**It is a derivative.** Flow-change this month rises with rain this month and falls with rain last
month: the model has learned that a fast-memory river responds to *changes* in rain, which is
exactly right for forecasting the residual over last month. A sustained slider then asks it a
question the structure cannot answer: with a year of extra rain, this month's rain and last month's
move together, the +0.21 and −0.19 nearly cancel, and what is left is decided by the shape of each
month's climatology. In April the March swing exceeds April's, so the net is negative (−5 points,
with `lag1` contributing −0.088 and the anomaly −0.082); in July the current month dominates
(+40 points, +0.31 from this month's rain alone); in October the roll term wins and it is negative
again. None of that is the model being wrong about rain.

### The pulse

Run the same gate as a **one-month pulse** — the delta applied to the current month only, which is
the question the residual model was trained on — and the picture changes:

| pair | sustained 12 mo | 1-month pulse |
|---|---|---|
| rain → streamflow | 8/12 | **12/12**, +4 to +44 points |
| heat → streamflow | 12/12 | 11/12 (one month at +0.57) |
| *pairs failing* | *5 of 18* | *5 of 18, three of them under a point* |
| PDSI → streamflow | 0/12 | **0/12**, −1.24 |
| rain → NDVI | 8/12 | 11/12 |
| heat → NDVI | 10/12 | 11/12 |
| PDSI → NDVI | 7/12 | 9/12 |

The gate now runs both: the app's default duration is gated, the pulse is reported beside it, and
the docstring says what the difference means. A pair that fails the sustained test and passes the
pulse is a **display limit of the residual architecture**: [§13](#13-layer-2-and-layer-3-built-measured)
chose not to integrate the learned residual over the scenario, so for a sustained scenario the card
shows a one-step change as if it were a level. A pair that fails both is a **wrong sign in the
model**. PDSI → streamflow is the second kind: the PDSI features carry mixed and net-negative
coefficients (`roll6` +0.083 against `roll3` −0.048, `lag3` −0.024, `roll12` −0.021, level −0.005),
which is collinearity with rain absorbing a negative partial effect. It is 1 to 3 points.

### What follows

- **The monotone constraint stays in `model_surface_water.py`.** It costs nothing, it is declared,
  and it would bind if the competition ever picked the XGBoost again. It is not a fix for anything
  that is wrong today.
- **For the ridge, the analogue is sign-constrained coefficients** (non-negative on the rain family,
  non-negative on PDSI), or dropping the collinear PDSI family from streamflow. Either is a declared
  experiment with the same rule. The PDSI one is worth 1–3 points; the rain one is not a model
  problem.
- **The sustained-scenario display is the real open item, and it is §13's decision revisited.**
  §13 declined to integrate the learned residual because, with human deltas still inside Layer 1,
  integration amplified D2's wrong signs. Layer 1 is now climate-only and two of the residual
  models are linear. Whether integrating the learned climate residual over the scenario, in the
  §4b mean-reverting form, now gives sustained responses with the pulse's signs is a testable
  question with the sign gate as its criterion. It is the next experiment for this card, and it
  is not taken here.

## 32. §13 revisited: integrating the learned residual does not ship, and cannot fix a sign `MEASURED`

[§31](#31-the-constraint-experiment-and-what-the-streamflow-model-turned-out-to-be) left one
experiment open: whether the sustained-scenario display of the residual models — a one-step change
shown as if it were a level — improves if Layer 3 integrates the learned climate residual over the
scenario the way it integrates the structural forcing. [§13](#13-layer-2-and-layer-3-built-measured)
had said no with human deltas still inside Layer 1; Layer 1 is climate-only now and two of the
residual models are linear, so the question was re-asked.
`scripts/phase3/experiment_integrate_learned.py` → `model/experiment_integrate_learned.json`: the
shipped stack with the switch flipped in memory, judged by criteria declared in its docstring —
no new sign failures, rain → streamflow must pass, no climate swing beyond 100 points, human gates
untouched.

### What the switch does

| climate slider → output | off | **on** | multiplier at 12 mo |
|---|---|---|---|
| rain → GRACE | +7.6 | +74.2 | 9.75 |
| rain → NDVI | +5.2 | +22.1 | 4.28 |
| rain → groundwater | −17.0 | **−187.1** | 11.02 |
| rain → streamflow | +40.1 | **+113.7** | 2.83 |
| heat → groundwater | +0.9 | +9.3 | 11.02 |
| PDSI → streamflow | −3.1 | −8.7 | 2.83 |

The human rows are bit-identical, the acceptance and no-double-count gates pass, and the
climate-sign gate reports **five failures before and five after, the same five**. Rain → streamflow
still fails.

```
  [PASS]  no new sign failures            5 vs 5
  [FAIL]  rain -> streamflow passes       FAIL
  [FAIL]  largest climate swing <= 100    187.1
  verdict: DOES NOT SHIP
```

### Why it was never going to fix the sign

§4b's closed form is `z_n = (s/λ)(1 − (1−λ)^n)`: the one-step residual `s` times a positive
constant. A positive multiplier cannot change a sign. The wrong-signed April is in `s` itself — the
derivative structure §31 read off the ridge, evaluated on a catalog where every lag already carries
the scenario — and integrating it just makes a wrong-signed month bigger. This should have been
said before the run rather than after, and it is said here so the next reader does not run it a
third time.

What integration does do is scale, and the scale is the second failure: for the slow outputs the
multiplier is 10 to 11, which takes the Cochise water table to −187 points (−3.9 ft for a wet
year, on a 2-ft card) and GRACE to +74. §13's +104 for streamflow is back as +114. The switch
stays off, for the same reason as §13 and one more: the models carry no dynamics of their own, and
a fitted λ applied to a one-step residual is not a substitute for them.

### What would actually answer the sustained question

A change model answers a level question only by being *rolled out*: month 1 with the scenario in
the current month and normals in the lags, month 2 with one month of scenario in the lags, and so
on, each step's prediction feeding the next step's anchor. That is [§4a](#4a-the-layer-3-experiment-measured)'s
rollout mode, which exists in `slider_sensitivity.py`, and which was unbounded for the XGBoost
models because they carry no restoring force. Whether it is bounded for the two ridges is a
question, not a plan; the fast outputs it matters for are still the XGBoost NDVI and the ridge
streamflow with its derivative, and the streamflow rollout would compound the same derivative. The
honest state of the sustained display is what §31 said: label the learned climate term as a
one-month response, and let the duration selector act on the structural layer, where it already
does.
