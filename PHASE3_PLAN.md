# Phase 3 Plan — The Sliders Do Not Work

**Status: complete. All seven steps of §6 are done — the sliders work, every sign is stable,
and each output card states where its number came from. The §7 gate passes on all 20
lever/output pairs.**

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
- **No model is retrained for deployment** under this plan. The learned layer is the part that works.

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

`groundwater` is the one model whose Layer 1 **already contains irrigation** — it is one of its
three human features — and it already responds: **+0.36 score points** across the policy range
(§9). Layer 2 will add a structural irrigation term on top of a learned model that already has
one, so for that single path the effect is counted twice.

At an estimated ~4 structural points ([PHASE3_PARAMS.md §2](PHASE3_PARAMS.md)) the learned share
is about 9%, so the pragmatic answer is probably to state it and accept it. But it has to be a
decision rather than an oversight, and the same check is owed to Mead → groundwater. GRACE is
clean: `extended: True` dropped irrigation from its feature set, so there is nothing to
double-count there — which is also why the GRACE sweep reads exactly 0.00 for that lever.

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

### 11.5 One open question, and the experiment that settles it `UNTESTED`

GRACE scored **better** with the deseasonalized human block than with its shipped feature set — on
both the residual it is trained on (−0.0119 → +0.0539) and the reconstructed level
(+0.4121 → +0.4462), on identical test rows.

That cannot be claimed against the deployed model. Every §10 variant used one fixed XGBoost
configuration rather than each model's own nested tuning, so the comparison is internally valid
but its "shipped" column is not the shipped model's real score. Settling it means running
`model_grace.py`'s actual pipeline on the human-block feature set and comparing to
PHASE2_REPORT.md's number. It is the one place where
[§8](#8-what-this-does-not-change)'s "no model is retrained for deployment" may be leaving
something on the table, and it should be tested rather than assumed either way.

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
- the [§11.3](#113-a-double-counting-hazard-the-plan-does-not-cover-verified) double-count of
  irrigation in the groundwater model is gone by construction rather than by accounting,
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
| irrigation → grace | corroborated | 18.76 |
| irrigation → ndvi | structural-only | 9.30 |
| irrigation → groundwater | corroborated | 7.82 |
| public supply → grace | structural-only | 5.63 |
| urbanization → ndvi | structural-only | 3.73 |
| population → grace | structural-only | 3.09 |
| public supply → groundwater | structural-only | 2.35 |
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
| Population | −3.09 | 0.00 | +1.29 | +1.74 | — | +0.43 |
| Irrigation | −18.76 | +9.30 | +7.82 | −2.37 | — | −0.59 |
| Public supply | −5.63 | 0.00 | +2.35 | −0.72 | — | −0.18 |
| Urbanization | 0.00 | −3.73 | 0.00 | +3.30 | — | +0.82 |
| Lake Mead | +1.32 | 0.00 | −0.55 | +0.17 | — | +0.04 |

**Every human slider now reaches four of the six outputs**, against §1's baseline where the
largest human effect anywhere was 1.7 points and four sliders were 0.00 everywhere. All 20
lever/output pairs hold their sign in all 12 months.

The wildlife column is small — 0.04 to 0.82 points — because it is a second-order effect reached
through one attenuating edge, and the conservative end of that edge's band shipped. At the band's
upper end (+0.1953, the significant fit) it is 3.6x larger. That range belongs on the card, which
is Layer 4.


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
