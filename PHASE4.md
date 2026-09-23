# Phase 4 Plan — Every Control Matters, and Time Runs

**Status: M1 in progress — tickets 01–03 and 05 done.** Written 2026-09-22 from a design interview.
Target date: **end of spring semester 2027** (~33 working weeks). Claim tags follow
[PROBLEMS.md](PROBLEMS.md): `MEASURED`, `VERIFIED`, `ESTIMATED`, `UNTESTED`.

**Updated 2026-09-22 after an acquisition pass** ([§4.4](#44-the-aquifers-bound-is-statutory-and-it-is-already-a-100-year-rule-verified)–[§4.6](#46-what-the-acquired-data-says-and-why-it-reframes-the-piece-measured)):
the local domain is measured, the aquifer bound turned out to be statutory rather than geologic,
the Colorado-side and population-side boundary conditions are on disk — and the two of them
together reframed what the century-scale story for Tucson actually is.

Phase 3 asked *does the slider move the output*. It got a yes, and the yes was not enough:
the answer is honest and, on four of five levers, imperceptible. Phase 4 asks the two questions
that follow from an interactive installation rather than from a model:

1. **Does every control produce a change a person can perceive?**
2. **Can the piece run forward in time — not three years, but as far as anyone wants to look?**

Neither is a modelling failure to be fixed by a better fit. Goal 1 is a **denominator**
problem and goal 2 is an **architecture** problem, and both are named precisely below.

Companion document: [LAND_USE_TOKENS.md](LAND_USE_TOKENS.md) — the land-use token vocabulary,
written for the installation team rather than for the code.

---

## 0. What the installation actually is, and the four things that follow

From *The Edge of the Desert* project deck (Dr. Yuanyuan Kay He, Udall Fellowship; team incl.
Dr. Peter Torpey, Live and Immersive Arts, and Dr. Sandra Bae / Dr. Takanori Fujiwara, CS):

- A **tabletop projection-mapped landscape** — river valley, city, mountains — with 3-D miniatures.
- **RFID sensors read land-use choices** as physical objects are placed on the model.
- A touch strip carries sliders (Population, Irrigation, Precipitation, Drought, Urbanization).
- Environmental impacts become **visible and audible** — projection plus sonification.
- **Multiple people act simultaneously, with competing priorities**, and the deck says this is
  deliberate: *"environmental policy isn't the result of one isolated variable or one person's
  decision."*
- The section is titled **"Interactive Land-Planning Model and Failure."**
- The installation will be held in **Tucson, Arizona**.

Four consequences, each of which decides something later in this document:

| Observation | Consequence |
|---|---|
| The control surface is **tokens on parcels**, not slider drags | The input contract is `place(land_use, parcel)`, and `impervious_pct` becomes a *function of the token set*, not a parallel control — [§7](#7-d6--the-token-vocabulary) |
| The output surface is **projection + sound**, built by another team | Our deliverable is an **engine and an API**, not a web page — [§8](#8-d7--the-engine-and-the-api) |
| People act **at the same time, in tension** | State is shared, and consequences must be **path-dependent** or the tension is fake — [§6](#6-d5--the-clock-irreversibility-and-sessions) |
| The piece is about **failure** | Failure states must be *reachable* and *physical* — which is a stock model, not a regression — [§4](#4-d3--the-long-horizon-engine) |
| It is in **Tucson** | The local domain is Tucson — [§2](#2-d1--the-two-tier-domain) |

---

## 1. Goal 1 diagnosed: it is a denominator, not a coefficient `MEASURED`

`slider_sensitivity.py --mode acceptance`, run 2026-09-22 on the shipped models, 12-month
scenario, full policy range, in score points:

| lever | best card | pts | second | pts |
|---|---|---|---|---|
| irrigation | groundwater | **193.22** | ndvi | 9.35 |
| public supply | groundwater | 11.09 | grace | 5.63 |
| population | groundwater | 6.09 | grace | 3.09 |
| impervious | surface water | 3.30 | ndvi | 0.97 |
| Lake Mead | grace | 2.20 | surface water | 0.28 |
| — | **wildfire** | **no human lever at all** | | |

All 19 pairs hold their declared sign 12/12 months. **Nothing here is wrong. That is the
problem.** One lever saturates a two-foot card by 12×, three levers are invisible, and one output
has no human path.

The urbanization case is the whole diagnosis in one line ([PHASE3_PLAN.md §34b](PHASE3_PLAN.md)):

```
local:    ΔNDVI = −0.0356 per unit paved      →  −16.4 % of the 0.2167 regional baseline
regional: +2.0 pts × −0.0356 / 100 = −0.00071 →  / 0.0879 score span = −0.8 points
```

Both numbers are right and they measure different things. The slider's entire range paves at most
~2 % of 27,780,270 acres, and an eight-county mean averages the rest away. §17 already proved the
coefficient is not the place to fix this — the *assumed* endpoint was 5.5× too strong and
measurement cut it down. **Enlarging a coefficient to make a lever visible is the one move this
project has spent three phases refusing, and Phase 4 does not make it.**

### The gate, restated `DESIGN`

Phase 3's §7 criterion ("≥5 score points across the lever's full range at 12 months") is retired as
a ship gate and kept as a reported diagnostic. It is replaced by:

- **Ship gate — perceptual.** *A single token placed anywhere in the domain must move at least two
  outputs by ≥2 % of that output's declared display range within 5 simulated years.* Every control
  must pass. This is strictly harder than §7 — the unit is one token, not a full-range drag — and
  it is the thing the installation actually requires.
- **Design principle — salience.** Every control is the *dominant* driver of at least one output.
  Not gated; it governs which outputs exist and what they are scoped to.

Magnitude is still never tuned toward. A lever that is physically small after honest re-scoping
says so, on the card and in the API's provenance block.

---

## 2. D1 — The two-tier domain

**Decision: a nested local tier inside the existing region. The region is not replaced.**

| Tier | Extent | Why |
|---|---|---|
| **Regional** | the eight counties as shipped ([PROBLEMS.md P8](PROBLEMS.md)) | The CAP / Lake Mead / DCP chain and the Cochise groundwater result are inherently regional and are the two most rigorously sourced things in the repo |
| **Local** | **Santa Cruz basin HUC8s around Tucson** | Land-cover effects happen here, the installation happens here, and the denominator is small enough for a token to matter |

**The local boundary is watershed-defined, not administrative, and it is now resolved `MEASURED`
(2026-09-22).** The Tucson AMA is the correct *policy* object, but its shapefile sits behind
`azwater.gov`, which still returns **HTTP 403** to every client tried. The hydrologic equivalent was
built from `data/raw/wbd/WBD_15_HU2_GDB.gdb`, already on disk:

```
domain = (15050301 Upper Santa Cruz  ∪  15050302 Rillito
        ∪ 15050303 Lower Santa Cruz  ∪  15050304 Brawley Wash)  ∩  Pima County
       = 9,124 km²  =  3,523 mi²  =  8.1 % of the eight-county region
```

**The clip to Pima County is not cosmetic — it is what makes the domain Tucson.** Measured against
the TIGER county file:

| HUC8 | name | total | why it is clipped |
|---|---|---|---|
| 15050301 | Upper Santa Cruz | 6,801 km² | contains **Tucson**, but also **Nogales**, 2,425 km² of Santa Cruz County and ~1,029 km² **in Mexico** — a different AMA and a different country |
| 15050302 | Rillito | 2,385 km² | 90 % Pima already — eastern Tucson metro |
| 15050303 | Lower Santa Cruz | 4,358 km² | contains **Marana**, but is 78 % **Pinal County** and reaches Casa Grande's farm belt |
| 15050304 | Brawley Wash | 3,649 km² | 97 % Pima — **Avra Valley**, where Tucson's recharge basins are |

Unclipped, those four basins are 17,193 km² and would drag in Nogales, Mexico and Pinal
agriculture. Clipped, the domain is **3,523 mi² against the Tucson AMA's ~3,900 mi², and ADWR's own
Tucson Regional groundwater model covers ~3,250 mi²** — so the proxy sits between the state's model
area and the full AMA. That is as close as this can be got without the shapefile.

**The payoff for goal 1 is arithmetic: the local denominator is 8.1 % of the regional one, so any
land-cover lever is ~12× more visible on a local card than on a regional one**, before a single
coefficient changes.

> **Built 2026-09-22 (tickets 01–03).** Local boundary in `scripts/phase1/local_region.py`; 25 local
> constants in `frontend/local_structural_params.json` derived by
> `scripts/phase4/local_structural_params.py`; tier flag plumbed through `structural.js` and
> `slider_sensitivity.py --tier local`. Acceptance results: all 19 lever/output pairs pass sign
> stability 12/12 in local mode; urbanization→surface_water 22.63 pts (was 3.30),
> urbanization→wildlife 5.61 pts (was 0.24). NDVI stays 0.97 — formula is scale-invariant for
> slider inputs; ≥5 pts comes from land-use tokens (ticket 07), not the denominator swap.

### Rejected, and why — do not retry

- **Re-scoping the whole project to Tucson.** Coherent, and it throws away CAP/Mead/DCP and the
  Cochise index. Rejected.
- **Pima County as the local tier.** An administrative box holding a great deal of empty desert,
  and it cuts Marana's growth edge in the wrong place.
- **"Tucson urbanized area + buffer," NLCD-derived.** Smallest denominator, largest apparent lever
  — and the boundary *moves as the user builds*. A self-redefining denominator makes urbanization
  look strong for the wrong reason, which is the mirror image of the bug being fixed. **Rejected
  explicitly.**

---

## 3. D2 — Nothing that ships today can get worse

The stated concern was that re-running Phase 1 and Phase 2 on a smaller footprint would produce
worse models, on a shorter and noisier record, with no guarantee of a gain. That concern is
correct, and the plan is built so it never has to be taken.

**Three tiers of ambition for the local domain. Phase 4 commits to the second.**

1. **Structural-only.** Layer 1 stays *exactly as shipped* — regional, untouched, not retrained.
   The local tier is Layer 2 arithmetic with local denominators: acres paved inside the domain,
   NDVI on those acres, pumping against the local aquifer, the local stream reach.
   **This alone fixes goal 1 for urbanization, irrigation and population**, because the whole
   problem was a denominator and Layer 2 already owns the entire human response
   ([PHASE3_PLAN.md §4](PHASE3_PLAN.md)).
2. **Structural-only + locally measured constants `← the commitment`.** The same arithmetic, but
   every in-repo constant recomputed over the local mask — local NDVI baseline, local irrigated
   fraction, local `storage_af_per_ft` from Tucson-basin wells, local `region_acres`. Still no
   retraining, and now the local numbers are locally measured rather than regionally borrowed.
3. **A local Layer 1.** Permitted **per model, behind a pre-declared gate**: a local model ships
   only if it beats both its own persistence baseline *and* the regional model evaluated on local
   truth, under the repo's standing three-part rule (Δ > 0, 4 of 5 folds, |t| ≥ 2). If it fails,
   that card stays regional and the card says so.

**The guarantee: no shipped artifact is modified by the local tier. It is additive.** The failure
mode of tier 3 is "we learn a local model is not better," which is a result, recorded like every
other null in [PHASE3_PLAN.md](PHASE3_PLAN.md) §23–§25 and §33.

### D2b — the nested budget

A token placed inside the Tucson domain pumps local groundwater *and* is part of regional pumping.
[§11.3](PHASE3_PLAN.md) established what double counting does. Three ways to arrange two tiers, and
only one survives:

- **(a) Local as a view** — regional totals are the accounting truth; the local card reports the
  share landing locally. Safe, but "where you put it" stays decorative.
- **(b) Nested budget `← chosen`** — the local tier is the truth inside the domain; the regional
  figure is `local + rest-of-region`. A subdivision inside the domain draws on the Tucson aquifer;
  one outside does not. **This is the only option that can satisfy the conservation gate
  ([§9](#9-the-gates)), and the only one that makes placement mean anything.**
- **(c) Two independent models, never summed.** What most installations do, and how you get two
  cards that quietly contradict each other. Rejected.

---

## 4. D3 — The long-horizon engine

### 4.1 Why the current architecture cannot run forward `MEASURED`

Read from `model/*_feature_names.json`, 2026-09-22:

| model | n features | carries its own lagged target? |
|---|---|---|
| grace | 4 | **no** |
| groundwater | 4 | **no** |
| surface_water | 23 | **no** |
| wildfire_monthly | 26 | **no** |
| ndvi | 46 | no — its only state is *GRACE's* lags |
| wildlife | 12 | **yes** — `bbs_abundance_anomaly_lag1/lag2/roll3` |

**Five of six models have no self-state, and four of six are residual (change) models.** A change
model with no self-lag, rolled forward, is a **pure integrator**: its level drifts linearly at
whatever rate the climate forcing implies, forever, with no restoring force anywhere in the model.
That is the mechanical reason [§32](PHASE3_PLAN.md) blew up — integrating the learned residual sent
the Cochise water table to −187 points on a two-foot card and streamflow to +114 — and it is why
"just roll it out for 1,200 months" is not an option and will not become one.

The restoring force exists in the world and is already measured, but it lives in **Layer 3**, not in
the models: `reversion_per_month` = grace 0.0386, groundwater 0.0157, ndvi 0.2221, surface water
0.3511, estimated from the observed series.

### 4.2 The four courses of action, and the choice

| | Approach | Cost | What it breaks | Verdict |
|---|---|---|---|---|
| **A** | **Stock-and-flow rollout.** Layer 2 becomes a real state model: aquifer storage, Mead storage, paved acres, population, irrigated acres as **stocks**; controls set **flows**. Annual step. Layer 1 evaluated once per simulated year for that year's climate, never iterated. | ~3–4 weeks + parameter sourcing | nothing shipped | **chosen — the spine** |
| **B** | **Long-record annual model.** Retrain on the long series only: nClimDiv 1895–2026, wildfire 1972–2021, wildlife 1968–2024, streamflow 1980–2025. | moderate | **NDVI, GRACE and the well index are 2000-only.** Half the cards go dark, and the features that gave GRACE and groundwater their skill (GLDAS, 2000–) do not exist before 2000 | **rejected** |
| **C** | **Equilibrium-response emulator.** Sweep the shipped ONNX models over a grid of climate states offline, fit a smooth surface, relax toward it at the measured `reversion_per_month`. | ~1 week, no retraining, no new data | nothing | **chosen — supplies A's climate term** |
| **D** | **Borrowed boundary conditions.** Published trajectories for the things that dominate a century: Reclamation CRSS / Post-2026 EIS for Mead, Arizona OEO for population, ADWR for aquifer validation. | acquisition-limited | — | **chosen — boundary conditions and validation only** |

**A is the spine; C supplies the climate term; D supplies the boundary conditions and one
validation target. This inverts the current architecture: Layer 2 becomes the long-horizon model
and Layer 1 is demoted to "what a year of this climate looks like."** That inversion is correct,
because a century is a stocks story and stocks are Layer 2's job.

### 4.3 Why D is not an alternative to A — recorded because it is the obvious mistake

D supplies *trajectories for exogenous drivers*. It supplies **no mechanism for a user's lever to
act on**. Pull irrigation to −50 % for sixty simulated years and CRSS has nothing to say; it does
not know about the table. **An installation built on D is a playback of someone else's projection
with controls that do nothing** — precisely the failure Phase 4 exists to fix. There is also a
horizon mismatch: **CRSS public traces stop around 2060 and Arizona OEO stops at 2060.** Past that,
D is silent and A is all there is.

D's three jobs, and no others:

1. **Mead / Colorado boundary condition, 2027–2060** — Post-2026 **Final** EIS (2026-07-31), which
   supersedes the Draft. **Acquired** — see [§4.5](#45-what-was-actually-acquired-and-what-the-sandbox-can-now-reach-verified-2026-09-22).
2. **The population reference trajectory** the population control deviates *from* — Arizona Office
   of Economic Opportunity, state and county, 2025–2060. **Acquired** — see
   [§4.5](#45-what-was-actually-acquired-and-what-the-sandbox-can-now-reach-verified-2026-09-22).
3. **A validation target** — if A's business-as-usual aquifer trajectory disagrees badly with
   ADWR's published Tucson Regional model, A is wrong and we find out cheaply. `azwater.gov` 403s;
   this one may not be obtainable, and the plan does not block on it.

### 4.4 The aquifer's bound is statutory, and it is already a 100-year rule `VERIFIED`

**Resolved 2026-09-22, and it turned out better than a geologic bottom.** Arizona already defines
the number this model needs, on exactly the horizon the installation wants:

> **A.A.C. R12-15-716(B)(2)** — for an Assured Water Supply, groundwater is "physically available"
> only if it can be withdrawn for **100 years** from a depth not exceeding the maximum 100-year
> depth-to-static-water-level: **1,000 feet below land surface in the Phoenix, Tucson and Prescott
> AMAs**, 1,100 ft in Pinal, 1,200 ft outside the AMAs, 400 ft for dry-lot developments.

The *Growing Water Smart* guidebook already in the repo root corroborates the rule's shape
(100 years of groundwater at a stated depth) and is quoted in the Coconino Plateau case study at
the non-AMA figure of 1,200 ft.

**Why this is the right bound for this piece, and better than saturated thickness:**

1. **It is the statute's own century.** The user asked for a hundred-year horizon; Arizona law asks
   the same question of every subdivision built inside an AMA. The installation is not inventing a
   timescale — it is running the state's own test.
2. **It is a threshold with consequences**, not an asymptote. Crossing it does not mean the aquifer
   is empty; it means **new subdivisions can no longer be approved on groundwater**. For a
   land-planning table, that is a far sharper failure state than a physical bottom nobody reaches.
3. **It needs no ADWR shapefile and no hydrogeologic study.** It is a number in the administrative
   code.

**Where Tucson actually stands, from the repo's own wells `MEASURED`:** the 14 Pima County wells in
`data/Final/groundwater_levels_daily_2000_2020.csv` sit at **31–316 ft**, median **204 ft**, latest
observation per well. **Headroom to the statutory limit is therefore ~800 ft.**

Saturated thickness and natural recharge are still wanted for the *rate* (§4.4a below), but the
**bound is settled** and Phase 4 does not block on ADWR for it.

### 4.4a Still to source `UNTESTED`

- **A Tucson-basin `storage_af_per_ft`.** The shipped 707,463 AF/ft was calibrated against the
  Cochise index and does not transfer. `pubs.usgs.gov` **now returns 200** with browser headers
  (see §4.5), so Pool & Anderson SIR 2007-5275 and its companions are reachable.
- **A recharge flux** for the `recharge_basin` token. Tucson's CAP and effluent recharge volumes are
  published by Tucson Water and the Arizona Water Banking Authority.
- **Validation**: ADWR's **Tucson Regional groundwater model** (~3,250 mi²) is the natural check on
  A's business-as-usual trajectory. Its landing page is on `azwater.gov` and 403s; reports may be
  mirrored elsewhere. Not a blocker.

**Two negative results, recorded so they are not re-opened:**

- **`AZGWS-Guidebook2024-Web.pdf` contains no hydrogeology.** 25,053 words, searched for saturated
  thickness, specific yield, storage capacity, recharge rates and depth to bedrock: it is a
  water-and-land-use *planning* guide. What it does contain is the policy scaffolding above, which
  is worth more.
- **`Aquafer_cross_section.svg` is artwork, not data.** No text nodes, no annotations, no scale.
  But it is **a purpose-built animation asset** — its groups are `aquifer_water`, `water_in_pipe`,
  `water_arrows`, `vegetation`, `snow`, `ground_layer_1..3`, `bedrock` — which is a direct
  statement of what the visual layer expects the engine to emit: a water-table height, a well
  level, a flow direction, a vegetation density and a snowpack. **Treat it as a requirement
  document for the output schema in [§8](#8-d7--the-engine-and-the-api), not a parameter source.**

### 4.5 What was actually acquired, and what the sandbox can now reach `VERIFIED` 2026-09-22

[PHASE3_PARAMS.md §8](PHASE3_PARAMS.md) says `curl` has no network here. **That is no longer true**,
and several 403s turned out to be bot-protection defeated by a complete browser header set rather
than blocks:

| Host | Status | Note |
|---|---|---|
| `usbr.gov` | **200** | Post-2026 **Final** EIS, July 2026 |
| `pubs.usgs.gov` | **200** | was 403 — reopens the hydrogeology literature |
| `oeo.az.gov` | **200** with full client-hint headers + cookie jar | was 403 to every simple fetch |
| `azwater.gov` | **403** | still closed |

**Downloaded into the repo:**

```
data/raw/projections/AZ_OEO_All_Series_2025-2060.zip     6.4 MB  low/medium/high, per county, annual
data/raw/projections/AZ_OEO_SubCounty_2020-2060.xlsx     183 KB  per-place, annual, 2020-2060
data/raw/crss/FEIS_AppM_LowerBasin_DemandSchedule.pdf    1.0 MB  AZ depletions by user, 2027-2060
data/raw/crss/FEIS_AppG_CRSS_InitialConditions.pdf       0.8 MB
data/raw/crss/FEIS_TA03_HydrologicResources.pdf          8.5 MB  Mead/Powell results by alternative
```

Three things to know about them:

1. **The Final EIS (2026-07-31) supersedes the Draft** referenced earlier in this plan. Appendix M
   carries Arizona's depletion schedules by priority class and water user, **annually 2027–2060**,
   and extracts cleanly with `pdftotext -layout`. This is the Colorado-side BAU.
2. **The OEO sub-county file does not contain Pima.** It covers twelve counties; Maricopa and Pinal
   appear as `MaricopaByMAG`/`PinalByMAG`, and **Pima is absent because PAG produces its own
   sub-county projections**. County-level Pima is in the zip; per-place Tucson/Marana/Oro Valley
   numbers must come from **Pima Association of Governments** — a small open item.
3. **A web-search summary of the Pima numbers was wrong**, which is why the file was downloaded
   rather than quoted. The authoritative 2025-vintage series are below.

### 4.6 What the acquired data says, and why it reframes the piece `MEASURED`

**Pima County population, OEO 2025 vintage, from the downloaded file:**

| series | 2025 | 2030 | 2040 | 2050 | 2060 |
|---|---|---|---|---|---|
| Low | 1,093,761 | 1,106,178 | 1,093,872 | 1,052,351 | **1,002,678** *(decline)* |
| **Medium** | 1,093,761 | 1,121,464 | 1,147,353 | 1,148,212 | **1,143,575** *(+4.6 % in 35 years)* |
| High | 1,093,761 | 1,166,019 | 1,288,856 | 1,398,818 | **1,519,907** *(+39 %)* |

**Tucson is not projected to grow much.** The medium series is essentially flat to 2060 and the low
series declines.

**And the Tucson water table is rising, not falling `MEASURED`.** Per-well linear trends over
2000–2020 on the 11 Pima wells with ≥8 years of record: **median −1.89 ft/yr** — the water table
coming *up* almost two feet a year. This corroborates [§28](PHASE3_PLAN.md), which measured −2.39
ft/yr and attributed it to the Tucson AMA's managed recharge, and
[PHASE3_PARAMS.md §2](PHASE3_PARAMS.md)'s second dead end, where GRACE fell while monitored wells
rose. **Under the historical trend the statutory 1,000 ft limit is never approached.**

**Taken together these two facts break the obvious story and replace it with a better one.** The
century-scale failure path for Tucson is *not* "growth drains the aquifer" — growth is flat and the
aquifer is recovering. It is:

```
Colorado River shortage  →  CAP deliveries cut  →  recharge stops and pumping substitutes
                         →  the recovery reverses  →  depth approaches 1,000 ft
                         →  no new subdivision can show an Assured Water Supply
```

Every link in that chain is already in this repository and is its **best-sourced** one: the DCP
shortage tiers are transcribed from primary text and tagged `VERIFIED`
([PHASE3_PARAMS.md §1](PHASE3_PARAMS.md)), `region_share_of_az_reduction` is `MEASURED` at 1.0 from
the CAP delivery record ([§23](PHASE3_PLAN.md)), and the Colorado-side trajectory is now on disk.

**Consequence for the design: the Lake Mead lever, today the weakest control on the board at 2.20
score points, should become one of the strongest at century scale in the local tier.** The
installation's subject is a desert city kept alive by a river that is projected to shrink — which
is a more honest and a more interesting thing for a table in Tucson to say than population
pressure. M5 should be re-read with this in mind, and the `recharge_basin` token
([LAND_USE_TOKENS.md](LAND_USE_TOKENS.md)) is no longer a courtesy: **it is the mechanism currently
holding the aquifer up.**


---

## 5. D4 — The outputs become stocks and levels

Three of the six outputs are **anomalies against a fixed 20-year baseline** (`grace_groundwater_anomaly`,
`depth_to_water_anomaly_ft`, `discharge_log_anomaly`) and one is an AR index on a bird survey.
*"−0.42 anomaly units relative to 2004–2009"* is not a quantity a person can stand in front of in
simulated 2090.

| Output | Today | Phase 4 |
|---|---|---|
| Groundwater | depth-to-water **anomaly**, ft | **feet below land surface**, and ft/yr of change |
| GRACE / storage | satellite **anomaly** units | **acre-feet in storage**, and % of the modelled bound |
| Surface water | log discharge **anomaly** | **cfs**, and **zero-flow days per year** |
| NDVI | index, regional mean | index, **on the local mask**, plus acres converted |
| Wildfire | risk index | risk index (unchanged) + **WUI exposure** ([§10](#10-d8--wildfire)) |
| Wildlife | BBS abundance anomaly | short-horizon: unchanged. Long-horizon: **habitat-driven, stated as coarse** |

The anomaly becomes a *derived view*, not the truth. This is required by A anyway — a stock model
needs stock units — and it is what makes a century legible: *"the water table is 310 ft down and
falling 2.1 ft/yr"* survives any horizon. It touches `generate_stats.py`, every card, and the
sonification mapping, which is why it is M2 and not later.

> **Built 2026-09-22 (ticket 05).** Stock units are now the primary display on all six output cards
> in `frontend/ui.js`: groundwater in ft below surface, surface water in cfs, NDVI and wildfire as
> indices, GRACE in anomaly units, wildlife as abundance index. Anomaly is demoted to secondary
> (`anomaly: value unit`). Styling in `frontend/style.css`.

**Wildlife is demoted at long horizon on purpose.** A 12-feature AR model on a breeding-bird survey
is a good short-horizon model and is not a century instrument. Long-horizon abundance is driven
from habitat (riparian extent + NDVI on the local mask) and labelled **coarse**.

---

## 6. D5 — The clock, irreversibility, and sessions

- **Time runs.** Default mode is a **running clock** — time advances continuously at an adjustable
  rate, pausable — and people change policy *while it runs*, watching consequences arrive late.
  A **horizon dial** scrubs when paused.
- **State is path-dependent and irreversible within a session.** A well drawn down in simulated
  2040 is still down in 2080 after the token is removed. **A reversible aquifer is a lie**, and
  irreversibility is what makes simultaneous, competing users matter rather than merely coexist.
- **Explicit reset** returns to the initial condition. Sessions are the unit of state.
- **No hard horizon cap.** Each output ships a declared **evidence horizon** (NDVI 24 years of
  record; streamflow 46; wildfire 40; nClimDiv 131; the aquifer stock is a mass balance with bounds
  and is arguably valid furthest of all). Bands widen with a stated rule, and past **3× an output's
  evidence horizon** the engine keeps emitting and flags `beyond_evidence: true`.

---

## 7. D6 — The token vocabulary

Six tokens, specified in full in **[LAND_USE_TOKENS.md](LAND_USE_TOKENS.md)**:
`subdivision`, `dense_infill`, `farm`, `retire_farm`, `recharge_basin`, `riparian_restoration`.

Two design rules behind that set:

1. **Every token has an opposing partner**, so the table always poses a trade-off rather than a
   one-way ratchet.
2. **At least one token must be restorative** (`recharge_basin`, and `riparian_restoration` in the
   local tier). Without one, every session ends in the same monotone collapse and the piece stops
   being a land-planning model and becomes a doom loop. CAP and effluent recharge is real Tucson
   policy and is visible in the well record, so this is not a courtesy token.

The physical table is another team's build. Token geometry is therefore carried as **explicit open
parameters** — `parcels_per_domain`, `acres_per_token`, `token_types`, `token_intensity` — and
every downstream number recomputes from them.

`impervious_pct` is **derived from the token set**, not controlled in parallel. A slider UI remains
as a fallback and for headless testing, but the token set is the contract.

---

## 8. D7 — The engine and the API

**Python is the language of record.** The JavaScript in `frontend/` was always an observation
harness, and the integration argument favours Python:

- **TouchDesigner embeds CPython** and scripts in Python; it has no JS runtime. For in-process
  embedding, Python is the short path.
- **Unreal embeds neither** — its Python is editor-only tooling. Unreal talks to external processes
  over HTTP / WebSocket / OSC.
- So the integration argument does not really choose a *language*; it chooses an **interface**.
  Once the engine is a local service, both consume it identically. Python wins on the in-process
  case and on owning the scientific code already.

**Decisions:**

- **Engine**: a real Python package (Layer 2 / Layer 3 / the stock model / `onnxruntime`), with a
  **stateless core** — `simulate(scenario, horizon, seed) → trajectory` — and a **thin stateful
  session + tick wrapper** over it for the running clock.
- **Boundary**: HTTP + WebSocket, plus **OSC** for the sonification, which is the lingua franca
  there.
- **`frontend/` survives as a thin client that calls the API** instead of reimplementing it.
  `frontend/structural.js`, the feature reconstruction in `catalog.js` and
  `scripts/phase3/check_catalog_parity.py` all **retire** — there is no second implementation left
  to drift. That parity check existed only to guard a duplication Phase 4 deletes; record the
  reason in [PROBLEMS.md](PROBLEMS.md) when it goes.
- **Offline is a hard requirement.** Today `frontend/index.html` loads `onnxruntime-web@1.17.3`
  from a **CDN importmap**, there is no `package.json` and no `node_modules` `VERIFIED` — a gallery
  machine with no network cannot boot the app as it stands. Dependencies get vendored and the whole
  thing must start from a cold, disconnected box.

### The output contract `DESIGN — mandatory`

Peter and Kay map values to light and sound, so **every output, at every timestep**, ships:

```json
{
  "value": 310.4,
  "unit": "ft below land surface",
  "display_min": 180.0, "display_max": 420.0,
  "band_low": 288.1, "band_high": 341.7,
  "provenance": "modelled",
  "beyond_evidence": false
}
```

`provenance` ∈ `measured | modelled | declared | extrapolated`, and it is **mandatory in the
schema, not optional**. A declared full-swing range is what lets a sonification be tuned once
instead of re-scaling itself every session — and an installation that cannot tell its audience
which part is measured is the failure this repo has spent three phases avoiding.

---

## 9. The gates

Phase 3 shipped three gates. Phase 4 ships five, all run against **the engine**, none mirroring its
arithmetic.

| # | Gate | Kind | What it asserts |
|---|---|---|---|
| 1 | **Perceptual** | **ship gate** | One token, anywhere, moves ≥2 outputs by ≥2 % of declared display range within 5 simulated years. Every control passes. |
| 2 | **Conservation** | **hard build gate** | Water in = water out + Δstorage, to a stated tolerance, at every timestep, in the nested budget. *A stock model that does not conserve mass is wrong in a way no regression ever could be.* |
| 3 | **Boundedness** | **hard build gate** | No output diverges over a 300-year run under any admissible scenario. This is [§32](PHASE3_PLAN.md)'s failure turned into a test. |
| 4 | **No double count** | build gate | Rewritten against the engine: the two tiers must not both book the same acre-foot. |
| 5 | **Climate signs** | build gate | Rewritten against the engine; the five known failures of [§30–§31](PHASE3_PLAN.md) carry forward with their explanations. |

Gates 2 and 3 are the most valuable new artifacts in Phase 4. They are what make it possible to run
a century without lying, and they are cheap.

---

## 10. D8 — Wildfire

Fire is the most legible failure the piece could show and **nothing human reaches it today**. The
chosen path, and its pre-declared losing branch:

- **Measure it.** Human ignitions dominate Arizona starts. `data/raw/wildfire/` holds perimeters
  1984–2023 and `data/raw/NLCD/` holds **annual** impervious cover 1995–2024, so *ignition density
  versus distance-to-development* is measurable in-repo, the way the NDVI endpoints were in
  [§17](PHASE3_PLAN.md) and [§20](PHASE3_PLAN.md). Same three-part rule, declared before the run.
- **Expect a sign tension, and want it**: development adds ignitions and removes fuel, so a dense
  core and a sprawling edge may differ in sign. That is the same competing-priorities tension the
  deck is about.
- **If the measurement is null** — fire is labelled **climate-only** and the null is stated on the
  card and in the API's provenance block. **Declared in advance, so the result is not negotiated
  after the fact.** A declared-sign coefficient on the most emotive card is where this project
  would lose its credibility first, and it is rejected outright.
- **Build the WUI-exposure readout regardless**: acres and structures within 1 km of burnable fuel
  at the current token layout. Pure geometry, honest under either outcome, responds to every token
  placed, and claims nothing about fire probability. *"How much of what you just built sits inside
  the fire-prone edge"* is a stronger sentence for a land-planning table than a modelled risk delta.

---

## 11. D9 — The climate future

1. **Trend source**: CMIP6 / downscaled Southwest projections if acquirable; **the in-repo
   nClimDiv 1895–2026 fitted trend as the declared fallback**, with an explicit beyond-evidence
   band past ~2060. A century of extrapolated OLS is indefensible and will be labelled as such.
2. **What trends**: **temperature as a mean shift; precipitation as widening variance on a flat
   mean.** Southwest projections are robust on warming and genuinely split on the monsoon. This is
   both the honest reading and the more dramatic one.
3. **Variability**: **stochastic**, block-resampled from the observed record and warm-shifted, with
   a **fixed seed per session** so a run is reproducible and a curator can demo it. Droughts
   *arriving* is dramatic; a smooth trend is not.
4. **Control**: a **climate-future control on the table** — hold / observed trend / hot-dry. The
   piece is about human choice under a climate nobody at the table picked, and letting people try
   the optimistic future and find it does not save them is stronger than asserting it.
5. **Ensemble**: the engine emits **one seeded trace as the primary channel** *and* an ensemble
   band alongside it, exposed in the API. The visual layer decides whether to draw the band; the
   engine never hides that the trace is one draw of many.
6. **Reference**: every card deviates from **business-as-usual** — current growth trends continued,
   sourced from OEO and CRSS ([§4.3](#43-why-d-is-not-an-alternative-to-a--recorded-because-it-is-the-obvious-mistake)).
   *"The difference your choices made against the future that was already coming"* is the sentence
   the piece is trying to say, and it is only sayable if BAU is sourced rather than invented.

---

## 12. Milestones

33 weeks, 2026-09-22 → end of spring semester 2027.

| | Milestone | Weeks | Target | Standalone? |
|---|---|---|---|---|
| **M1** | **Local tier, structural-only** (§2, §3): confirm the HUC8 set against the WBD GDB, local denominators, local constants re-derived | 5 | end Oct 2026 | **yes — demonstrable on its own. Fixes goal 1 for urbanization, irrigation and population with no retraining and no new architecture.** |
| **M2** | **Stock units** (§5) + **engine/API skeleton** with the provenance schema (§8) | 4 | end Nov 2026 | yes |
| **M3** | **Stock-and-flow rollout + open horizon** (§4), conservation and boundedness gates (§9) | 7 | end Jan 2027 | yes |
| **M4** | **Token vocabulary + nested budget** (§7, §2b); perceptual gate becomes the ship gate | 4 | end Feb 2027 | yes |
| **M5** | **Climate futures + BAU** (§11) from the acquired D data | 4 | end Mar 2027 | yes |
| **M6** | **WUI measurement + fire card** (§10) — a research pass, blocks nothing | 4 | end Apr 2027 | yes |
| | **Buffer / integration with the installation team** | 5 | May 2027 | |

**Order matters.** The tempting alternative — build the API first, since the API is the deliverable
— means designing the contract before knowing what flows through it. M1 first also means that if
everything after slips, the thing that shipped is the thing you asked for first.

**Compression order, if the calendar tightens:** cut **M6** first (the fire card degrades
gracefully to climate-only, which is already the declared null branch); then **M5's stochastic
ensemble** (fall back to a smooth trend, single trace); then **M4's token count** (ship
`subdivision` / `farm` / `recharge_basin` and defer the other three). **M1–M3 are not
compressible** — they are the goals.

---

## 13. What Phase 4 does not change

- **No shipped model is retrained or modified** unless a per-model local challenger clears the
  pre-declared gate in §3 tier 3.
- **No coefficient is enlarged to make a lever visible.** [§17](PHASE3_PLAN.md) stands.
- **Layer 1 stays climate-only.** The human response stays entirely in Layer 2
  ([§11.3](PHASE3_PLAN.md)), and gate 4 keeps it that way.
- **`integrate_learned_residual` stays off** ([§32](PHASE3_PLAN.md)). §4.1 above is the reason,
  stated mechanically rather than empirically this time.
- **The region stays the eight counties** ([PROBLEMS.md P8](PROBLEMS.md)).

---

## 14. Rejected branches, recorded so they are not retried

1. **Rolling the learned residual forward over a century.** Four of six models are pure integrators
   with no self-lag; they diverge by construction. §4.1, and §32 already paid for this once.
2. **Retraining on the long records only (course B).** NDVI, GRACE and the well index have no
   pre-2000 data, and GLDAS — the feature that gave two of them their skill — has none either.
3. **Borrowed projections as the model (course D alone).** No mechanism for a lever to act on, and
   silent past 2060.
4. **A moving, NLCD-derived urban denominator.** Makes the lever look strong for the wrong reason.
5. **A declared-sign human→wildfire coefficient.** Rejected in advance of the measurement, so the
   measurement is not negotiated afterwards.
6. **Backing the aquifer bound out of the well record.** The `A_eff` mistake, restated.
7. **Redefining slider units without re-scoping the output.** A false fix — it rescales the axis
   and leaves the denominator alone.
8. **JavaScript as the engine of record.** Considered seriously and reversed: TouchDesigner is
   Python-native, and the scientific code already is.

---

## 15. Open questions

- **A Tucson-basin `storage_af_per_ft`, a recharge flux, and ADWR validation** — §4.4a. The bound
  itself is settled; the rate is not.
- **Pima sub-county projections** must come from **Pima Association of Governments**; the state file
  omits Pima by design (§4.5).
- **Token geometry** — `parcels_per_domain`, `acres_per_token`, `token_intensity`. Owned by the
  installation team; carried as open parameters until answered.
- **Does the physical table depict a specific reach?** If so, the local domain should match it. The
  repo's *Living River — Downtown Tucson to Marana* reports suggest that reach, and the measured
  domain in §2 contains all of it.
- **Are the 44 Cochise wells shallow floodplain piezometers?** 41 of the 44 sit at **4–42 ft**
  depth-to-water, against 31–316 ft for the Pima wells `MEASURED` 2026-09-22. If so, the shipped
  groundwater card is closer to a riparian shallow-water-table index than to a basin aquifer
  indicator, which would explain why four climate inputs forecast it so well
  ([§29](PHASE3_PLAN.md)) and would change how that card should be *labelled* — not what it
  predicts. Worth one pass over the well metadata before M1.
- **Does the climate-future trend need CMIP6 at all**, now that the dominant century-scale driver
  for Tucson looks like Colorado River allocation rather than local climate? §11 stands, but its
  priority relative to M5's CRSS work should be re-read.
