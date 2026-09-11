# Discussion — how the inputs reach the outputs, and what we are assuming

This is the readable companion to [PHASE3_PLAN.md](PHASE3_PLAN.md). The plan is a record of what
was measured and when; this is an argument about *why the numbers look the way they do*, written
for someone who has the app open and is asking "wait, why did that happen?"

Everything here is current as of the latest run of
`python scripts/phase3/slider_sensitivity.py --mode sweep`. Where a number is an assumption rather
than a measurement, it says so.

---

## 0. Two things that trip everyone up first

Both of these have caused a "this model is broken" reaction, and neither is a bug.

**1. The sweep table reports a `min → max` swing, not "more of this thing".** Irrigation's slider
runs from −60% to +40% of baseline withdrawal, so its `surface_water` entry of **−2.37** means
*going from far less irrigation to somewhat more* costs 2.37 points of streamflow. More irrigation,
less flow. The same applies to public supply. If you read the table as "irrigation raises
streamflow" you have read the direction backwards — which is easy, and is why the app's sliders are
the better way to look at it.

**2. "Groundwater Depth vs Normal" measures depth *downward*.** `depth_to_water_anomaly_ft` is how
far you drill before hitting water. A **bigger** number means the water table is **lower**, i.e.
less water. So pumping harder makes this go **up**, and that is correct. Three of the six outputs
are named for a quantity that rises when things get worse, which is why every card now prints a
direction line under its title:

| output | bar rises → |
|---|---|
| GRACE Groundwater Anomaly | more water stored |
| NDVI Vegetation Health | greener |
| **Groundwater Depth vs Normal** | **water table DEEPER — less water** |
| Streamflow vs Normal | more flow past the gages |
| **Wildfire Risk Index** | **more burned area** |
| Bird Abundance vs Normal | more birds |

---

## 1. The whole map, in one table

Change in the 0–100 score for a full slider swing, at a 12-month scenario:

| | GRACE | NDVI | Groundwater depth | Streamflow | Wildfire | Wildlife |
|---|---|---|---|---|---|---|
| **Population** | −3.09 | 0.00 | **+9.90** | +1.74 | — | +0.43 |
| **Irrigation** | **−18.76** | **+9.30** | **+60.11** | −2.37 | — | −0.59 |
| **Public supply** | −5.63 | 0.00 | **+18.03** | −0.72 | — | −0.18 |
| **Urbanization** | 0.00 | −0.97 | 0.00 | **+3.30** | — | +0.82 |
| **Lake Mead level** | +1.32 | 0.00 | −4.22 | +0.17 | — | +0.04 |
| Precipitation | −3.38 | +4.96 | −0.20 | **+36.70** | −14.50 | +10.37 |
| Temperature | +0.18 | +0.47 | −0.31 | −0.28 | **+9.56** | +1.05 |
| Drought (PDSI) | −1.73 | −4.80 | −0.15 | −1.06 | +0.01 | **+7.35** |

Two architectural facts explain the shape of this table, and are worth stating before any
individual row:

- **The climate rows come from six trained XGBoost/Ridge models. The human rows do not.** The
  learned models are run with every human lever held at its climatological normal, and a separate
  structural layer — water balance, land-cover arithmetic, published policy — supplies the entire
  human response. This is not a stylistic choice; the panel simply cannot identify human
  coefficients. Of 21 lever/target pairs tested, 12 had no effect at all and only 3 had a stable
  sign. Letting the models "learn" human effects produced coefficients that flipped sign with the
  month.
- **The human response accumulates; the climate response does not.** Each output has a fitted
  mean-reversion rate λ, and a sustained human forcing integrates as `(s/λ)(1−(1−λ)ⁿ)`. Those rates
  differ enormously, and that difference does more work than any coefficient.

---

## 2. Memory is why the columns look so different

| output | λ per month | effective memory | consequence |
|---|---|---|---|
| Groundwater depth | **0.0157** | **~64 months** | pumping accumulates for years |
| GRACE storage | 0.0386 | ~26 months | accumulates, but less |
| NDVI | 0.2221 | ~4.5 months | a growing season, then reset |
| Streamflow | **0.3511** | **~2.8 months** | this month's weather, near enough |

An aquifer remembers; a river does not. That single fact is why irrigation moves groundwater depth
by 60 points and streamflow by 2, even though both are driven by the same pumping. Streamflow
forgets the pumping almost as fast as it happens; the water table integrates it.

It is also why **climate dominates streamflow (+36.70 for precipitation) and humans dominate
groundwater.** Those are not competing claims about which matters more in the world — they are
claims about which one a 12-month integral is capable of accumulating.

---

## 3. Output by output

### 3.1 Groundwater depth — the clean one, and the biggest human effect in the app

**Every pumping lever pushes the water table down, and nothing else does much.**

| | at slider max | in feet, 12 months |
|---|---|---|
| Irrigation +40% | +24.0 pts | **+4.62 ft deeper** |
| Public supply +60% | +10.8 pts | +2.08 ft |
| Population +3 M | +8.5 pts | +1.63 ft |
| Lake Mead −95 ft | −1.1 pts | −0.22 ft |

The mechanism is the most direct in the model and has no intermediate steps:

```
Δpumping (AF/month)  ÷  S_y·A (AF per foot)  =  Δdepth per month   →  integrated over the scenario
```

The arithmetic is checkable by hand. +40% irrigation is +1,098 MGD ≈ 1.2 million acre-feet over a
year; divided by the storage coefficient of 244,082 AF/ft that is ~5 feet of drawdown, less a
little as recharge partially catches up — 4.62 ft. Nothing in that chain is fitted to make the
answer come out.

**Why it is trustworthy.** This is one of only three `corroborated` levers, meaning the structural
mechanism *and* an independent empirical check agree on the sign after climate and trend controls.
It was measured twice: a distributed-lag regression on the project's own panel (t = +3.16), and a
horizon sweep that measured the storage coefficient at the scenario's own duration.

**Lake Mead is negative here and that is the interesting part.** A *lower* reservoir means Arizona
takes a Colorado River cut, part of which gets replaced by pumping — so a falling Mead drives the
water table down. The slider reads elevation, so raising Mead relieves pumping and the depth
recovers. It is the weakest of the four despite being the best-documented mechanism (published DCP
shortage tiers), because two untested multipliers cut it to 30% of its nominal strength. See §5.

**The honest caveat.** Groundwater's *climate* column is nearly blank (−0.20 for precipitation),
and that is partly real and partly not. Pumping genuinely is the dominant driver of Arizona water
tables. But the groundwater model also has essentially no forecast skill (+0.0086), so its climate
term is weak in the app partly because the learned layer cannot predict it. Do not read "humans
matter 300× more than rain for groundwater" off this column.

### 3.2 GRACE — the same physics, a different instrument, and they disagree

GRACE measures total water storage from orbit as an equivalent water height. Physically it should
move with the water table. In the app the two respond to the same levers with opposite signs —
irrigation is **+60.11** on groundwater depth and **−18.76** on GRACE — and *both mean less water*,
because depth is measured downward and storage upward. They agree.

Where they genuinely do not agree is in the data. The observed GRACE series runs **+0.71**
correlation with the well-depth anomaly, where physics wants negative. The most likely explanation
is a scale mismatch: GRACE's ~300 km footprint is leakage-smeared across the whole region, while
the well network sits in pumped agricultural basins. They are not measuring the same water.

**GRACE is also the only model in the project with negative forecast skill (−0.0349).** The target
is sound — a strong depletion trend (r = −0.84), correct drought and Mead signs — it simply cannot
be predicted from what we have, because monthly change in a storage integral is driven by pumping
that nobody observes monthly. Three attempts to fix that with features have now returned nulls. Its
human response, which *is* real, comes entirely from the structural layer.

### 3.3 Streamflow — the tug-of-war, and two signs that look wrong

This is the output that generates the most "that must be backwards" reactions. Every path:

| lever | sign | points at max |
|---|---|---|
| Population — effluent | + | **+1.82** |
| Population — stream capture | − | −0.34 |
| **→ net for population** | | **+1.49** |
| Urbanization — storm runoff | + | **+2.74** |
| Irrigation — stream capture | − | −0.96 |
| Public supply — stream capture | − | −0.43 |
| Lake Mead — stream capture | + | +0.04 |

**Urbanization raises streamflow, and that is textbook.** Pavement prevents infiltration, so a
larger share of every storm becomes direct runoff instead of soaking in. Urban catchments produce
more total runoff and much higher peaks. The Phase 2 feature set already carried
`precip × impervious` as the dominant urban-desert discharge term for the same reason.

**Population raises streamflow, and that is regionally specific.** The perennial reaches of the
Santa Cruz through Tucson are treated wastewater. This is not a modelling liberty — the project
carries the source: Pima County's reclamation system *"continues to produce high-quality
effluent,"* and *"releasing effluent into the river provides habitat and helps replenish the
aquifer."* More people means more effluent means more flow past the gages.

It is worth being precise about what makes this defensible rather than convenient: **population
reaches streamflow twice, through effluent (+) and through the municipal pumping it drives (−), and
the net sign is an output of the model rather than an input to it.** It survives the entire
parameter band — +0.65 to +2.14 across every corner, and still +0.65 at the corner least favourable
to it. If the two paths were close, the sign would be a coin flip and we would have to say so.

**A real weakness, recorded rather than hidden.** The effluent lever converts added discharge into
a share of total gaged flow, spreading it evenly across a 103-gage network. But the target is the
*mean over gages* of `ln(Q/Q_normal)`, and effluent does not arrive evenly — it enters a handful of
Santa Cruz reaches. Concentrated flow moves a mean-of-logs differently from the same volume spread
across everything. The direction is unaffected and the magnitude is the right order, but the
normalisation is an approximation, and a per-gage version would be the honest fix.

### 3.4 NDVI — where irrigation and urbanization point in opposite directions

| lever | points | why |
|---|---|---|
| Irrigation | **+9.30** | irrigated cropland is much greener than desert |
| Urbanization | −0.97 | pavement is not green |

**Irrigation being *good* for vegetation while being the worst thing in the app for groundwater is
the single most useful tension the tool exposes.** The same slider that drives the water table down
4.6 feet also greens the region. Any policy conversation about cutting agricultural water use has
to hold both, and a tool that showed only one would be lying by omission. This is the reason the
NDVI lever was built even though it is small.

**Urbanization's −0.97 is the most argued-over number in the model, and it has now been measured
three ways.** Cross-sectional regression across 287 months: −0.0250 NDVI per unit impervious
fraction. Difference-in-differences, each cell against its own past: −0.0356. Matched
urbanised-versus-control contrast: −0.0419. The DiD is what ships, because it removes the confound
that cities are built on valley floors rather than on a random sample of the desert.

The finding underneath it is better than the coefficient: **it is not the concrete, it is what the
concrete replaced.**

| what the cell was before paving | baseline NDVI | effect of fully paving it |
|---|---|---|
| dry desert | 0.153 | **−0.0027** (indistinguishable from zero) |
| typical | 0.238 | −0.0221 |
| green | 0.360 | −0.1224 |
| cropland / riparian | 0.520 | **−0.1792** |

A 45× difference. Paving desert costs essentially nothing, because desert NDVI is already about
what pavement reads — there is nothing to lose. Paving farmland costs 83% of that cell's entire
vegetation signal.

**So why is the app's number still under a point?** Because of dilution, which is arithmetic rather
than physics:

```
  local effect of fully paving a cell        −16% of the vegetation signal
  × the slider's entire range                = about 2% of the region's area
  = region-wide                              −0.97 points
```

The same coefficient applied to the whole eight counties would be **−40 points**. The physics is
not weak; the *instrument* is. An eight-county mean NDVI is built to average away exactly the kind
of small-area, high-intensity change urbanization is. Two calibration checks confirm the framing
rather than the coefficient is what limits it: the slider's maximum is already **four times the
entire observed 24-year change** in regional impervious cover, and the NDVI bar's 0–100 range is
climate's yardstick — drought month to monsoon month — which puts climate and urbanization at about
**123 : 1** on this output.

The card therefore now states both scales, with the regional number as the value and *"on the land
actually paved, NDVI falls 16%"* underneath it.

### 3.5 Wildlife — everything arrives through one edge

Wildlife is reached by a **transfer edge**, not by its own lever set: streamflow → bird abundance,
coefficient +0.0536. Every human lever reaches wildlife only by first moving streamflow, which is
why the whole column is small (+0.04 to +0.82) and why its signs are inherited.

The mechanism is real — riparian corridors carry disproportionate bird abundance in the Southwest —
and it was estimated on the project's own annual panel, n = 44: **+0.1953 (t = +2.19, p = 0.029)**
with drought controlled, attenuating to **+0.0536 (t = +0.42)** when a linear trend is added.
Positive in both. **The conservative end ships**, which is why wildlife is the most uncertain thing
in the application: its band spans about 6×, and the card says so.

A second edge, NDVI → wildlife, was estimated and came back a **null** (t = −0.19 / +0.17, p ≈ 0.85,
sign flips). It is deliberately not implemented. An earlier version of the code passed NDVI into the
wildlife model anyway, where it was silently dropped because wildlife's 12 features do not include
it — a claim the interface was making and the arithmetic was not.

### 3.6 Wildfire — a deliberate blank

**No human lever reaches wildfire, and this is a finding rather than an omission.** Human ignitions
are the majority of US fire *counts*, but ignition is not the limiting factor for large-fire
*extent* in the Southwest — fuel and weather are — and the target (MTBS burned area) measures
extent. No defensible coefficient from population or impervious cover to large-fire extent could be
sourced, and the panel measured both at under 0.05 sd.

So it ships as `climate-only`, declared in the parameters, and the card says *"no structural path to
this output"* rather than showing a row of zeros that looks like a bug. The model itself is fine —
+0.3716 skill, one of the better ones. It is simply a climate story.

---

## 4. Why irrigation dominates everything

Irrigation is the only lever that reaches four outputs, and it is the largest effect in three of
them. Two reasons, and they compound:

1. **Scale.** The 2020 deseasonalized baseline is **2,744 MGD**, against 823 MGD for public supply.
   Agriculture is most of the region's water use, so a percentage change in it is a much larger
   absolute change than the same percentage anywhere else.
2. **It is the only lever with two independent lines of evidence**, so it is not attenuated by the
   conservatism applied to everything else. Where a lever rests on an untested multiplier, that
   multiplier is set at a defensible middle and the band is wide; irrigation was measured.

The tension it creates is the point of the whole tool: **cutting irrigation raises the water table,
restores GRACE storage, slightly increases streamflow — and makes the region browner.**

---

## 5. What we are assuming, and where it hurts

Of the twelve constants shipping with a declared band, **ten still feed a lever and seven of those
are `UNTESTED`.** The three that are measured are the three that were argued about most.

| constant | value | band | status | what it decides |
|---|---|---|---|---|
| `storage_af_per_ft` | 244,082 | 184,023 – 289,336 | **MEASURED** | every groundwater lever |
| `ndvi_impervious` | 0.1811 | 0.1710 – 0.2075 | **MEASURED** | urbanization → NDVI |
| `stream_capture_fraction` | 0.10 | 0.05 – 0.25 | UNTESTED | all four negative streamflow paths |
| `effluent_return_fraction` | 0.55 | 0.45 – 0.70 | UNTESTED | population → streamflow |
| `region_share_of_az_reduction` | 0.60 | 0.40 – 0.80 | UNTESTED | every Lake Mead path |
| `groundwater_substitution_fraction` | 0.50 | 0.30 – 0.70 | UNTESTED | every Lake Mead path |
| `ndvi_irrigated_crop` | 0.55 | 0.45 – 0.65 | UNTESTED | irrigation → NDVI |
| `runoff_coefficient_impervious` / `_natural` | 0.85 / 0.15 | 0.75–0.95 / 0.05–0.25 | UNTESTED | urbanization → streamflow |
| `transfer_surface_water_to_wildlife` | 0.0536 | 0.0536 – 0.1953 | MEASURED | the entire wildlife column |

**Where the assumptions hurt most:**

- **Lake Mead is the best-documented mechanism and the weakest lever in the app**, because its two
  untested multipliers multiply: 0.60 × 0.50 = 30% of nominal. Its effect on groundwater spans
  **7×** across the band (+0.9 to +6.5 points). That lever is not *small*, it is *unknown* — a
  materially different statement, and the reason the card shows the range rather than just the
  point.
- **Wildlife spans about 6×**, because it compounds the transfer edge's own band with every
  upstream streamflow parameter.
- **`ndvi_irrigated_crop` is the one endpoint still assumed**, and it is not for lack of trying:
  unlike impervious cover, there is no cropland mask in the repo to regress against. The in-repo
  route is HUC12 irrigation withdrawal against HUC12-mean NDVI, which is real but coarser.

Two decisions deliberately *not* revisited, both written down so they are not quietly reversed
later: `storage_af_per_ft` was measured at the scenario's own duration rather than picked from
either physical limit, and the NDVI coefficient came from the design that identifies it best rather
than the one that gives the biggest number. In both cases the answer came out larger, and in both
cases the reason for the choice was fixed before the number was known.

---

## 6. The lesson that keeps recurring

This project has now hit the same shape of problem four times, and it is worth naming because it
will happen again:

> **Everything gets scored on whatever was easy to score, and the thing that actually matters goes
> unmeasured until someone looks.**

- Every model was scored on *forecast accuracy*, so nobody noticed that the sliders moved the
  outputs by at most 1.7 points — the entire premise of the interface — until it was measured.
- The aquifer storage coefficient was shipping at the end of a 16.9× range that was 7.7× past
  anything the data supported, because the range was treated as an uncertainty rather than a
  question.
- Urbanization's effect was measured against the desert *around* cities rather than against the
  same land *before* it was paved, which conflated paving with siting.
- Every check ran under Node, so five green gates certified a frontend that could not start in a
  browser at all.

The countermeasure is the same each time: **ask what the number would look like if the thing you
are not measuring were badly wrong, and then go measure that.**

---

## 7. Open questions

Ranked, with the reasoning, in [PHASE3_PLAN.md §16](PHASE3_PLAN.md). In short:

1. **`ndvi_irrigated_crop`** — the last assumed NDVI endpoint; needs the HUC12 route.
2. **CAP monthly deliveries** — one acquisition that serves two unrelated problems: the only
   remaining monthly pumping proxy for GRACE, and the county-level data
   `region_share_of_az_reduction` explicitly asks for.
3. **Corroborate `stream_capture_fraction` and `effluent_return_fraction`** with the method that
   already worked for irrigation. These are the widest bands on the streamflow card.
4. **An urban-footprint NDVI output** — proposed, not built. It would move hard under the
   urbanization slider because it removes the area dilution that flattens the regional number. The
   blockers are real: no learned model predicts it, a moving mask changes the denominator, and
   climate would still dominate it.

---

*Every number here is reproducible from the repo. The sweep is
`python scripts/phase3/slider_sensitivity.py --mode sweep`; the gates are listed in the
[README](README.md); the measurements behind each constant are in `model/*.json` with the script
that produced them named in the constant's own `source` field.*
