# Policy Scenario Proposal

Date: 2026-06-03

## Purpose

This proposal shifts the frontend from seven low-level hydrology and demand knobs into a smaller policy-oriented scenario system.

The current app exposes model-facing variables directly:

- `powell_pool_elevation`
- `snow_water_equivalent_in`
- `precipitation_mm_day`
- `temperature_2m_c`
- `irrigation_total_withdrawal_mgd`
- `public_supply_groundwater_mgd`
- `grace_groundwater_anomaly`

Those controls are useful for testing the model, but they are not how planners, residents, or elected officials usually think about water policy. The revised interface should expose five major attributes that describe recognizable water-policy conditions. Each attribute then translates into the existing model features under the hood.

The goal is not to make the model pretend to decide policy automatically. The goal is to make it easier to ask:

> Under a scenario like this, which policy package appears most relevant, and what model drivers does it affect?

## Proposed Interaction Model

The public interface should have two layers:

1. Five high-level policy attributes
2. Four scenario modes that set the five attributes to predefined policy bundles

The existing raw feature knobs can remain available as an advanced/debug view, but they should no longer be the main experience.

## Five Major Attributes

All five attributes should use a simple `0-100` scale in the UI. A higher value means "more of this condition." The runtime translates those values into model-facing feature changes using the existing feature metadata, historical medians, and p5/p95 bounds.

| Attribute | User-facing meaning | Primary model features affected | Policy interpretation |
|---|---|---|---|
| Urban Expansion Pressure | How quickly population, development intensity, and municipal water demand grow | `population` / `AZPOP_pct_change`, `public_supply_groundwater_mgd`, optional future urban-heat adjustment to `temperature_2m_c` | Growth management, development review, water availability checks, water impact fees, low-water building/site standards |
| Water-Intensive Agriculture And Turf | How much demand comes from high-water crops, turf, golf courses, large HOA landscapes, and irrigated open areas | `irrigation_total_withdrawal_mgd`, optional `ndvi`, slow-pressure adjustment to `grace_groundwater_anomaly` | Crop choice, turf conversion, reclaimed water for large turf areas, HOA water budgeting, non-potable irrigation |
| Conservation And Efficiency | How strongly homes, businesses, HOAs, and public facilities reduce avoidable water use | Decreases `public_supply_groundwater_mgd`; can also decrease `irrigation_total_withdrawal_mgd` | Rebates, audits, smart controllers, low-flow fixtures, drought education, enforcement of water-waste rules |
| Reuse And Aquifer Recharge | How much reclaimed water, recharge capacity, stormwater capture, and non-potable substitution are used | Increases or stabilizes `grace_groundwater_anomaly`; can reduce `public_supply_groundwater_mgd` and `irrigation_total_withdrawal_mgd` | Water reclamation expansion, recharge facility expansion, reclaimed water goals, non-potable irrigation systems |
| Hot-Dry Watershed Stress | Severity of heat, low precipitation, poor snowpack, reservoir stress, and regional hydrologic shortage | Increases `temperature_2m_c`; decreases `precipitation_mm_day`, `snow_water_equivalent_in`, `powell_pool_elevation`; optional decrease to `streamflow_cfs` and `grace_groundwater_anomaly` | External climate and watershed context that determines how aggressive the policy response needs to be |

## General Plan Basis For The Five Attributes

Each high-level attribute should be traceable to a policy concern or implementation strategy in the Avondale General Plan 2030. Some attributes represent desired policy action, while others represent stressors the plan is trying to manage. The policy references below come from the local `General_Plan_2030.pdf`. Main source sections are Resilience to Extreme Heat, pp. 81-83; Environmental Planning & Conservation, pp. 84-88; Public Buildings, Services and Facilities, pp. 117-118; Water Resources, pp. 125-129; and Implementation Strategies, pp. 151 and 159-160.

| Attribute | Specific General Plan policies and strategies | Why this supports the knob |
|---|---|---|
| Urban Expansion Pressure | Water Resources Element Goal 2, Policies A-B and F: plan potable/non-potable systems for residential, commercial, and industrial growth; update and implement the Water Resource Master Plan; maintain control of water-resource opportunities. Water Resources Goal 3, Policies A and C: balance growth with water/reclaimed-water supplies and guide land-use decisions to conserve water. Water Resources Goal 1, Policy C: use water, water-resource, and development fees so new growth helps pay for capital improvements. CF/WR Strategies 1, 9, 10, and 13: regularly update water master plans, implement water-conserving land-use policies, require ecological/groundwater impact statements for rezonings and plan amendments, and evaluate new development against water availability and wastewater treatment capacity. Public Buildings, Services and Facilities Element Goal 4, Policy B: match water and wastewater capacity to current and future community needs. | This knob represents the pressure created by population growth, development intensity, and municipal water demand. The plan does not say "stop growth"; it says growth should be checked against water supply, wastewater capacity, reclaimed-water supply, and capital planning. |
| Water-Intensive Agriculture And Turf | CF/WR Strategy 5: consider requiring golf courses and large turfed areas to use reclaimed or other non-potable water for irrigation. Water Resources Goal 3, Policy B: support water-use efficiency and reclaimed-water reuse efficiency. Water Resources Goal 4, Policy C: expand conservation incentives and education for non-residential users including HOAs, schools, public facilities, and commercial users. Environmental Planning & Conservation Element Goal 7, Policy B: explore non-potable water for landscaping. Environmental Planning & Conservation Goal 9, Policies A-B: support gardens and sustainable agriculture businesses. Water Conservation Programs and Landscaping Ordinance: xeriscape and irrigation timer rebates, landscape consultations, HOA water budgeting assistance, native/low-water plant requirements, and required water conservation plans for new developments. | This knob should represent irrigated-landscape and crop-demand pressure, not an endorsed policy goal. The plan supports local food systems and landscapes, but it repeatedly points toward low-water plants, efficient irrigation, non-potable substitution, turf conversion, and special attention to large turf/golf/HOA/non-residential users. |
| Conservation And Efficiency | Water Resources Goal 4, Policies A-C: encourage conservation through education, technical assistance, outreach, incentives, and programs for residential and non-residential users. Water Conservation Programs: xeriscape landscape rebates, irrigation timer rebates, plumbing rebates, high-efficiency clothes washer rebates, landscape consultations, HOA water budgeting assistance, water-wasting investigations, low-water-use fixtures, homeowner classes, school assemblies, xeriscape demonstrations, educator resources, Project WET, and self-audit kits. Water Conservation Regulation: Water Use Plans, Waste of Water Prohibited, and water rate structure. SD-EP Strategy 14: continue low-flow showerheads, low-flow toilet rebates, turf conversions, smart controllers, school/adult education, home audits, and printed materials. CF/WR Strategy 12: continue resident programs, techniques, and education for water conservation. Public Buildings, Services and Facilities Goal 5, Policy D: incorporate energy and water conservation measures into public buildings and facilities. | This knob is the direct policy-action knob for reducing avoidable household, commercial, HOA, and public-facility demand. It maps cleanly to lower public-supply groundwater demand and lower outdoor irrigation demand. |
| Reuse And Aquifer Recharge | Water Resources Goal 1, Policies B and D: maximize beneficial use of reclaimed water and coordinate regionally on effluent. Water Resources Goal 2, Policies A, D, and F: plan potable/non-potable systems, coordinate with agencies and neighboring communities, and maintain water-resource opportunities. Water Resources Goal 3, Policies A-B: balance growth with water and reclaimed-water supplies and support reuse efficiency. Environmental Planning & Conservation Goal 7, Policies A-B: balance groundwater pumping and replenishment using renewable supplies and explore non-potable water for landscaping. Public Buildings, Services and Facilities Goal 4, Policy C: ensure wastewater reclamation facilities meet effluent-recharge requirements. CF/WR Strategies 2-4: expand water reclamation facilities, investigate recharge-capacity expansion to maximize reclaimed-water reuse, and pursue regional reclaimed-water goals. CF/WR Strategy 5 also supports non-potable irrigation substitution for large turf. | This knob is the plan's strongest direct tie to aquifer health. The plan describes the McDowell Recharge Facility, four recharge facilities, reclaimed-water recharge, Crystal Gardens Wetlands treatment, and Water Reclamation Facility expansion as part of Avondale's long-term water strategy. |
| Hot-Dry Watershed Stress | Resilience to Extreme Heat Element Goal 1, Policies A-E: tree canopy, cool pavement/cool roofs, reliable access to air-conditioned spaces and water during heat events, heat mitigation in the Hazard Mitigation Plan, and shade-canopy zoning for new development. Resilience to Extreme Heat Goal 2, Policies A-E: vulnerable-population support, weatherization and energy assistance, portable AC emergency support, interagency coordination, and more cooling centers. Water Conservation Programs section: Stage 1 Drought Preparedness uses an education-first approach focused on wise water use and water-efficient practices. Environmental Planning & Conservation Element stormwater discussion and Goal 3, Policy B: prevent stormwater pollution; low-impact development can use stormwater as a resource and mitigate heat and flooding. Public Buildings, Services and Facilities Goal 3, Policies A-D and CF/WR Strategies 14-16: flood/stormwater management, updated flood studies, regional flood coordination, and warning information. | This knob is not a local policy lever; it represents external climate and watershed stress. The plan ties that stress to heat, drought preparedness, stormwater, water access, flood planning, shade, tree canopy, and emergency resilience. In the model it should increase temperature and reduce precipitation, snowpack, reservoir level, streamflow, and aquifer health. |

## Suggested Under-The-Hood Mapping

The simplest implementation is to keep the ONNX model unchanged and add a translation layer before projection.

Each high-level attribute creates feature deltas from the July 2020 baseline or current selected baseline. The deltas should be clamped to the existing p5/p95 ranges from `display_metadata.json` so the app does not generate unrealistic inputs too easily.

Suggested mapping rules:

| Attribute | At 0 | At 50 | At 100 |
|---|---|---|---|
| Urban Expansion Pressure | Flat or low population growth; lower public-supply demand | Baseline growth and public demand | High population growth; high public-supply groundwater demand |
| Water-Intensive Agriculture And Turf | Low irrigation withdrawal; low turf/crop pressure | Baseline irrigation demand | High irrigation withdrawal; high turf/crop pressure |
| Conservation And Efficiency | Minimal conservation response | Existing conservation programs | Aggressive demand reduction across residential, HOA, commercial, and public uses |
| Reuse And Aquifer Recharge | Little extra reuse/recharge beyond baseline | Existing reuse/recharge operations | Strong recharge, reclaimed water, and non-potable substitution |
| Hot-Dry Watershed Stress | Cooler/wetter conditions; stronger snowpack and reservoir context | Baseline watershed conditions | Hotter/drier conditions; lower snowpack, precipitation, Powell, streamflow, and aquifer health |

When multiple attributes affect the same feature, combine their effects additively and then clamp:

```text
feature_value = clamp(
  baseline_value + urban_delta + agriculture_delta + conservation_delta + reuse_delta + climate_delta,
  feature_p5,
  feature_p95
)
```

Priority notes:

- Conservation should reduce demand features, not climate features.
- Reuse and recharge should improve groundwater conditions slowly, not instantly solve drought.
- Hot-Dry Watershed Stress should be treated as an external condition, not a policy lever.
- Urban Expansion Pressure should increase demand even when conservation is also high, but conservation can offset some of that increase.
- Water-Intensive Agriculture And Turf should combine agricultural demand and large irrigated-landscape demand because both map cleanly to irrigation pressure.

## Four Scenario Modes

Scenario modes are presets. Selecting a mode sets the five attributes, then the user can still fine-tune them manually.

| Scenario mode | Urban Expansion Pressure | Water-Intensive Agriculture And Turf | Conservation And Efficiency | Reuse And Aquifer Recharge | Hot-Dry Watershed Stress |
|---|---:|---:|---:|---:|---:|
| Current Trajectory | 45 | 45 | 40 | 45 | 45 |
| Growth-First Buildout | 85 | 70 | 25 | 40 | 50 |
| Water-Wise Buildout | 55 | 25 | 85 | 85 | 55 |
| Drought Emergency Response | 35 | 15 | 95 | 80 | 90 |

### Current Trajectory

This mode represents continuation of current programs and moderate growth.

Policy bundle:

- Maintain existing conservation rebates and education.
- Continue water availability review for new development.
- Continue existing recharge and reclaimed-water operations.
- Keep Stage 1 drought preparedness education as the primary public response.
- Treat large turf and agricultural demand as present but not aggressively reduced.

General Plan basis:

- Water Resources Element Goal 4 backs ongoing conservation education, technical assistance, outreach, and incentives.
- The Water Conservation Programs section backs rebates, landscape consultations, HOA water budgeting, water audits, low-water fixtures, and school/adult education.
- The Stage 1 Drought Preparedness discussion backs an education-first current-response scenario.
- CF/WR Strategies 1, 12, and 13 back regular water master planning, resident conservation programs, and development review against water availability and treatment capacity.

Best use:

- Baseline comparison
- Public education
- Showing how much stronger policy action would change the scenario

### Growth-First Buildout

This mode represents rapid development and high water demand with limited additional conservation.

Policy bundle:

- Prioritize near-term housing, commercial, and employment growth.
- Expand water and wastewater infrastructure to serve growth.
- Keep existing conservation requirements but avoid major new restrictions.
- Allow substantial irrigated landscapes, turf, or water-intensive crops unless separately mitigated.
- Use development fees and infrastructure planning as the main response.

General Plan basis:

- Water Resources Element Goal 2 backs planning potable and non-potable systems for continued residential, commercial, and industrial growth.
- Water Resources Element Goal 3 backs providing water and wastewater services to newly developing areas while balancing growth with water and reclaimed-water supplies.
- Water Resources Element Goal 1, Policy C backs using water, water resources, and development fees for capital improvements related to growth.
- CF/WR Strategy 13 backs evaluating new development needs against water availability and wastewater treatment capacity.
- This mode is a stress-test scenario, not the plan's preferred outcome. It intentionally reduces the conservation and reuse emphasis so users can compare it against the more plan-aligned Water-Wise Buildout mode.

Best use:

- Testing whether growth assumptions push the system toward drought vulnerability
- Showing why water-impact review matters
- Comparing demand expansion against available water and recharge capacity

### Water-Wise Buildout

This mode represents continued growth paired with aggressive conservation, reuse, and recharge.

Policy bundle:

- Require stronger water conservation plans for new development.
- Expand turf conversion, smart controller, and low-flow fixture incentives.
- Increase HOA water budgeting and landscape consultations.
- Require or incentivize reclaimed/non-potable water for golf courses and large turf areas where practical.
- Expand water reclamation and recharge capacity.
- Use groundwater/ecological impact statements in rezoning and plan amendment review.

General Plan basis:

- Water Resources Element Goal 3, Policies A-C back balancing growth with water supplies, reuse efficiency, and land-use decisions that conserve water resources.
- Water Resources Element Goal 4 backs efficient water use and conservation throughout the community.
- Environmental Planning & Conservation Element Goal 7 backs balancing groundwater pumping and replenishment and exploring non-potable water for landscaping.
- CF/WR Strategies 2, 3, and 5 back water reclamation expansion, recharge-capacity evaluation, and reclaimed or non-potable irrigation for golf courses and large turf areas.
- CF/WR Strategies 9, 10, 12, and 13 back land-use conservation policies, groundwater/ecological impact statements, conservation education, and water-availability review for new development.

Best use:

- Showing a constructive policy path that does not require stopping growth
- Demonstrating demand-offset strategies
- Turning the model into a planning conversation tool

### Drought Emergency Response

This mode represents severe hot-dry watershed stress and an aggressive municipal/regional response.

Policy bundle:

- Escalate drought preparedness measures beyond education-only outreach.
- Reduce nonessential outdoor water use.
- Prioritize public messaging, public dashboards, and neighborhood-level participation.
- Accelerate turf conversion and water-waste enforcement.
- Maximize reclaimed water, recharge, and non-potable substitution where feasible.
- Coordinate with regional partners on reclaimed water, shortage response, and flood/stormwater planning.

General Plan basis:

- The Stage 1 Drought Preparedness discussion backs drought-response framing and wise-water-use education.
- Water Resources Element Goal 4 and CF/WR Strategy 12 back aggressive conservation outreach, incentives, and practical public guidance.
- Environmental Planning & Conservation Element Goal 7 and CF/WR Strategies 2-4 back reclaimed water, recharge, and regional effluent coordination.
- Resilience to Extreme Heat Element Goal 1 backs heat mitigation, reliable access to water during heat events, hazard mitigation, and development standards tied to shade.
- Resilience to Extreme Heat Element Goal 2 backs emergency resilience, vulnerable-population support, interagency coordination, and cooling centers.
- Public Participation Element Goal 2 and QL/PP Strategies 6-8 back public communication, regional planning links, and partnerships with surrounding communities.
- The exact emergency restrictions should be checked against Avondale's separate Drought Preparedness Plan before being presented as official City policy.

Best use:

- Stress testing
- Emergency planning
- Public workshops about tradeoffs during prolonged drought

## General Plan Policy Hooks

The scenario system should explicitly connect back to the Avondale General Plan 2030 so the visualization feels policy-relevant rather than only educational.

Useful hooks from the plan:

- Review and update the Water Resources Master Plan and Water Infrastructure Master Plan regularly.
- Expand water reclamation facilities as needed.
- Investigate increasing recharge capacity to maximize reuse of reclaimed water and improve flexibility for receiving surface water.
- Consider requiring golf courses and large turf areas to use reclaimed or other non-potable water where practical and cost-effective.
- Continue implementing land-use policies that conserve water resources.
- Require statements describing impacts to ecological systems and groundwater supplies as part of rezoning and plan amendment requests.
- Provide residents with programs, techniques, and education to conserve water.
- Evaluate new development needs against water availability and wastewater treatment capacity.
- Develop GIS-based public information tools for planning applications and regional planning participation.

## Frontend Changes

Recommended UI structure:

1. Replace the seven visible raw knobs with five larger policy-attribute knobs.
2. Add a scenario mode control with four options:
   - Current Trajectory
   - Growth-First Buildout
   - Water-Wise Buildout
   - Drought Emergency Response
3. Add a compact "policy effects" panel that shows:
   - active scenario mode
   - five attribute values
   - underlying model features being changed
   - relevant General Plan strategies
   - whether the scenario is within historical bounds
4. Keep the existing historical/projection time series.
5. Keep the cross-section visual, but drive it from the translated low-level features.
6. Keep raw model knobs behind an "advanced" or "model inputs" view for debugging and instructor review.

## Runtime Changes

Recommended data additions:

- `frontend/public/model/policy_attributes.json`
- `frontend/public/model/scenario_modes.json`

`policy_attributes.json` should define:

- attribute id
- label
- description
- default value
- feature mappings
- policy notes
- General Plan references

`scenario_modes.json` should define:

- mode id
- label
- description
- policy bundle
- five attribute values

The runtime should then:

1. Store `policyAttributeValues` as the main UI state.
2. Convert those values into the existing `controlValues`.
3. Feed the existing projection engine without changing the ONNX contract.
4. Store both the high-level attributes and translated low-level controls with each projected point.
5. Display both layers in the point-detail legend.

## Important Framing

The model target is an inverted U.S. Drought Monitor score, not a direct measurement of long-term water sustainability. The app should say, in plain language, that the projection is a decision-support signal based on historical drought, climate, hydrology, and demand patterns.

The strongest public claim should be:

> This tool helps compare policy scenarios and identify which water-management levers are most relevant under different growth and drought conditions.

Avoid claiming:

> This tool proves that a specific policy will cause a specific sustainability score.

## Open Questions

- Should population return as a model-facing controlled feature under Urban Expansion Pressure?
- Should `streamflow_cfs` become indirectly controlled by Hot-Dry Watershed Stress, or remain fixed as a non-knob model feature?
- Should NDVI be used as a proxy for irrigated landscape intensity, ecosystem condition, or both?
- Should the four scenario modes be framed as city policy choices, regional futures, or workshop presets?
- Should the app include a staff-report export for planning meetings?

## Recommended Next Step

Implement the policy layer as metadata first, without retraining the model. That keeps the current strong ONNX model intact while making the interface more actionable.

The first build target should be:

- five policy knobs
- four mode buttons
- translation from attributes to existing control values
- a small explanation panel showing which existing model features changed
- raw seven-knob controls preserved only as an advanced view
